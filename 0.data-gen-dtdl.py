#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mass-scale DTDL generator via Ollama (batched, resumable, flushed JSONL).

What it does
------------
1) Discovers digital-twin topics in batches until reaching --target-topics.
   - Writes each topic to topics.jsonl (flush + fsync).
   - Deduplicates by id/title across runs.
2) For each topic, asks the model to emit DTDL v2 interfaces (no fixed count).
   - Writes ONLY the raw Interface object per line to interfaces.jsonl (flush + fsync).
   - Forces dockerImage.value to "registry.local/dtm/<topic_id>/<component_name>:v1.0.0".
   - Marks topic done in interfaces_done.txt (flush + fsync).

Run examples
------------
# Build 10,000 topics in batches of 50, then generate interfaces:
python dtdl_mass_generator.py --target-topics 10000 --topics-batch 50

# Seed from a file (e.g., CLS350 CDI description) + build the rest:
python dtdl_mass_generator.py --target-topics 10000 --topics-batch 50 --seed-file cls350_seed.txt

# Resume later (reads existing files and continues):
python dtdl_mass_generator.py --resume
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Set
from string import Template

import requests

DEFAULT_HOST = "http://10.1.1.1:60002"
DEFAULT_MODEL = "gpt-oss:120b"

TOPICS_JSONL = "topics.jsonl"
INTERFACES_JSONL = "interfaces.jsonl"
DONE_TOPICS_LOG = "interfaces_done.txt"

# ---------------------------
# Ollama API
# ---------------------------

def ollama_generate(host: str, model: str, prompt: str, timeout: int = 180) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    print(f"[DEBUG] POST {url} (model={model})")
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Ollama request failed: {e}", file=sys.stderr)
        raise
    data = r.json()
    text = data.get("response", "")
    print(f"[DEBUG] Ollama response chars: {len(text)}")
    return text

# ---------------------------
# Prompt templates
# ---------------------------

TOPIC_PROMPT = Template(
    """You are a domain expert in digital-twin modeling.
Propose a diverse set of NEW digital-twin topics across vehicles, energy, manufacturing, buildings, healthcare devices, robotics, and environment.

Avoid duplicates with these known ids and titles:
- ids to avoid: $avoid_ids
- titles to avoid: $avoid_titles

Return STRICT JSON ONLY (no markdown, no commentary).

Output format:
[
  {"id":"short_snake_case_id","title":"Human Friendly Title","brief":"1-sentence what to twin"},
  ...
]

Constraints:
- $num_topics total topics (exactly this many)
- Non-overlapping, clearly distinct domains
- ids must be lowercase snake_case, 2–5 words
- Titles concise (<= 6 words)
- Keep "brief" <= 18 words
"""
)

INTERFACE_PROMPT = Template(
    """You are generating DTDL v2 Interface definitions for the digital twin topic below.

Topic:
id=$topic_id
title=$topic_title
brief=$topic_brief

OUTPUT REQUIREMENTS:
- Return STRICT JSONL ONLY (no markdown, no extra text).
- Each line is ONE valid DTDL v2 Interface JSON object (no wrapper fields).
- Emit as many distinct component interfaces as make sense (no fixed limit).
- Use 2–3 sentences of detailed, specific description for each interface.

DTDL RULES FOR EACH INTERFACE OBJECT:
{
  "@context": "dtmi:dtdl:context;2",
  "@id": "dtmi:$topic_id:<component_name>;1",
  "@type": "Interface",
  "displayName": "<ComponentModelName>",
  "description": "<2–3 sentences describing what this component twins in detail>",
  "contents": [
    {"@type":"Property","name":"dockerImage","schema":"string","value":"registry.local/dtm/$topic_id/<component_name>:v1.0.0"},
    <other_properties>,
    <other_telemetries>
  ]
}

ADDITIONAL CONSTRAINTS:
- The 'dockerImage' property MUST include the 'value' exactly like above (dummy path).
- Use meaningful component names; keep JSON valid per line; no comments/fences/trailing commas.
- Generate 2-5 other reasonable properties and 2-5 reasonable telemetries based on the current digital twin interface's nature
"""
)

SEED_TO_TOPIC_PROMPT = Template(
    """You are making a single digital-twin topic object from the following seed description:

---
$seed_text
---

Return STRICT JSON ONLY (no markdown), format:
{
  "id": "short_snake_case_id",
  "title": "Seeded Title (<= 6 words)",
  "brief": "1-sentence summary (<= 18 words)"
}
Guidelines:
- id must reflect the seed (e.g., cls350_cdi_2011_uk)
- concise and accurate
"""
)

# ---------------------------
# Helpers
# ---------------------------

def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def extract_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Fallback to first [...] block
    start, end = text.find('['), text.rfind(']')
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    raise ValueError("Failed to extract JSON array from model output.")

def iter_jsonl_objects(text: str):
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception as e:
            print(f"[WARN] JSONL parse failed at line {lineno}: {e}")

def looks_like_interface(obj: Dict[str, Any]) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("@type") == "Interface"
        and obj.get("@context") == "dtmi:dtdl:context;2"
        and isinstance(obj.get("contents"), list)
        and isinstance(obj.get("@id"), str)
        and isinstance(obj.get("displayName"), str)
        and isinstance(obj.get("description"), str)
    )

COMP_RE = re.compile(r"^dtmi:([^:;]+):([^;]+);[0-9]+$")

def parse_component_from_id(dtmi_id: str) -> str:
    m = COMP_RE.match(dtmi_id.strip())
    if not m:
        return ""
    return m.group(2)

def force_fill_docker_image_value(interface_obj: Dict[str, Any], topic_id: str) -> None:
    dtmi_id = interface_obj.get("@id", "")
    component = parse_component_from_id(dtmi_id) or "component"
    desired = f"registry.local/dtm/{topic_id}/{component}:v1.0.0"
    for entry in interface_obj.get("contents", []):
        if entry.get("@type") == "Property" and entry.get("name") == "dockerImage" and entry.get("schema") == "string":
            if not entry.get("value"):
                entry["value"] = desired
            return

def fsync_append_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def fsync_append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())

# ---------------------------
# Disk state (resume)
# ---------------------------

def load_existing_topics(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                topics.append(json.loads(line))
            except Exception:
                continue
    return topics

def load_done_topics(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.add(s)
    return out

# ---------------------------
# Topic discovery (batched)
# ---------------------------

def deduplicate_topics(incoming: List[Dict[str, str]], seen_ids: Set[str], seen_titles: Set[str]) -> List[Dict[str, str]]:
    unique: List[Dict[str, str]] = []
    for t in incoming:
        if not isinstance(t, dict):
            continue
        tid, title, brief = t.get("id"), t.get("title"), t.get("brief")
        if not (isinstance(tid, str) and isinstance(title, str) and isinstance(brief, str)):
            continue
        nid, nt = normalize(tid), normalize(title)
        if " " in nid or "-" in nid:
            continue
        if nid in seen_ids or nt in seen_titles:
            continue
        seen_ids.add(nid)
        seen_titles.add(nt)
        unique.append({"id": nid, "title": title.strip(), "brief": brief.strip()})
    return unique

def discover_topics_batch(host: str, model: str, batch_size: int, avoid_ids: List[str], avoid_titles: List[str]) -> List[Dict[str, str]]:
    prompt = TOPIC_PROMPT.substitute(
        num_topics=batch_size,
        avoid_ids=", ".join(avoid_ids) or "(none)",
        avoid_titles=", ".join(avoid_titles) or "(none)",
    )
    text = ollama_generate(host, model, prompt)
    arr = extract_json_array(text)
    return [t for t in arr if isinstance(t, dict)]

# ---------------------------
# Interface generation
# ---------------------------

def generate_interfaces_for_topic(host: str, model: str, topic: Dict[str, str]) -> List[Dict[str, Any]]:
    tid, title, brief = topic["id"], topic["title"], topic["brief"]
    prompt = INTERFACE_PROMPT.substitute(topic_id=tid, topic_title=title, topic_brief=brief)
    print(f"[STEP] Generating interfaces for topic '{tid}' ...")
    text = ollama_generate(host, model, prompt)
    results: List[Dict[str, Any]] = []
    for obj in iter_jsonl_objects(text):
        if looks_like_interface(obj):
            force_fill_docker_image_value(obj, tid)
            results.append(obj)
    print(f"[DEBUG] Valid interfaces parsed: {len(results)}")
    return results

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Mass-scale DTDL generator via Ollama (batched/resumable).")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host base URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--target-topics", type=int, default=6000, help="Total unique topics to reach")
    ap.add_argument("--topics-batch", type=int, default=50, help="Topics requested per model call")
    ap.add_argument("--seed-file", default=None, help="Optional path to seed a first topic (e.g., CLS350 CDI description)")
    ap.add_argument("--resume", default=True, action="store_true", help="Resume from existing files if present")
    ap.add_argument("--topics-file", default=TOPICS_JSONL, help="Path to topics JSONL")
    ap.add_argument("--interfaces-file", default=INTERFACES_JSONL, help="Path to interfaces JSONL")
    ap.add_argument("--done-file", default=DONE_TOPICS_LOG, help="Path to processed topics log")
    args = ap.parse_args()

    print("[CONFIG] Host:", args.host)
    print("[CONFIG] Model:", args.model)
    print("[CONFIG] Target topics:", args.target_topics)
    print("[CONFIG] Topics per call:", args.topics_batch)
    print("[CONFIG] Files:", args.topics_file, args.interfaces_file, args.done_file)
    print("[CONFIG] Resume:", args.resume)
    print("[CONFIG] Seed file:", args.seed_file)

    # Load / init state
    if args.resume:
        existing_topics = load_existing_topics(args.topics_file)
        done_topics = load_done_topics(args.done_file)
    else:
        existing_topics, done_topics = [], set()
        # clear files
        open(args.topics_file, "w", encoding="utf-8").close()
        open(args.interfaces_file, "w", encoding="utf-8").close()
        open(args.done_file, "w", encoding="utf-8").close()
        print("[STEP] Initialized output files (fresh run).")

    # Build sets for dedup
    seen_ids: Set[str] = set()
    seen_titles: Set[str] = set()
    topics: List[Dict[str, str]] = []

    for t in existing_topics:
        tid, title = t.get("id"), t.get("title")
        if isinstance(tid, str) and isinstance(title, str):
            seen_ids.add(normalize(tid))
            seen_titles.add(normalize(title))
            topics.append({"id": normalize(tid), "title": title, "brief": t.get("brief","")})

    # Seed
    if args.seed_file and os.path.exists(args.seed_file):
        with open(args.seed_file, "r", encoding="utf-8") as fh:
            seed_text = fh.read()
        seed_prompt = SEED_TO_TOPIC_PROMPT.substitute(seed_text=seed_text.strip())
        try:
            seed_resp = ollama_generate(args.host, args.model, seed_prompt)
            seed_obj = json.loads(seed_resp.strip())
            sid, stitle, sbrief = normalize(seed_obj["id"]), seed_obj["title"].strip(), seed_obj["brief"].strip()
            if sid not in seen_ids and normalize(stitle) not in seen_titles:
                seen_ids.add(sid)
                seen_titles.add(normalize(stitle))
                seed_topic = {"id": sid, "title": stitle, "brief": sbrief}
                topics.append(seed_topic)
                fsync_append_line(args.topics_file, seed_topic)
                print(f"[STEP] Seed topic added: {seed_topic}")
        except Exception as e:
            print(f"[WARN] Seed ingest failed: {e}")

    # Discover until target
    while len(topics) < args.target_topics:
        need = args.target_topics - len(topics)
        batch = min(args.topics_batch, need)
        avoid_ids = sorted(seen_ids)
        avoid_titles = sorted(seen_titles)
        try:
            raw = discover_topics_batch(args.host, args.model, batch, avoid_ids, avoid_titles)
            unique = deduplicate_topics(raw, seen_ids, seen_titles)
            print(f"[DEBUG] Got {len(unique)}/{batch} unique topics this round")
            for t in unique:
                topics.append(t)
                fsync_append_line(args.topics_file, t)  # flush immediately
        except Exception as e:
            print(f"[WARN] Topic discovery round failed: {e}")
        # small backoff to be kind to the server
        time.sleep(0.8)

        # If model struggles to produce uniques, don't loop forever
        if batch > 0 and len(unique) == 0:
            print("[WARN] No unique topics returned; backing off...")
            time.sleep(2.0)

    print(f"[INFO] Total unique topics available: {len(topics)}")

    # Generate interfaces for each topic (skip done)
    processed = 0
    for idx, t in enumerate(topics, start=1):
        tid = t["id"]
        if tid in done_topics:
            continue
        print(f"\n[TOPIC {idx}/{len(topics)}] id={tid} title={t['title']}")
        try:
            interfaces = generate_interfaces_for_topic(args.host, args.model, t)
            if not interfaces:
                print("[WARN] No valid interfaces returned; skipping.")
            else:
                for iface in interfaces:
                    # Write ONLY the interface object
                    fsync_append_line(args.interfaces_file, iface)
            # Mark topic done no matter what (to avoid infinite retries)
            fsync_append_text(args.done_file, tid)
            done_topics.add(tid)
            processed += 1
            print(f"[STEP] Done topic '{tid}'. (processed this run: {processed})")
        except Exception as e:
            print(f"[ERROR] Interface generation failed for {tid}: {e}", file=sys.stderr)
            # do not mark done; will retry on next run
        time.sleep(0.5)

    print("\n[DONE] Completed interface generation pass.")
    print(f"[INFO] Topics processed this run: {processed}")
    print(f"[INFO] topics.jsonl lines: {len(topics)} ; interfaces_done.txt lines: {len(done_topics)}")
    print(f"[INFO] interfaces.jsonl current size: {os.path.getsize(args.interfaces_file) if os.path.exists(args.interfaces_file) else 0} bytes")

if __name__ == "__main__":
    main()
