#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turn `interfaces.jsonl` into the fill-in dataset `fill-eval.jsonl`.

For every DTDL Interface, two Ollama calls produce one record:

    answer : a full instance JSON — the interface id, a plausible value for every
             Property, and 0 for every Telemetry.
    anchor : a prose paragraph that states the Property values *only*. It never
             mentions the interface id, the dockerImage path, or any telemetry, so a
             model reading it can only recover what the answer legitimately contains.

Records are keyed by a SHA-256 of the input line and logged to `<output>.ckpt`, so an
interrupted run resumes without regenerating or duplicating anything.

    python 0.data-gen-fill.py --input data/interfaces.jsonl --output data/fill-eval.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import requests

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://10.1.1.1:60002")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
DEFAULT_INPUT = os.path.join("data", "interfaces.jsonl")
DEFAULT_OUTPUT = os.path.join("data", "fill-eval.jsonl")


# -------------------------
# Ollama client (/api/chat)
# -------------------------
class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.8) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
            return data["messages"][-1].get("content", "")
        raise RuntimeError(f"Unexpected Ollama response format: {data}")


# -------------------------
# Helpers
# -------------------------
def json_only(s: str) -> str:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S)
    if fence:
        return fence.group(1)
    brace = re.search(r"(\{.*\})", s, re.S)
    if brace:
        return brace.group(1)
    return s.strip()


def cast_value(v: Any, schema: str) -> Any:
    t = (schema or "").lower()
    try:
        if t in ("double", "float"):
            return float(v)
        if t in ("integer", "long", "int"):
            return int(float(v))
        if t in ("boolean", "bool"):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v).strip().lower()
            if s in ("true", "yes", "y", "1"):
                return True
            if s in ("false", "no", "n", "0"):
                return False
            return bool(s)
        if t == "string":
            return str(v)
        return v
    except Exception:
        return str(v)


def extract_fields(interface_obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    interface_id = interface_obj.get("@id") or interface_obj.get("id") or ""
    contents = interface_obj.get("contents", []) or []
    properties, telemetries = [], []
    for item in contents:
        if not isinstance(item, dict):
            continue
        t = item.get("@type") or item.get("type")
        if t == "Property":
            properties.append(item)
        elif t == "Telemetry":
            telemetries.append(item)
    return interface_id, properties, telemetries


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# -------------------------
# Prompts
# -------------------------
SYSTEM_JSON_VALUES = (
    "You generate realistic, domain-appropriate JSON values ONLY.\n"
    "You MUST return a single compact JSON object with no commentary."
)

USER_JSON_VALUES_TEMPLATE = """
You are given a DTDL-like interface with the following Property fields (name -> schema).
Generate realistic, coherent values for these properties as a single JSON object.

- DO NOT invent new keys.
- Preserve dockerImage EXACTLY as provided (do not alter version or registry).
- Follow schema types strictly (string/double/integer/boolean).
- Values should be internally consistent and plausible for the domain described by 'displayName' and 'description'.
- For numeric fields, produce reasonable magnitudes.
- For string identifiers, prefer short, slug-like IDs.

Interface summary:
displayName: {display_name}
description: {description}

Properties (JSON array of objects with name/schema):
{properties_json}

If dockerImage_original is not empty, set the property 'dockerImage' to EXACTLY that:
dockerImage_original: {docker_image_original}

Return ONLY JSON.
"""

# Anchor rules: no interface id / DTMI, no dockerImage path, no telemetry whatsoever.
# Anything the anchor leaks that the answer also contains would make the fill-in task
# solvable by copying rather than by extraction.
SYSTEM_ANCHOR = (
    "You are a technical writer for digital twins. "
    "Write one concise paragraph (6-8 sentences) describing the instance, "
    "weaving in ALL key PROPERTY fields with their concrete values. "
    "Strict rules:\n"
    "1) DO NOT mention the interface id, DTMI, or any interface name.\n"
    "2) DO NOT mention docker image names, docker registries, or any dockerImage path.\n"
    "3) DO NOT mention telemetry names or telemetry values; completely ignore telemetry.\n"
    "4) Avoid lists and headings. No JSON, no bullet points."
)

USER_ANCHOR_TEMPLATE = """
Write a paragraph that naturally mentions these PROPERTY fields and values and reflects the semantics
of the system. Keep it objective, technical, and readable to engineers.

Use ONLY the provided fields; do not infer or add interface ids or docker image paths.
Do NOT mention telemetry.

allowed fields JSON:
{allowed_instance_json}
"""


# -------------------------
# Instance / anchor generation
# -------------------------
def build_instance_with_llm(client: OllamaClient, interface_obj: Dict[str, Any]) -> Dict[str, Any]:
    interface_id, properties, telemetries = extract_fields(interface_obj)
    display_name = interface_obj.get("displayName", "")
    description = interface_obj.get("description", "")

    docker_image_default = ""
    for p in properties:
        if p.get("name") == "dockerImage" and isinstance(p.get("value", ""), str):
            docker_image_default = p["value"]
            break

    props_slim = [{"name": p.get("name"), "schema": p.get("schema", "string")} for p in properties]

    user_prompt = USER_JSON_VALUES_TEMPLATE.format(
        display_name=display_name,
        description=description,
        properties_json=json.dumps(props_slim, ensure_ascii=False, indent=2),
        docker_image_original=docker_image_default,
    )
    messages = [{"role": "system", "content": SYSTEM_JSON_VALUES},
                {"role": "user", "content": user_prompt}]

    gen_props: Dict[str, Any] = {}
    for attempt in range(3):
        try:
            raw = client.chat(messages, temperature=0.6)
            gen_props = json.loads(json_only(raw))
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)

    # Full instance, used as the `answer`.
    instance: Dict[str, Any] = {"interface": interface_id}
    for p in properties:
        name = p.get("name")
        schema = p.get("schema", "string")
        if name == "dockerImage":
            if docker_image_default:
                instance[name] = docker_image_default
            else:
                instance[name] = cast_value(gen_props.get(name, ""), "string")
        else:
            instance[name] = cast_value(gen_props.get(name, ""), schema)

    # Telemetry is runtime data, not something a spec paragraph states: initialise to 0
    # so the instance is complete, and exclude it from the anchor and from scoring.
    for t in telemetries:
        instance[t.get("name")] = 0

    return instance


def build_anchor_with_llm(
    client: OllamaClient,
    interface_obj: Dict[str, Any],
    full_instance: Dict[str, Any],
) -> str:
    # Allowed fields: Properties minus dockerImage; never `interface`, never Telemetry.
    _, properties, telemetries = extract_fields(interface_obj)
    telemetry_names = {t.get("name") for t in telemetries if isinstance(t, dict)}
    property_names = {p.get("name") for p in properties if isinstance(p, dict)}

    allowed = {}
    for k, v in full_instance.items():
        if k in ("interface", "dockerImage") or k in telemetry_names:
            continue
        if k in property_names:
            allowed[k] = v

    user_prompt = USER_ANCHOR_TEMPLATE.format(
        allowed_instance_json=json.dumps(allowed, ensure_ascii=False, indent=2),
    )
    messages = [{"role": "system", "content": SYSTEM_ANCHOR},
                {"role": "user", "content": user_prompt}]
    text = client.chat(messages, temperature=0.8).strip()
    text = re.sub(r"^```.*?\n|\n```$", "", text, flags=re.S)
    return " ".join(text.split())


# -------------------------
# Checkpointing
# -------------------------
def sha256_line(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_checkpoint(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h = line.strip()
            if h:
                done.add(h)
    return done


def append_checkpoint(path: str, h: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(h + "\n")
        f.flush()
        os.fsync(f.fileno())


class ProgressReporter:
    """One-line progress with an ETA extrapolated from completed samples only."""

    def __init__(self, total_lines: int):
        self.total = total_lines
        self.written = 0
        self.skipped = 0
        self.failed = 0
        self.sample_seconds = 0.0
        self.started = time.time()

    @property
    def avg(self) -> float:
        return (self.sample_seconds / self.written) if self.written else 0.0

    def report(self, line_num: int, message: str) -> None:
        remaining = max(0, self.total - (self.written + self.skipped + self.failed))
        eta = self.avg * remaining
        eta_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta))
        pct = (line_num / self.total * 100) if self.total else 0.0
        print(f"[{line_num}/{self.total} | {pct:.1f}%] {message} | "
              f"written={self.written} skipped={self.skipped} failed={self.failed} | "
              f"avg/sample={format_duration(self.avg)} | "
              f"ETA {format_duration(eta)} (about {eta_at})", flush=True)


# -------------------------
# Main loop
# -------------------------
def process_interfaces_file(
    input_path: str,
    output_path: str,
    host: str,
    model: str,
    timeout: int = 120,
    limit: int = 0,
) -> None:
    client = OllamaClient(host=host, model=model, timeout=timeout)

    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            total_lines = sum(1 for _ in fin)
    except OSError as e:
        print(f"[FATAL] Cannot read input file: {e}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = output_path + ".ckpt"
    processed_hashes = load_checkpoint(ckpt_path)
    if processed_hashes:
        print(f"[INFO] Resuming: {len(processed_hashes)} lines already recorded in {ckpt_path}")

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    progress = ProgressReporter(total_lines)

    # Append mode: the checkpoint is what makes this safe to re-run.
    with open(output_path, "a", encoding="utf-8") as fout, \
            open(input_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                progress.skipped += 1
                progress.report(line_num, "blank line, skipped")
                continue

            h = sha256_line(raw)
            if h in processed_hashes:
                progress.skipped += 1
                progress.report(line_num, "already processed (resume hit), skipped")
                continue

            try:
                iface = json.loads(raw)
            except json.JSONDecodeError as e:
                progress.failed += 1
                append_checkpoint(ckpt_path, h)
                processed_hashes.add(h)
                progress.report(line_num, f"parse failed, skipped: {e}")
                continue

            sample_start = time.time()
            try:
                full_instance = build_instance_with_llm(client, iface)
                anchor = build_anchor_with_llm(client, iface, full_instance)
                record = {"anchor": anchor, "answer": full_instance}

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())

                # Checkpoint only after the record is durably on disk, so a crash
                # between the two never loses a line or duplicates one.
                append_checkpoint(ckpt_path, h)
                processed_hashes.add(h)

                progress.written += 1
                progress.sample_seconds += time.time() - sample_start
                progress.report(line_num, "written")

                if limit and progress.written >= limit:
                    print(f"[INFO] Reached --limit={limit}, stopping.", flush=True)
                    break

            except Exception as e:
                progress.failed += 1
                append_checkpoint(ckpt_path, h)
                processed_hashes.add(h)
                progress.report(line_num, f"generation failed, skipped: {e}")
                continue

    elapsed = time.time() - progress.started
    print(f"Done. total lines={total_lines} | written={progress.written} "
          f"skipped={progress.skipped} failed={progress.failed} | "
          f"elapsed={format_duration(elapsed)} | avg/sample={format_duration(progress.avg)} | "
          f"output: {output_path}", flush=True)


# -------------------------
# CLI
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT, help=f"Input interfaces JSONL (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSONL (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    ap.add_argument("--timeout", type=int, default=120, help="Ollama request timeout in seconds")
    ap.add_argument("--limit", type=int, default=0, help="Stop after writing N records (0 = all)")
    args = ap.parse_args()

    process_interfaces_file(
        input_path=args.input,
        output_path=args.output,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
