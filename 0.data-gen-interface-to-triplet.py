#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a (query, positive, negative) triplet database from interfaces.jsonl.

query     : "I'm looking for a digital-twin interface that <summary description>. <specifications>"
            <summary description> comes from the interface `description`.
            <specifications>      comes from the matching `anchor` text in fill-eval.jsonl.
positive  : interface JSON of the described interface
negative  : interface JSON sampled from a random pool, verified by an Ollama model
            to belong to a DIFFERENT topic than the positive.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------
# Defaults
# ---------------------------

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.10.10.4:11434")
# gpt-oss:20b is the verifier: ~1.9x faster per call than gpt-oss:120b (median 1.39s vs 2.59s)
# and it agrees with 120b on 15/20 near-duplicate topic pairs, vs 12/20 for gemma3:27b and
# llama3.2:3b. The lexical prefilter below covers most of the remaining gap.
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

INTERFACES_JSONL = os.path.join("data", "interfaces.jsonl")
FILL_EVAL_JSONL = os.path.join("data", "fill-eval.jsonl")
OUTPUT_JSONL = os.path.join("data", "triplet.jsonl")


# ---------------------------
# Ollama API
# ---------------------------

def ollama_generate(host: str, model: str, prompt: str, timeout: int = 120) -> str:
    """Call /api/generate, falling back to /api/chat. Returns the raw text answer."""
    host = host.rstrip("/")
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        if r.ok:
            obj = r.json()
            if isinstance(obj, dict) and "response" in obj:
                return str(obj["response"])
    except requests.RequestException:
        pass

    payload2 = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }
    r2 = requests.post(f"{host}/api/chat", json=payload2, timeout=timeout)
    r2.raise_for_status()
    obj2 = r2.json()
    if isinstance(obj2, dict) and isinstance(obj2.get("message"), dict):
        return str(obj2["message"].get("content", ""))
    raise RuntimeError(f"Unexpected Ollama response: {obj2}")


VERIFY_PROMPT = """You are validating training data for a digital-twin interface retrieval model.

Below are two DTDL interfaces. Decide whether they belong to DIFFERENT topics
(different application domains / different digital-twin systems).

Interface A (positive):
  id: {a_id}
  displayName: {a_name}
  description: {a_desc}

Interface B (candidate negative):
  id: {b_id}
  displayName: {b_name}
  description: {b_desc}

They are the SAME topic if they model the same system, the same asset family, or two
closely-related parts of one digital twin (e.g. "Battery Pack" and "Cell Module" of the
same EV battery twin). They are DIFFERENT topics if a user searching for A would never
accept B as an answer.

Reply with JSON only, no prose:
{{"different_topic": true|false, "reason": "<one short sentence>"}}
"""


# ---------------------------
# Progress bar
# ---------------------------

class Progress:
    """Dependency-free progress bar with rate and ETA, rendered on stderr."""

    def __init__(self, total: int, enabled: bool = True, width: int = 26, min_interval: float = 0.1):
        self.total = max(0, total)
        self.width = width
        self.min_interval = min_interval
        self.done = 0
        self.start = time.time()
        self._last_draw = 0.0
        self._lock = threading.Lock()
        self.tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.enabled = enabled and self.total > 0

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN / inf
            return "--:--"
        seconds = int(max(0, seconds))
        h, rem = divmod(seconds, 3600)
        mnt, sec = divmod(rem, 60)
        return f"{h:d}:{mnt:02d}:{sec:02d}" if h else f"{mnt:02d}:{sec:02d}"

    def _render(self, suffix: str) -> str:
        elapsed = time.time() - self.start
        frac = self.done / self.total if self.total else 0.0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        return (f"[{bar}] {frac * 100:5.1f}% {self.done}/{self.total} "
                f"{rate:.1f}/s elapsed {self._fmt_time(elapsed)} eta {self._fmt_time(eta)}"
                + (f" | {suffix}" if suffix else ""))

    def update(self, n: int = 1, suffix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.done += n
            now = time.time()
            if not force and self.done < self.total and now - self._last_draw < self.min_interval:
                return
            self._last_draw = now
            line = self._render(suffix)
            if self.tty:
                sys.stderr.write("\r\033[2K" + line)
            else:
                # Not a terminal (piped/redirected): emit periodic lines instead of redrawing.
                if force or self.done % 200 == 0 or self.done == self.total:
                    sys.stderr.write(line + "\n")
            sys.stderr.flush()

    def log(self, message: str) -> None:
        """Print a message without leaving a half-drawn bar behind."""
        with self._lock:
            if self.enabled and self.tty:
                sys.stderr.write("\r\033[2K")
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
            self._last_draw = 0.0  # force a redraw on the next update

    def close(self) -> None:
        if self.enabled and self.tty:
            with self._lock:
                sys.stderr.write("\n")
                sys.stderr.flush()


def parse_verdict(text: str) -> Optional[bool]:
    """Extract the boolean verdict from a model reply; None if unparseable."""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = obj.get("different_topic")
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "y", "1")
        except json.JSONDecodeError:
            pass
    low = text.lower()
    if "different_topic" in low:
        tail = low.split("different_topic", 1)[1]
        if re.search(r"\btrue\b", tail):
            return True
        if re.search(r"\bfalse\b", tail):
            return False
    return None


# ---------------------------
# Data loading
# ---------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def topic_of(interface_id: str) -> str:
    """dtmi:<topic>:<Name>;1 -> <topic>"""
    parts = (interface_id or "").split(":")
    return parts[1] if len(parts) > 2 else interface_id


# Generic words that carry no domain meaning in a topic id.
TOPIC_STOPWORDS = {
    "twin", "twins", "monitor", "monitoring", "model", "modeling", "system", "network",
    "analysis", "simulation", "optimization", "opt", "management", "tracking", "assessment",
    "the", "and", "for", "with",
}


def topic_tokens(topic: str) -> set:
    return {w for w in topic.split("_") if len(w) > 2 and w not in TOPIC_STOPWORDS}


def same_topic_lexically(a_topic: str, b_topic: str) -> bool:
    """Cheap near-duplicate check on topic ids, applied before spending an LLM call.

    The corpus holds pairs like `stormwater_drainage_network` / `municipal_stormwater_drainage`
    that are the same topic under two names; a smaller verifier tends to call those "different".
    Rejecting them here costs one extra resample and only fires on ~1% of random topic pairs.
    """
    ta, tb = topic_tokens(a_topic), topic_tokens(b_topic)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:      # one topic is a refinement of the other
        return True
    return len(ta & tb) >= 2      # two shared domain words


def load_specifications(path: str) -> Dict[str, List[str]]:
    """Map interface @id -> list of `anchor` specification paragraphs from fill-eval.jsonl."""
    specs: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = (row.get("answer") or {}).get("interface")
            anchor = (row.get("anchor") or "").strip()
            if iid and anchor:
                specs.setdefault(iid, []).append(anchor)
    return specs


# ---------------------------
# Query construction
# ---------------------------

def summarize_description(interface: Dict[str, Any]) -> str:
    """First sentence of the description, phrased to follow 'an interface that ...'."""
    desc = (interface.get("description") or "").strip()
    if not desc:
        name = interface.get("displayName") or topic_of(interface.get("@id", ""))
        return f"models a {name}"
    first = re.split(r"(?<=[.!?])\s+", desc)[0].strip().rstrip(".")
    if first:
        first = first[0].lower() + first[1:]
    return first


def fallback_specifications(interface: Dict[str, Any]) -> str:
    """Used when fill-eval.jsonl has no anchor for this interface."""
    props, telem = [], []
    for item in interface.get("contents", []) or []:
        t = item.get("@type")
        types = t if isinstance(t, list) else [t]
        name = item.get("name")
        if not name:
            continue
        if "Telemetry" in types:
            telem.append(name)
        elif "Property" in types:
            props.append(name)
    bits = []
    if props:
        bits.append("It must expose the properties " + ", ".join(props) + ".")
    if telem:
        bits.append("It must stream the telemetry " + ", ".join(telem) + ".")
    return " ".join(bits) or "It must match the described digital-twin asset."


def build_query(interface: Dict[str, Any], specs: Dict[str, List[str]], rng: random.Random) -> str:
    summary = summarize_description(interface)
    candidates = specs.get(interface.get("@id", ""))
    spec_text = rng.choice(candidates) if candidates else fallback_specifications(interface)
    return f"I'm looking for a digital-twin interface that {summary}. {spec_text}"


# ---------------------------
# Negative sampling
# ---------------------------

class NegativeSampler:
    """Samples a negative from the interface pool and verifies the topic differs."""

    def __init__(self, pool: List[Dict[str, Any]], host: str, model: str,
                 timeout: int, verify: bool, max_attempts: int, log=None):
        self.pool = pool
        self.host = host
        self.model = model
        self.timeout = timeout
        self.verify = verify
        self.max_attempts = max_attempts
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        self._cache: Dict[Tuple[str, str], bool] = {}
        self._lock = threading.Lock()
        self.stats = {"llm_calls": 0, "cache_hits": 0, "prefiltered": 0, "rejected": 0, "unverified": 0}

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] += n

    def _different_topic(self, a: Dict[str, Any], b: Dict[str, Any]) -> Optional[bool]:
        ta, tb = sorted((topic_of(a["@id"]), topic_of(b["@id"])))
        key: Tuple[str, str] = (ta, tb)
        with self._lock:
            if key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[key]

        prompt = VERIFY_PROMPT.format(
            a_id=a.get("@id", ""), a_name=a.get("displayName", ""), a_desc=a.get("description", ""),
            b_id=b.get("@id", ""), b_name=b.get("displayName", ""), b_desc=b.get("description", ""),
        )
        try:
            text = ollama_generate(self.host, self.model, prompt, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001 - network/model failures must not kill the run
            self.log(f"[WARN] Ollama verification failed: {e}")
            return None
        self._bump("llm_calls")
        verdict = parse_verdict(text)
        if verdict is None:
            self.log(f"[WARN] Unparseable verdict: {text[:160]!r}")
            return None
        with self._lock:
            self._cache[key] = verdict
        return verdict

    def sample(self, positive: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
        pos_id = positive.get("@id", "")
        pos_topic = topic_of(pos_id)
        tried = set()
        for _ in range(self.max_attempts):
            cand = rng.choice(self.pool)
            cand_id = cand.get("@id", "")
            if cand_id == pos_id or cand_id in tried:
                continue
            tried.add(cand_id)
            # Cheap prefilters, no LLM call: identical topic segment, or two topic ids that
            # are near-duplicate names for one topic.
            cand_topic = topic_of(cand_id)
            if cand_topic == pos_topic or same_topic_lexically(pos_topic, cand_topic):
                self._bump("prefiltered")
                continue
            if not self.verify:
                return cand
            verdict = self._different_topic(positive, cand)
            if verdict is True:
                return cand
            if verdict is False:
                self._bump("rejected")
            else:
                self._bump("unverified")
        return None


# ---------------------------
# Resume support
# ---------------------------

def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(json.loads(row["positive"])["@id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interfaces", default=INTERFACES_JSONL, help="Input interfaces JSONL")
    ap.add_argument("--fill-eval", default=FILL_EVAL_JSONL, help="JSONL holding the `anchor` specifications")
    ap.add_argument("--out", default=OUTPUT_JSONL, help="Output triplet JSONL")
    ap.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--timeout", type=int, default=120, help="Ollama request timeout (s)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel verification workers")
    ap.add_argument("--max-attempts", type=int, default=5, help="Negative candidates to try per triplet")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N interfaces (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--no-verify", action="store_true", help="Skip the Ollama topic check")
    ap.add_argument("--resume", action="store_true", help="Append to --out, skipping interfaces already present")
    ap.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    args = ap.parse_args()

    print(f"[INFO] Loading interfaces from {args.interfaces}")
    interfaces = load_jsonl(args.interfaces)
    print(f"[INFO] Loaded {len(interfaces)} interfaces")

    print(f"[INFO] Loading specifications from {args.fill_eval}")
    specs = load_specifications(args.fill_eval)
    print(f"[INFO] Specifications available for {len(specs)} interfaces")

    targets = interfaces[: args.limit] if args.limit else interfaces

    done = load_done_ids(args.out) if args.resume else set()
    if done:
        print(f"[INFO] Resuming: {len(done)} triplets already written")
        targets = [i for i in targets if i.get("@id") not in done]

    progress = Progress(total=len(targets), enabled=not args.no_progress)
    sampler = NegativeSampler(
        pool=interfaces,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        verify=not args.no_verify,
        max_attempts=args.max_attempts,
        log=progress.log,
    )
    if sampler.verify:
        print(f"[INFO] Verifying negatives with {args.model} @ {args.host} ({args.workers} workers)")
    else:
        print("[INFO] Ollama verification disabled (--no-verify)")

    def work(index_interface: Tuple[int, Dict[str, Any]]) -> Optional[Dict[str, str]]:
        idx, interface = index_interface
        rng = random.Random(args.seed + idx)  # per-item RNG keeps runs reproducible under threading
        negative = sampler.sample(interface, rng)
        if negative is None:
            progress.log(f"[WARN] No verified negative for {interface.get('@id')}")
            return None
        return {
            "query": build_query(interface, specs, rng),
            "positive": json.dumps(interface, ensure_ascii=False),
            "negative": json.dumps(negative, ensure_ascii=False),
        }

    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    written = skipped = 0
    write_lock = threading.Lock()

    with open(args.out, mode, encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(work, item) for item in enumerate(targets)]
            for n, fut in enumerate(as_completed(futures), 1):
                triplet = fut.result()
                if triplet is None:
                    skipped += 1
                else:
                    with write_lock:
                        out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                        written += 1
                    if n % 200 == 0:
                        out_f.flush()
                progress.update(suffix=f"ok={written} skip={skipped} llm={sampler.stats['llm_calls']}",
                                force=(n == len(targets)))
    progress.close()

    print(f"\n[DONE] Triplet database: {written} entries written to {args.out}")
    print(f"   skipped={skipped}  llm_calls={sampler.stats['llm_calls']}  "
          f"cache_hits={sampler.stats['cache_hits']}  prefiltered={sampler.stats['prefiltered']}  "
          f"rejected={sampler.stats['rejected']}  unverified={sampler.stats['unverified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
