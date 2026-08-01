"""Shared driver for the property fill-in inference scripts.

`2.perf-eval-fill-gen-local.py` and `2.perf-eval-fill-gen-gemini.py` differ only in how
they turn one (anchor, interface) pair into a JSON object. Everything else — the schema
lookup, the prompt, the type coercion, the resume log, the timing stats — is identical
and lives here, so the two backends stay comparable by construction.

Resume model
------------
`<out_dir>/filled-output-<label>.done` holds one completed 0-based input line index per
line. Output rows are appended in the order they are produced, which is the input order,
because the loop is sequential and skips indices already in the `.done` file. A row is
written for *every* input line, including failures (as `{}`), so the prediction file
stays line-aligned with `fill-eval.jsonl` — `2.perf-eval-result-eval.py` requires that.

Units
-----
Every duration here is **seconds**. Earlier revisions timed in milliseconds but wrote
them under `*_seconds_per_sample` keys; result files under `results/` produced by those
revisions are off by 1000. See the README.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# One extraction: (anchor, interface_id, properties_spec) -> {property_name: value}
ExtractFn = Callable[[str, str, List[Dict[str, str]]], Dict[str, Any]]


def fsync_file(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def human_time(seconds: float) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m{s:02d}s"


def load_interfaces(path: Path) -> Dict[str, Dict[str, Any]]:
    """interface @id -> the Property fields a model is asked to fill.

    `dockerImage` is dropped: it is a deployment artifact that the anchor paragraph is
    explicitly forbidden from mentioning, so asking for it would only measure guessing.
    """
    id_to_schema: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            interface_id = obj.get("@id")
            props = []
            for c in obj.get("contents", []) or []:
                if not isinstance(c, dict):
                    continue
                if c.get("@type") == "Property":
                    name = c.get("name")
                    if not name or name == "dockerImage":
                        continue
                    props.append({"name": name, "schema": c.get("schema", "string")})
            if interface_id:
                id_to_schema[interface_id] = {"properties": props}
    return id_to_schema


def coerce_type(value: Any, schema_type: str) -> Any:
    """DTDL schema -> Python type. Returns None for missing or uncoercible values."""
    if value is None:
        return None
    st = (schema_type or "string").lower()
    try:
        if st in ("double", "float"):
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
            return None
        if st in ("integer", "int", "long"):
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                return int(float(value.strip()))
            return None
        if st in ("boolean", "bool"):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "yes", "y", "1"):
                    return True
                if v in ("false", "no", "n", "0"):
                    return False
                return None
            if isinstance(value, (int, float)):
                return bool(value)
            return None
        return str(value)
    except (TypeError, ValueError):
        return None


def build_prompt(anchor: str, interface_id: str, properties_spec: List[Dict[str, str]]) -> str:
    """The flat text prompt both backends send.

    Kept character-for-character in step with the snippet in the published model card
    (`1.model-push-to-huggingface.py:build_prompt`) — the GRPO checkpoint is sensitive to
    the encoding, so what is documented has to be what is actually sent.
    """
    fields_desc = "\n".join(f'- "{p["name"]}" ({p["schema"]})' for p in properties_spec)
    return (
        "You are an information extraction assistant.\n"
        "Extract values exactly from the ANCHOR text to fill ONLY the following fields "
        f"for interface \"{interface_id}\".\n"
        "If a value is not stated, use null.\n"
        "Do not infer unstated facts.\n"
        "Return ONLY minified JSON (no markdown, no comments), "
        "with EXACT keys listed below.\n\n"
        "Fields:\n"
        f"{fields_desc}\n\n"
        "ANCHOR:\n"
        f"{anchor}\n"
    )


def parse_json_object(text: str) -> Dict[str, Any]:
    """First JSON object in a completion, fenced or not; {} when there is none."""
    text = (text or "").strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def clean_against_spec(obj: Dict[str, Any], properties_spec: List[Dict[str, str]]) -> Dict[str, Any]:
    """Keep exactly the requested keys, coerced to their DTDL schema."""
    return {p["name"]: coerce_type(obj.get(p["name"]), p.get("schema", "string"))
            for p in properties_spec}


class Stats:
    """Per-sample timing, in seconds, carried across resumed runs."""

    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.count = 0
        self.total = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self.hist_avg: Optional[float] = None
        self.hist_min: Optional[float] = None
        self.hist_max: Optional[float] = None
        self._load_history()

    def _load_history(self) -> None:
        if not self.progress_path.exists():
            return
        try:
            with self.progress_path.open("r", encoding="utf-8") as f:
                hist = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if hist.get("processed", 0) > 0:
            self.hist_avg = float(hist.get("avg_seconds_per_sample") or 0) or None
            hm, hx = hist.get("min_seconds_per_sample"), hist.get("max_seconds_per_sample")
            self.hist_min = float(hm) if hm is not None else None
            self.hist_max = float(hx) if hx is not None else None

    def record(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.min = seconds if self.min is None else min(self.min, seconds)
        self.max = seconds if self.max is None else max(self.max, seconds)

    @property
    def avg(self) -> float:
        if self.count:
            return self.total / self.count
        return self.hist_avg or 0.0

    @property
    def best(self) -> Optional[float]:
        return self.min if self.min is not None else self.hist_min

    @property
    def worst(self) -> Optional[float]:
        return self.max if self.max is not None else self.hist_max


def write_stats_txt(path: Path, stats: Stats) -> None:
    def num(v: Optional[float]) -> str:
        return f"{v:.6f}" if v is not None else ""

    lines = [
        f"min_seconds_per_sample={num(stats.best)}",
        f"max_seconds_per_sample={num(stats.worst)}",
        f"avg_seconds_per_sample={num(stats.avg or None)}",
        "",
        f"min_human={human_time(stats.best) if stats.best is not None else '-'}",
        f"max_human={human_time(stats.worst) if stats.worst is not None else '-'}",
        f"avg_human={human_time(stats.avg) if stats.avg > 0 else '-'}",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        fsync_file(f)


def save_progress_meta(path: Path, processed: int, total: int, stats: Stats, started_ts: float) -> None:
    meta = {
        "processed": processed,
        "total": total,
        "avg_seconds_per_sample": stats.avg,
        "min_seconds_per_sample": stats.best,
        "max_seconds_per_sample": stats.worst,
        "started_at_unix": started_ts,
        "updated_at_unix": time.time(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        fsync_file(f)


def load_done_indices(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    done.add(int(ln))
                except ValueError:
                    continue
    return done


def run(
    extract: ExtractFn,
    label: str,
    interfaces_path: Path,
    fill_eval_path: Path,
    out_dir: Path,
    limit: int = 0,
) -> int:
    """Run `extract` over every record in `fill_eval_path`. Returns a process exit code."""
    for path in (interfaces_path, fill_eval_path):
        if not path.exists():
            print(f"[ERROR] Not found: {path}", file=sys.stderr, flush=True)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"filled-output-{label}.jsonl"
    done_path = out_dir / f"filled-output-{label}.done"
    progress_path = out_dir / f"progress-{label}.json"
    stats_path = out_dir / f"sample_time_stats-{label}.txt"

    id_to_schema = load_interfaces(interfaces_path)
    with fill_eval_path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    total = len(lines)
    if total == 0:
        print("[INFO] Input is empty, nothing to do.", flush=True)
        return 0

    done_indices = load_done_indices(done_path)
    stats = Stats(progress_path)
    started_ts = time.time()
    print(f"[INFO] {label}: {total} records, {len(done_indices)} already done -> {output_path}",
          flush=True)
    write_stats_txt(stats_path, stats)

    processed_this_run = 0
    with output_path.open("a", encoding="utf-8") as out_f, \
            done_path.open("a", encoding="utf-8") as done_f:

        def commit(idx: int, result: Dict[str, Any], elapsed: float, note: str) -> None:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            fsync_file(out_f)
            done_f.write(f"{idx}\n")
            fsync_file(done_f)
            done_indices.add(idx)
            stats.record(elapsed)
            save_progress_meta(progress_path, len(done_indices), total, stats, started_ts)
            write_stats_txt(stats_path, stats)
            remaining = total - len(done_indices)
            eta = remaining * stats.avg
            print(f"[DONE] {idx + 1}/{total}  {len(done_indices) / total * 100:.2f}%  "
                  f"took={human_time(elapsed)}  ETA={human_time(eta)}  "
                  f"(min={human_time(stats.best)} max={human_time(stats.worst)} "
                  f"avg={human_time(stats.avg)}){note}", flush=True)

        for idx, raw in enumerate(lines):
            if idx in done_indices:
                continue

            t0 = time.time()
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                commit(idx, {}, time.time() - t0, f"  [WARN] line {idx + 1} is not JSON: {e}")
                continue

            interface_id = (item.get("answer") or {}).get("interface") or item.get("interface")
            anchor = item.get("anchor", "")

            if not interface_id or interface_id not in id_to_schema:
                commit(idx, {}, time.time() - t0,
                       f"  [WARN] unknown interface {interface_id!r}")
                continue

            props_spec = id_to_schema[interface_id]["properties"]
            note = ""
            try:
                filled = extract(anchor, interface_id, props_spec)
            except Exception as e:  # noqa: BLE001 - one bad sample must not end the run
                filled = {}
                note = f"  [ERROR] extraction failed: {e}"

            commit(idx, {k: v for k, v in filled.items() if k != "dockerImage"},
                   time.time() - t0, note)

            processed_this_run += 1
            if limit and processed_this_run >= limit:
                print(f"[INFO] Reached --limit={limit}, stopping.", flush=True)
                break

    print(f"\n[COMPLETE] {len(done_indices)}/{total}  elapsed={human_time(time.time() - started_ts)}  "
          f"output={output_path}  (min={human_time(stats.best)} max={human_time(stats.worst)} "
          f"avg={human_time(stats.avg)})", flush=True)
    return 0
