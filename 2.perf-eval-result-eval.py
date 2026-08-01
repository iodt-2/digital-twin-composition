#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score filled-output files against the answers in `fill-eval.jsonl`.

    # every model directory under results/
    python 2.perf-eval-result-eval.py

    # one specific prediction file
    python 2.perf-eval-result-eval.py --pred results/gemini-2.5-pro/filled-output-gemini-2.5-pro.jsonl

Prediction files must be line-aligned with the eval file: line N of the prediction is
the answer to line N of `fill-eval.jsonl`. The fill-in scripts guarantee this by
writing `{}` for records they could not process; those rows are then excluded here
rather than counted as total failures, so a backend is scored on what it attempted.

What is compared
----------------
Keys of `answer` excluding `interface` and `dockerImage` (neither is stated in the
anchor) and excluding zero-valued keys, which are the telemetry placeholders the
generator initialises to 0. A value mismatch counts as both a false positive and a
false negative, so precision and recall both see it.
"""

import argparse
import json
import math
import os
import pathlib
from typing import Any, Dict, List, Set, Tuple

IGNORE_KEYS = {"interface", "dockerImage"}
DEFAULT_EVAL = os.path.join("data", "fill-eval.jsonl")
DEFAULT_PRED_ROOT = "results"


def is_zero_value(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0


def almost_equal(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if tol == 0:
            return a == b
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse failed at {path}:{line_no}: {e}")
    return out


def evaluate_pair(
    gold_row: Dict[str, Any],
    pred_row: Dict[str, Any],
    tol: float,
) -> Tuple[int, int, int, int, List[Dict[str, Any]], Dict[str, Dict[str, int]], int]:
    """-> TP, FP, FN, required, diffs, per-key stats, present_required"""
    gold_ans: Dict[str, Any] = gold_row.get("answer", {})
    pred: Dict[str, Any] = pred_row or {}

    required_keys: Set[str] = {
        k for k, v in gold_ans.items()
        if k not in IGNORE_KEYS and not is_zero_value(v)
    }
    pred_keys: Set[str] = {k for k in pred.keys() if k not in IGNORE_KEYS}

    tp = fp = fn = 0
    diffs: List[Dict[str, Any]] = []
    per_key_stats: Dict[str, Dict[str, int]] = {}

    def slot(key: str) -> Dict[str, int]:
        return per_key_stats.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "required": 0, "pred": 0})

    for k in required_keys:
        slot(k)["required"] += 1
    for k in pred_keys:
        slot(k)["pred"] += 1

    present_required = 0
    for k in required_keys:
        gold_v = gold_ans.get(k)
        if k in pred_keys:
            present_required += 1
            pred_v = pred.get(k)
            if almost_equal(gold_v, pred_v, tol):
                tp += 1
                slot(k)["tp"] += 1
            else:
                fp += 1
                fn += 1
                diffs.append({"type": "mismatch", "key": k, "gold": gold_v, "pred": pred_v})
                slot(k)["fp"] += 1
                slot(k)["fn"] += 1
        else:
            fn += 1
            diffs.append({"type": "missing", "key": k, "gold": gold_v, "pred": None})
            slot(k)["fn"] += 1

    for k in pred_keys - required_keys:
        fp += 1
        diffs.append({"type": "extra", "key": k, "gold": None, "pred": pred.get(k)})
        slot(k)["fp"] += 1

    return tp, fp, fn, len(required_keys), diffs, per_key_stats, present_required


def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def model_label_from_path(p: str) -> str:
    path = pathlib.Path(p)
    return path.parent.name or path.stem


def discover_pred_files(root: str) -> List[str]:
    """Every `filled-output*.jsonl` under `root`, at any depth.

    New runs land in `results/<label>/`; the runs shipped with the repo sit one level
    deeper, under `results/sentence-transformers/<label>/`.
    """
    root_path = pathlib.Path(root)
    if not root_path.exists():
        return []
    return [str(p) for p in sorted(root_path.rglob("filled-output*.jsonl"))]


def load_time_stats_from_dir(model_dir: pathlib.Path) -> Dict[str, float]:
    """Parse `key=value` lines out of the `sample_time_stats-*.txt` in a model directory.

    Keys are reported with whatever unit their name declares. Files written before the
    unit fix carry millisecond values under `*_seconds_per_sample` names — see the
    README — so treat a suspiciously large "seconds" figure as milliseconds.
    """
    stats: Dict[str, float] = {}
    if not model_dir.is_dir():
        return stats

    for txt_file in sorted(model_dir.glob("*.txt")):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    k, v = (part.strip() for part in line.split("=", 1))
                    if v.lower().endswith("ms"):
                        k, v = (k if k.endswith("_ms") else f"{k}_ms"), v[:-2].strip()
                    try:
                        stats[k] = float(v)
                    except ValueError:
                        continue
        except OSError:
            continue
    return stats


def evaluate_one(gold_path: str, pred_path: str, tol: float, top_percent: float = 100.0) -> Dict[str, Any]:
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"[{pred_path}] line counts differ: eval={len(gold)}, pred={len(pred)}. "
                         "Predictions must be line-aligned with the eval file.")

    if not (0 < top_percent <= 100):
        raise ValueError(f"--top_percent must be in (0, 100], got {top_percent}")
    use_n = max(1, int(math.floor(len(gold) * top_percent / 100.0)))
    gold, pred = gold[:use_n], pred[:use_n]

    global_tp = global_fp = global_fn = global_required = 0
    rows_exact_match = rows_evaluated = rows_skipped_empty_pred = 0
    per_key_agg: Dict[str, Dict[str, int]] = {}

    for g, p in zip(gold, pred):
        # An empty object means the backend never produced a prediction for this row.
        if isinstance(p, dict) and not p:
            rows_skipped_empty_pred += 1
            continue

        tp, fp, fn, req, _diffs, per_key_stats, _present = evaluate_pair(g, p, tol)

        rows_evaluated += 1
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_required += req
        if fp == 0 and fn == 0:
            rows_exact_match += 1

        for k, s in per_key_stats.items():
            agg = per_key_agg.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "required": 0, "pred": 0})
            for kk in agg:
                agg[kk] += s.get(kk, 0)

    precision = safe_div(global_tp, global_tp + global_fp)
    recall = safe_div(global_tp, global_tp + global_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "model": model_label_from_path(pred_path),
        "evaluated_rows": rows_evaluated,
        "skipped_empty_pred_rows": rows_skipped_empty_pred,
        "required_fields": global_required,
        "TP": global_tp, "FP": global_fp, "FN": global_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy_union": safe_div(global_tp, global_tp + global_fp + global_fn),
        "em_field": safe_div(global_tp, global_required),
        "em_row": safe_div(rows_exact_match, rows_evaluated),
        "time_stats": load_time_stats_from_dir(pathlib.Path(pred_path).parent),
    }


def print_one_metrics(m: Dict[str, Any]) -> None:
    print(f"\n================ {m['model']} ================\n")
    print("=== Overall ===")
    print(f"Number of rows                  : {m['evaluated_rows']}")
    print(f"Skipped empty predictions       : {m['skipped_empty_pred_rows']}")
    print(f"Required fields                 : {m['required_fields']}")
    print(f"TP: {m['TP']} | FP: {m['FP']} | FN: {m['FN']}")
    print(f"Precision                       : {m['precision']:.4f}")
    print(f"Recall                          : {m['recall']:.4f}")
    print(f"F1                              : {m['f1']:.4f}")
    print(f"Exact Match Rate (per field)    : {m['em_field']:.4f}")
    print(f"Exact Match Rate (per row)      : {m['em_row']:.4f}")
    print(f"Accuracy (required u predicted) : {m['accuracy_union']:.4f}")

    ts = m.get("time_stats") or {}
    if ts:
        print("\n=== Time stats (unit per key name) ===")
        for k in sorted(ts):
            print(f"{k:30s}: {ts[k]:.3f}")
    print()


def print_summary_table(ms: List[Dict[str, Any]]) -> None:
    if not ms:
        return
    cols = ["model", "precision", "recall", "f1", "accuracy_union", "em_row"]
    widths = {
        c: max(len(c), max(len(f"{m[c]:.4f}") if isinstance(m[c], float) else len(str(m[c])) for m in ms))
        for c in cols
    }
    print("\n====== Summary ======")
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for m in ms:
        print(" | ".join(
            (f"{m[c]:.4f}" if isinstance(m[c], float) else str(m[c])).ljust(widths[c])
            for c in cols
        ))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=DEFAULT_EVAL, help=f"fill-eval JSONL with answers (default: {DEFAULT_EVAL})")
    ap.add_argument("--pred", nargs="*", help=f"Prediction files (default: scan {DEFAULT_PRED_ROOT}/*/)")
    ap.add_argument("--pred-root", default=DEFAULT_PRED_ROOT, help="Directory scanned when --pred is omitted")
    ap.add_argument("--tol", type=float, default=0.0, help="Absolute tolerance for float comparison")
    ap.add_argument("--top_percent", type=float, default=70.0,
                    help="Compare only the first N%% of rows, so partially-finished runs stay "
                         "comparable with each other (default: 70)")
    args = ap.parse_args()

    pred_list = args.pred or discover_pred_files(args.pred_root)
    if not pred_list:
        print(f"[ERROR] No prediction files found under {args.pred_root}/. Pass --pred explicitly.")
        return

    results: List[Dict[str, Any]] = []
    for pred_path in (str(pathlib.Path(p)) for p in pred_list):
        if not os.path.exists(pred_path):
            print(f"[WARN] Not found, skipping: {pred_path}")
            continue
        try:
            metrics = evaluate_one(args.eval, pred_path, args.tol, args.top_percent)
            print_one_metrics(metrics)
            results.append(metrics)
        except Exception as e:  # noqa: BLE001 - one bad file must not hide the others
            print(f"[ERROR] Failed: {pred_path} -> {e}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
