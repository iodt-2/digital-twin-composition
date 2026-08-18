#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

IGNORE_KEYS = {"interface", "dockerImage"}
DEFAULT_EVAL = os.path.join("data", "fill-eval.jsonl")
DEFAULT_FILL_DATASET = os.path.join("data", "llm-fill-ft-80-20.ds")
DEFAULT_PRED_ROOT = "results"

# Must stay equal to the defaults of `1.fine-tune-GRPO-llm.py`; the shards only line up
# with what training saw if every one of the three matches.
SPLIT_TEST_SIZE = 0.3
SPLIT_SEED = 42
TRAINED_SPLIT = "test"  # the shard `1.fine-tune-GRPO-llm.py --split test` trains on


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


def gold_sources(args) -> Dict[int, Dict[str, Any]]:
    """Row count -> the gold a prediction file of that length was produced against.

    The two fill-in backends read different inputs: the Gemini one `fill-eval.jsonl`
    (27,770 lines), the local/Ollama one a split of the fine-tuning dataset (5,554 rows
    in `test`). A run is line-aligned with whatever it read and nothing else, so the row
    count identifies its gold unambiguously - which is what lets one invocation score
    every backend into one table. An input that is not on disk is simply skipped, so
    this works with either half of the pair present.
    """
    sources: Dict[int, Dict[str, Any]] = {}

    if os.path.exists(args.eval):
        gold = load_jsonl(args.eval)
        if args.ft_dataset:
            # Fatal on purpose: if the mapping from shard index to eval line is wrong,
            # every score below is computed on the wrong rows.
            verify_ft_alignment(args.ft_dataset, gold)
        shard = (f"whole file ({len(gold)} rows)" if args.split == "all"
                 else f"'{args.split}' shard of "
                      f"train_test_split(test_size={args.test_size}, seed={args.seed})")
        sources[len(gold)] = {
            "gold": gold, "name": os.path.basename(args.eval), "scope": f"{args.eval}, {shard}",
            "rows": select_rows(len(gold), args.split, args.test_size, args.seed,
                                args.top_percent),
            "shard_warning": (
                f"The '{TRAINED_SPLIT}' shard is what 1.fine-tune-GRPO-llm.py trains on. "
                "Scores for a fine-tuned checkpoint measure memorisation, not extraction."
                if args.split == TRAINED_SPLIT else
                f"--split all includes the {math.ceil(args.test_size * len(gold))} training "
                "rows. Fine-tuned checkpoints are flattered by this; other backends are "
                "unaffected." if args.split == "all" else ""),
        }

    if os.path.isdir(args.dataset):
        gold = load_dataset_gold(args.dataset, args.dataset_split)
        rows = list(range(len(gold)))
        if args.top_percent < 100:
            # A saved split is already in permutation order, so a prefix of it is a
            # uniform sub-sample, exactly as for a reproduced shard.
            rows = rows[:max(1, int(math.floor(len(rows) * args.top_percent / 100.0)))]
        if len(gold) in sources:
            raise ValueError(
                f"{args.eval} and {args.dataset}[{args.dataset_split}] both hold "
                f"{len(gold)} rows, so a prediction file cannot be attributed to one of "
                "them. Pass --eval or --dataset to score against a single input.")
        sources[len(gold)] = {
            "gold": gold, "rows": rows,
            "name": f"{os.path.basename(args.dataset.rstrip('/\\'))}[{args.dataset_split}]",
            "scope": f"{args.dataset}, '{args.dataset_split}' split ({len(gold)} rows)",
            # The split was taken when the dataset was built; there is no shard to warn
            # about, only the half a fine-tune consumed.
            "shard_warning": ("'train' is the split an 80/20 fine-tune consumed. Scores for "
                              "a checkpoint trained on it measure memorisation, not "
                              "extraction." if args.dataset_split == "train" else ""),
        }

    return sources


def load_dataset_gold(path: str, split: str) -> List[Dict[str, Any]]:
    """Gold answers from a `save_to_disk` dataset, in row order.

    `2.perf-eval-fill-gen-local.py` runs over one split of the fine-tuning dataset, so
    its predictions are line-aligned with that split, not with `fill-eval.jsonl`. The
    `ground_truth` column is already the answer with `interface`, `dockerImage` and the
    zeroed telemetry removed — the same set `evaluate_pair` calls required — so it is
    read straight into the shape the scorer expects.
    """
    from datasets import load_from_disk

    data = load_from_disk(path)
    if isinstance(data, dict):
        if split not in data:
            raise ValueError(f"{path} has no split {split!r}: {list(data)}")
        data = data[split]
    if "ground_truth" not in data.column_names:
        raise ValueError(f"{path}[{split}] has no `ground_truth` column: {data.column_names}")
    return [{"answer": json.loads(gt)} for gt in data["ground_truth"]]


def split_indices(n_rows: int, test_size: float, seed: int) -> Dict[str, List[int]]:
    if not (0 < test_size < 1):
        raise ValueError(f"--test-size must be in (0, 1), got {test_size}")

    try:
        from datasets import Dataset
    except ImportError:
        pass
    else:
        shards = Dataset.from_dict({"row": list(range(n_rows))}).train_test_split(
            test_size=test_size, seed=seed
        )
        return {name: list(shards[name]["row"]) for name in ("train", "test")}

    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover - one of the two is always installed
        raise RuntimeError(
            "Reproducing the training split needs either `datasets` or `numpy` installed."
        ) from e

    n_test = math.ceil(test_size * n_rows)
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return {
        "test": [int(i) for i in permutation[:n_test]],
        "train": [int(i) for i in permutation[n_test:]],
    }


def select_rows(n_rows: int, split: str, test_size: float, seed: int,
                top_percent: float) -> List[int]:
    """0-based line numbers of `fill-eval.jsonl` to score, ascending."""
    if not (0 < top_percent <= 100):
        raise ValueError(f"--top_percent must be in (0, 100], got {top_percent}")

    rows = list(range(n_rows)) if split == "all" else split_indices(n_rows, test_size, seed)[split]

    if top_percent < 100:
        # A shard is in permutation order, so a prefix of it is a uniform random
        # sub-sample of the shard - a cheap smoke test that is not biased towards the
        # front of the corpus. For `--split all` there is no permutation and this stays
        # the positional prefix it always was.
        rows = rows[:max(1, int(math.floor(len(rows) * top_percent / 100.0)))]

    return sorted(rows)


def _ft_prompt_text(prompt: Any) -> str:
    """The fine-tuning dataset's `prompt` column, flat or chat-shaped, as one string."""
    if isinstance(prompt, list):
        return " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in prompt
        )
    return str(prompt or "")


def verify_ft_alignment(ft_path: str, gold: Sequence[Dict[str, Any]], sample: int = 200) -> None:
    """Check that the fine-tuning dataset really is `fill-eval.jsonl`, row for row.

    Reproducing the split is only meaningful if shard index i names line i+1 of the eval
    file. That holds because the dataset is built one row per line in file order, but it
    is an assumption, and a silently wrong one would hand back a "held-out" set that
    overlaps training. So: same row count, and every sampled row's prompt still carries
    that line's anchor verbatim (`build_prompt` embeds it unchanged).
    """
    from datasets import load_from_disk

    ft = load_from_disk(ft_path)
    if isinstance(ft, dict):  # DatasetDict - the fill-in dataset is saved single-split
        ft = ft.get("train") or next(iter(ft.values()))

    if len(ft) != len(gold):
        raise ValueError(
            f"[{ft_path}] holds {len(ft)} rows but the eval file holds {len(gold)}. The "
            "split cannot be mapped onto eval lines; pass the dataset training actually used."
        )
    if "prompt" not in ft.column_names:
        print(f"[WARN] {ft_path} has no `prompt` column; row alignment left unchecked.")
        return

    step = max(1, len(ft) // max(1, sample))
    checked = 0
    for i in range(0, len(ft), step):
        anchor = str(gold[i].get("anchor", "")).strip()
        if anchor and anchor not in _ft_prompt_text(ft[i]["prompt"]):
            raise ValueError(
                f"[{ft_path}] row {i} does not contain the anchor of eval line {i + 1}. The "
                "fine-tuning dataset is not row-aligned with the eval file, so the shard "
                "indices do not name eval lines."
            )
        checked += 1
    print(f"[INFO] Row alignment verified against {ft_path}: {len(ft)} rows, {checked} anchors sampled.")


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


def evaluate_one(gold: Sequence[Dict[str, Any]], pred_path: str, tol: float,
                 rows: Iterable[int], allow_partial: bool = False,
                 pred: Optional[List[Dict[str, Any]]] = None,
                 gold_name: str = "") -> Dict[str, Any]:
    pred = load_jsonl(pred_path) if pred is None else pred
    if allow_partial and len(pred) < len(gold):
        # A dataset run is resumable and expensive, so an unfinished one is a normal
        # state to want a number from. Its rows are appended in input order and never
        # out of it, so the file is always a prefix of the split - scoring the rows it
        # reached is still scoring the right rows.
        print(f"[WARN] {pred_path} holds {len(pred)} of {len(gold)} rows — an unfinished "
              f"run, scored over that prefix against {gold_name or 'the gold'}. Its length "
              "names no input, so check that is the one it was generated from.")
        rows = [i for i in rows if i < len(pred)]
    elif len(gold) != len(pred):
        raise ValueError(f"[{pred_path}] line counts differ: eval={len(gold)}, pred={len(pred)}. "
                         "Predictions must be line-aligned with the eval file.")

    global_tp = global_fp = global_fn = global_required = 0
    rows_exact_match = rows_evaluated = rows_skipped_empty_pred = rows_in_scope = 0
    per_key_agg: Dict[str, Dict[str, int]] = {}

    for i in rows:
        rows_in_scope += 1
        g, p = gold[i], pred[i]

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
        "gold": gold_name,
        "scored_rows": rows_in_scope,
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
    if m.get("gold"):
        print(f"Scored against                  : {m['gold']}")
    print(f"Rows in scored shard            : {m['scored_rows']}")
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


def print_summary_table(ms: List[Dict[str, Any]], caption: str) -> None:
    if not ms:
        return
    # `gold` earns its column only when the files did not all come from one input.
    cols = ["model", "precision", "recall", "f1", "accuracy_union", "em_row"]
    if len({m.get("gold", "") for m in ms}) > 1:
        cols.insert(1, "gold")
    widths = {
        c: max(len(c), max(len(f"{m[c]:.4f}") if isinstance(m[c], float) else len(str(m[c])) for m in ms))
        for c in cols
    }
    print("\n====== Summary ======")
    print(caption)
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
    ap.add_argument("--dataset", default=DEFAULT_FILL_DATASET,
                    help="save_to_disk dataset holding the gold for predictions from "
                         f"2.perf-eval-fill-gen-local.py (default: {DEFAULT_FILL_DATASET}). "
                         "Files are matched to it or to --eval by row count, so both kinds "
                         "can be scored in one run; --split, --test-size and --seed apply "
                         "only to --eval")
    ap.add_argument("--dataset-split", default="test",
                    help="Split of --dataset the predictions were generated from (default: test)")
    ap.add_argument("--partial", action="store_true",
                    help="Also score prediction files that stop short of the end, over the "
                         "prefix they reached. Only sound for --dataset runs, whose rows are "
                         "appended in input order; a short fill-eval file stays an error")
    ap.add_argument("--pred", nargs="*", help=f"Prediction files (default: scan {DEFAULT_PRED_ROOT}/*/)")
    ap.add_argument("--pred-root", default=DEFAULT_PRED_ROOT, help="Directory scanned when --pred is omitted")
    ap.add_argument("--tol", type=float, default=0.0, help="Absolute tolerance for float comparison")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="Which shard of the same train_test_split the fine-tuning script uses. "
                         f"'{TRAINED_SPLIT}' is what training consumed, so the default 'train' is "
                         "the held-out 70%% (default: train)")
    ap.add_argument("--test-size", type=float, default=SPLIT_TEST_SIZE,
                    help=f"Fraction held out by train_test_split (default: {SPLIT_TEST_SIZE}; "
                         "must match 1.fine-tune-GRPO-llm.py)")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED,
                    help=f"Split seed (default: {SPLIT_SEED}; must match 1.fine-tune-GRPO-llm.py)")
    ap.add_argument("--ft-dataset", default=None,
                    help="Fine-tuning dataset directory (e.g. llm-fill-ft.ds). When given, its "
                         "row alignment with the eval file is verified before scoring")
    ap.add_argument("--top_percent", type=float, default=100.0,
                    help="Score only this %% of the selected shard, as a cheap smoke test. Drawn in "
                         "permutation order, so it is a random sub-sample of the shard rather than "
                         "a prefix of the corpus (default: 100)")
    args = ap.parse_args()

    pred_list = args.pred or discover_pred_files(args.pred_root)
    if not pred_list:
        print(f"[ERROR] No prediction files found under {args.pred_root}/. Pass --pred explicitly.")
        return

    try:
        sources = gold_sources(args)
    except (OSError, ValueError, ImportError) as e:
        print(f"[ERROR] Could not read the gold answers: {e}")
        return
    if not sources:
        print(f"[ERROR] Neither {args.eval} nor {args.dataset} exists; nothing to score against.")
        return

    for source in sources.values():
        print(f"[INFO] {source['scope']}: {len(source['rows'])} of "
              f"{len(source['gold'])} rows scored.")
        if source["shard_warning"]:
            print(f"[WARN] {source['shard_warning']}")
    if args.top_percent < 100:
        print(f"[INFO] --top_percent {args.top_percent}: a sub-sample, for a quick check only.")

    results: List[Dict[str, Any]] = []
    for pred_path in (str(pathlib.Path(p)) for p in pred_list):
        if not os.path.exists(pred_path):
            print(f"[WARN] Not found, skipping: {pred_path}")
            continue
        try:
            pred = load_jsonl(pred_path)
            source = sources.get(len(pred))
            if source is None and args.partial:
                # Unfinished, so its length names no input: the longest gold it could be a
                # prefix of is the only candidate.
                source = next((s for n, s in sorted(sources.items()) if n > len(pred)), None)
            if source is None:
                raise ValueError(
                    f"{len(pred)} rows match no input ("
                    + ", ".join(f"{s['name']}={n}" for n, s in sorted(sources.items()))
                    + "). Predictions must be line-aligned with what produced them"
                    + ("" if args.partial else "; pass --partial to score an unfinished run"))
            metrics = evaluate_one(source["gold"], pred_path, args.tol, source["rows"],
                                   allow_partial=args.partial, pred=pred,
                                   gold_name=source["name"])
            print_one_metrics(metrics)
            results.append(metrics)
        except Exception as e:  # noqa: BLE001 - one bad file must not hide the others
            print(f"[ERROR] Failed: {pred_path} -> {e}")

    print_summary_table(results, "Scored against the input each file is aligned with.")


if __name__ == "__main__":
    main()
