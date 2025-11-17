#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, os, pathlib
from typing import Any, Dict, Tuple, List, Set

IGNORE_KEYS = {"interface", "dockerImage"}

# 保留默认列表作为兜底（如果 finished 目录不存在或为空时）
DEFAULT_PREDS = [
    r'finished\gpt-oss-20b\filled-output.jsonl',
    r'finished\gpt-oss-120b\filled-output.jsonl',
    r'finished\gemini-2.5-flash-lite\filled-output-gemini-2.5-flash-lite.jsonl',
    r'finished\gemini-2.5-flash\filled-output-gemini-2.5-flash.jsonl',
    r'finished\gemini-2.5-pro\filled-output-gemini-2.5-pro.jsonl',
    r'finished\Qwen2-0.5B\filled-output-Qwen2-0.5B-Instruct.jsonl',
    r'finished\Qwen2-0.5B-GRPO-1500\filled-output-checkpoint-1500.jsonl',
    r'finished\Qwen2-0.5B-GRPO-5500\filled-output-checkpoint-5500.jsonl',
]


def is_zero_value(v: Any) -> bool:
    return isinstance(v, (int, float)) and v == 0


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
                raise ValueError(f"JSON 解析失败 {path}:{line_no}: {e}")
    return out


def evaluate_pair(
    gold_row: Dict[str, Any],
    pred_row: Dict[str, Any],
    tol: float
) -> Tuple[int, int, int, int, List[Dict[str, Any]], Dict[str, Dict[str, int]], int]:
    """
    返回：
      TP, FP, FN, TOTAL_REQUIRED, diffs, per_key_stats, present_required
    """
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

    def upd_stats(key: str, which: str):
        d = per_key_stats.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "required": 0, "pred": 0})
        d[which] += 1

    for k in required_keys:
        per_key_stats.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "required": 0, "pred": 0})
        per_key_stats[k]["required"] += 1
    for k in pred_keys:
        per_key_stats.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "required": 0, "pred": 0})
        per_key_stats[k]["pred"] += 1

    present_required = 0
    for k in required_keys:
        gold_v = gold_ans.get(k)
        if k in pred_keys:
            present_required += 1
            pred_v = pred.get(k)
            if almost_equal(gold_v, pred_v, tol):
                tp += 1
                upd_stats(k, "tp")
            else:
                fp += 1
                fn += 1
                diffs.append({"type": "mismatch", "key": k, "gold": gold_v, "pred": pred_v})
                upd_stats(k, "fp")
                upd_stats(k, "fn")
        else:
            fn += 1
            diffs.append({"type": "missing", "key": k, "gold": gold_v, "pred": None})
            upd_stats(k, "fn")

    extra_keys = pred_keys - required_keys
    for k in extra_keys:
        fp += 1
        diffs.append({"type": "extra", "key": k, "gold": None, "pred": pred.get(k)})
        upd_stats(k, "fp")

    total_required = len(required_keys)
    return tp, fp, fn, total_required, diffs, per_key_stats, present_required


def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def model_label_from_path(p: str) -> str:
    p = pathlib.Path(p)
    if p.parent and p.parent.name:
        return p.parent.name
    return p.stem


# ===== 新增：从 finished/ 自动发现 jsonl 文件 =====
def discover_pred_files(root: str = "finished") -> List[str]:
    root_path = pathlib.Path(root)
    preds: List[str] = []
    if not root_path.exists():
        return preds

    # 每个子目录下找所有 *.jsonl
    for sub in sorted(root_path.iterdir()):
        if sub.is_dir():
            for jsonl in sorted(sub.glob("*.jsonl")):
                preds.append(str(jsonl))
    return preds


# ===== 新增：读取每个模型目录下的 txt 性能数据 =====
def load_time_stats_from_dir(model_dir: pathlib.Path) -> Dict[str, float]:
    """
    读取目录中所有 .txt 文件，解析形如：
        key=value
        key=value ms
    的行，返回 {key: float_value_ms}
    """
    stats: Dict[str, float] = {}
    if not model_dir.exists() or not model_dir.is_dir():
        return stats

    for txt_file in model_dir.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # 去掉单位 ms / MS
                    if v.lower().endswith("ms"):
                        v = v[:-2].strip()
                    try:
                        stats[k] = float(v)
                    except ValueError:
                        # 解析失败就跳过该行
                        continue
        except OSError:
            # 某个 txt 打不开就忽略
            continue

    return stats


def evaluate_one(
    gold_path: str,
    pred_path: str,
    tol: float,
    top_percent: float = 100.0
) -> Dict[str, Any]:
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"[{pred_path}] 两文件行数不一致：eval={len(gold)}, pred={len(pred)}。请检查输入。")

    n = len(gold)
    if not (0 < top_percent <= 100):
        raise ValueError(f"--top_percent 必须在 (0, 100] 区间内，当前为 {top_percent}")
    use_n = int(math.floor(n * top_percent / 100.0))
    use_n = max(1, use_n)
    gold = gold[:use_n]
    pred = pred[:use_n]

    global_tp = global_fp = global_fn = global_required = 0
    rows_exact_match = 0
    rows_evaluated = 0
    rows_skipped_empty_pred = 0

    per_key_agg: Dict[str, Dict[str, int]] = {}

    for i, (g, p) in enumerate(zip(gold, pred), 1):
        # 跳过空预测行
        if isinstance(p, dict) and len(p.keys()) == 0:
            rows_skipped_empty_pred += 1
            continue

        tp, fp, fn, req, diffs, per_key_stats, present_req = evaluate_pair(g, p, tol)

        rows_evaluated += 1
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_required += req

        if fp == 0 and fn == 0:
            rows_exact_match += 1

        for k, s in per_key_stats.items():
            agg = per_key_agg.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "required": 0, "pred": 0})
            for kk in agg.keys():
                agg[kk] += s.get(kk, 0)

    precision = safe_div(global_tp, (global_tp + global_fp))
    recall = safe_div(global_tp, (global_tp + global_fn))
    f1 = safe_div(2 * precision * recall, (precision + recall)) if (precision + recall) else 0.0
    em_field = safe_div(global_tp, global_required)  # = recall
    accuracy_union = safe_div(global_tp, (global_tp + global_fp + global_fn))
    em_row = safe_div(rows_exact_match, rows_evaluated)

    # 新增：读取对应目录下的 txt 时间统计
    model_dir = pathlib.Path(pred_path).parent
    time_stats = load_time_stats_from_dir(model_dir)

    metrics = {
        "model": model_label_from_path(pred_path),
        "evaluated_rows": rows_evaluated,
        "skipped_empty_pred_rows": rows_skipped_empty_pred,
        "required_fields": global_required,
        "TP": global_tp, "FP": global_fp, "FN": global_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy_union": accuracy_union,
        "em_field": em_field, "em_row": em_row,
        "time_stats": time_stats,  # 新增
    }
    return metrics


def print_one_metrics(m: Dict[str, Any]):
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
    print(f"Accuracy (required ∪ predicted) : {m['accuracy_union']:.4f}")


    ts = m.get("time_stats") or {}
    if ts:
        print("\n=== Time stats (ms) ===")
        for k in sorted(ts.keys()):
            print(f"{k:30s}: {ts[k]:.3f}")
    print()


def print_summary_table(ms: List[Dict[str, Any]]):
    if not ms:
        return
    cols = [
        "model", "precision", "recall", "f1",
        "accuracy_union", "em_row"
    ]
    widths = {
        c: max(len(c), max(len(f"{m[c]:.4f}") if isinstance(m[c], float) else len(str(m[c])) for m in ms))
        for c in cols
    }
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print("\n====== Summary ======")
    print(header)
    print(sep)
    for m in ms:
        row = []
        for c in cols:
            v = f"{m[c]:.4f}" if isinstance(m[c], float) else str(m[c])
            row.append(v.ljust(widths[c]))
        print(" | ".join(row))
    print()


def main():
    ap = argparse.ArgumentParser(description="Compare multiple LLM filled outputs against eval answers.")
    ap.add_argument("--eval", default="fill-eval.jsonl", help="fill-eval.jsonl（含 answer）")
    ap.add_argument("--pred", nargs="*", help="一个或多个 filled-output.jsonl（留空则自动扫描 finished/）")
    ap.add_argument("--tol", type=float, default=0.0, help="浮点比较的绝对容差（默认 0）")
    ap.add_argument(
        "--top_percent",
        type=float,
        default=70.0,
        help="仅比较前多少百分比的数据 (0-100]，默认 70"
    )
    args = ap.parse_args()

    if args.pred and len(args.pred) > 0:
        pred_list = args.pred
    else:
        # 优先从 finished/ 自动发现
        auto_preds = discover_pred_files("finished")
        if auto_preds:
            pred_list = auto_preds
        else:
            # 兜底使用原来的 DEFAULT_PREDS
            pred_list = DEFAULT_PREDS

    pred_list = [str(pathlib.Path(p)) for p in pred_list]

    results: List[Dict[str, Any]] = []
    for pred_path in pred_list:
        if not os.path.exists(pred_path):
            print(f"[WARN] 文件不存在，跳过：{pred_path}")
            continue
        try:
            metrics = evaluate_one(args.eval, pred_path, args.tol, args.top_percent)
            print_one_metrics(metrics)
            results.append(metrics)
        except Exception as e:
            print(f"[ERROR] 失败：{pred_path} -> {e}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
