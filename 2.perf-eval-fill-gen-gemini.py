import json
import sys
import os
import time
from pathlib import Path
import requests

# ----------------- 配置 -----------------

OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)

OPENAI_API_KEY = "api_key"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "models/gemini-2.5-pro")

INTERFACES_PATH = Path("interfaces.jsonl")
FILL_EVAL_PATH = Path(f"fill-eval.jsonl")

APD = OPENAI_MODEL.split("/")[1] if '/' in OPENAI_MODEL else OPENAI_MODEL
if not os.path.isdir(APD):
    os.mkdir(APD)

OUTPUT_PATH = Path(f"{APD}/filled-output-{APD}.jsonl")       # 结果文件：一行一个 JSON
DONE_INDEX_PATH = Path(f"{APD}/filled-output-{APD}.done")    # 进度索引文件：一行一个已完成的输入行号（0-based）
PROGRESS_META_PATH = Path(f"{APD}/progress-{APD}.json")      # 进度元数据文件：统计信息等
STATS_TXT_PATH = Path(f"{APD}/sample_time_stats-{APD}.txt")  # 保存单样本 min/max/avg 的 txt

HTTP_TIMEOUT = 120
# ----------------------------------------


def fsync_file(f):
    f.flush()
    os.fsync(f.fileno())


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m{s:02d}s"


def load_interfaces(path: Path):
    id_to_schema = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            interface_id = obj.get("@id")
            contents = obj.get("contents", [])
            props = []
            for c in contents:
                if type(c) == str:
                    continue
                if c.get("@type") == "Property":
                    name = c.get("name")
                    schema = c.get("schema", "string")
                    if name == "dockerImage":
                        continue
                    props.append({"name": name, "schema": schema})
            if interface_id:
                id_to_schema[interface_id] = {"properties": props}
    return id_to_schema


def coerce_type(value, schema_type: str):
    if value is None:
        return None
    st = (schema_type or "string").lower()
    try:
        if st in ("double", "float"):
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
            return None
        elif st in ("integer", "int", "long"):
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                return int(float(value.strip()))
            return None
        elif st in ("boolean", "bool"):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "yes", "y", "1"):
                    return True
                if v in ("false", "no", "n", "0"):
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return None
        else:
            return str(value)
    except Exception:
        return None


def call_openai_extract(anchor: str, interface_id: str, properties_spec: list):
    """
    调用 OpenAI 兼容 chat.completions 完成抽取。
    仅返回声明的属性（已在外层过滤 dockerImage）。
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "缺少 API Key：请设置环境变量 OPENAI_API_KEY 或 GENAI_API_KEY"
        )

    fields_desc = "\n".join(
        [f'- "{p["name"]}" ({p["schema"]})' for p in properties_spec]
    )

    system_prompt = (
        "You are an information extraction assistant. "
        "Extract values exactly from user-provided ANCHOR text. "
        "If a value is not explicitly stated, output null. "
        "Do not hallucinate or infer unstated facts. "
        "Return ONLY a minified JSON object with EXACT keys requested."
    )
    user_prompt = f"""
Interface: "{interface_id}"
Required fields and types:
{fields_desc}

ANCHOR:
\"\"\"{anchor}\"\"\"
""".strip()

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # 兼容 openai 格式
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    # 兜底抓取 JSON
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        obj = json.loads(text)
    except Exception:
        obj = {}

    clean = {}
    for p in properties_spec:
        name = p["name"]
        schema = p.get("schema", "string")
        val = obj.get(name, None)
        clean[name] = coerce_type(val, schema)
    return clean


def read_all_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def load_done_indices(path: Path):
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    done.add(int(ln))
                except ValueError:
                    continue
    return done


def append_done_index(f_done, idx: int):
    f_done.write(f"{idx}\n")
    fsync_file(f_done)


def write_stats_txt(min_sec, max_sec, avg_sec):
    """
    将单样本耗时统计写入 txt（覆盖写），并 fsync。
    """
    with STATS_TXT_PATH.open("w", encoding="utf-8") as f:
        content = [
            f"min_seconds_per_sample={min_sec:.6f}" if min_sec is not None else "min_seconds_per_sample=",
            f"max_seconds_per_sample={max_sec:.6f}" if max_sec is not None else "max_seconds_per_sample=",
            f"avg_seconds_per_sample={avg_sec:.6f}" if avg_sec is not None else "avg_seconds_per_sample=",
            "",
            f"min_human={human_time(min_sec) if min_sec is not None else '-'}",
            f"max_human={human_time(max_sec) if max_sec is not None else '-'}",
            f"avg_human={(f'{avg_sec:.2f}s/样本' if (avg_sec is not None and avg_sec > 0) else '-')}",
        ]
        f.write("\n".join(content))
        fsync_file(f)


def save_progress_meta(processed, total, avg_sec, min_sec, max_sec, started_ts, path=PROGRESS_META_PATH):
    meta = {
        "processed": processed,
        "total": total,
        "avg_seconds_per_sample": avg_sec,
        "min_seconds_per_sample": min_sec,
        "max_seconds_per_sample": max_sec,
        "started_at_unix": started_ts,
        "updated_at_unix": time.time_ns() / 1_000_000,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        fsync_file(f)


def main():
    if not INTERFACES_PATH.exists():
        print(f"[ERROR] 未找到 {INTERFACES_PATH}", file=sys.stderr, flush=True)
        sys.exit(1)
    if not FILL_EVAL_PATH.exists():
        print(f"[ERROR] 未找到 {FILL_EVAL_PATH}", file=sys.stderr, flush=True)
        sys.exit(1)

    id_to_schema = load_interfaces(INTERFACES_PATH)
    lines = read_all_lines(FILL_EVAL_PATH)
    total = len(lines)
    if total == 0:
        print("[INFO] 输入为空，无需处理。", flush=True)
        return

    done_indices = load_done_indices(DONE_INDEX_PATH)

    out_f = OUTPUT_PATH.open("a", encoding="utf-8", buffering=1)
    done_f = DONE_INDEX_PATH.open("a", encoding="utf-8", buffering=1)

    started_ts = time.time_ns() / 1_000_000

    # 运行期统计
    cumulative_seconds = 0.0
    session_processed = 0
    min_sec = None
    max_sec = None

    # 历史统计（用于 SKIP 与冷启动 ETA）
    hist_avg = None
    hist_min = None
    hist_max = None
    if PROGRESS_META_PATH.exists():
        try:
            with PROGRESS_META_PATH.open("r", encoding="utf-8") as pf:
                hist = json.load(pf)
                if hist.get("processed", 0) > 0:
                    hist_avg = float(hist.get("avg_seconds_per_sample", 0) or 0)
                    hm = hist.get("min_seconds_per_sample", None)
                    hx = hist.get("max_seconds_per_sample", None)
                    hist_min = float(hm) if hm is not None else None
                    hist_max = float(hx) if hx is not None else None
        except Exception:
            pass

    def get_avg():
        if session_processed > 0:
            return cumulative_seconds / session_processed
        return hist_avg or 0.0

    def get_min():
        return min_sec if min_sec is not None else hist_min

    def get_max():
        return max_sec if max_sec is not None else hist_max

    # 启动时先把当前（历史）统计写一份 txt，便于观察
    write_stats_txt(get_min(), get_max(), get_avg())

    for idx, raw in enumerate(lines):
        # 预估用于 SKIP 打印
        avg_sec = get_avg()
        cur_min = get_min()
        cur_max = get_max()

        if idx in done_indices:
            remaining = total - len(done_indices)
            eta = (remaining * avg_sec) if avg_sec > 0 else 0
            percent = (len(done_indices) / total) * 100
            min_str = human_time(cur_min) if cur_min is not None else "-"
            max_str = human_time(cur_max) if cur_max is not None else "-"
            avg_str = f"{avg_sec:.2f}s/样本" if avg_sec > 0 else "-"
            print(
                f"[SKIP] {idx+1}/{total}  已完成={len(done_indices)}  {percent:.2f}%  ETA={human_time(eta)}  "
                f"(单样本: 最快={min_str}, 最慢={max_str}, 平均={avg_str})",
                flush=True,
            )
            continue

        t0 = time.time_ns() / 1_000_000
        try:
            item = json.loads(raw)
        except Exception as e:
            print(f"[WARN] 第 {idx+1} 行解析 JSON 失败：{e}; 将跳过。", flush=True)
            out_f.write("{}\n"); fsync_file(out_f)
            append_done_index(done_f, idx)
            done_indices.add(idx)

            dt = time.time_ns() / 1_000_000 - t0
            session_processed += 1
            cumulative_seconds += dt
            min_sec = dt if (min_sec is None or dt < min_sec) else min_sec
            max_sec = dt if (max_sec is None or dt > max_sec) else max_sec
            avg_sec = get_avg()
            remaining = total - len(done_indices)
            eta = (remaining * avg_sec) if avg_sec > 0 else 0
            percent = (len(done_indices) / total) * 100
            save_progress_meta(len(done_indices), total, avg_sec, min_sec, max_sec, started_ts)
            write_stats_txt(min_sec, max_sec, avg_sec)
            print(
                f"[DONE] {idx+1}/{total}  用时={human_time(dt)}  已完成={len(done_indices)}  {percent:.2f}%  ETA={human_time(eta)}  "
                f"(单样本: 本次={human_time(dt)}, 最快={human_time(min_sec)}, 最慢={human_time(max_sec)}, 平均={avg_sec:.2f}s/样本)",
                flush=True,
            )
            continue

        interface_id = item.get("answer", {}).get("interface") or item.get("interface")
        anchor = item.get("anchor", "")

        if not interface_id or interface_id not in id_to_schema:
            print(f"[WARN] 第 {idx+1} 行找不到有效 interface 定义：{interface_id}; 将输出空对象。", flush=True)
            out_f.write("{}\n"); fsync_file(out_f)
            append_done_index(done_f, idx)
            done_indices.add(idx)

            dt = time.time_ns() / 1_000_000 - t0
            session_processed += 1
            cumulative_seconds += dt
            min_sec = dt if (min_sec is None or dt < min_sec) else min_sec
            max_sec = dt if (max_sec is None or dt > max_sec) else max_sec
            avg_sec = get_avg()
            remaining = total - len(done_indices)
            eta = (remaining * avg_sec) if avg_sec > 0 else 0
            percent = (len(done_indices) / total) * 100
            save_progress_meta(len(done_indices), total, avg_sec, min_sec, max_sec, started_ts)
            write_stats_txt(min_sec, max_sec, avg_sec)
            print(
                f"[DONE] {idx+1}/{total}  用时={human_time(dt)}  已完成={len(done_indices)}  {percent:.2f}%  ETA={human_time(eta)}  "
                f"(单样本: 本次={human_time(dt)}, 最快={human_time(min_sec)}, 最慢={human_time(max_sec)}, 平均={avg_sec:.2f}s/样本)",
                flush=True,
            )
            continue

        props_spec = id_to_schema[interface_id]["properties"]

        try:
            filled_props = call_openai_extract(anchor, interface_id, props_spec)
        except Exception as e:
            print(f"[ERROR] 第 {idx+1} 行调用 OpenAI API 出错：{e}; 输出空对象并继续。", flush=True)
            filled_props = {}

        result = {k: v for k, v in filled_props.items() if k != "dockerImage"}

        out_f.write(json.dumps(result, ensure_ascii=False) + "\n"); fsync_file(out_f)
        append_done_index(done_f, idx)
        done_indices.add(idx)

        dt = time.time_ns() / 1_000_000 - t0
        session_processed += 1
        cumulative_seconds += dt
        min_sec = dt if (min_sec is None or dt < min_sec) else min_sec
        max_sec = dt if (max_sec is None or dt > max_sec) else max_sec

        avg_sec = get_avg()
        remaining = total - len(done_indices)
        eta = (remaining * avg_sec) if avg_sec > 0 else 0
        percent = (len(done_indices) / total) * 100

        save_progress_meta(len(done_indices), total, avg_sec, min_sec, max_sec, started_ts)
        write_stats_txt(min_sec, max_sec, avg_sec)

        print(
            f"[DONE] {idx+1}/{total}  用时={human_time(dt / 1000)}  已完成={len(done_indices)}  {percent:.2f}%  ETA={human_time(eta / 1000)}  "
            f"(单样本: 本次={dt:.2f}ms, 最快={min_sec:.2f}ms, 最慢={max_sec:.2f}ms, 平均={avg_sec:.2f}ms/样本)",
            flush=True,
        )

    out_f.close()
    done_f.close()

    total_time = time.time_ns() / 1_000_000 - started_ts
    min_str = min_sec if min_sec is not None else "-"
    max_str = max_sec if max_sec is not None else "-"
    avg_str = f"{(cumulative_seconds / session_processed):.2f}ms/样本" if session_processed > 0 else "-"

    # 结束时再写一次 stats.txt（确保最终值）
    write_stats_txt(min_sec, max_sec, (cumulative_seconds / session_processed) if session_processed > 0 else None)

    print(
        f"\n[COMPLETE] 全部完成：{len(done_indices)}/{total}  总耗时={human_time(total_time / 1000)}  结果文件={OUTPUT_PATH}  "
        f"(单样本统计: 最快={min_str}, 最慢={max_str}, 平均={avg_str})",
        flush=True,
    )


if __name__ == "__main__":
    main()
