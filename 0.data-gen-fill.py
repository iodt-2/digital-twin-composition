#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple
import requests
import re
import time
import os
import hashlib

# -------------------------
# Ollama 客户端（/api/chat）
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
# 工具方法
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
# LLM 提示词
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

# 锚文规则：不包含 interface/DTMI，不包含 dockerImage 路径，不包含任何 Telemetry
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
# 生成 instance / anchor
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

    for attempt in range(3):
        try:
            raw = client.chat(messages, temperature=0.6)
            raw_json = json_only(raw)
            gen_props = json.loads(raw_json)
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.0)

    # 构造完整 instance（供 answer 使用）
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

    for t in telemetries:
        name = t.get("name")
        instance[name] = 0

    return instance

def build_anchor_with_llm(
    client: OllamaClient,
    interface_obj: Dict[str, Any],
    full_instance: Dict[str, Any]
) -> str:
    # 仅允许的字段：Property 去掉 dockerImage；排除 interface；完全不含 Telemetry
    _, properties, telemetries = extract_fields(interface_obj)
    telemetry_names = {t.get("name") for t in telemetries if isinstance(t, dict)}
    property_names = {p.get("name") for p in properties if isinstance(p, dict)}

    allowed = {}
    for k, v in full_instance.items():
        if k == "interface":
            continue
        if k in telemetry_names:
            continue
        if k == "dockerImage":
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
# 断点相关
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

def append_checkpoint(path: str, h: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(h + "\n")
        f.flush()
        os.fsync(f.fileno())

# -------------------------
# 主流程（含 ETA 预测）
# -------------------------
def process_interfaces_file(
    input_path: str,
    output_path: str,
    host: str,
    model: str,
    limit: int = 0
):
    client = OllamaClient(host=host, model=model)

    # 统计总行数用于进度
    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            total_lines = sum(1 for _ in fin)
    except Exception as e:
        print(f"[FATAL] 无法读取输入文件: {e}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = output_path + ".ckpt"
    processed_hashes = load_checkpoint(ckpt_path)

    # 输出为追加模式，支持 resume
    fout = open(output_path, "a", encoding="utf-8")
    # 统计
    written = 0
    skipped = 0
    failed = 0

    # ETA 统计：按「成功写出样本」的平均耗时来估算
    total_sample_time = 0.0  # 成功样本累计耗时（秒）
    overall_start_ts = time.time()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                skipped += 1
                remaining = max(0, total_lines - (written + skipped + failed))
                avg = (total_sample_time / written) if written > 0 else 0.0
                eta_secs = avg * remaining
                eta_ts = time.time() + eta_secs
                print(f"[{line_num}/{total_lines}] 空行，跳过 | written={written} skipped={skipped} failed={failed} | "
                      f"ETA 剩余约 {format_duration(eta_secs)}，预计完成于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_ts))}",
                      flush=True)
                continue

            h = sha256_line(raw)
            if h in processed_hashes:
                skipped += 1
                remaining = max(0, total_lines - (written + skipped + failed))
                avg = (total_sample_time / written) if written > 0 else 0.0
                eta_secs = avg * remaining
                eta_ts = time.time() + eta_secs
                print(f"[{line_num}/{total_lines}] 已处理（resume 命中），跳过 | written={written} skipped={skipped} failed={failed} | "
                      f"ETA 剩余约 {format_duration(eta_secs)}，预计完成于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_ts))}",
                      flush=True)
                continue

            try:
                iface = json.loads(raw)
            except Exception as e:
                failed += 1
                append_checkpoint(ckpt_path, h)
                processed_hashes.add(h)
                remaining = max(0, total_lines - (written + skipped + failed))
                avg = (total_sample_time / written) if written > 0 else 0.0
                eta_secs = avg * remaining
                eta_ts = time.time() + eta_secs
                print(f"[{line_num}/{total_lines}] 解析失败，跳过：{e} | written={written} skipped={skipped} failed={failed} | "
                      f"ETA 剩余约 {format_duration(eta_secs)}，预计完成于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_ts))}",
                      flush=True)
                continue

            # === 计时开始：单个成功样本 ===
            sample_start = time.time()
            try:
                full_instance = build_instance_with_llm(client, iface)
                anchor = build_anchor_with_llm(client, iface, full_instance)
                record = {"anchor": anchor, "answer": full_instance}

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())

                append_checkpoint(ckpt_path, h)
                processed_hashes.add(h)

                written += 1
                # 更新样本耗时
                total_sample_time += (time.time() - sample_start)

                pct = (line_num / total_lines) * 100 if total_lines else 0
                remaining = max(0, total_lines - (written + skipped + failed))
                avg = (total_sample_time / written) if written > 0 else 0.0
                eta_secs = avg * remaining
                eta_ts = time.time() + eta_secs
                print(f"[{line_num}/{total_lines} | {pct:.1f}%] 写入成功 | written={written} skipped={skipped} failed={failed} | "
                      f"avg/sample={format_duration(avg)} | "
                      f"ETA 剩余约 {format_duration(eta_secs)}，预计完成于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_ts))}",
                      flush=True)

                if limit and written >= limit:
                    print(f"[INFO] 达到 --limit={limit}，结束。", flush=True)
                    break

            except Exception as e:
                failed += 1
                append_checkpoint(ckpt_path, h)
                processed_hashes.add(h)
                remaining = max(0, total_lines - (written + skipped + failed))
                avg = (total_sample_time / written) if written > 0 else 0.0
                eta_secs = avg * remaining
                eta_ts = time.time() + eta_secs
                print(f"[{line_num}/{total_lines}] 处理失败，跳过：{e} | written={written} skipped={skipped} failed={failed} | "
                      f"ETA 剩余约 {format_duration(eta_secs)}，预计完成于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_ts))}",
                      flush=True)
                continue

    fout.close()
    total_elapsed = time.time() - overall_start_ts
    avg = (total_sample_time / written) if written > 0 else 0.0
    print(f"完成。总行数={total_lines} | 写入={written} 跳过={skipped} 失败={failed} | "
          f"总耗时={format_duration(total_elapsed)} | avg/sample={format_duration(avg)} | 输出: {output_path}",
          flush=True)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="interfaces.jsonl", help="输入 interfaces.jsonl 路径（默认：interfaces.jsonl）")
    ap.add_argument("--output", default="fill-eval.jsonl", help="输出 jsonl 路径（默认：fill-eval.jsonl）")
    ap.add_argument("--host", default="http://10.1.1.1:60002", help="Ollama 主机地址（默认：http://10.1.1.1:60002）")
    ap.add_argument("--model", default="gpt-oss:120b", help="Ollama 模型名（默认：gpt-oss:120b）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少条（0=全部）")
    args = ap.parse_args()

    process_interfaces_file(
        input_path=args.input,
        output_path=args.output,
        host=args.host,
        model=args.model,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()
