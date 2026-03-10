#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import ast
import time
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional, Set
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Configs from env ---
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://10.1.1.49:60002")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
DEFAULT_DATASET_PATH = os.getenv("ETE_EVAL_PATH", "./data/dataset_mid.jsonl")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./models/faiss.index")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./models/embeddings.npy")
METADATA_PATH = os.getenv("METADATA_PATH", "./models/metadata.json")
SENTENCE_TRANSFORMER_PATH = os.getenv("SENTENCE_TRANSFORMER_PATH", "./models/MiniLM-L6-based-new-triplets-final")

# local qwen model path
DEFAULT_QWEN_PATH = os.getenv("QWEN_MODEL_PATH", "./models/Qwen2-0.5B-GRPO-Fill-In")

# dataset_original for subsystem exact-match evaluation
DATASET_ORIGINAL_PATH = os.getenv("DATASET_ORIGINAL_PATH", "./data/dataset_original.jsonl")

# outputs
DEFAULT_EVAL_OUT_PATH = os.getenv("EVAL_OUT_PATH", "./outputs/evaluation_results.jsonl")
DEFAULT_DEBUG_OUT_PATH = os.getenv("DEBUG_OUT_PATH", "./outputs/debug_results.jsonl")

# Ignore keys for direct strict compare (hard ignore; not prompt-based)
IGNORE_KEYS = {"@id", "displayName", "dockerImage", "interface"}


# -------------------------
# logging / progress helpers
# -------------------------
class Logger:
    def __init__(self, mode: str = "brief"):
        self.mode = mode

    def brief(self, msg: str = "") -> None:
        print(msg)

    def verbose(self, msg: str = "") -> None:
        if self.mode == "verbose":
            print(msg)

    def section(self, msg: str = "") -> None:
        print(msg)


def pretty(obj: Any) -> str:
    if obj is None:
        return "<None>"
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return str(obj)


def minified_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def fmt_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_progress(current: int, total: int, elapsed_s: float, width: int = 24) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0.0), 1.0)
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    avg = elapsed_s / current if current > 0 else 0.0
    eta = avg * (total - current) if current > 0 else 0.0
    pct = ratio * 100.0
    return f"[{bar}] {current}/{total} {pct:6.2f}% | elapsed {fmt_seconds(elapsed_s)} | eta {fmt_seconds(eta)}"


# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def load_metadata(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("items", "docs", "data", "metadata"):
            v = obj.get(key)
            if isinstance(v, list):
                return v
    raise ValueError(f"Unsupported metadata format in {path}. Expected a list, or a dict containing a list.")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# dataset_original helpers
# -------------------------
def extract_group_id_from_dtmi(dtmi: str) -> str:
    if not isinstance(dtmi, str):
        return ""
    s = dtmi.strip()
    if not s:
        return ""
    parts = s.split(":")
    if len(parts) >= 3 and parts[0] == "dtmi":
        return parts[1].strip()
    return ""


def load_dataset_original_group_index(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATASET_ORIGINAL_PATH not found: {path}")

    rows = load_jsonl(path)
    group_map: Dict[str, Dict[str, Any]] = {}

    for idx0, row in enumerate(rows):
        iface = row.get("interface")
        if not isinstance(iface, dict):
            continue

        dtmi = iface.get("@id")
        if not isinstance(dtmi, str):
            continue

        gid = extract_group_id_from_dtmi(dtmi)
        if not gid:
            continue

        if gid not in group_map:
            group_map[gid] = {
                "faiss_ids_zero_based": [],
                "line_numbers_one_based": [],
                "interfaces": [],
            }

        group_map[gid]["faiss_ids_zero_based"].append(idx0)
        group_map[gid]["line_numbers_one_based"].append(idx0 + 1)
        group_map[gid]["interfaces"].append(
            {
                "faiss_id": idx0,
                "line_number": idx0 + 1,
                "@id": dtmi,
                "displayName": iface.get("displayName"),
            }
        )

    return group_map


# -------------------------
# FAISS + embedding helpers
# -------------------------
def guess_metric(index: "faiss.Index") -> str:
    if hasattr(index, "metric_type"):
        if index.metric_type == faiss.METRIC_L2:
            return "l2"
        if index.metric_type == faiss.METRIC_INNER_PRODUCT:
            return "ip"
    return "ip"


def maybe_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def build_query_vectors(model: SentenceTransformer, queries: List[str], normalize: bool) -> np.ndarray:
    q = model.encode(
        queries,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if normalize:
        q = maybe_normalize(q)
    return q


def faiss_search(index: "faiss.Index", qvecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if qvecs.dtype != np.float32:
        qvecs = qvecs.astype(np.float32)
    D, I = index.search(qvecs, k)
    return D, I


# -------------------------
# Interface parsing helpers
# -------------------------
def _try_parse_json_or_pyobj(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return s
    ss = s.strip()
    if not ss:
        return s

    try:
        return json.loads(ss)
    except Exception:
        pass

    try:
        return ast.literal_eval(ss)
    except Exception:
        return s


def unwrap_interface(interface_obj: Any) -> Any:
    if isinstance(interface_obj, dict) and "parsed" in interface_obj and isinstance(interface_obj["parsed"], dict):
        return interface_obj["parsed"]
    return interface_obj


def extract_interface_payload(meta: Optional[Dict[str, Any]]) -> Any:
    if not meta:
        return None
    for key in ("positive", "interface", "dtdl", "text", "content", "document", "chunk"):
        if key in meta and meta[key] is not None:
            return _try_parse_json_or_pyobj(meta[key])
    return meta


def interface_id_display(interface_obj: Any) -> str:
    interface_obj = unwrap_interface(interface_obj)
    if isinstance(interface_obj, dict):
        _id = interface_obj.get("@id") or interface_obj.get("id") or ""
        dn = interface_obj.get("displayName") or ""
        return f"@id={_id!r}, displayName={dn!r}"
    return "<non-dict interface>"


def get_contents_list_from_interface(interface_obj: Any) -> List[Dict[str, Any]]:
    interface_obj = unwrap_interface(interface_obj)
    if not isinstance(interface_obj, dict):
        return []

    if isinstance(interface_obj.get("contents"), list):
        return [c for c in interface_obj["contents"] if isinstance(c, dict)]

    contents: List[Dict[str, Any]] = []
    for k, v in interface_obj.items():
        if k.endswith("_properties_and_telemetries") and isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    contents.append(item)
    return contents


def get_property_fields_from_interface(interface_obj: Any) -> List[Dict[str, Any]]:
    contents = get_contents_list_from_interface(interface_obj)
    return [c for c in contents if c.get("@type") == "Property" and "name" in c]


def get_telemetry_field_names_from_interface(interface_obj: Any) -> Set[str]:
    contents = get_contents_list_from_interface(interface_obj)
    names: Set[str] = set()
    for c in contents:
        if c.get("@type") == "Telemetry" and "name" in c:
            names.add(str(c["name"]))
    return names


def safe_name(s: str) -> str:
    s = s.strip()
    if not s:
        return "subsystem"
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "subsystem"
    return s[:60]


def choose_subsystem_name(interface_obj: Any, fallback: str) -> str:
    interface_obj = unwrap_interface(interface_obj)
    if isinstance(interface_obj, dict):
        _id = interface_obj.get("@id") or ""
        dn = interface_obj.get("displayName") or ""
        if isinstance(_id, str) and _id:
            tail = _id.split(":")[-1]
            tail = tail.split(";")[0]
            return safe_name(tail) or fallback
        if isinstance(dn, str) and dn:
            return safe_name(dn) or fallback
    return fallback


def get_subsystem_blocks_from_composed_interface(composed_iface: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    blocks = []
    for k, v in composed_iface.items():
        if k.endswith("_properties_and_telemetries") and isinstance(v, list):
            blocks.append((k, [x for x in v if isinstance(x, dict)]))
    return blocks


def get_property_names_from_contents(contents: List[Dict[str, Any]]) -> List[str]:
    return [str(c["name"]) for c in contents if isinstance(c, dict) and c.get("@type") == "Property" and "name" in c]


# -------------------------
# text normalization helpers
# -------------------------
def normalize_text_for_compare(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    for ch in ["‐", "-", "‒", "–", "—", "−"]:
        s = s.replace(ch, "-")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool, dict, list)):
        return v
    if isinstance(v, str):
        s = normalize_text_for_compare(v)
        if s.lower() in ("null", "none", ""):
            return None
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            return s
    return v


# -------------------------
# Ollama helpers
# -------------------------
def ollama_generate(prompt: str, host: str, model: str, timeout_s: int = 120) -> str:
    host = host.rstrip("/")
    generate_url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    if requests is not None:
        try:
            r = requests.post(generate_url, json=payload, timeout=timeout_s)
            if r.ok:
                obj = r.json()
                if isinstance(obj, dict) and "response" in obj:
                    return str(obj["response"])
        except Exception:
            pass

        chat_url = f"{host}/api/chat"
        payload2 = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
        r2 = requests.post(chat_url, json=payload2, timeout=timeout_s)
        r2.raise_for_status()
        obj2 = r2.json()
        if isinstance(obj2, dict) and isinstance(obj2.get("message"), dict):
            return str(obj2["message"].get("content", ""))
        raise RuntimeError(f"Unexpected /api/chat response: {obj2}")

    import urllib.request

    def _post(url: str, data: dict) -> dict:
        b = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=b, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    try:
        obj = _post(generate_url, payload)
        if isinstance(obj, dict) and "response" in obj:
            return str(obj["response"])
    except Exception:
        pass

    chat_url = f"{host}/api/chat"
    payload2 = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
    obj2 = _post(chat_url, payload2)
    if isinstance(obj2, dict) and isinstance(obj2.get("message"), dict):
        return str(obj2["message"].get("content", ""))
    raise RuntimeError(f"Unexpected ollama response (no requests): {obj2}")


# -------------------------
# Local Qwen helpers
# -------------------------
_QWEN_CACHE: Dict[str, Any] = {}


def qwen_generate(
    prompt: str,
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise SystemExit(
            "Local Qwen backend requires transformers + torch.\n"
            "Install with: pip install transformers torch\n"
        )

    key = f"{model_path}"
    if key not in _QWEN_CACHE:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto",
        )
        _QWEN_CACHE[key] = (tok, mdl)
    tok, mdl = _QWEN_CACHE[key]

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
        eos_token_id=getattr(tok, "eos_token_id", None),
    )
    with torch.no_grad():
        out = mdl.generate(**inputs, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def parse_model_json_output(text: str) -> Any:
    s = text.strip()

    def _slice_braces(s0: str) -> str:
        l1, r1 = s0.find("{"), s0.rfind("}")
        l2, r2 = s0.find("["), s0.rfind("]")
        candidates = []
        if l1 != -1 and r1 != -1 and r1 > l1:
            candidates.append((l1, r1))
        if l2 != -1 and r2 != -1 and r2 > l2:
            candidates.append((l2, r2))
        if not candidates:
            return s0
        l, r = sorted(candidates, key=lambda x: x[0])[0]
        return s0[l:r + 1]

    if not (s.startswith("{") or s.startswith("[")):
        s = _slice_braces(s)

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        raise ValueError(f"Cannot parse model output as JSON/Python literal.\nRaw output:\n{text}")


# -------------------------
# Decomposition / Composition
# -------------------------
def build_decompose_prompt(description_text: str, max_parts: int) -> str:
    return (
        "You are a decomposition assistant.\n"
        "Given a DESCRIPTION of a desired composed a large digital twin system, split it into sub-queries for subsystem "
        "digital twin interfaces used to search sub-components, reformat and extract information for the subsystem"
        "so the sub-query can be used in the FAISS search.\n"
        "Important constraints:\n"
        f"- Return ONLY minified JSON array of strings.\n"
        f"- Array length must be between 1 and {max_parts}.\n"
        "- Each sub-query must be self-contained and target ONE subsystem/interface.\n"
        "- You MAY summarize wording, BUT you MUST preserve ALL stated facts and constraints.\n"
        "- Do NOT drop or generalize any information.\n"
        "- Do NOT simplify information.\n"
        "- Keep such values verbatim when present.\n"
        "- Do not add new explanations.\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )


def compose_interfaces(
    subsystem_interfaces: List[Dict[str, Any]],
    composed_id: str,
    composed_display: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    composed: Dict[str, Any] = {
        "@context": "dtmi:dtdl:context;2",
        "@id": composed_id,
        "@type": "Interface",
        "displayName": composed_display,
    }

    manifest: List[Dict[str, Any]] = []
    used_keys = set(composed.keys())

    for i, pack in enumerate(subsystem_interfaces):
        faiss_id = pack.get("faiss_id")
        iface = pack.get("interface")
        iface_u = unwrap_interface(iface)

        fallback = f"sub_system_{i + 1}"
        name = choose_subsystem_name(iface_u, fallback=fallback)
        key = f"{name}_properties_and_telemetries"
        if key in used_keys:
            key = f"{name}_{i + 1}_properties_and_telemetries"
        used_keys.add(key)

        contents = []
        if isinstance(iface_u, dict) and isinstance(iface_u.get("contents"), list):
            contents = [c for c in iface_u["contents"] if isinstance(c, dict)]
        else:
            contents = get_contents_list_from_interface(iface_u)

        composed[key] = contents
        manifest.append(
            {
                "subsystem_idx": i + 1,
                "faiss_id": faiss_id,
                "sub_query": pack.get("sub_query"),
                "score": pack.get("score"),
                "subsystem_key": key,
                "contents_len": len(contents),
                "interface_id_display": interface_id_display(iface_u),
                "interface_id": iface_u.get("@id") if isinstance(iface_u, dict) else None,
                "displayName": iface_u.get("displayName") if isinstance(iface_u, dict) else None,
            }
        )

    return composed, manifest


# -------------------------
# Fill-in prompts
# -------------------------
def build_fillin_prompt_direct_flat_instance(description_text: str, interface_obj: Dict[str, Any]) -> str:
    props = get_property_fields_from_interface(interface_obj)
    prop_names = [str(p["name"]) for p in props if "name" in p]
    required_keys = ["interface"] + prop_names

    iface_id = interface_obj.get("@id") if isinstance(interface_obj.get("@id"), str) else ""

    prompt = (
        "You are an information extraction assistant.\n"
        "Return a FULL initiated instance for the given Interface.\n"
        "Rules:\n"
        "- Use ONLY values explicitly stated in DESCRIPTION. Do not infer unstated facts.\n"
        "- If a value is not stated, use null.\n"
        "- Return ONLY minified JSON (no markdown, no comments).\n"
        f"- JSON must contain EXACTLY these keys (and no others): {required_keys}\n"
        f"- The 'interface' field MUST be exactly: {iface_id!r}\n\n"
        "INTERFACE (for context):\n"
        f"{minified_json(interface_obj)}\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )
    return prompt


def build_fillin_prompt_composed_nested_instance(description_text: str, composed_interface: Dict[str, Any]) -> str:
    blocks = get_subsystem_blocks_from_composed_interface(composed_interface)
    iface_id = composed_interface.get("@id") if isinstance(composed_interface.get("@id"), str) else ""

    schema_spec: Dict[str, List[str]] = {}
    for key, contents in blocks:
        subsystem_name = key.replace("_properties_and_telemetries", "")
        schema_spec[subsystem_name] = get_property_names_from_contents(contents)

    prompt = (
        "You are an information extraction assistant.\n"
        "Return a FULL initiated instance for a COMPOSED Interface with multiple subsystems.\n"
        "Rules:\n"
        "- Use ONLY values explicitly stated in DESCRIPTION. Do not infer unstated facts.\n"
        "- If a value is not stated, use null.\n"
        "- Do NOT output any telemetry fields.\n"
        "- Return ONLY minified JSON (no markdown, no comments).\n"
        "- Output must follow EXACTLY this JSON shape:\n"
        '  {"interface": "<interface_id>", "subsystems": { "<subsystem_name>": { "<property>": value_or_null, ... }, ... }}\n'
        f"- The 'interface' field MUST be exactly: {iface_id!r}\n"
        f"- The 'subsystems' keys MUST be exactly: {list(schema_spec.keys())}\n"
        "- For each subsystem, include EXACTLY the listed property keys (and no others), in this spec:\n"
        f"{minified_json(schema_spec)}\n\n"
        "COMPOSED INTERFACE (for context):\n"
        f"{minified_json(composed_interface)}\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )
    return prompt


# -------------------------
# Evaluation helpers
# -------------------------
def strict_compare_direct_instance(
    predicted_instance: Dict[str, Any],
    expected_output: Dict[str, Any],
    interface_obj: Dict[str, Any],
) -> Dict[str, Any]:
    telemetry_keys = get_telemetry_field_names_from_interface(interface_obj)

    expected_keys = [k for k in expected_output.keys() if k not in IGNORE_KEYS and k not in telemetry_keys]
    pred_keys = [k for k in predicted_instance.keys() if k not in IGNORE_KEYS and k not in telemetry_keys]

    expected_set = set(expected_keys)
    pred_set = set(pred_keys)

    missing = sorted(list(expected_set - pred_set))
    extra = sorted(list(pred_set - expected_set))
    fields: Dict[str, Any] = {}

    tp = 0
    fp = 0
    fn = 0

    for k in sorted(expected_set & pred_set):
        pv = normalize_value(predicted_instance.get(k))
        gv = normalize_value(expected_output.get(k))
        ok = pv == gv
        fields[k] = {"pred": pv, "gt": gv, "ok": ok}
        if ok:
            tp += 1
        else:
            fp += 1
            fn += 1

    fp += len(extra)
    fn += len(missing)

    evaluated_total = tp + fp
    recall_total = tp + fn
    precision = tp / evaluated_total if evaluated_total else 0.0
    recall = tp / recall_total if recall_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    overall_ok = (len(missing) == 0 and len(extra) == 0 and fp == 0)

    return {
        "mode": "direct_strict_compare",
        "overall_ok": overall_ok,
        "missing_keys": missing,
        "extra_keys": extra,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fields": fields,
    }


def evaluate_subsystem_exact_match(
    desired_group_id: Optional[str],
    manifest: Optional[List[Dict[str, Any]]],
    dataset_original_group_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    retrieved_ids: List[int] = []
    if manifest:
        for m in manifest:
            fid = m.get("faiss_id")
            if isinstance(fid, int) and fid >= 0:
                retrieved_ids.append(fid)

    report: Dict[str, Any] = {
        "desired_group_id": desired_group_id,
        "retrieved_faiss_ids": retrieved_ids,
        "retrieved_count": len(retrieved_ids),
        "expected_count": 0,
        "overall_ok": False,
        "reason": "",
        "correct_faiss_ids_from_expected_lines": [],
        "valid_retrieved_faiss_ids": [],
        "invalid_retrieved_faiss_ids": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    if not desired_group_id:
        report["reason"] = "desired group_id missing in ETE_EVAL_PATH row"
        return report

    target = dataset_original_group_map.get(desired_group_id)
    if not target:
        report["reason"] = f"group_id={desired_group_id!r} not found in dataset_original"
        return report

    expected_line_numbers_one_based = [int(x) for x in target.get("line_numbers_one_based", [])]
    allowed_faiss_ids = sorted({ln - 1 for ln in expected_line_numbers_one_based if isinstance(ln, int) and ln >= 1})

    valid_retrieved = [fid for fid in retrieved_ids if fid in set(allowed_faiss_ids)]
    invalid_retrieved = [fid for fid in retrieved_ids if fid not in set(allowed_faiss_ids)]

    tp = len(valid_retrieved)
    fp = len(invalid_retrieved)
    fn = max(0, len(allowed_faiss_ids) - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    report["expected_count"] = len(allowed_faiss_ids)
    report["correct_faiss_ids_from_expected_lines"] = allowed_faiss_ids
    report["valid_retrieved_faiss_ids"] = valid_retrieved
    report["invalid_retrieved_faiss_ids"] = invalid_retrieved
    report["overall_ok"] = (len(invalid_retrieved) == 0)
    report["reason"] = (
        "all retrieved faiss_ids are in the correct lines"
        if report["overall_ok"]
        else "some retrieved faiss_ids are not correct"
    )
    report["tp"] = tp
    report["fp"] = fp
    report["fn"] = fn
    report["precision"] = precision
    report["recall"] = recall
    report["f1"] = f1
    return report


def build_verify_prompt_for_direct(
    description_text: str,
    interface_obj: Dict[str, Any],
    predicted_instance: Dict[str, Any],
    expected_output: Dict[str, Any],
) -> str:
    telemetry_keys = sorted(list(get_telemetry_field_names_from_interface(interface_obj)))
    expected_keys = [
        k for k in expected_output.keys()
        if k not in IGNORE_KEYS and k not in telemetry_keys
    ]

    prompt = (
        "You are a strict but semantically-aware verification assistant.\n"
        "You will be given:\n"
        "1) DESCRIPTION text\n"
        "2) An Interface schema\n"
        "3) A PREDICTED instance JSON\n"
        "4) An EXPECTED instance JSON\n\n"
        "Task:\n"
        "- Verify each field in PREDICTED against EXPECTED.\n"
        "- Final judgment must be semantic, not only string-exact.\n"
        "- Treat equivalent expressions as MATCH when they clearly refer to the same value.\n"
        "- Examples of MATCH:\n"
        "  - abbreviation vs full form: 'nickel-manganese-cobalt (NMC)' == 'NMC'\n"
        "  - different unicode hyphen characters vs normal hyphen\n"
        "  - harmless formatting differences, capitalization differences, spacing differences\n"
        "  - numeric string vs numeric value when they represent the same value\n"
        "- Mark MISMATCH only if the values are genuinely different in meaning.\n"
        "- Ignore keys in this fixed ignore list: "
        f"{sorted(list(IGNORE_KEYS))}\n"
        "- Ignore telemetry fields entirely: "
        f"{telemetry_keys}\n"
        "- Only evaluate these keys:\n"
        f"{expected_keys}\n\n"
        "Output requirements:\n"
        "- Return ONLY minified JSON.\n"
        "- Use EXACTLY this JSON shape:\n"
        '{"overall_result":"PASS|FAIL","summary":{"matched":0,"mismatched":0,"accuracy":0.0},"fields":{"<key>":{"pred":null,"gt":null,"result":"PASS|FAIL","reason":""}},"notes":""}\n'
        "- Keep reason concise.\n"
        "- If values are semantically equivalent, result MUST be PASS.\n\n"
        "INTERFACE:\n"
        f"{minified_json(interface_obj)}\n\n"
        "PREDICTED INSTANCE:\n"
        f"{minified_json(predicted_instance)}\n\n"
        "EXPECTED INSTANCE:\n"
        f"{minified_json(expected_output)}\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )
    return prompt


def build_direct_final_eval(
    raw_strict_eval: Dict[str, Any],
    llm_verify_obj: Dict[str, Any],
) -> Dict[str, Any]:
    fields = llm_verify_obj.get("fields", {}) if isinstance(llm_verify_obj, dict) else {}
    tp = 0
    fp = 0
    fn = 0

    if isinstance(fields, dict):
        for _, v in fields.items():
            if isinstance(v, dict) and v.get("result") == "PASS":
                tp += 1
            else:
                fp += 1
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    overall_ok = (llm_verify_obj.get("overall_result") == "PASS")

    return {
        "mode": "direct_llm_verify",
        "overall_ok": overall_ok,
        "final_source": "llm_verify",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "raw_strict_compare": raw_strict_eval,
        "llm_verify": llm_verify_obj,
    }


def build_verify_prompt_for_composed(description_text: str, composed_interface: Dict[str, Any], instance_obj: Dict[str, Any]) -> str:
    blocks = get_subsystem_blocks_from_composed_interface(composed_interface)
    schema_spec: Dict[str, List[str]] = {}
    for key, contents in blocks:
        subsystem_name = key.replace("_properties_and_telemetries", "")
        schema_spec[subsystem_name] = get_property_names_from_contents(contents)

    prompt = (
        "You are a strict verification assistant.\n"
        "You will be given:\n"
        "1) DESCRIPTION text\n"
        "2) A COMPOSED Interface schema for subsystems and their Property keys\n"
        "3) A filled INSTANCE JSON\n\n"
        "Task:\n"
        "- Verify whether each subsystem Property in INSTANCE is filled with a value that is explicitly stated in DESCRIPTION.\n"
        "- Mark as mismatch if value is not stated, incorrect, wrong unit, wrong ID, or inferred.\n"
        "- null is OK if DESCRIPTION does not state the value, include and state it in the output.\n"
        "- Do NOT evaluate any telemetry fields.\n\n"
        "Output requirements:\n"
        "- Return ONLY minified JSON (no markdown, no comments).\n"
        "- Use EXACTLY this JSON shape:\n"
        '{"overall_result":"PASS|FAIL","subsystems":{"<name>":{"result":"PASS|FAIL","matches":[{"key":"...","value":...,"evidence":"..."}],"mismatches":[{"key":"...","value":...,"reason":"..."}],"notes":""}},"notes":""}\n'
        "- Keep evidence short (<= 15 words), copied from DESCRIPTION when possible.\n\n"
        "SUBSYSTEM PROPERTY SPEC:\n"
        f"{minified_json(schema_spec)}\n\n"
        "COMPOSED INTERFACE (context):\n"
        f"{minified_json(composed_interface)}\n\n"
        "INSTANCE:\n"
        f"{minified_json(instance_obj)}\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )
    return prompt


def build_paper_record_base(
    qi: int,
    query: str,
    path: str,
    top1_faiss_id: Optional[int],
    top1_score: Optional[float],
    status: str,
    route_time_seconds: Optional[float],
) -> Dict[str, Any]:
    return {
        "query_index": qi,
        "path": path,
        "top1_faiss_id": top1_faiss_id,
        "top1_score": top1_score,
        "status": status,
        "query_length": len(query),
        "route_time_seconds": route_time_seconds,
    }


def build_paper_record_direct(
    qi: int,
    query: str,
    top1_faiss_id: Optional[int],
    top1_score: Optional[float],
    status: str,
    final_eval: Optional[Dict[str, Any]],
    route_time_seconds: Optional[float],
) -> Dict[str, Any]:
    rec = build_paper_record_base(qi, query, "direct", top1_faiss_id, top1_score, status, route_time_seconds)
    if isinstance(final_eval, dict):
        rec.update({
            "interface_match_ok": bool(final_eval.get("overall_ok", False)),
            "tp": final_eval.get("tp"),
            "fp": final_eval.get("fp"),
            "fn": final_eval.get("fn"),
            "tn": final_eval.get("tn"),
            "precision": final_eval.get("precision"),
            "recall": final_eval.get("recall"),
            "f1": final_eval.get("f1"),
        })
    else:
        rec.update({
            "interface_match_ok": False,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "precision": None,
            "recall": None,
            "f1": None,
        })
    return rec


def build_paper_record_composed(
    qi: int,
    query: str,
    top1_faiss_id: Optional[int],
    top1_score: Optional[float],
    status: str,
    subsystem_eval: Optional[Dict[str, Any]],
    verify_eval: Optional[Dict[str, Any]],
    route_time_seconds: Optional[float],
) -> Dict[str, Any]:
    rec = build_paper_record_base(qi, query, "decompose+compose", top1_faiss_id, top1_score, status, route_time_seconds)

    subsystem_ok = False
    tp = fp = fn = tn = precision = recall = f1 = None
    if isinstance(subsystem_eval, dict):
        subsystem_ok = bool(subsystem_eval.get("overall_ok", False))
        tp = subsystem_eval.get("tp")
        fp = subsystem_eval.get("fp")
        fn = subsystem_eval.get("fn")
        tn = subsystem_eval.get("tn")
        precision = subsystem_eval.get("precision")
        recall = subsystem_eval.get("recall")
        f1 = subsystem_eval.get("f1")

    llm_verify_ok = None
    if isinstance(verify_eval, dict):
        llm_verify_ok = (verify_eval.get("overall_result") == "PASS")

    rec.update({
        "interface_match_ok": subsystem_ok,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "llm_verify_ok": llm_verify_ok,
    })
    return rec


def build_debug_record(
    qi: int,
    query: str,
    desired_group_id: Optional[str],
    path: str,
    top1: Optional[Dict[str, Any]],
    status: str,
    route_time_seconds: Optional[float],
    manifest: Optional[List[Dict[str, Any]]] = None,
    direct_eval: Optional[Dict[str, Any]] = None,
    subsystem_eval: Optional[Dict[str, Any]] = None,
    verify_eval: Optional[Dict[str, Any]] = None,
    predicted_instance: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "query_index": qi,
        "query": query,
        "desired_group_id": desired_group_id,
        "path": path,
        "status": status,
        "route_time_seconds": route_time_seconds,
        "top1": top1,
        "error": error,
    }
    if manifest is not None:
        rec["manifest"] = manifest
    if direct_eval is not None:
        rec["direct_eval"] = direct_eval
    if subsystem_eval is not None:
        rec["subsystem_exact_match_eval"] = subsystem_eval
    if verify_eval is not None:
        rec["verify_eval"] = verify_eval
    if predicted_instance is not None:
        rec["predicted_instance"] = predicted_instance
    return rec


def compute_time_stats(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    vals = [float(r["route_time_seconds"]) for r in rows if isinstance(r.get("route_time_seconds"), (int, float))]
    if not vals:
        return {
            "avg_route_time_seconds": None,
            "median_route_time_seconds": None,
            "p95_route_time_seconds": None,
            "min_route_time_seconds": None,
            "max_route_time_seconds": None,
        }

    arr = np.array(vals, dtype=np.float64)
    return {
        "avg_route_time_seconds": float(np.mean(arr)),
        "median_route_time_seconds": float(np.median(arr)),
        "p95_route_time_seconds": float(np.percentile(arr, 95)),
        "min_route_time_seconds": float(np.min(arr)),
        "max_route_time_seconds": float(np.max(arr)),
    }


def summarize_paper_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    direct = [r for r in records if r.get("path") == "direct"]
    composed = [r for r in records if r.get("path") == "decompose+compose"]

    def avg(vals: List[Optional[float]]) -> Optional[float]:
        xs = [float(x) for x in vals if isinstance(x, (int, float))]
        return (sum(xs) / len(xs)) if xs else None

    def sum_metric(rows: List[Dict[str, Any]], key: str) -> int:
        total = 0
        for r in rows:
            v = r.get(key)
            if isinstance(v, int):
                total += v
        return total

    def ratio_true(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        if not rows:
            return None
        cnt = sum(1 for r in rows if r.get(key) is True)
        return cnt / len(rows)

    all_rows = list(records)

    overall_tp = sum_metric(all_rows, "tp")
    overall_fp = sum_metric(all_rows, "fp")
    overall_fn = sum_metric(all_rows, "fn")

    overall_micro_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else None
    overall_micro_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else None
    overall_micro_f1 = (
        2 * overall_micro_precision * overall_micro_recall / (overall_micro_precision + overall_micro_recall)
        if (
            overall_micro_precision is not None
            and overall_micro_recall is not None
            and (overall_micro_precision + overall_micro_recall) > 0
        )
        else None
    )

    direct_tp = sum_metric(direct, "tp")
    direct_fp = sum_metric(direct, "fp")
    direct_fn = sum_metric(direct, "fn")
    direct_micro_precision = direct_tp / (direct_tp + direct_fp) if (direct_tp + direct_fp) else None
    direct_micro_recall = direct_tp / (direct_tp + direct_fn) if (direct_tp + direct_fn) else None
    direct_micro_f1 = (
        2 * direct_micro_precision * direct_micro_recall / (direct_micro_precision + direct_micro_recall)
        if (
            direct_micro_precision is not None
            and direct_micro_recall is not None
            and (direct_micro_precision + direct_micro_recall) > 0
        )
        else None
    )

    composed_tp = sum_metric(composed, "tp")
    composed_fp = sum_metric(composed, "fp")
    composed_fn = sum_metric(composed, "fn")
    composed_micro_precision = composed_tp / (composed_tp + composed_fp) if (composed_tp + composed_fp) else None
    composed_micro_recall = composed_tp / (composed_tp + composed_fn) if (composed_tp + composed_fn) else None
    composed_micro_f1 = (
        2 * composed_micro_precision * composed_micro_recall / (composed_micro_precision + composed_micro_recall)
        if (
            composed_micro_precision is not None
            and composed_micro_recall is not None
            and (composed_micro_precision + composed_micro_recall) > 0
        )
        else None
    )

    summary = {
        "overall_summary": {
            "count": len(all_rows),
            "num_direct": len(direct),
            "num_composed": len(composed),
            "interface_match_accuracy": ratio_true(all_rows, "interface_match_ok"),
            "llm_verify_accuracy_on_composed_only": ratio_true(composed, "llm_verify_ok"),
            "tp_total": overall_tp,
            "fp_total": overall_fp,
            "fn_total": overall_fn,
            "macro_precision": avg([r.get("precision") for r in all_rows]),
            "macro_recall": avg([r.get("recall") for r in all_rows]),
            "macro_f1": avg([r.get("f1") for r in all_rows]),
            "micro_precision": overall_micro_precision,
            "micro_recall": overall_micro_recall,
            "micro_f1": overall_micro_f1,
            **compute_time_stats(all_rows),
        },
        "direct_route": {
            "count": len(direct),
            "interface_match_accuracy": ratio_true(direct, "interface_match_ok"),
            "tp_total": direct_tp,
            "fp_total": direct_fp,
            "fn_total": direct_fn,
            "macro_precision": avg([r.get("precision") for r in direct]),
            "macro_recall": avg([r.get("recall") for r in direct]),
            "macro_f1": avg([r.get("f1") for r in direct]),
            "micro_precision": direct_micro_precision,
            "micro_recall": direct_micro_recall,
            "micro_f1": direct_micro_f1,
            **compute_time_stats(direct),
        },
        "decompose_route": {
            "count": len(composed),
            "interface_match_accuracy": ratio_true(composed, "interface_match_ok"),
            "llm_verify_accuracy": ratio_true(composed, "llm_verify_ok"),
            "tp_total": composed_tp,
            "fp_total": composed_fp,
            "fn_total": composed_fn,
            "macro_precision": avg([r.get("precision") for r in composed]),
            "macro_recall": avg([r.get("recall") for r in composed]),
            "macro_f1": avg([r.get("f1") for r in composed]),
            "micro_precision": composed_micro_precision,
            "micro_recall": composed_micro_recall,
            "micro_f1": composed_micro_f1,
            **compute_time_stats(composed),
        },
    }
    return summary


# -------------------------
# main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="Top-k results per query (default: 1).")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N queries (0 = all).")
    parser.add_argument("--start", type=int, default=0, help="Start offset into the JSONL dataset.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize query embeddings (recommended for IP/cosine).")
    parser.add_argument("--metric", choices=["auto", "ip", "l2"], default="auto", help="FAISS metric: ip or l2.")
    parser.add_argument("--timeout", type=int, default=120, help="Ollama request timeout (seconds).")

    parser.add_argument("--min_sim", type=float, default=0.80, help="If top1 sim < min_sim, run decomposition/composition.")
    parser.add_argument("--max_subsystems", type=int, default=5, help="Max number of subsystems after decomposition.")

    parser.add_argument("--no_fillin", action="store_true", help="Disable fill-in + evaluation entirely.")
    parser.add_argument("--no_decompose", action="store_true", help="Disable decomposition/composition even if sim is low.")
    parser.add_argument("--no_verify", action="store_true", help="Disable composed-case LLM verify step (still prints instance).")

    parser.add_argument(
        "--fill_backend",
        choices=["ollama", "qwen"],
        default="ollama",
        help="Fill-in model backend: ollama or local qwen (default: ollama).",
    )
    parser.add_argument("--qwen_path", type=str, default=DEFAULT_QWEN_PATH, help="Local Qwen model path (HF format).")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=1024, help="Max new tokens for local Qwen generation.")
    parser.add_argument("--qwen_temperature", type=float, default=0.0, help="Temperature for local Qwen generation.")

    parser.add_argument("--dataset_original_path", type=str, default=DATASET_ORIGINAL_PATH, help="dataset_original.jsonl path.")
    parser.add_argument("--eval_out", type=str, default=DEFAULT_EVAL_OUT_PATH, help="Compact paper-style evaluation JSONL.")
    parser.add_argument("--debug_out", type=str, default=DEFAULT_DEBUG_OUT_PATH, help="Debug JSONL output path.")
    parser.add_argument("--summary_out", type=str, default="", help="Compact summary JSON output path. Default: <eval_out>.summary.json")
    parser.add_argument("--debug_summary_out", type=str, default="", help="Debug summary JSON output path. Default: <debug_out>.summary.json")
    parser.add_argument("--print_mode", choices=["brief", "verbose"], default="brief", help="brief=compact runtime logs, verbose=current style detailed logs")
    args = parser.parse_args()

    logger = Logger(mode=args.print_mode)

    summary_out = args.summary_out.strip() if isinstance(args.summary_out, str) else ""
    if not summary_out:
        summary_out = args.eval_out + ".summary.json"

    debug_summary_out = args.debug_summary_out.strip() if isinstance(args.debug_summary_out, str) else ""
    if not debug_summary_out:
        debug_summary_out = args.debug_out + ".summary.json"

    for path in [args.eval_out, args.debug_out, summary_out, debug_summary_out]:
        ensure_parent_dir(path)

    with open(args.eval_out, "w", encoding="utf-8") as f:
        pass
    with open(args.debug_out, "w", encoding="utf-8") as f:
        pass

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS_INDEX_PATH not found: {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)

    if os.path.exists(EMBEDDINGS_PATH):
        try:
            emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
            if hasattr(index, "ntotal") and emb.shape[0] != index.ntotal:
                logger.brief(
                    f"[warn] embeddings rows ({emb.shape[0]}) != index.ntotal ({index.ntotal})."
                )
        except Exception as e:
            logger.brief(f"[warn] Could not load embeddings from {EMBEDDINGS_PATH}: {e}")

    metadata: List[Dict[str, Any]] = []
    if os.path.exists(METADATA_PATH):
        metadata = load_metadata(METADATA_PATH)
        if hasattr(index, "ntotal") and len(metadata) != index.ntotal:
            logger.brief(
                f"[warn] metadata items ({len(metadata)}) != index.ntotal ({index.ntotal})."
            )
    else:
        logger.brief(f"[warn] METADATA_PATH not found: {METADATA_PATH}. Will print ids only.")

    metric = guess_metric(index) if args.metric == "auto" else args.metric
    normalize = args.normalize or (metric == "ip")

    if not os.path.exists(DEFAULT_DATASET_PATH):
        raise FileNotFoundError(f"DEFAULT_DATASET_PATH not found: {DEFAULT_DATASET_PATH}")
    rows = load_jsonl(DEFAULT_DATASET_PATH)
    rows = rows[args.start:] if args.start > 0 else rows
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    dataset_original_group_map = load_dataset_original_group_index(args.dataset_original_path)

    queries: List[str] = []
    expected_outputs: List[Any] = []
    desired_group_ids: List[Optional[str]] = []

    for r in rows:
        q = r.get("query")
        if isinstance(q, str) and q.strip():
            queries.append(q.strip())
            expected_outputs.append(r.get("expected_output"))
            iface = r.get("interface")
            gid = None
            if isinstance(iface, dict):
                raw_gid = iface.get("group_id")
                if isinstance(raw_gid, str) and raw_gid.strip():
                    gid = raw_gid.strip()
            desired_group_ids.append(gid)

    if not queries:
        raise ValueError("No valid 'query' strings found in dataset slice.")

    if not os.path.exists(SENTENCE_TRANSFORMER_PATH):
        raise FileNotFoundError(f"SENTENCE_TRANSFORMER_PATH not found: {SENTENCE_TRANSFORMER_PATH}")
    embed_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)

    def fillin_generate(prompt: str) -> str:
        if args.fill_backend == "ollama":
            return ollama_generate(prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
        if not os.path.exists(args.qwen_path):
            raise FileNotFoundError(f"--qwen_path not found: {args.qwen_path}")
        return qwen_generate(
            prompt,
            model_path=args.qwen_path,
            max_new_tokens=args.qwen_max_new_tokens,
            temperature=args.qwen_temperature,
        )

    logger.section(f"Ollama host/model: {DEFAULT_HOST} / {DEFAULT_MODEL}")
    logger.section(f"Fill-in backend: {args.fill_backend}" + (f" (qwen_path={args.qwen_path})" if args.fill_backend == "qwen" else ""))
    logger.section(f"Dataset: {DEFAULT_DATASET_PATH}")
    logger.section(f"dataset_original: {args.dataset_original_path}")
    logger.section(f"Eval out: {args.eval_out}")
    logger.section(f"Debug out: {args.debug_out}")
    logger.section(f"Summary out: {summary_out}")
    logger.section(f"Debug summary out: {debug_summary_out}")
    logger.section(f"Index: {FAISS_INDEX_PATH}   metric={metric}   normalize_queries={normalize}")
    logger.section("=" * 130)

    paper_records: List[Dict[str, Any]] = []
    debug_records_count = 0

    run_start = time.time()
    total_queries = len(queries)

    for qi, q in enumerate(queries):
        query_start = time.time()
        current_done_before = qi
        logger.brief(render_progress(current_done_before, total_queries, time.time() - run_start))
        logger.brief(f"[{qi}] start | desired_group_id={desired_group_ids[qi]!r}")

        exp = expected_outputs[qi]
        desired_group_id = desired_group_ids[qi]

        logger.verbose(f"query:\n{q}\n")
        logger.verbose("expected_output:")
        logger.verbose(pretty(exp))
        logger.verbose("")

        qvec = build_query_vectors(embed_model, [q], normalize=normalize)
        D, I = faiss_search(index, qvec, k=args.k)
        doc_id = int(I[0, 0])
        score = float(D[0, 0])
        label = "sim" if metric == "ip" else "l2"

        if doc_id < 0:
            route_time_seconds = time.time() - query_start
            paper_record = build_paper_record_base(qi, q, "direct", None, None, "no_result", route_time_seconds)
            debug_record = build_debug_record(qi, q, desired_group_id, "direct", None, "no_result", route_time_seconds, error="top1 no result")
            append_jsonl(args.eval_out, paper_record)
            append_jsonl(args.debug_out, debug_record)
            paper_records.append(paper_record)
            debug_records_count += 1
            logger.brief(f"[{qi}] no result")
            logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
            logger.brief("-" * 130)
            continue

        meta = metadata[doc_id] if metadata and 0 <= doc_id < len(metadata) else None
        interface_obj_any = extract_interface_payload(meta)
        interface_obj = unwrap_interface(interface_obj_any)

        top1 = {
            "faiss_id": doc_id,
            "score": score,
            "metric_label": label,
            "interface_id_display": interface_id_display(interface_obj),
            "interface_id": interface_obj.get("@id") if isinstance(interface_obj, dict) else None,
            "displayName": interface_obj.get("displayName") if isinstance(interface_obj, dict) else None,
        }

        logger.brief(f"[{qi}] top1 | faiss_id={doc_id} {label}={score:.6f} | {top1['interface_id']}")
        logger.verbose("top1 interface:")
        logger.verbose(pretty(interface_obj_any))
        logger.verbose("")

        use_decompose = (not args.no_decompose) and (metric == "ip") and (score < args.min_sim)

        used_path = "direct"
        composed_iface: Optional[Dict[str, Any]] = None
        manifest: Optional[List[Dict[str, Any]]] = None
        predicted_instance: Optional[Dict[str, Any]] = None
        direct_eval: Optional[Dict[str, Any]] = None
        subsystem_exact_match_eval: Optional[Dict[str, Any]] = None
        verify_eval: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None
        status = "started"

        if use_decompose:
            used_path = "decompose+compose"
            logger.brief(f"[{qi}] route=decompose+compose | top1 {label}={score:.6f} < min_sim {args.min_sim:.3f}")
            decomp_prompt = build_decompose_prompt(description_text=q, max_parts=args.max_subsystems)
            logger.verbose("[decomposition] prompt:\n" + decomp_prompt + "\n")

            try:
                decomp_text = ollama_generate(decomp_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                decomp_obj = parse_model_json_output(decomp_text)
            except Exception as e:
                logger.brief(f"[{qi}] decomposition error -> fallback direct | {e}")
                decomp_obj = []
                error_msg = f"decomposition_error: {e}"

            if not isinstance(decomp_obj, list) or not decomp_obj:
                used_path = "direct"
            else:
                sub_queries = [str(x).strip() for x in decomp_obj if str(x).strip()][:args.max_subsystems]
                logger.brief(f"[{qi}] decomposition produced {len(sub_queries)} sub-queries")
                logger.verbose(f"[decomposition] sub-queries:\n{pretty(sub_queries)}\n")

                subsystem_packs: List[Dict[str, Any]] = []

                for i, sq in enumerate(sub_queries):
                    sq_vec = build_query_vectors(embed_model, [sq], normalize=normalize)
                    d2, i2 = faiss_search(index, sq_vec, k=1)
                    sid = int(i2[0, 0])
                    sscore = float(d2[0, 0])
                    smeta = metadata[sid] if metadata and 0 <= sid < len(metadata) else None
                    siface_any = extract_interface_payload(smeta)
                    siface = unwrap_interface(siface_any)

                    subsystem_packs.append({"faiss_id": sid, "interface": siface_any, "score": sscore, "sub_query": sq})
                    logger.verbose(
                        f"sub-query[{i + 1}] -> faiss_id={sid} {label}={sscore:.6f} ({interface_id_display(siface)})"
                    )

                composed_id = "dtmi:composed_system:Interface;1"
                composed_display = "ComposedSystem"
                composed_iface, manifest = compose_interfaces(
                    subsystem_interfaces=subsystem_packs,
                    composed_id=composed_id,
                    composed_display=composed_display,
                )

                subsystem_exact_match_eval = evaluate_subsystem_exact_match(
                    desired_group_id=desired_group_id,
                    manifest=manifest,
                    dataset_original_group_map=dataset_original_group_map,
                )

                logger.brief(
                    f"[{qi}] subsystem_match={subsystem_exact_match_eval.get('overall_ok')} "
                    f"| tp={subsystem_exact_match_eval.get('tp')} fp={subsystem_exact_match_eval.get('fp')} fn={subsystem_exact_match_eval.get('fn')} "
                    f"| f1={subsystem_exact_match_eval.get('f1'):.4f}"
                )
                logger.verbose("[composition] manifest:")
                logger.verbose(pretty(manifest))
                logger.verbose("[evaluation] subsystem exact match:")
                logger.verbose(pretty(subsystem_exact_match_eval))
                logger.verbose("")

        if args.no_fillin:
            status = "fillin_disabled"
            route_time_seconds = time.time() - query_start
            if used_path == "direct":
                paper_record = build_paper_record_direct(qi, q, doc_id, score, status, None, route_time_seconds)
            else:
                paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, None, route_time_seconds)

            debug_record = build_debug_record(
                qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                manifest=manifest,
                subsystem_eval=subsystem_exact_match_eval,
                error=error_msg,
            )

            append_jsonl(args.eval_out, paper_record)
            append_jsonl(args.debug_out, debug_record)
            paper_records.append(paper_record)
            debug_records_count += 1

            logger.brief(f"[{qi}] fill-in disabled | route={used_path}")
            logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
            logger.brief("-" * 130)
            continue

        if used_path == "direct":
            if not isinstance(interface_obj, dict):
                status = "direct_interface_not_dict"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_direct(qi, q, doc_id, score, status, None, route_time_seconds)
                debug_record = build_debug_record(qi, q, desired_group_id, used_path, top1, status, route_time_seconds, error=status)
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] direct interface not dict")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            fill_prompt = build_fillin_prompt_direct_flat_instance(description_text=q, interface_obj=interface_obj)
            logger.verbose("[fill-in] path=direct")
            logger.verbose("[fill-in] prompt:\n" + fill_prompt + "\n")

            try:
                model_text = fillin_generate(fill_prompt)
                inst_any = parse_model_json_output(model_text)
            except Exception as e:
                status = f"direct_fillin_error: {e}"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_direct(qi, q, doc_id, score, status, None, route_time_seconds)
                debug_record = build_debug_record(qi, q, desired_group_id, used_path, top1, status, route_time_seconds, error=str(e))
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] direct fill-in error | {e}")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            if not isinstance(inst_any, dict):
                status = "direct_fillin_output_not_dict"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_direct(qi, q, doc_id, score, status, None, route_time_seconds)
                debug_record = build_debug_record(qi, q, desired_group_id, used_path, top1, status, route_time_seconds, error="fillin output not dict")
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] direct fill-in output not dict")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            predicted_instance = inst_any
            logger.verbose("initiated instance:")
            logger.verbose(pretty(predicted_instance))
            logger.verbose("")

            if not isinstance(exp, dict):
                status = "direct_expected_output_not_dict"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_direct(qi, q, doc_id, score, status, None, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    predicted_instance=predicted_instance,
                    error="expected_output is not dict",
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] expected_output not dict")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            raw_strict_eval = strict_compare_direct_instance(
                predicted_instance=predicted_instance,
                expected_output=exp,
                interface_obj=interface_obj,
            )

            logger.brief(
                f"[{qi}] direct raw | tp={raw_strict_eval.get('tp')} fp={raw_strict_eval.get('fp')} fn={raw_strict_eval.get('fn')} "
                f"| f1={raw_strict_eval.get('f1'):.4f}"
            )
            logger.verbose("evaluation (direct raw strict compare):")
            logger.verbose(pretty(raw_strict_eval))
            logger.verbose("")

            direct_verify_prompt = build_verify_prompt_for_direct(
                description_text=q,
                interface_obj=interface_obj,
                predicted_instance=predicted_instance,
                expected_output=exp,
            )
            logger.verbose("[verify-direct] prompt:\n" + direct_verify_prompt + "\n")

            try:
                direct_verify_text = ollama_generate(
                    direct_verify_prompt,
                    host=DEFAULT_HOST,
                    model=DEFAULT_MODEL,
                    timeout_s=args.timeout,
                )
                direct_verify_obj = parse_model_json_output(direct_verify_text)
            except Exception as e:
                direct_eval = {
                    "mode": "direct_llm_verify",
                    "overall_ok": False,
                    "final_source": "llm_verify",
                    "tp": None,
                    "fp": None,
                    "fn": None,
                    "tn": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "raw_strict_compare": raw_strict_eval,
                    "llm_verify_error": str(e),
                }
                status = f"direct_verify_error: {e}"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_direct(qi, q, doc_id, score, status, direct_eval, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    direct_eval=direct_eval,
                    predicted_instance=predicted_instance,
                    error=str(e),
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] direct verify error | {e}")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            direct_eval = build_direct_final_eval(
                raw_strict_eval=raw_strict_eval,
                llm_verify_obj=direct_verify_obj,
            )
            status = "ok"
            route_time_seconds = time.time() - query_start

            logger.brief(
                f"[{qi}] direct final | interface_match={direct_eval.get('overall_ok')} "
                f"| tp={direct_eval.get('tp')} fp={direct_eval.get('fp')} fn={direct_eval.get('fn')} "
                f"| f1={direct_eval.get('f1'):.4f} | route_time={route_time_seconds:.3f}s"
            )
            logger.verbose("verify result (direct LLM semantic verify):")
            logger.verbose(pretty(direct_verify_obj))
            logger.verbose("")
            logger.verbose("final evaluation (direct):")
            logger.verbose(pretty(direct_eval))

            paper_record = build_paper_record_direct(qi, q, doc_id, score, status, direct_eval, route_time_seconds)
            debug_record = build_debug_record(
                qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                direct_eval=direct_eval,
                predicted_instance=predicted_instance,
            )

            append_jsonl(args.eval_out, paper_record)
            append_jsonl(args.debug_out, debug_record)
            paper_records.append(paper_record)
            debug_records_count += 1

        else:
            if composed_iface is None or not isinstance(composed_iface, dict):
                status = "composed_interface_missing"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, None, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    manifest=manifest,
                    subsystem_eval=subsystem_exact_match_eval,
                    error=status,
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] composed interface missing")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            fill_prompt = build_fillin_prompt_composed_nested_instance(description_text=q, composed_interface=composed_iface)
            logger.verbose("[fill-in] path=decompose+compose")
            logger.verbose("[fill-in] prompt:\n" + fill_prompt + "\n")

            try:
                model_text = fillin_generate(fill_prompt)
                inst_any = parse_model_json_output(model_text)
            except Exception as e:
                status = f"composed_fillin_error: {e}"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, None, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    manifest=manifest,
                    subsystem_eval=subsystem_exact_match_eval,
                    error=str(e),
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] composed fill-in error | {e}")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            if not isinstance(inst_any, dict):
                status = "composed_fillin_output_not_dict"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, None, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    manifest=manifest,
                    subsystem_eval=subsystem_exact_match_eval,
                    error="fillin output not dict",
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] composed fill-in output not dict")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            predicted_instance = inst_any
            logger.verbose("initiated instance (composed):")
            logger.verbose(pretty(predicted_instance))
            logger.verbose("")

            if args.no_verify:
                status = "ok"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, None, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    manifest=manifest,
                    subsystem_eval=subsystem_exact_match_eval,
                    predicted_instance=predicted_instance,
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] verify skipped | route_time={route_time_seconds:.3f}s")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            verify_prompt = build_verify_prompt_for_composed(
                description_text=q,
                composed_interface=composed_iface,
                instance_obj=predicted_instance,
            )
            logger.verbose("[verify] prompt:\n" + verify_prompt + "\n")

            try:
                verify_text = ollama_generate(verify_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                verify_eval = parse_model_json_output(verify_text)
            except Exception as e:
                status = f"verify_error: {e}"
                route_time_seconds = time.time() - query_start
                paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, None, route_time_seconds)
                debug_record = build_debug_record(
                    qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                    manifest=manifest,
                    subsystem_eval=subsystem_exact_match_eval,
                    predicted_instance=predicted_instance,
                    error=str(e),
                )
                append_jsonl(args.eval_out, paper_record)
                append_jsonl(args.debug_out, debug_record)
                paper_records.append(paper_record)
                debug_records_count += 1
                logger.brief(f"[{qi}] verify error | {e}")
                logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
                logger.brief("-" * 130)
                continue

            status = "ok"
            route_time_seconds = time.time() - query_start
            llm_verify_ok = isinstance(verify_eval, dict) and verify_eval.get("overall_result") == "PASS"
            logger.brief(
                f"[{qi}] composed final | interface_match={subsystem_exact_match_eval.get('overall_ok') if subsystem_exact_match_eval else None} "
                f"| llm_verify_ok={llm_verify_ok} | route_time={route_time_seconds:.3f}s"
            )
            logger.verbose("verify result (LLM):")
            logger.verbose(pretty(verify_eval))

            paper_record = build_paper_record_composed(qi, q, doc_id, score, status, subsystem_exact_match_eval, verify_eval, route_time_seconds)
            debug_record = build_debug_record(
                qi, q, desired_group_id, used_path, top1, status, route_time_seconds,
                manifest=manifest,
                subsystem_eval=subsystem_exact_match_eval,
                verify_eval=verify_eval,
                predicted_instance=predicted_instance,
            )

            append_jsonl(args.eval_out, paper_record)
            append_jsonl(args.debug_out, debug_record)
            paper_records.append(paper_record)
            debug_records_count += 1

        query_elapsed = time.time() - query_start
        logger.brief(f"[{qi}] done | route={used_path} | query_time={fmt_seconds(query_elapsed)}")
        logger.brief(render_progress(qi + 1, total_queries, time.time() - run_start))
        logger.brief("-" * 130)

    compact_summary = summarize_paper_metrics(paper_records)

    debug_summary = {
        "num_debug_records": debug_records_count,
        "num_paper_records": len(paper_records),
        "run_elapsed_seconds": time.time() - run_start,
        "print_mode": args.print_mode,
        "eval_out": args.eval_out,
        "debug_out": args.debug_out,
        "summary_out": summary_out,
        "debug_summary_out": debug_summary_out,
        "dataset": DEFAULT_DATASET_PATH,
        "dataset_original": args.dataset_original_path,
        "metric": metric,
        "normalize_queries": normalize,
        "min_sim": args.min_sim,
        "fill_backend": args.fill_backend,
        "overall_time_stats": compute_time_stats(paper_records),
        "direct_time_stats": compute_time_stats([r for r in paper_records if r.get("path") == "direct"]),
        "decompose_time_stats": compute_time_stats([r for r in paper_records if r.get("path") == "decompose+compose"]),
    }

    write_json(summary_out, compact_summary)
    write_json(debug_summary_out, debug_summary)

    logger.section("\n" + "=" * 130)
    logger.section("FINAL PAPER SUMMARY")
    logger.section(pretty(compact_summary))
    logger.section(f"Compact evaluation saved to: {args.eval_out}")
    logger.section(f"Debug evaluation saved to: {args.debug_out}")
    logger.section(f"Compact summary saved to: {summary_out}")
    logger.section(f"Debug summary saved to: {debug_summary_out}")


if __name__ == "__main__":
    main()