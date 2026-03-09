#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import ast
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

# evaluation output
DEFAULT_EVAL_OUT_PATH = os.getenv("EVAL_OUT_PATH", "./outputs/evaluation_results.jsonl")

# Ignore keys for direct strict compare (hard ignore; not prompt-based)
IGNORE_KEYS = {"@id", "displayName", "dockerImage", "interface"}


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


def pretty(obj: Any) -> str:
    if obj is None:
        return "<None>"
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return str(obj)


def minified_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


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
    """
    Example:
      dtmi:smart_window_tint_control_system:tint_controller;1
    -> smart_window_tint_control_system
    """
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
    """
    Read dataset_original.jsonl only once and build:
      group_id -> {
        "faiss_ids_zero_based": [...],
        "line_numbers_one_based": [...],
        "interfaces": [...]
      }

    Assumption:
      faiss_id corresponds to zero-based line index/order in dataset_original.jsonl.
      One-based line numbers are also recorded for debugging/inspection.
    """
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
    # wrapper like {"raw": "...", "parsed": {...}}
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
    """
    Works for:
      - standard DTDL: {"contents": [...]}
      - composed interface: multiple "*_properties_and_telemetries": [ ... ]
    """
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
    """
    Return list of (subsystem_key, contents_list)
    subsystem_key is the key like "<name>_properties_and_telemetries"
    """
    blocks = []
    for k, v in composed_iface.items():
        if k.endswith("_properties_and_telemetries") and isinstance(v, list):
            blocks.append((k, [x for x in v if isinstance(x, dict)]))
    return blocks


def get_property_names_from_contents(contents: List[Dict[str, Any]]) -> List[str]:
    return [str(c["name"]) for c in contents if isinstance(c, dict) and c.get("@type") == "Property" and "name" in c]


def get_telemetry_names_from_contents(contents: List[Dict[str, Any]]) -> List[str]:
    return [str(c["name"]) for c in contents if isinstance(c, dict) and c.get("@type") == "Telemetry" and "name" in c]


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
    """
    subsystem_interfaces: list of packs:
      {"faiss_id": int, "interface": Any, "score": float, "sub_query": str}

    Returns:
      - composed interface dict (required format)
      - manifest for debug prints
    """
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
def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool, dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("null", "none", ""):
            return None
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            return v
    return v


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

    correct = 0
    total = 0
    for k in sorted(expected_set & pred_set):
        total += 1
        pv = normalize_value(predicted_instance.get(k))
        gv = normalize_value(expected_output.get(k))
        ok = pv == gv
        fields[k] = {"pred": pv, "gt": gv, "ok": ok}
        if ok:
            correct += 1

    overall_ok = (len(missing) == 0 and len(extra) == 0 and correct == total)

    return {
        "mode": "direct_strict_compare",
        "overall_ok": overall_ok,
        "missing_keys": missing,
        "extra_keys": extra,
        "summary": {"correct": correct, "total": total, "accuracy": (correct / total) if total else 0.0},
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
        "overall_ok": False,
        "reason": "",
        "correct_faiss_ids_from_expected_lines": [],
        "invalid_retrieved_faiss_ids": [],
        "valid_retrieved_faiss_ids": [],
    }

    if not desired_group_id:
        report["reason"] = "desired group_id missing in ETE_EVAL_PATH row"
        return report

    target = dataset_original_group_map.get(desired_group_id)
    if not target:
        report["reason"] = f"group_id={desired_group_id!r} not found in dataset_original"
        return report

    expected_line_numbers_one_based = [
        int(x) for x in target.get("line_numbers_one_based", [])
    ]
    allowed_faiss_ids = sorted({ln - 1 for ln in expected_line_numbers_one_based if isinstance(ln, int) and ln >= 1})

    valid_retrieved = [fid for fid in retrieved_ids if fid in set(allowed_faiss_ids)]
    invalid_retrieved = [fid for fid in retrieved_ids if fid not in set(allowed_faiss_ids)]

    report["correct_faiss_ids_from_expected_lines"] = allowed_faiss_ids
    report["valid_retrieved_faiss_ids"] = valid_retrieved
    report["invalid_retrieved_faiss_ids"] = invalid_retrieved
    report["overall_ok"] = (len(invalid_retrieved) == 0)
    report["reason"] = (
        "all retrieved faiss_ids are in the correct lines"
        if report["overall_ok"]
        else "some retrieved faiss_ids are not correct"
    )
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
    overall_ok = (llm_verify_obj.get("overall_result") == "PASS")

    return {
        "mode": "direct_llm_verify",
        "overall_ok": overall_ok,
        "final_source": "llm_verify",
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


def update_summary_counts(summary: Dict[str, Any], record: Dict[str, Any]) -> None:
    summary["queries_total"] += 1

    path = record.get("path")
    if path == "direct":
        summary["direct_total"] += 1
        direct_eval = record.get("direct_eval")
        if isinstance(direct_eval, dict):
            if direct_eval.get("overall_ok") is True:
                summary["direct_pass"] += 1
            else:
                summary["direct_fail"] += 1
        else:
            summary["direct_fail"] += 1

    elif path == "decompose+compose":
        summary["composed_total"] += 1

        subsystem_eval = record.get("subsystem_exact_match_eval")
        if isinstance(subsystem_eval, dict):
            if subsystem_eval.get("overall_ok") is True:
                summary["subsystem_exact_match_pass"] += 1
            else:
                summary["subsystem_exact_match_fail"] += 1
        else:
            summary["subsystem_exact_match_fail"] += 1

        verify_eval = record.get("verify_eval")
        if isinstance(verify_eval, dict):
            if verify_eval.get("overall_result") == "PASS":
                summary["verify_pass"] += 1
            else:
                summary["verify_fail"] += 1
        elif record.get("verify_skipped"):
            summary["verify_skipped"] += 1
        else:
            summary["verify_fail"] += 1


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

    parser.add_argument("--min_sim", type=float, default=0.75, help="If top1 sim < min_sim, run decomposition/composition.")
    parser.add_argument("--max_subsystems", type=int, default=5, help="Max number of subsystems after decomposition.")

    parser.add_argument("--no_fillin", action="store_true", help="Disable Ollama fill-in + evaluation entirely.")
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
    parser.add_argument("--eval_out", type=str, default=DEFAULT_EVAL_OUT_PATH, help="Per-query evaluation output JSONL path.")
    parser.add_argument("--summary_out", type=str, default="", help="Summary JSON output path. Default: <eval_out>.summary.json")
    args = parser.parse_args()

    summary_out = args.summary_out.strip() if isinstance(args.summary_out, str) else ""
    if not summary_out:
        summary_out = args.eval_out + ".summary.json"

    ensure_parent_dir(args.eval_out)
    ensure_parent_dir(summary_out)

    # reset per-run output files
    with open(args.eval_out, "w", encoding="utf-8") as f:
        pass

    # Load index
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS_INDEX_PATH not found: {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Optional sanity check
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
            if hasattr(index, "ntotal") and emb.shape[0] != index.ntotal:
                print(
                    f"[warn] embeddings rows ({emb.shape[0]}) != index.ntotal ({index.ntotal}). "
                    "Make sure metadata/index alignment is correct."
                )
        except Exception as e:
            print(f"[warn] Could not load embeddings from {EMBEDDINGS_PATH}: {e}")

    # Load metadata
    metadata: List[Dict[str, Any]] = []
    if os.path.exists(METADATA_PATH):
        metadata = load_metadata(METADATA_PATH)
        if hasattr(index, "ntotal") and len(metadata) != index.ntotal:
            print(
                f"[warn] metadata items ({len(metadata)}) != index.ntotal ({index.ntotal}). "
                "If you used an ID map or filtered docs, ensure you map ids correctly."
            )
    else:
        print(f"[warn] METADATA_PATH not found: {METADATA_PATH}. Will print ids only.")

    metric = guess_metric(index) if args.metric == "auto" else args.metric
    normalize = args.normalize or (metric == "ip")

    # Load dataset
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

    # Load embed model
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

    print(f"Ollama host/model: {DEFAULT_HOST} / {DEFAULT_MODEL}")
    print(f"Fill-in backend: {args.fill_backend}" + (f" (qwen_path={args.qwen_path})" if args.fill_backend == "qwen" else ""))
    print(f"Dataset: {DEFAULT_DATASET_PATH}")
    print(f"dataset_original: {args.dataset_original_path}")
    print(f"Eval out: {args.eval_out}")
    print(f"Summary out: {summary_out}")
    print(f"Index: {FAISS_INDEX_PATH}   metric={metric}   normalize_queries={normalize}")
    print("=" * 130)

    summary: Dict[str, Any] = {
        "queries_total": 0,
        "direct_total": 0,
        "direct_pass": 0,
        "direct_fail": 0,
        "composed_total": 0,
        "subsystem_exact_match_pass": 0,
        "subsystem_exact_match_fail": 0,
        "verify_pass": 0,
        "verify_fail": 0,
        "verify_skipped": 0,
        "eval_out": args.eval_out,
        "summary_out": summary_out,
        "dataset": DEFAULT_DATASET_PATH,
        "dataset_original": args.dataset_original_path,
    }

    for qi, q in enumerate(queries):
        print(f"[{qi}] query:\n{q}\n")

        exp = expected_outputs[qi]
        desired_group_id = desired_group_ids[qi]

        print("expected_output (ground truth instance):")
        print(pretty(exp))
        print("")
        print(f"desired_group_id: {desired_group_id!r}")
        print("")

        record: Dict[str, Any] = {
            "query_index": qi,
            "query": q,
            "expected_output": exp,
            "desired_group_id": desired_group_id,
            "path": None,
            "top1": None,
            "manifest": None,
            "composed_interface": None,
            "predicted_instance": None,
            "direct_eval": None,
            "subsystem_exact_match_eval": None,
            "verify_eval": None,
            "verify_skipped": False,
            "status": "started",
        }

        qvec = build_query_vectors(embed_model, [q], normalize=normalize)
        D, I = faiss_search(index, qvec, k=args.k)
        doc_id = int(I[0, 0])
        score = float(D[0, 0])
        label = "sim" if metric == "ip" else "l2"

        if doc_id < 0:
            print("top1: <no result>")
            record["status"] = "no_result"
            append_jsonl(args.eval_out, record)
            update_summary_counts(summary, record)
            print("-" * 130)
            continue

        meta = metadata[doc_id] if metadata and 0 <= doc_id < len(metadata) else None
        interface_obj_any = extract_interface_payload(meta)
        interface_obj = unwrap_interface(interface_obj_any)

        record["top1"] = {
            "faiss_id": doc_id,
            "score": score,
            "metric_label": label,
            "interface_id_display": interface_id_display(interface_obj),
            "interface_id": interface_obj.get("@id") if isinstance(interface_obj, dict) else None,
            "displayName": interface_obj.get("displayName") if isinstance(interface_obj, dict) else None,
        }

        print(f"top1: id={doc_id}  {label}={score:.6f}  ({interface_id_display(interface_obj)})")
        print("top1 interface (raw/wrapper possible):")
        print(pretty(interface_obj_any))
        print("")

        use_decompose = (not args.no_decompose) and (metric == "ip") and (score < args.min_sim)

        used_path = "direct"
        composed_iface: Optional[Dict[str, Any]] = None
        manifest: Optional[List[Dict[str, Any]]] = None

        if use_decompose:
            used_path = "decompose+compose"
            print(f"[decomposition] top1 sim {score:.6f} < min_sim {args.min_sim:.3f} -> start decomposition")
            decomp_prompt = build_decompose_prompt(description_text=q, max_parts=args.max_subsystems)
            print("[decomposition] prompt:\n" + decomp_prompt + "\n")

            try:
                decomp_text = ollama_generate(decomp_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                decomp_obj = parse_model_json_output(decomp_text)
            except Exception as e:
                print(f"[decomposition] ERROR: {e}")
                print("[decomposition] Fallback: use direct interface (no decompose).")
                decomp_obj = []

            if not isinstance(decomp_obj, list) or not decomp_obj:
                print("[decomposition] got empty/invalid list; fallback to direct interface.")
                used_path = "direct"
            else:
                sub_queries = [str(x).strip() for x in decomp_obj if str(x).strip()][:args.max_subsystems]
                print(f"[decomposition] sub-queries (n={len(sub_queries)}):")
                for i, sq in enumerate(sub_queries):
                    print(f"  ({i + 1}) {sq}")
                print("")

                subsystem_packs: List[Dict[str, Any]] = []
                print("[decomposition] retrieving subsystem interfaces from FAISS...")

                for i, sq in enumerate(sub_queries):
                    sq_vec = build_query_vectors(embed_model, [sq], normalize=normalize)
                    d2, i2 = faiss_search(index, sq_vec, k=1)
                    sid = int(i2[0, 0])
                    sscore = float(d2[0, 0])
                    smeta = metadata[sid] if metadata and 0 <= sid < len(metadata) else None
                    siface_any = extract_interface_payload(smeta)
                    siface = unwrap_interface(siface_any)

                    subsystem_packs.append({"faiss_id": sid, "interface": siface_any, "score": sscore, "sub_query": sq})

                    print(
                        f"  sub-query[{i + 1}] -> faiss_id={sid}  {label}={sscore:.6f}  "
                        f"({interface_id_display(siface)})"
                    )
                print("")

                composed_id = "dtmi:composed_system:Interface;1"
                composed_display = "ComposedSystem"
                composed_iface, manifest = compose_interfaces(
                    subsystem_interfaces=subsystem_packs,
                    composed_id=composed_id,
                    composed_display=composed_display,
                )
                record["manifest"] = manifest
                record["composed_interface"] = composed_iface

                print("[composition] manifest:")
                for m in manifest:
                    print(
                        f"  subsystem[{m['subsystem_idx']}] faiss_id={m['faiss_id']} "
                        f"key={m['subsystem_key']} contents_len={m['contents_len']} "
                        f"({m['interface_id_display']})"
                    )
                print("\n[composition] composed interface:")
                print(pretty(composed_iface))
                print("")

                subsystem_exact_match_eval = evaluate_subsystem_exact_match(
                    desired_group_id=desired_group_id,
                    manifest=manifest,
                    dataset_original_group_map=dataset_original_group_map,
                )
                record["subsystem_exact_match_eval"] = subsystem_exact_match_eval
                print("[evaluation] subsystem exact match vs desired interface group:")
                print(pretty(subsystem_exact_match_eval))
                print("")

        record["path"] = used_path

        if args.no_fillin:
            record["status"] = "fillin_disabled"
            append_jsonl(args.eval_out, record)
            update_summary_counts(summary, record)
            print(f"[info] fill-in disabled; path={used_path}")
            print("-" * 130)
            continue

        if used_path == "direct":
            if not isinstance(interface_obj, dict):
                print("[warn] direct interface is not a dict; skip fill-in/eval.")
                record["status"] = "direct_interface_not_dict"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            fill_prompt = build_fillin_prompt_direct_flat_instance(description_text=q, interface_obj=interface_obj)
            print("[fill-in] path=direct")
            print("[fill-in] prompt:\n" + fill_prompt + "\n")

            try:
                model_text = fillin_generate(fill_prompt)
                inst_any = parse_model_json_output(model_text)
            except Exception as e:
                print(f"[fill-in] ERROR: {e}")
                record["status"] = f"direct_fillin_error: {e}"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            if not isinstance(inst_any, dict):
                print("[fill-in] ERROR: model output is not a JSON object/dict")
                print("raw model output:")
                print(model_text)
                record["status"] = "direct_fillin_output_not_dict"
                record["predicted_instance"] = model_text
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            predicted_instance: Dict[str, Any] = inst_any
            record["predicted_instance"] = predicted_instance

            print("initiated instance (model output):")
            print(pretty(predicted_instance))
            print("")

            if not isinstance(exp, dict):
                print("[warn] expected_output is not a dict; cannot strict-compare.")
                record["status"] = "direct_expected_output_not_dict"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            raw_strict_eval = strict_compare_direct_instance(
                predicted_instance=predicted_instance,
                expected_output=exp,
                interface_obj=interface_obj,
            )

            print(
                "evaluation (direct raw strict compare vs expected_output):")
            print(pretty(raw_strict_eval))
            print("")

            direct_verify_prompt = build_verify_prompt_for_direct(
                description_text=q,
                interface_obj=interface_obj,
                predicted_instance=predicted_instance,
                expected_output=exp,
            )
            print("[verify-direct] prompt:\n" + direct_verify_prompt + "\n")

            try:
                direct_verify_text = ollama_generate(
                    direct_verify_prompt,
                    host=DEFAULT_HOST,
                    model=DEFAULT_MODEL,
                    timeout_s=args.timeout,
                )
                direct_verify_obj = parse_model_json_output(direct_verify_text)
            except Exception as e:
                print(f"[verify-direct] ERROR: {e}")
                record["direct_eval"] = {
                    "mode": "direct_llm_verify",
                    "overall_ok": False,
                    "final_source": "llm_verify",
                    "raw_strict_compare": raw_strict_eval,
                    "llm_verify_error": str(e),
                }
                record["status"] = f"direct_verify_error: {e}"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            final_direct_eval = build_direct_final_eval(
                raw_strict_eval=raw_strict_eval,
                llm_verify_obj=direct_verify_obj,
            )

            record["direct_eval"] = final_direct_eval
            record["status"] = "ok"

            print("verify result (direct LLM semantic verify):")
            print(pretty(direct_verify_obj))
            print("")
            print("final evaluation (direct; final result based on LLM verify):")
            print(pretty(final_direct_eval))

            append_jsonl(args.eval_out, record)
            update_summary_counts(summary, record)
            print("-" * 130)

        else:
            if composed_iface is None or not isinstance(composed_iface, dict):
                print("[warn] composed interface missing; cannot fill-in composed instance.")
                record["status"] = "composed_interface_missing"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            fill_prompt = build_fillin_prompt_composed_nested_instance(description_text=q, composed_interface=composed_iface)
            print("[fill-in] path=decompose+compose")
            print("[fill-in] prompt:\n" + fill_prompt + "\n")

            try:
                model_text = fillin_generate(fill_prompt)
                inst_any = parse_model_json_output(model_text)
            except Exception as e:
                print(f"[fill-in] ERROR: {e}")
                record["status"] = f"composed_fillin_error: {e}"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            if not isinstance(inst_any, dict):
                print("[fill-in] ERROR: model output is not a JSON object/dict")
                print("raw model output:")
                print(model_text)
                record["status"] = "composed_fillin_output_not_dict"
                record["predicted_instance"] = model_text
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            composed_instance: Dict[str, Any] = inst_any
            record["predicted_instance"] = composed_instance

            print("initiated instance (composed; model output):")
            print(pretty(composed_instance))
            print("")

            if args.no_verify:
                print("[verify] disabled (--no_verify).")
                record["verify_skipped"] = True
                record["status"] = "ok"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            verify_prompt = build_verify_prompt_for_composed(
                description_text=q,
                composed_interface=composed_iface,
                instance_obj=composed_instance,
            )
            print("[verify] prompt:\n" + verify_prompt + "\n")

            try:
                verify_text = ollama_generate(verify_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                verify_obj = parse_model_json_output(verify_text)
            except Exception as e:
                print(f"[verify] ERROR: {e}")
                record["status"] = f"verify_error: {e}"
                append_jsonl(args.eval_out, record)
                update_summary_counts(summary, record)
                print("-" * 130)
                continue

            record["verify_eval"] = verify_obj
            record["status"] = "ok"

            print("verify result (LLM):")
            print(pretty(verify_obj))
            append_jsonl(args.eval_out, record)
            update_summary_counts(summary, record)
            print("-" * 130)

    write_json(summary_out, summary)

    print("\n" + "=" * 130)
    print("FINAL EVALUATION SUMMARY")
    print(pretty(summary))
    print(f"Per-query evaluation saved to: {args.eval_out}")
    print(f"Summary saved to: {summary_out}")


if __name__ == "__main__":
    main()