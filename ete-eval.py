#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import ast
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise SystemExit(
        "faiss is not installed. Install with: pip install faiss-cpu (or faiss-gpu)\n"
        f"Original error: {e}"
    )

from sentence_transformers import SentenceTransformer

try:
    import requests  # type: ignore
except ImportError:
    requests = None


# --- Configs from env ---
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://10.1.1.49:60002")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
DEFAULT_DATASET_PATH = os.getenv("ETE_EVAL_PATH", "./data/syntactic.jsonl")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./models/faiss.index")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./models/embeddings.npy")
METADATA_PATH = os.getenv("METADATA_PATH", "./models/metadata.json")
SENTENCE_TRANSFORMER_PATH = os.getenv("SENTENCE_TRANSFORMER_PATH", "./models/MiniLM-L6-based-new-triplets-final")

# Ignore keys for direct strict compare (hard ignore; not prompt-based)
IGNORE_KEYS = {"@id", "displayName", "dockerImage"}


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


def get_telemetry_field_names_from_interface(interface_obj: Any) -> set:
    contents = get_contents_list_from_interface(interface_obj)
    names = set()
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
        return s0[l : r + 1]

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

        fallback = f"sub_system_{i+1}"
        name = choose_subsystem_name(iface_u, fallback=fallback)
        key = f"{name}_properties_and_telemetries"
        if key in used_keys:
            key = f"{name}_{i+1}_properties_and_telemetries"
        used_keys.add(key)

        contents = []
        if isinstance(iface_u, dict) and isinstance(iface_u.get("contents"), list):
            contents = [c for c in iface_u["contents"] if isinstance(c, dict)]
        else:
            # if iface_u isn't a standard interface dict, try generic extraction
            contents = get_contents_list_from_interface(iface_u)

        composed[key] = contents
        manifest.append(
            {
                "subsystem_idx": i + 1,
                "faiss_id": faiss_id,
                "sub_query": pack.get("sub_query"),
                "subsystem_key": key,
                "contents_len": len(contents),
                "interface_id_display": interface_id_display(iface_u),
            }
        )

    return composed, manifest


# -------------------------
# Fill-in prompts (UPDATED)
# -------------------------
def build_fillin_prompt_direct_flat_instance(description_text: str, interface_obj: Dict[str, Any]) -> str:
    """
    Direct (no composition) case:
    - Ask model to return a FULL initiated instance that matches expected_output format (flat keys),
      but ONLY for property fields (telemetries may be included by expected_output; we ignore telemetries in compare).
    """
    props = get_property_fields_from_interface(interface_obj)
    prop_names = [str(p["name"]) for p in props if "name" in p]
    # also include 'interface' key expected by your ground truth
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
    """
    Composed case:
    - Ask model to output an initiated instance JSON with nested per-subsystem properties to avoid conflicts.
    - Output format MUST be:
      {"interface": "...", "subsystems": {"<subsystem_name>": {"prop": value_or_null, ...}, ...}}
    - We DO NOT request telemetries values (not checked).
    """
    blocks = get_subsystem_blocks_from_composed_interface(composed_interface)
    iface_id = composed_interface.get("@id") if isinstance(composed_interface.get("@id"), str) else ""

    # Build exact schema spec: subsystem_name -> property list
    schema_spec: Dict[str, List[str]] = {}
    for key, contents in blocks:
        # subsystem name is key prefix
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
# Evaluation (UPDATED)
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
        # attempt numeric parsing
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
    """
    Direct case evaluation:
    - Compare keys & values one-to-one against expected_output
    - Ignore IGNORE_KEYS
    - Ignore telemetry keys (derived from interface)
    """
    telemetry_keys = get_telemetry_field_names_from_interface(interface_obj)

    # Expected keys we care about: all keys from expected_output excluding ignored + telemetries
    expected_keys = [k for k in expected_output.keys() if k not in IGNORE_KEYS and k not in telemetry_keys]

    # Pred must have exactly these keys (no more, no less) for strictness
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


def build_verify_prompt_for_composed(description_text: str, composed_interface: Dict[str, Any], instance_obj: Dict[str, Any]) -> str:
    """
    Composed case verification:
    - LLM checks each subsystem's properties are filled reasonably from DESCRIPTION, no hallucination/inference.
    - No comparison to expected_output.
    - Output JSON format:
      {
        "overall_result": "PASS"|"FAIL",
        "subsystems": {
          "<name>": {
            "result": "PASS"|"FAIL",
            "matches": [{"key": "...", "value": ..., "evidence": "quote or short evidence"}...],
            "mismatches": [{"key": "...", "value": ..., "reason": "..."}...],
            "notes": "..."
          }, ...
        },
        "notes": "..."
      }
    """
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

    # UPDATED default min_sim
    parser.add_argument("--min_sim", type=float, default=0.75, help="If top1 sim < min_sim, run decomposition/composition.")
    parser.add_argument("--max_subsystems", type=int, default=5, help="Max number of subsystems after decomposition.")

    parser.add_argument("--no_fillin", action="store_true", help="Disable Ollama fill-in + evaluation entirely.")
    parser.add_argument("--no_decompose", action="store_true", help="Disable decomposition/composition even if sim is low.")
    parser.add_argument("--no_verify", action="store_true", help="Disable composed-case LLM verify step (still prints instance).")
    args = parser.parse_args()

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

    queries: List[str] = []
    expected_outputs: List[Any] = []
    for r in rows:
        q = r.get("query")
        if isinstance(q, str) and q.strip():
            queries.append(q.strip())
            expected_outputs.append(r.get("expected_output"))

    if not queries:
        raise ValueError("No valid 'query' strings found in dataset slice.")

    # Load embed model
    if not os.path.exists(SENTENCE_TRANSFORMER_PATH):
        raise FileNotFoundError(f"SENTENCE_TRANSFORMER_PATH not found: {SENTENCE_TRANSFORMER_PATH}")
    embed_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)

    print(f"Ollama host/model: {DEFAULT_HOST} / {DEFAULT_MODEL}")
    print(f"Dataset: {DEFAULT_DATASET_PATH}")
    print(f"Index: {FAISS_INDEX_PATH}   metric={metric}   normalize_queries={normalize}")
    print("=" * 130)

    for qi, q in enumerate(queries):
        print(f"[{qi}] query:\n{q}\n")

        exp = expected_outputs[qi]
        print("expected_output (ground truth instance):")
        print(pretty(exp))
        print("")

        # initial retrieval
        qvec = build_query_vectors(embed_model, [q], normalize=normalize)
        D, I = faiss_search(index, qvec, k=args.k)
        doc_id = int(I[0, 0])
        score = float(D[0, 0])
        label = "sim" if metric == "ip" else "l2"

        if doc_id < 0:
            print("top1: <no result>")
            print("-" * 130)
            continue

        meta = metadata[doc_id] if metadata and 0 <= doc_id < len(metadata) else None
        interface_obj_any = extract_interface_payload(meta)
        interface_obj = unwrap_interface(interface_obj_any)

        print(f"top1: id={doc_id}  {label}={score:.6f}  ({interface_id_display(interface_obj)})")
        print("top1 interface (raw/wrapper possible):")
        print(pretty(interface_obj_any))
        print("")

        # Determine whether composition is used
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
                sub_queries = [str(x).strip() for x in decomp_obj if str(x).strip()][: args.max_subsystems]
                print(f"[decomposition] sub-queries (n={len(sub_queries)}):")
                for i, sq in enumerate(sub_queries):
                    print(f"  ({i+1}) {sq}")
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
                        f"  sub-query[{i+1}] -> faiss_id={sid}  {label}={sscore:.6f}  "
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

        # Stop early if fill-in disabled
        if args.no_fillin:
            print(f"[info] fill-in disabled; path={used_path}")
            print("-" * 130)
            continue

        # -------------------------
        # Fill-in
        # -------------------------
        if used_path == "direct":
            if not isinstance(interface_obj, dict):
                print("[warn] direct interface is not a dict; skip fill-in/eval.")
                print("-" * 130)
                continue

            # Direct fill-in: output flat instance matching expected_output schema
            fill_prompt = build_fillin_prompt_direct_flat_instance(description_text=q, interface_obj=interface_obj)
            print("[fill-in] path=direct")
            print("[fill-in] prompt:\n" + fill_prompt + "\n")

            try:
                model_text = ollama_generate(fill_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                inst_any = parse_model_json_output(model_text)
            except Exception as e:
                print(f"[fill-in] ERROR: {e}")
                print("-" * 130)
                continue

            if not isinstance(inst_any, dict):
                print("[fill-in] ERROR: model output is not a JSON object/dict")
                print("raw model output:")
                print(model_text)
                print("-" * 130)
                continue

            predicted_instance: Dict[str, Any] = inst_any
            print("initiated instance (model output):")
            print(pretty(predicted_instance))
            print("")

            # -------------------------
            # Evaluation (direct): strict compare vs expected_output
            # -------------------------
            if not isinstance(exp, dict):
                print("[warn] expected_output is not a dict; cannot strict-compare.")
                print("-" * 130)
                continue

            eval_report = strict_compare_direct_instance(
                predicted_instance=predicted_instance,
                expected_output=exp,
                interface_obj=interface_obj,
            )
            print("evaluation (direct strict compare vs expected_output; ignore @id/displayName/dockerImage; ignore telemetries):")
            print(pretty(eval_report))
            print("-" * 130)

        else:
            # Composed fill-in: output nested instance to avoid conflicts
            if composed_iface is None or not isinstance(composed_iface, dict):
                print("[warn] composed interface missing; cannot fill-in composed instance.")
                print("-" * 130)
                continue

            fill_prompt = build_fillin_prompt_composed_nested_instance(description_text=q, composed_interface=composed_iface)
            print("[fill-in] path=decompose+compose")
            print("[fill-in] prompt:\n" + fill_prompt + "\n")

            try:
                model_text = ollama_generate(fill_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                inst_any = parse_model_json_output(model_text)
            except Exception as e:
                print(f"[fill-in] ERROR: {e}")
                print("-" * 130)
                continue

            if not isinstance(inst_any, dict):
                print("[fill-in] ERROR: model output is not a JSON object/dict")
                print("raw model output:")
                print(model_text)
                print("-" * 130)
                continue

            composed_instance: Dict[str, Any] = inst_any
            print("initiated instance (composed; model output):")
            print(pretty(composed_instance))
            print("")

            # -------------------------
            # Verification (composed): LLM verify reasonableness, no expected_output compare
            # -------------------------
            if args.no_verify:
                print("[verify] disabled (--no_verify).")
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
                print("-" * 130)
                continue

            print("verify result (LLM):")
            print(pretty(verify_obj))
            print("-" * 130)


if __name__ == "__main__":
    main()