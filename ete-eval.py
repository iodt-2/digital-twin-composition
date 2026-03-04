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
DEFAULT_DATASET_PATH = os.getenv("ETE_EVAL_PATH", "./data/merged.jsonl")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./models/faiss.index")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./models/embeddings.npy")
METADATA_PATH = os.getenv("METADATA_PATH", "./models/metadata.json")
SENTENCE_TRANSFORMER_PATH = os.getenv(
    "SENTENCE_TRANSFORMER_PATH",
    "./models/MiniLM-L6-based-new-triplets-final",
)

IGNORE_KEYS = {"@id", "displayName", "dockerImage"}


# -----------------------------
# IO + parsing helpers
# -----------------------------
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
    raise ValueError(
        f"Unsupported metadata format in {path}. Expected a list, or a dict containing a list."
    )


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


def pretty(obj: Any) -> str:
    if obj is None:
        return "<None>"
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return str(obj)


def minified_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def unwrap_interface(interface_obj: Any) -> Any:
    """
    Retrieve real interface dict from wrapper like:
      {'raw': '...', 'parsed': {...}}
    """
    if isinstance(interface_obj, dict):
        if "parsed" in interface_obj and isinstance(interface_obj["parsed"], dict):
            return interface_obj["parsed"]
    return interface_obj


def extract_interface_payload(meta: Optional[Dict[str, Any]]) -> Any:
    """
    Retrieve interface from metadata; prioritize 'positive'.
    """
    if not meta:
        return None
    for key in ("positive", "interface", "dtdl", "text", "content", "document", "chunk"):
        if key in meta and meta[key] is not None:
            return _try_parse_json_or_pyobj(meta[key])
    return meta


# -----------------------------
# FAISS helpers
# -----------------------------
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


def is_retrieval_good(metric: str, score: float, min_sim: float, max_l2: float) -> bool:
    if metric == "ip":
        return score >= min_sim
    # l2: smaller is better
    return score <= max_l2


# -----------------------------
# Interface content extraction (supports composed format)
# -----------------------------
def iter_contents_from_interface(interface_obj: Any) -> List[Dict[str, Any]]:
    """
    Support:
      - Standard DTDL interface: interface_obj["contents"] -> list
      - Composed interface:
          { ..., "<subsystem>_properties_and_telemetries": [ ... ], ... }
    """
    interface_obj = unwrap_interface(interface_obj)
    if not isinstance(interface_obj, dict):
        return []

    all_contents: List[Dict[str, Any]] = []

    # Standard DTDL
    contents = interface_obj.get("contents")
    if isinstance(contents, list):
        for c in contents:
            if isinstance(c, dict):
                all_contents.append(c)

    # Composed fields
    for k, v in interface_obj.items():
        if isinstance(k, str) and k.endswith("_properties_and_telemetries") and isinstance(v, list):
            for c in v:
                if isinstance(c, dict):
                    all_contents.append(c)

    return all_contents


def get_property_fields_from_interface(interface_obj: Any) -> List[Dict[str, Any]]:
    props: List[Dict[str, Any]] = []
    for c in iter_contents_from_interface(interface_obj):
        if c.get("@type") == "Property" and "name" in c:
            props.append(c)
    return props


def get_telemetry_field_names_from_interface(interface_obj: Any) -> set:
    names = set()
    for c in iter_contents_from_interface(interface_obj):
        if c.get("@type") == "Telemetry" and "name" in c:
            names.add(str(c["name"]))
    return names


def extract_eval_keys(interface_obj: Any) -> List[str]:
    props = get_property_fields_from_interface(interface_obj)
    return [str(p["name"]) for p in props if "name" in p]


# -----------------------------
# Ollama call + prompts
# -----------------------------
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
    else:
        import urllib.request

        def _post(url: str, data: dict) -> Optional[dict]:
            b = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=b, headers={"Content-Type": "application/json"}, method="POST"
            )
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
    if not s.startswith("{") and not s.startswith("["):
        left = s.find("{")
        right = s.rfind("}")
        l2 = s.find("[")
        r2 = s.rfind("]")
        # prefer array if present
        if l2 != -1 and r2 != -1 and r2 > l2:
            s = s[l2 : r2 + 1]
        elif left != -1 and right != -1 and right > left:
            s = s[left : right + 1]

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    raise ValueError(f"Cannot parse model output as JSON/pyobj. Raw output:\n{text}")


def build_fillin_prompt(description_text: str, interface_obj: Any) -> str:
    """
    Fill ONLY property keys (flat), even if interface_obj is composed.
    """
    prop_fields = get_property_fields_from_interface(interface_obj)
    prop_names = [str(p.get("name")) for p in prop_fields if p.get("name") is not None]

    # For context, include composed interface JSON
    interface_obj_unwrapped = unwrap_interface(interface_obj)
    interface_str = minified_json(interface_obj_unwrapped) if isinstance(interface_obj_unwrapped, dict) else str(interface_obj_unwrapped)

    prompt = (
        "You are an information extraction assistant.\n"
        "Task: Extract values from the DESCRIPTION text to fill ONLY the interface Property fields.\n"
        "Rules:\n"
        "- Use ONLY values explicitly stated in DESCRIPTION. Do not infer unstated facts.\n"
        "- If a value is not stated, use null.\n"
        "- Return ONLY minified JSON (no markdown, no comments).\n"
        f"- JSON must contain EXACTLY these keys (and no others): {prop_names}\n\n"
        "INTERFACE (for context):\n"
        f"{interface_str}\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )
    return prompt


def build_decompose_prompt(user_query: str, max_n: int) -> str:
    """
    Return JSON array of objects:
      [{"name":"BatteryPack","query":"..."}, ...]
    """
    prompt = (
        "You are a system decomposition assistant.\n"
        "Given a user request, decompose it into a small set of subsystem search queries.\n"
        "Rules:\n"
        f"- Output ONLY minified JSON array.\n"
        f"- Array length MUST be between 1 and {max_n}.\n"
        "- Each item MUST be an object with EXACT keys: name, query.\n"
        "- 'name' should be a short subsystem identifier (letters/numbers/underscore preferred).\n"
        "- 'query' should be a concise retrieval query targeting a single interface.\n"
        "- Do NOT include any extra keys.\n\n"
        "USER_QUERY:\n"
        f"{user_query}\n"
    )
    return prompt


# -----------------------------
# Composition
# -----------------------------
def sanitize_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "sub_system"
    # keep letters numbers underscore; replace others with underscore
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "sub_system"


def infer_subsystem_name(interface_obj: Any, fallback: str) -> str:
    interface_obj = unwrap_interface(interface_obj)
    if isinstance(interface_obj, dict):
        dn = interface_obj.get("displayName")
        if isinstance(dn, str) and dn.strip():
            return sanitize_name(dn)
        iid = interface_obj.get("@id")
        if isinstance(iid, str) and iid.strip():
            # dtmi:xxx:BatteryPack;1 -> BatteryPack
            tail = iid.split(":")[-1]
            tail = tail.split(";")[0]
            if tail.strip():
                return sanitize_name(tail)
    return sanitize_name(fallback)


def compose_interfaces(
    interfaces: List[Any],
    names: List[str],
    composed_id: str = "dtmi:composed:System;1",
    composed_display_name: str = "ComposedSystem",
) -> Dict[str, Any]:
    """
    Output format required:
    {
      "@context": "dtmi:dtdl:context;2",
      "@id": "xxx",
      "@type": "Interface",
      "displayName": "xxx",
      "<subsystem>_properties_and_telemetries": [ ... ],
      ...
    }
    """
    out: Dict[str, Any] = {
        "@context": "dtmi:dtdl:context;2",
        "@id": composed_id,
        "@type": "Interface",
        "displayName": composed_display_name,
    }

    for i, iface in enumerate(interfaces):
        iface_u = unwrap_interface(iface)
        subsys = sanitize_name(names[i] if i < len(names) else f"sub_system_{i+1}")
        key = f"{subsys}_properties_and_telemetries"

        contents: List[Dict[str, Any]] = []
        if isinstance(iface_u, dict):
            c = iface_u.get("contents")
            if isinstance(c, list):
                contents = [x for x in c if isinstance(x, dict)]
        out[key] = contents

    return out


# -----------------------------
# Evaluation
# -----------------------------
def normalize_expected_value(v: Any) -> Any:
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


def compare_fillin_to_groundtruth(
    fillin: Dict[str, Any],
    expected_output: Dict[str, Any],
    interface_obj: Any,
) -> Tuple[Dict[str, Any], float]:
    if not isinstance(expected_output, dict):
        raise ValueError("expected_output must be a JSON object/dict for comparison.")

    prop_keys = extract_eval_keys(interface_obj)
    telemetry_keys = get_telemetry_field_names_from_interface(interface_obj)

    eval_keys = [k for k in prop_keys if k not in IGNORE_KEYS and k not in telemetry_keys]

    report: Dict[str, Any] = {"evaluated_keys": eval_keys, "fields": {}}
    correct = 0
    total = len(eval_keys)

    for k in eval_keys:
        pred_v = normalize_expected_value(fillin.get(k, None))
        gt_v = normalize_expected_value(expected_output.get(k, None))
        ok = pred_v == gt_v
        report["fields"][k] = {"pred": pred_v, "gt": gt_v, "ok": ok}
        if ok:
            correct += 1

    acc = (correct / total) if total > 0 else 0.0
    report["summary"] = {"correct": correct, "total": total, "accuracy": acc}
    return report, acc


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="Top-k results per query (default: 1).")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N queries (0 = all).")
    parser.add_argument("--start", type=int, default=0, help="Start offset into the JSONL dataset.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize query embeddings (recommended for IP/cosine).")
    parser.add_argument("--metric", choices=["auto", "ip", "l2"], default="auto", help="FAISS metric: ip or l2.")

    # Retrieval-good thresholds
    parser.add_argument("--min_sim", type=float, default=0.75, help="If metric=ip, require top1 sim >= min_sim, else decompose.")
    parser.add_argument("--max_l2", type=float, default=1e9, help="If metric=l2, require top1 l2 <= max_l2, else decompose.")

    # Decompose/compose controls
    parser.add_argument("--max_subsystems", type=int, default=5, help="Max number of subsystems produced by decomposition.")
    parser.add_argument("--timeout", type=int, default=120, help="Ollama request timeout (seconds).")

    parser.add_argument("--no_fillin", action="store_true", help="Disable Ollama fill-in + evaluation.")
    parser.add_argument("--no_decompose", action="store_true", help="Disable decomposition/composition fallback (always fill-in on top1).")
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
    rows = rows[args.start :] if args.start > 0 else rows
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    queries: List[str] = []
    expected_outputs: List[Any] = []
    for r in rows:
        q = r.get("query")
        if isinstance(q, str) and q.strip():
            queries.append(q.strip())
            expected_outputs.append(r.get("expected_output"))

    if not queries:
        raise ValueError("No valid 'query' strings found in dataset slice.")

    # Embedding model
    if not os.path.exists(SENTENCE_TRANSFORMER_PATH):
        raise FileNotFoundError(f"SENTENCE_TRANSFORMER_PATH not found: {SENTENCE_TRANSFORMER_PATH}")
    embed_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)

    print(f"Ollama host/model: {DEFAULT_HOST} / {DEFAULT_MODEL}")
    print(f"Dataset: {DEFAULT_DATASET_PATH}")
    print(f"Index: {FAISS_INDEX_PATH}   metric={metric}   normalize_queries={normalize}")
    print("=" * 100)

    overall_correct = 0
    overall_total = 0

    for qi, q in enumerate(queries):
        print(f"[{qi}] query:\n{q}\n")
        exp = expected_outputs[qi]
        print("expected_output (ground truth instance):")
        print(pretty(exp))
        print("")

        # 1) normal top1 retrieval
        qvec = build_query_vectors(embed_model, [q], normalize=normalize)
        D, I = faiss_search(index, qvec, k=max(args.k, 1))
        doc_id = int(I[0, 0])
        score = float(D[0, 0])
        label = "sim" if metric == "ip" else "l2"

        meta = metadata[doc_id] if (doc_id >= 0 and metadata and doc_id < len(metadata)) else None
        top1_interface = extract_interface_payload(meta)
        top1_interface_u = unwrap_interface(top1_interface)

        print(f"top1: id={doc_id}  {label}={score:.6f}")
        print("top1 interface (raw):")
        print(pretty(top1_interface))
        print("")

        # Decide whether retrieval is good
        good = is_retrieval_good(metric, score, args.min_sim, args.max_l2)
        composed_interface: Optional[Dict[str, Any]] = None
        used_interface_for_fillin: Any = top1_interface_u

        if (not good) and (not args.no_decompose):
            print(f"[info] top1 not good enough (metric={metric}, score={score:.6f}). Trigger decomposition.")
            de_prompt = build_decompose_prompt(q, max_n=args.max_subsystems)

            try:
                de_text = ollama_generate(de_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
                de_obj = parse_model_json_output(de_text)
            except Exception as e:
                print(f"[error] decomposition failed: {e}")
                de_obj = None

            if not isinstance(de_obj, list) or not de_obj:
                print("[warn] decomposition returned invalid output; fallback to top1 fill-in.")
            else:
                sub_items = de_obj[: args.max_subsystems]
                sub_queries: List[str] = []
                sub_names: List[str] = []

                for idx, item in enumerate(sub_items):
                    if isinstance(item, dict) and "query" in item and "name" in item:
                        sq = item["query"]
                        sn = item["name"]
                        if isinstance(sq, str) and sq.strip():
                            sub_queries.append(sq.strip())
                            sub_names.append(str(sn))
                    elif isinstance(item, str) and item.strip():
                        # tolerate plain string list; assign default names
                        sub_queries.append(item.strip())
                        sub_names.append(f"sub_system_{idx+1}")

                if not sub_queries:
                    print("[warn] no valid sub-queries; fallback to top1 fill-in.")
                else:
                    # retrieve each sub-query -> get its interface (assumed exists)
                    sub_ifaces: List[Any] = []
                    for idx, sq in enumerate(sub_queries):
                        sqvec = build_query_vectors(embed_model, [sq], normalize=normalize)
                        sD, sI = faiss_search(index, sqvec, k=1)
                        sid = int(sI[0, 0])
                        sscore = float(sD[0, 0])
                        smeta = metadata[sid] if (sid >= 0 and metadata and sid < len(metadata)) else None
                        siface = extract_interface_payload(smeta)
                        siface_u = unwrap_interface(siface)

                        # infer/clean a subsystem name
                        inferred = infer_subsystem_name(siface_u, fallback=sub_names[idx] if idx < len(sub_names) else f"sub_system_{idx+1}")
                        sub_names[idx] = inferred

                        sub_ifaces.append(siface_u)

                        print(f"  sub[{idx}] name={sub_names[idx]!r}  top1_id={sid}  {label}={sscore:.6f}")
                        # 不额外打印接口内容，避免日志爆炸；需要的话你可以打开下一行
                        # print(pretty(siface_u))

                    # compose
                    composed_interface = compose_interfaces(
                        interfaces=sub_ifaces,
                        names=sub_names,
                        composed_id="dtmi:composed:System;1",
                        composed_display_name="ComposedSystem",
                    )

                    print("\ncomposed interface (required format):")
                    print(pretty(composed_interface))
                    print("")

                    used_interface_for_fillin = composed_interface

        # 2) Fill-in (skip if disabled)
        if args.no_fillin:
            print("-" * 100)
            continue

        if not isinstance(exp, dict):
            print("[warn] expected_output is not a dict; skip evaluation.")
            print("-" * 100)
            continue

        if not isinstance(used_interface_for_fillin, dict):
            print("[warn] interface for fill-in is not a dict; skip fill-in.")
            print("-" * 100)
            continue

        # 3) Fill-in via Ollama
        fill_prompt = build_fillin_prompt(description_text=q, interface_obj=used_interface_for_fillin)

        try:
            fill_text = ollama_generate(fill_prompt, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout_s=args.timeout)
            fill_obj = parse_model_json_output(fill_text)
        except Exception as e:
            print(f"[error] fill-in failed: {e}")
            print("-" * 100)
            continue

        if not isinstance(fill_obj, dict):
            print("[error] fill-in output is not a JSON dict.")
            print("raw fill-in output:")
            print(fill_text)
            print("-" * 100)
            continue

        print("fill-in (model output):")
        print(pretty(fill_obj))
        print("")

        # 4) Compare (properties only; ignore telemetries; ignore @id/displayName/dockerImage)
        report, acc = compare_fillin_to_groundtruth(
            fillin=fill_obj,
            expected_output=exp,
            interface_obj=used_interface_for_fillin,
        )

        print("comparison (properties only; ignore @id/displayName/dockerImage; ignore telemetries):")
        print(pretty(report))
        print("")
        print(f"per-sample accuracy: {acc:.4f}")
        print("-" * 100)

        summary = report.get("summary", {})
        overall_correct += int(summary.get("correct", 0))
        overall_total += int(summary.get("total", 0))

    if overall_total > 0:
        print(f"OVERALL property accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.4f}")
    else:
        print("OVERALL property accuracy: n/a (no evaluated fields)")


if __name__ == "__main__":
    main()