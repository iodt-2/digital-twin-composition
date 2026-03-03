#!/usr/bin/env python3

import os
import json
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


# --- Configs from env (as provided) ---
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://10.1.1.49:60002")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
DEFAULT_DATASET_PATH = os.getenv("ETE_EVAL_PATH", "./data/ete-eval.jsonl")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./models/faiss.index")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./models/embeddings.npy")  # optional sanity check
METADATA_PATH = os.getenv("METADATA_PATH", "./models/metadata.json")
SENTENCE_TRANSFORMER_PATH = os.getenv("SENTENCE_TRANSFORMER_PATH", "./models/MiniLM-L6-based-new-triplets-final")


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
    """
    Supports:
      - list[dict] directly
      - dict with a top-level list under common keys like "items", "docs", "data"
    """
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


def guess_metric(index: "faiss.Index") -> str:
    """
    Heuristic:
      - If index.metric_type exists and is INNER_PRODUCT / L2, use it.
      - Otherwise, default to 'ip' (common for cosine/IP setups).
    """
    metric = None
    if hasattr(index, "metric_type"):
        metric = index.metric_type
        # faiss.METRIC_INNER_PRODUCT == 0, faiss.METRIC_L2 == 1 (typically)
        if metric == faiss.METRIC_L2:
            return "l2"
        if metric == faiss.METRIC_INNER_PRODUCT:
            return "ip"
    return "ip"


def maybe_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def build_query_vectors(
    model: SentenceTransformer,
    queries: List[str],
    normalize: bool,
) -> np.ndarray:
    q = model.encode(
        queries,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll control normalization explicitly
    ).astype(np.float32)

    if q.ndim == 1:
        q = q.reshape(1, -1)

    if normalize:
        q = maybe_normalize(q)

    return q


def faiss_search(
    index: "faiss.Index",
    qvecs: np.ndarray,
    k: int,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores_or_distances, ids) as (D, I).
      - metric == "ip": D is similarity (higher is better)
      - metric == "l2": D is squared L2 distance (lower is better)
    """
    if qvecs.dtype != np.float32:
        qvecs = qvecs.astype(np.float32)
    D, I = index.search(qvecs, k)
    return D, I


def format_hit(
    doc_id: int,
    score: float,
    meta: Optional[Dict[str, Any]],
    metric: str,
) -> str:
    # For L2, smaller is better. Still print the raw value; you can invert if you want.
    label = "sim" if metric == "ip" else "l2"
    if meta is None:
        return f"  id={doc_id}  {label}={score:.6f}"

    # Try common fields; fall back to a compact JSON snippet
    title = meta.get("title") or meta.get("name") or meta.get("id") or ""
    text = meta.get("text") or meta.get("content") or meta.get("chunk") or meta.get("document") or ""
    text_preview = (text[:180] + "…") if isinstance(text, str) and len(text) > 180 else text

    extra = ""
    if title:
        extra += f"  title={title!r}"
    if text_preview:
        extra += f"  text={text_preview!r}"

    return f"  id={doc_id}  {label}={score:.6f}{extra}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Top-k results per query.")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Only process the first N queries (0 = all).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset into the JSONL dataset (useful for sharding).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize query embeddings (recommended for cosine/IP setups).",
    )
    parser.add_argument(
        "--metric",
        choices=["auto", "ip", "l2"],
        default="auto",
        help="FAISS metric to interpret scores: ip=inner product (similarity), l2=distance.",
    )
    parser.add_argument(
        "--print_expected",
        action="store_true",
        help="Also print expected_output from the dataset lines.",
    )
    args = parser.parse_args()

    # Load index
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS_INDEX_PATH not found: {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Optional: sanity check index size vs embeddings.npy
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

    # Load metadata (must align with index ids)
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

    # Determine metric
    metric = guess_metric(index) if args.metric == "auto" else args.metric

    # If metric is IP and you want cosine similarity, normalize both query + indexed vectors.
    # This script can normalize query vectors; ensure your index vectors were normalized too.
    normalize = args.normalize or (metric == "ip")

    # Load dataset queries
    if not os.path.exists(DEFAULT_DATASET_PATH):
        raise FileNotFoundError(f"DEFAULT_DATASET_PATH not found: {DEFAULT_DATASET_PATH}")
    rows = load_jsonl(DEFAULT_DATASET_PATH)

    # Slice
    rows = rows[args.start :] if args.start > 0 else rows
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    queries: List[str] = []
    for r in rows:
        q = r.get("generated_user_query")
        if not isinstance(q, str) or not q.strip():
            continue
        queries.append(q.strip())

    if not queries:
        raise ValueError("No valid 'generated_user_query' strings found in dataset slice.")

    # Embed queries
    if not os.path.exists(SENTENCE_TRANSFORMER_PATH):
        raise FileNotFoundError(f"SENTENCE_TRANSFORMER_PATH not found: {SENTENCE_TRANSFORMER_PATH}")

    model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
    qvecs = build_query_vectors(model, queries, normalize=normalize)

    # Search
    D, I = faiss_search(index, qvecs, k=args.k, metric=metric)

    # Print results
    print(f"Host/model (unused here): {DEFAULT_HOST} / {DEFAULT_MODEL}")
    print(f"Dataset: {DEFAULT_DATASET_PATH}")
    print(f"Index: {FAISS_INDEX_PATH}   metric={metric}   normalize_queries={normalize}")
    print("-" * 80)

    for qi, q in enumerate(queries):
        print(f"[{qi}] query: {q!r}")

        for rank in range(args.k):
            doc_id = int(I[qi, rank])
            score = float(D[qi, rank])

            # FAISS uses -1 for invalid ids sometimes
            if doc_id < 0:
                continue

            meta = metadata[doc_id] if metadata and 0 <= doc_id < len(metadata) else None
            print(format_hit(doc_id, score, meta, metric))

        print("-" * 80)


if __name__ == "__main__":
    main()