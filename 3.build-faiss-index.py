#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the FAISS retrieval index that `3.system-eval.py` and the agent demo search.

    python 3.build-faiss-index.py --model models/MiniLM-L6-based-new-triplets-final

Writes four index-aligned artifacts — row *i* of each refers to the same interface:

    models/faiss.index          inner-product index over L2-normalised embeddings
    models/embeddings.npy       the same vectors, float32, for inspection or re-indexing
    models/metadata.json        [{"interface": {...}, "@id": ..., "displayName": ...}, ...]
    data/dataset_original.jsonl one {"interface": {...}} per line

`dataset_original.jsonl` exists because `3.system-eval.py` scores retrieval by FAISS id:
it groups that file's line numbers by the topic segment of each `@id` and checks whether
the ids that came back belong to the query's topic. Line number - 1 *is* the FAISS id,
so the file must be written here, from the same list, in the same order.

What gets embedded
------------------
The full interface JSON, serialised exactly as `0.data-gen-interface-to-triplet.py`
serialises the `positive` column. The retrieval model was trained with MNRL against
those strings, so anything else — the description alone, a summary — is off the
distribution it learned.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_INTERFACES = os.path.join("data", "interfaces.jsonl")
DEFAULT_MODEL = os.getenv("SENTENCE_TRANSFORMER_PATH",
                          os.path.join("models", "MiniLM-L6-based-new-triplets-final"))
DEFAULT_INDEX = os.getenv("FAISS_INDEX_PATH", os.path.join("models", "faiss.index"))
DEFAULT_EMBEDDINGS = os.getenv("EMBEDDINGS_PATH", os.path.join("models", "embeddings.npy"))
DEFAULT_METADATA = os.getenv("METADATA_PATH", os.path.join("models", "metadata.json"))
DEFAULT_DATASET_ORIGINAL = os.getenv("DATASET_ORIGINAL_PATH",
                                     os.path.join("data", "dataset_original.jsonl"))


def load_interfaces(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno} skipped, invalid JSON: {e}", file=sys.stderr)
                continue
            if isinstance(obj, dict) and obj.get("@id"):
                rows.append(obj)
            else:
                print(f"[WARN] {path}:{lineno} skipped, no @id", file=sys.stderr)
    return rows


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interfaces", default=DEFAULT_INTERFACES, help="Interface catalogue JSONL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer path or hub id")
    ap.add_argument("--index-out", default=DEFAULT_INDEX)
    ap.add_argument("--embeddings-out", default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--metadata-out", default=DEFAULT_METADATA)
    ap.add_argument("--dataset-original-out", default=DEFAULT_DATASET_ORIGINAL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="Index only the first N interfaces (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(args.interfaces):
        print(f"[ERROR] Not found: {args.interfaces}", file=sys.stderr)
        return 1

    interfaces = load_interfaces(args.interfaces)
    if args.limit:
        interfaces = interfaces[: args.limit]
    if not interfaces:
        print(f"[ERROR] No usable interfaces in {args.interfaces}", file=sys.stderr)
        return 1
    print(f"[INFO] {len(interfaces)} interfaces from {args.interfaces}")

    documents = [json.dumps(iface, ensure_ascii=False) for iface in interfaces]

    print(f"[INFO] Encoding with {args.model}")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        # Normalised here so an inner-product index returns cosine similarity directly;
        # 3.system-eval.py's --min_sim threshold is stated in cosine terms.
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"[INFO] Embeddings: {embeddings.shape}")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"[INFO] Index built: ntotal={index.ntotal}")

    for path in (args.index_out, args.embeddings_out, args.metadata_out, args.dataset_original_out):
        ensure_parent_dir(path)

    faiss.write_index(index, args.index_out)
    np.save(args.embeddings_out, embeddings)

    metadata = [
        {
            "faiss_id": i,
            "interface": iface,
            "@id": iface.get("@id"),
            "displayName": iface.get("displayName"),
        }
        for i, iface in enumerate(interfaces)
    ]
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    with open(args.dataset_original_out, "w", encoding="utf-8") as f:
        for iface in interfaces:
            f.write(json.dumps({"interface": iface}, ensure_ascii=False) + "\n")

    print(f"[DONE] {args.index_out}\n"
          f"       {args.embeddings_out}\n"
          f"       {args.metadata_out}\n"
          f"       {args.dataset_original_out}  ({len(interfaces)} lines; line N -> faiss id N-1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
