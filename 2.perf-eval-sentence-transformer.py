#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark embedding models on the held-out triplet test split.

    python 2.perf-eval-sentence-transformer.py
    python 2.perf-eval-sentence-transformer.py --models all-MiniLM-L6-v2 ./models/my-run-final

The split is re-derived from `data/triplet.jsonl` with the same seed
`1.fine-tune-sentence-transformer.py` uses, so the test set here is the 10% that
training never saw and never evaluated on.
"""

import argparse
import os
import sys
import time

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SimilarityFunction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies.SuperTripletEvaluator import SuperTripletEvaluator  # noqa: E402

DEFAULT_DATAFILE = os.path.join("data", "triplet.jsonl")
DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-distilroberta-v1",
    "multi-qa-distilbert-cos-v1",
    "multi-qa-MiniLM-L6-cos-v1",
    "./models/MiniLM-L6-based-new-triplets-final",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATAFILE, help=f"Triplet JSONL (default: {DEFAULT_DATAFILE})")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model names or local paths")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate on the first N test triplets (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="Split seed; must match the training script")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"[ERROR] Triplet file not found: {args.data}")

    full_dataset = load_dataset("json", data_files=args.data)["train"]
    train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=args.seed)
    test_dataset = train_testvalid["test"].train_test_split(test_size=0.5, seed=args.seed)["test"]
    if args.limit:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    print(f"[INFO] Evaluating on {len(test_dataset)} held-out triplets\n")

    evaluator = SuperTripletEvaluator(
        anchors=test_dataset["query"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        main_similarity_function=SimilarityFunction.COSINE,
        name="triplet-test",
    )

    for name in args.models:
        try:
            model = SentenceTransformer(name)
        except Exception as e:  # noqa: BLE001 - a missing local checkpoint should not end the sweep
            print(f"[WARN] {name}: could not load ({e})")
            continue
        start = time.time()
        results = evaluator(model)
        results["time_taken_seconds"] = time.time() - start
        print(f"{name}: {results}")


if __name__ == "__main__":
    main()
