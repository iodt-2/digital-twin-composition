#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a SentenceTransformer to retrieve DTDL interfaces from natural-language queries.

    python 1.fine-tune-sentence-transformer.py
    python 1.fine-tune-sentence-transformer.py --base-model ./deberta-base --run-name deberta-v1
    python 1.fine-tune-sentence-transformer.py --no-wandb --epochs 1

Input is the triplet database from `0.data-gen-interface-to-triplet.py`:
`{query, positive, negative}`, where positive/negative are interface JSON *strings*.
The trained model is what `3.system-eval.py` loads as `SENTENCE_TRANSFORMER_PATH`.

Objectives
----------
`mnrl`   MultipleNegativesRankingLoss over (query, positive, negative) triplets. The
         default: it uses every other positive in the batch as an extra negative, so
         the effective negative count scales with `--batch-size`.
`cosent` CoSENTLoss over (sentence1, sentence2, score) pairs. Only for a scored-pair
         dataset — it will not run on the triplet file.
"""

import argparse
import os
import sys

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SimilarityFunction, TripletEvaluator
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies.SuperTripletEvaluator import SuperTripletEvaluator  # noqa: E402
from dependencies.ThresholdedTripletEvaluator import ThresholdedTripletEvaluator  # noqa: E402

DEFAULT_DATAFILE = os.path.join("data", "triplet.jsonl")
DEFAULT_BASE_MODEL = "nreimers/MiniLM-L6-H384-uncased"


def build_evaluator(kind: str, eval_dataset, threshold: float):
    """Pick the metric set to watch during training.

    `super` is the default: it reports triplet accuracy *and* the raw positive/negative
    cosines, which is what the curves in `results/sentence-transformers/` are plotted from.
    """
    if kind == "super":
        return SuperTripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_similarity_function=SimilarityFunction.COSINE,
            name="triplet-dev",
        )
    if kind == "triplet":
        return TripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_similarity_function=SimilarityFunction.COSINE,
            name="triplet-dev",
        )
    if kind == "thresholded":
        return ThresholdedTripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            threshold=threshold,
            name="triplet-dev",
            batch_size=64,
            show_progress_bar=True,
        )
    raise ValueError(f"Unknown evaluator: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATAFILE, help=f"Triplet JSONL (default: {DEFAULT_DATAFILE})")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model name or local path")
    ap.add_argument("--run-name", default="MiniLM-L6-based-v2", help="Run name; also the output directory under models/")
    ap.add_argument("--loss", choices=["mnrl", "cosent"], default="mnrl", help="Training objective")
    ap.add_argument("--evaluator", choices=["super", "triplet", "thresholded"], default="super",
                    help="Evaluator to run during training")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Similarity threshold for --evaluator thresholded; match 3.system-eval.py --min_sim")
    ap.add_argument("--epochs", type=float, default=5, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Per-device batch size. MNRL benefits from the largest that fits in VRAM.")
    ap.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--eval-steps", type=int, default=100, help="Evaluate every N steps")
    ap.add_argument("--save-steps", type=int, default=1000, help="Checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument("--bf16", action="store_true", help="Train in bfloat16")
    ap.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction,
                    help="Log to Weights & Biases (default: on; --no-wandb to disable)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"[ERROR] Triplet file not found: {args.data}\n"
                         f"        Build it with: python 0.data-gen-interface-to-triplet.py")

    model = SentenceTransformer(args.base_model)

    dataset = load_dataset("json", data_files=args.data)
    full_dataset = dataset["train"]
    # 80 / 10 / 10. The test half of the eval split is reserved for
    # 2.perf-eval-sentence-transformer.py, which re-derives it with the same seed.
    train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=args.seed)
    test_valid_split = train_testvalid["test"].train_test_split(test_size=0.5, seed=args.seed)
    train_dataset = train_testvalid["train"]
    eval_dataset = test_valid_split["train"]
    print(f"[INFO] {len(train_dataset)} train / {len(eval_dataset)} eval / "
          f"{len(test_valid_split['test'])} held-out test triplets")

    loss = CoSENTLoss(model) if args.loss == "cosent" else MultipleNegativesRankingLoss(model)

    report_to = []
    if args.wandb:
        import wandb
        wandb.init(project="sentence-transformers", name=args.run_name)
        report_to = ["wandb"]

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"models/{args.run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        fp16=False,
        bf16=args.bf16,
        # MNRL treats other in-batch rows as negatives, so a duplicated query in a batch
        # would be scored as its own negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=500,
        run_name=args.run_name,
        report_to=report_to,
        push_to_hub=False,
    )

    dev_evaluator = build_evaluator(args.evaluator, eval_dataset, args.threshold)
    print(f"[INFO] Baseline before training: {dev_evaluator(model)}")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    final_dir = f"models/{args.run_name}-final"
    model.save_pretrained(final_dir)
    print(f"[DONE] Saved to {final_dir}")


if __name__ == "__main__":
    main()
