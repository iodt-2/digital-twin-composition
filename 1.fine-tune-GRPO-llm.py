#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO fine-tuning of Qwen2-0.5B-Instruct for DTDL property fill-in.

    python 1.fine-tune-GRPO-llm.py --dataset llm-fill-ft.ds --output-dir Qwen2-0.5B-GRPO

The dataset is a `datasets` directory on disk with at least a `prompt` column and a
`ground_truth` column holding the expected answer object as a JSON string.

Reward functions
----------------
Two are available, and the choice is not cosmetic — it changes what the model learns:

`shipped` (default) reproduces the run that produced the released
`Qwen2-0.5B-GRPO-Fill-In` checkpoint, quirks included. It only ever scores fenced
```json output, because the bare-JSON branch indexes a list of chat messages as if it
were a dict, raises, and lands in the `except` at -1. A perfect fenced answer is worth
0.8, which is exactly where the logged reward curve plateaus
(`results/llm-fill-in/reward.csv`). The published model card documents this, so keep
this setting to reproduce or extend that checkpoint.

`corrected` scores both fenced and bare JSON, compares float values with a tolerance
rather than by identity, and tops out at 1.0. Use it for new runs. It will not
reproduce the published reward curve.
"""

import argparse
import json
import math
import re
from typing import Any, Dict, List

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

FENCED_JSON = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
FLOAT_TOLERANCE = 1e-3


def _completion_text(completion: Any) -> str:
    """GRPO hands back either a string or a list of chat messages, depending on setup."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _values_agree(gold: Any, pred: Any) -> bool:
    if isinstance(gold, bool) or isinstance(pred, bool):
        return gold == pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(gold), float(pred), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
    return gold == pred


def reward_shipped(completions, prompts, ground_truth, **kwargs) -> List[float]:
    """The reward function the released checkpoint was trained with. See module docstring.

    Preserved verbatim in behaviour: `completions[i][0]['content']` succeeds only for
    chat-shaped rollouts, and the `else` branch's `completions[i]['content']` raises
    `TypeError` on a list, so unfenced output scores -1 rather than 1.0.
    """
    rewards = []
    for i in range(len(completions)):
        r = 1
        try:
            match = FENCED_JSON.search(completions[i][0]["content"])
            if match:
                p = json.loads(match.group(1))
                r = 0.8
            else:
                p = json.loads(completions[i]["content"])
            count = len(p)
            for k in p:
                g = json.loads(ground_truth[i])
                if k not in g:
                    count -= 1
                    continue
                if isinstance(g, float):
                    try:
                        if float(g) - float(p) > FLOAT_TOLERANCE:
                            count -= 1
                    except Exception:
                        count -= 1
                else:
                    if g[k] != p[k]:
                        count -= 1
            for k in g:
                if k not in p:
                    count -= 1
            frac = count / len(p)
            r = r / frac if r < 0 else r * frac
        except Exception:
            r = -1
        rewards.append(r)
    return rewards


def reward_corrected(completions, prompts, ground_truth, **kwargs) -> List[float]:
    """Symmetric key/value agreement in [-1, 1]; unparseable output scores -1.

    Both fenced and bare JSON are accepted, so the format factor no longer caps a
    correct answer below 1.0.
    """
    rewards = []
    for completion, gold_raw in zip(completions, ground_truth):
        text = _completion_text(completion)
        match = FENCED_JSON.search(text)
        payload = match.group(1) if match else text.strip()
        try:
            pred: Dict[str, Any] = json.loads(payload)
            gold: Dict[str, Any] = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
        except (json.JSONDecodeError, TypeError):
            rewards.append(-1.0)
            continue

        if not isinstance(pred, dict) or not isinstance(gold, dict) or not pred:
            rewards.append(-1.0)
            continue

        # Start from the predicted key count and dock a point for every disagreement in
        # either direction: an invented key, a wrong value, or a ground-truth key omitted.
        count = len(pred)
        for k, pv in pred.items():
            if k not in gold or not _values_agree(gold[k], pv):
                count -= 1
        count -= sum(1 for k in gold if k not in pred)

        rewards.append(max(-1.0, count / len(pred)))
    return rewards


REWARDS = {"shipped": reward_shipped, "corrected": reward_corrected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="llm-fill-ft.ds", help="`datasets` directory saved with save_to_disk")
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct", help="Base model to fine-tune")
    ap.add_argument("--output-dir", default="Qwen2-0.5B-GRPO", help="Checkpoint output directory")
    ap.add_argument("--epochs", type=float, default=1, help="Training epochs")
    ap.add_argument("--reward", choices=sorted(REWARDS), default="shipped",
                    help="Reward function; 'shipped' reproduces the released checkpoint (see docstring)")
    ap.add_argument("--split", choices=["train", "test"], default="test",
                    help="Which half of the 70/30 split to train on. The released checkpoint used "
                         "'test' — the 30%% shard — leaving the rest unseen for evaluation.")
    ap.add_argument("--test-size", type=float, default=0.3, help="Fraction held out by train_test_split")
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    args = ap.parse_args()

    full_dataset = load_from_disk(args.dataset)
    split = full_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    dataset = split[args.split]
    print(f"[INFO] Training on the '{args.split}' shard: {len(dataset)} examples "
          f"of {len(full_dataset)}")
    print(f"[INFO] Reward function: {args.reward}")
    if args.reward == "shipped":
        print("[WARN] The 'shipped' reward only rewards fenced ```json output and caps at 0.8. "
              "Use --reward corrected for new runs.")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=REWARDS[args.reward],
        args=training_args,
        train_dataset=dataset,
    )
    # Left padding is required for batched generation during GRPO rollouts.
    tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
    tokenizer.padding_side = "left"
    trainer.train()


if __name__ == "__main__":
    main()
