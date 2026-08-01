#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill DTDL properties from anchor paragraphs with a local Transformers model.

    python 2.perf-eval-fill-gen-local.py --model models/Qwen2-0.5B-GRPO-Fill-In
    LOCAL_MODEL=models/Qwen2-0.5B-GRPO/checkpoint-5500 python 2.perf-eval-fill-gen-local.py

Reads `data/fill-eval.jsonl`, looks the property schema up in `data/interfaces.jsonl`,
and writes one JSON object per input line to
`results/<label>/filled-output-<label>.jsonl`, where `<label>` is the last path segment
of the model. Resumable: re-running skips indices recorded in the `.done` file.

Score the output with `2.perf-eval-result-eval.py`.
"""

import argparse
import os
import sys
from pathlib import Path

from transformers import pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies import fill_eval_runner as runner  # noqa: E402

DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "models/Qwen2-0.5B-GRPO-Fill-In")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Model path or hub id (default: $LOCAL_MODEL or {DEFAULT_MODEL})")
    ap.add_argument("--interfaces", default=os.path.join("data", "interfaces.jsonl"))
    ap.add_argument("--fill-eval", default=os.path.join("data", "fill-eval.jsonl"))
    ap.add_argument("--out-root", default="results", help="Parent directory for the output folder")
    ap.add_argument("--label", default=None, help="Output folder name (default: last segment of --model)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new samples (0 = all)")
    args = ap.parse_args()

    label = args.label or Path(args.model).name

    # Greedy decoding: this is extraction, and the checkpoint's generation_config
    # inherits Qwen2-Instruct's chat sampling defaults, which are not what we want.
    pipe = pipeline("text-generation", model=args.model)

    def extract(anchor, interface_id, props_spec):
        prompt = runner.build_prompt(anchor, interface_id, props_spec)
        outputs = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        obj = runner.parse_json_object(outputs[0]["generated_text"])
        return runner.clean_against_spec(obj, props_spec)

    return runner.run(
        extract=extract,
        label=label,
        interfaces_path=Path(args.interfaces),
        fill_eval_path=Path(args.fill_eval),
        out_dir=Path(args.out_root) / label,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
