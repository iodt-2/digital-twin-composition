#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill DTDL properties from anchor paragraphs through an OpenAI-compatible chat endpoint.

    export OPENAI_API_KEY=...
    python 2.perf-eval-fill-gen-gemini.py --model models/gemini-2.5-pro

Defaults target Gemini's OpenAI-compatible surface; point `--base-url` at any other
provider that speaks `/chat/completions`. Output layout, resume behaviour and timing
stats are identical to `2.perf-eval-fill-gen-local.py`, so the two are directly
comparable.

Score the output with `2.perf-eval-result-eval.py`.
"""

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies import fill_eval_runner as runner  # noqa: E402

DEFAULT_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "models/gemini-2.5-pro")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    ap.add_argument("--interfaces", default=os.path.join("data", "interfaces.jsonl"))
    ap.add_argument("--fill-eval", default=os.path.join("data", "fill-eval.jsonl"))
    ap.add_argument("--out-root", default="results", help="Parent directory for the output folder")
    ap.add_argument("--label", default=None, help="Output folder name (default: last segment of --model)")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new samples (0 = all)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GENAI_API_KEY")
    if not api_key:
        print("[ERROR] Set OPENAI_API_KEY (or GENAI_API_KEY) before running.", file=sys.stderr)
        return 2

    label = args.label or Path(args.model).name
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    system_prompt = (
        "You are an information extraction assistant. "
        "Extract values exactly from user-provided ANCHOR text. "
        "If a value is not explicitly stated, output null. "
        "Do not hallucinate or infer unstated facts. "
        "Return ONLY a minified JSON object with EXACT keys requested."
    )

    def extract(anchor, interface_id, props_spec):
        fields_desc = "\n".join(f'- "{p["name"]}" ({p["schema"]})' for p in props_spec)
        user_prompt = (
            f'Interface: "{interface_id}"\n'
            f"Required fields and types:\n{fields_desc}\n\n"
            f'ANCHOR:\n"""{anchor}"""'
        )
        payload = {
            "model": args.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
        resp.raise_for_status()
        text = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        obj = runner.parse_json_object(text)
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
