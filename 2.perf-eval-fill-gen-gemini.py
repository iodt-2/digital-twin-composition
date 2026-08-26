#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill DTDL properties from the fine-tuning dataset with a hosted Gemini model.

    python 2.perf-eval-fill-gen-gemini.py --model gemini-2.5-pro
    python 2.perf-eval-fill-gen-gemini.py --model gemini-2.5-flash --thinking-budget 0
    GEMINI_API_KEY=... python 2.perf-eval-fill-gen-gemini.py --gemini-api

Reads the same input as `2.perf-eval-fill-gen-local.py`: a `datasets` directory saved
with `save_to_disk` (default `data/llm-fill-ft-80-20.ds`, split `test` - the 20% the
80/20 fine-tune held out), whose rows already carry the flat fill-in `prompt` and the
`ground_truth` JSON. The row's prompt is sent verbatim, and the field names and DTDL
types the answer is coerced to are read back out of it, so a Gemini run and a local or
Ollama run over the same split are line-aligned, scored against the same gold, and
differ in nothing but the model.

Generation goes through `google-genai` (`pip install google-genai`). Vertex AI is the
default surface - it needs `--project` (or `$GOOGLE_CLOUD_PROJECT`), a location, and
application-default credentials from `gcloud auth application-default login`. The
`--gemini-api` flag switches to the Gemini Developer API and reads `$GEMINI_API_KEY` /
`$GOOGLE_API_KEY` instead.

Writes into `results/<label>/`, where `<label>` is the last path segment of the model:

    filled-output-<label>.jsonl   one JSON object per row of the split, in row order
    filled-output-<label>.done    resume log - re-running skips the rows listed here
    progress-<label>.json, sample_time_stats-<label>.txt

Score the output with `2.perf-eval-result-eval.py`, which matches a prediction file to
the input it was produced from by row count and so needs no flag:

    python 2.perf-eval-result-eval.py
    python 2.perf-eval-result-eval.py --partial --pred results/<label>/filled-...jsonl

The first scores everything under `results/`; the second adds a run that is still going,
scored over the rows it has reached.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies import fill_eval_runner as runner  # noqa: E402

DEFAULT_DATASET = os.environ.get("FILL_DATASET", os.path.join("data", "llm-fill-ft-80-20.ds"))
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
# No project id in the source: it is a CLI flag with an env-var default, like every
# other host in this repo.
DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "iodt2-497608")
DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# The `- "name" (schema)` lines of the prompt. The dataset stores the prompt but not the
# property spec, and this is the very list `fill_eval_runner.build_prompt` wrote into it,
# so recovering it here keeps predictions coerced exactly as the other backends coerce
# them - same keys, same DTDL types, same comparison against the gold answer.
FIELD_RE = re.compile(r'^- "([^"]+)" \(([^)]+)\)$', re.M)

# Retrying one of these is worth it; anything else is either this row's fault or fatal.
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
# Missing credentials, wrong project, unknown model: every remaining row would fail the
# same way, so there is nothing to retry and nothing to record.
FATAL_STATUS = {401, 403, 404}


class ServerUnavailable(RuntimeError):
    """The transport failed, not the record.

    A rate limit that outlives the retries, an expired credential or a 404 for the model
    says nothing about the sample. Recording it the way a bad answer is recorded would
    write `{}` *and* mark the row done, so an endpoint that is unreachable quietly burns
    all 5554 rows, unrecoverably. The run aborts on it instead, having written nothing
    for the row, so a re-run after fixing the cause resumes exactly here.
    """


def response_text(response) -> str:
    """The answer's text parts, thinking excluded.

    `response.text` warns or returns None when a candidate holds anything other than a
    single text part - which is what a thinking model returns - so the parts are walked
    here. A prompt stopped by a safety filter, or a candidate that spent its whole output
    budget on reasoning, yields "": a real answer of nothing, not a transport failure.
    """
    chunks = []
    for candidate in (getattr(response, "candidates", None) or []):
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def gemini_generator(model: str, use_gemini_api: bool, project: str, location: str,
                     max_output_tokens: int, thinking_budget, timeout: int,
                     retries: int, retry_wait: float):
    """One `generate_content` completion per prompt.

    Temperature 0 because this is extraction, and `application/json` because the prompt
    already demands a bare JSON object - together they leave the hosted model as little
    room to drift from the local backend as the API allows.
    """
    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError as e:
        raise SystemExit(f"[ERROR] google-genai is not installed ({e}): pip install google-genai")

    if use_gemini_api:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("[ERROR] --gemini-api needs GEMINI_API_KEY (or GOOGLE_API_KEY).")
        # No `api_version` here: the Developer API serves the 2.5 generation from the
        # SDK's own default, and pinning v1 would drop what only v1beta exposes.
        client = genai.Client(api_key=api_key,
                              http_options=types.HttpOptions(timeout=timeout * 1000))  # ms
    else:
        if not project:
            raise SystemExit("[ERROR] Vertex AI needs a project: pass --project or set "
                             "$GOOGLE_CLOUD_PROJECT (or use --gemini-api instead).")
        client = genai.Client(vertexai=True, project=project, location=location,
                              http_options=types.HttpOptions(api_version="v1",
                                                             timeout=timeout * 1000))  # ms

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        # The prompt carries no tools; left enabled, the SDK still walks every response
        # looking for calls to make on our behalf.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=(types.ThinkingConfig(thinking_budget=thinking_budget)
                         if thinking_budget is not None else None),
    )

    def generate(prompt: str) -> str:
        last = None
        for attempt in range(retries + 1):
            try:
                return response_text(client.models.generate_content(
                    model=model, contents=prompt, config=config))
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                if code in FATAL_STATUS:
                    raise ServerUnavailable(f"{model}: {code} {e}") from e
                if code not in TRANSIENT_STATUS:
                    raise  # a 400 on this prompt is this row's answer: `{}`
                last = e
            except OSError as e:  # connection reset, DNS failure, socket timeout
                last = e
            if attempt < retries:
                # Linear backoff: a 429 here is a per-minute quota, and doubling past it
                # only idles the run for longer than the window it is waiting out.
                time.sleep(retry_wait * (attempt + 1))
        raise ServerUnavailable(f"{model}: gave up after {retries + 1} attempts ({last})")

    return generate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Model id (default: $GEMINI_MODEL or {DEFAULT_MODEL})")
    ap.add_argument("--gemini-api", action="store_true",
                    help="Use the Gemini Developer API with $GEMINI_API_KEY instead of Vertex AI")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help="Vertex AI project (default: $GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--location", default=DEFAULT_LOCATION,
                    help=f"Vertex AI location (default: $GOOGLE_CLOUD_LOCATION or {DEFAULT_LOCATION})")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"save_to_disk directory (default: {DEFAULT_DATASET})")
    ap.add_argument("--split", default="test", help="Split to run (default: test)")
    ap.add_argument("--out-root", default="results", help="Parent directory for the output folder")
    ap.add_argument("--label", default=None, help="Output folder name (default: last segment of --model)")
    ap.add_argument("--max-output-tokens", type=int, default=2048,
                    help="Output budget per sample (default: 2048). On a 2.5 model the "
                         "thinking tokens come out of this budget too, so the 512 the "
                         "local backend uses would leave a thinking answer empty")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="Thinking tokens: 0 disables thinking on the models that allow "
                         "that, -1 is dynamic (default: leave the model's own setting alone)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="HTTP timeout in seconds (default: 900). A thinking model on a "
                         "long prompt answers in minutes, not seconds; a timeout that "
                         "outlives the retries aborts the run rather than losing rows")
    ap.add_argument("--retries", type=int, default=5,
                    help="Retries per sample on a rate limit or a 5xx (default: 5)")
    ap.add_argument("--retry-wait", type=float, default=10.0,
                    help="Seconds before the first retry, twice that before the second, "
                         "and so on (default: 10)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new samples (0 = all)")
    args = ap.parse_args()

    label = args.label or Path(args.model).name.replace(":", "-")

    from datasets import load_from_disk

    data = load_from_disk(args.dataset)
    if isinstance(data, dict):  # a DatasetDict; a single-split save is already a Dataset
        if args.split not in data:
            print(f"[ERROR] {args.dataset} has no split {args.split!r}: {list(data)}",
                  file=sys.stderr, flush=True)
            return 1
        data = data[args.split]
    prompts = data["prompt"]
    total = len(prompts)
    if total == 0:
        print("[INFO] Split is empty, nothing to do.", flush=True)
        return 0

    out_dir = Path(args.out_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"filled-output-{label}.jsonl"
    done_path = out_dir / f"filled-output-{label}.done"
    progress_path = out_dir / f"progress-{label}.json"
    stats_path = out_dir / f"sample_time_stats-{label}.txt"

    generate = gemini_generator(args.model, args.gemini_api, args.project, args.location,
                                args.max_output_tokens, args.thinking_budget,
                                args.timeout, args.retries, args.retry_wait)

    done = runner.load_done_indices(done_path)
    stats = runner.Stats(progress_path)
    started_ts = time.time()
    where = "Gemini API" if args.gemini_api else f"Vertex AI {args.project}/{args.location}"
    print(f"[INFO] {label} ({where} {args.model}): {args.dataset}[{args.split}], "
          f"{total} rows, {len(done)} already done -> {output_path}", flush=True)
    runner.write_stats_txt(stats_path, stats)

    processed_this_run = 0
    with output_path.open("a", encoding="utf-8") as out_f, \
            done_path.open("a", encoding="utf-8") as done_f:

        for idx in range(total):
            if idx in done:
                continue

            prompt = prompts[idx]
            props_spec = [{"name": name, "schema": schema}
                          for name, schema in FIELD_RE.findall(prompt)]

            t0 = time.time()
            note = ""
            try:
                filled = runner.clean_against_spec(
                    runner.parse_json_object(generate(prompt)), props_spec)
            except ServerUnavailable as e:
                print(f"\n[ABORT] row {idx + 1}/{total}: {e}\n"
                      f"[ABORT] {len(done)} rows are safely recorded and nothing was "
                      f"written for this one. Fix the endpoint (or raise --timeout / "
                      f"--retries) and re-run to resume here.", file=sys.stderr, flush=True)
                return 2
            except Exception as e:  # noqa: BLE001 - one bad sample must not end the run
                filled = {}
                note = f"  [ERROR] generation failed: {e}"
            elapsed = time.time() - t0

            # Payload first, resume marker second: a crash between the two must lose a
            # record, never claim one that was never written.
            out_f.write(json.dumps(filled, ensure_ascii=False) + "\n")
            runner.fsync_file(out_f)
            done_f.write(f"{idx}\n")
            runner.fsync_file(done_f)
            done.add(idx)

            stats.record(elapsed)
            runner.save_progress_meta(progress_path, len(done), total, stats, started_ts)
            runner.write_stats_txt(stats_path, stats)
            print(f"[DONE] {idx + 1}/{total}  {len(done) / total * 100:.2f}%  "
                  f"took={runner.human_time(elapsed)}  "
                  f"ETA={runner.human_time((total - len(done)) * stats.avg)}  "
                  f"(min={runner.human_time(stats.best)} max={runner.human_time(stats.worst)} "
                  f"avg={runner.human_time(stats.avg)}){note}", flush=True)

            processed_this_run += 1
            if args.limit and processed_this_run >= args.limit:
                print(f"[INFO] Reached --limit={args.limit}, stopping.", flush=True)
                break

    print(f"\n[COMPLETE] {len(done)}/{total}  elapsed={runner.human_time(time.time() - started_ts)}  "
          f"output={output_path}  (min={runner.human_time(stats.best)} "
          f"max={runner.human_time(stats.worst)} avg={runner.human_time(stats.avg)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
