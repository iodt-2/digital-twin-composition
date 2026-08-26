#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill DTDL properties from the fine-tuning dataset, with a local model or via Ollama.

    python 2.perf-eval-fill-gen-local.py --model models/Qwen2-0.5B-GRPO-Fill-In
    python 2.perf-eval-fill-gen-local.py --ollama --model gpt-oss:120b
    LOCAL_MODEL=models/Qwen2-0.5B-GRPO/checkpoint-5500 python 2.perf-eval-fill-gen-local.py

Reads a `datasets` directory saved with `save_to_disk` (default
`data/llm-fill-ft-80-20.ds`, split `test` — the 20% the 80/20 fine-tune held out).
Its rows already carry the flat fill-in `prompt` and the `ground_truth` JSON, so
neither `fill-eval.jsonl` nor the `interfaces.jsonl` schema lookup is needed here: the
field names and DTDL types are read back out of the prompt that is sent.

Writes into `results/<label>/`, where `<label>` is the last path segment of the model
(`:` replaced, so an Ollama tag is a usable directory name):

    filled-output-<label>.jsonl   one JSON object per row of the split, in row order
    filled-output-<label>.done    resume log — re-running skips the rows listed here
    progress-<label>.json, sample_time_stats-<label>.txt

The predictions are line-aligned with the split rather than with `fill-eval.jsonl`
(invariant 2 holds against whatever the input file is). The scorer matches a prediction
file to the input it was produced from by row count, so it needs no flag — and scores
these runs alongside the `fill-eval.jsonl` ones in the same table:

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
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies import fill_eval_runner as runner  # noqa: E402

DEFAULT_DATASET = os.environ.get("FILL_DATASET", os.path.join("data", "llm-fill-ft-80-20.ds"))
DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2-0.5B-Instruct")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.10.10.4:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

# The `- "name" (schema)` lines of the prompt. The dataset stores the prompt but not the
# property spec, and this is the very list `fill_eval_runner.build_prompt` wrote into it,
# so recovering it here keeps predictions coerced exactly as the jsonl backends coerce
# them — same keys, same DTDL types, same comparison against the gold answer.
FIELD_RE = re.compile(r'^- "([^"]+)" \(([^)]+)\)$', re.M)


def local_generator(model: str, max_new_tokens: int):
    """Greedy decoding: this is extraction, and the checkpoint's generation_config
    inherits Qwen2-Instruct's chat sampling defaults, which are not what we want.

    Both settings go on the pipeline once, not into every call: the pipeline already
    sends its own `generation_config`, and transformers 5 deprecates passing generation
    arguments alongside it. `max_length` is then cleared because the pipeline fills it
    with the global default of 20, which loses to `max_new_tokens` and says so on every
    sample. What is generated is unchanged - greedy, 512 new tokens, everything else
    (eos/pad ids, the checkpoint's repetition_penalty) still from generation_config.json.
    """
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model,
                    do_sample=False, max_new_tokens=max_new_tokens)
    pipe.generation_config.max_length = None

    def generate(prompt: str) -> str:
        return pipe(prompt, return_full_text=False)[0]["generated_text"]

    return generate


class ServerUnavailable(RuntimeError):
    """The transport failed, not the record.

    A refused connection, a read timeout or a 404 for the model says nothing about the
    sample. Recording it the way a bad answer is recorded would write `{}` *and* mark the
    row done, so a server that is down or a `--timeout` set too low for the model quietly
    burns all 5554 rows, unrecoverably. The run aborts on it instead, having written
    nothing for the row, so a re-run after fixing the server resumes exactly here.
    """


def ollama_generator(host: str, model: str, max_new_tokens: int, timeout: int):
    """One `/api/generate` completion per prompt, temperature 0 for the same reason."""
    url = host.rstrip("/") + "/api/generate"

    def generate(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0, "num_predict": max_new_tokens}}
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                answer = json.loads(response.read().decode("utf-8"))
        except OSError as e:  # URLError, HTTPError and socket timeouts are all OSError
            raise ServerUnavailable(f"{url} ({model}): {e}") from e
        # A thinking model that spends its whole `num_predict` budget on reasoning
        # returns an empty `response`; that is a real answer of nothing, not a failure.
        return str(answer.get("response") or "")

    return generate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help=f"Model path or hub id, or the Ollama tag with --ollama "
                         f"(default: {DEFAULT_MODEL}, or {OLLAMA_MODEL} with --ollama)")
    ap.add_argument("--ollama", action="store_true",
                    help="Generate on an Ollama server instead of loading the model here")
    ap.add_argument("--ollama-host", default=OLLAMA_HOST,
                    help=f"Ollama base URL (default: $OLLAMA_HOST or {OLLAMA_HOST})")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"save_to_disk directory (default: {DEFAULT_DATASET})")
    ap.add_argument("--split", default="test", help="Split to run (default: test)")
    ap.add_argument("--out-root", default="results", help="Parent directory for the output folder")
    ap.add_argument("--label", default=None, help="Output folder name (default: last segment of --model)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=900,
                    help="Ollama HTTP timeout in seconds (default: 900). A thinking model "
                         "on a host that cannot hold it in VRAM answers in minutes, not "
                         "seconds; a timeout now aborts the run rather than losing rows")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new samples (0 = all)")
    args = ap.parse_args()

    model = args.model or (OLLAMA_MODEL if args.ollama else DEFAULT_MODEL)
    label = args.label or Path(model).name.replace(":", "-")

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

    generate = (ollama_generator(args.ollama_host, model, args.max_new_tokens, args.timeout)
                if args.ollama else local_generator(model, args.max_new_tokens))

    done = runner.load_done_indices(done_path)
    stats = runner.Stats(progress_path)
    started_ts = time.time()
    where = f"{args.ollama_host} {model}" if args.ollama else model
    print(f"[INFO] {label} ({where}): {args.dataset}[{args.split}], {total} rows, "
          f"{len(done)} already done -> {output_path}", flush=True)
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
                      f"written for this one. Fix the server (or raise --timeout) and "
                      f"re-run to resume here.", file=sys.stderr, flush=True)
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
