# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**InterTwin** — a research pipeline that takes a natural-language request for a digital
twin and returns an instantiated [DTDL v2](https://github.com/Azure/opendigitaltwins-dtdl)
Interface. It is a paper artifact, not a product: the outputs under `results/` and the
checkpoints on HuggingFace are *published evidence*, so changing what a script computes
changes what has been claimed. Read "Do not silently change" below before editing
scoring, reward, or prompt code.

Run everything from the repository root. Scripts are standalone and prefixed by stage;
there is no package entry point, no test suite, no CI.

```
0.*  data generation      1.*  fine-tuning + model publishing
2.*  per-model evaluation 3.*  index build + system evaluation      4.*  deployment
dependencies/            shared modules imported as `dependencies.<name>`
```

## The data contract

Four invariants hold the pipeline together. Most bugs here are violations of one of them.

1. **`interfaces.jsonl` is the catalogue.** One DTDL Interface JSON object per line, no
   wrapper. Its `@id` has the form `dtmi:<topic>:<Name>;1`, and `<topic>` is the join key
   to `topics.jsonl` and to every `group_id` in the evaluation sets. Splitting on `:` and
   taking index 1 is how every script recovers the topic.

2. **`fill-eval.jsonl` is line-aligned with predictions.** `{"anchor", "answer"}` per
   line. Every fill-in backend must emit exactly one JSON object per input line, in input
   order, writing `{}` when it fails — `2.perf-eval-result-eval.py` rejects files whose
   line count differs, and scores by position.

3. **`dockerImage` and telemetry are excluded from extraction.** The anchor paragraph is
   generated under a prompt that forbids mentioning the interface id, the dockerImage
   path, and telemetry; the answer zeroes telemetry. So asking a model to extract them
   would measure guessing. Every stage filters `dockerImage` by name and skips
   zero-valued keys. If you add a field to the answer, decide which side of that line it
   falls on.

4. **FAISS id == line number - 1 of `data/dataset_original.jsonl`.** `3.system-eval.py`
   scores retrieval by id: it groups that file's lines by topic and asks whether the
   returned ids belong to the query's topic. `3.build-faiss-index.py` writes the index,
   the embeddings, the metadata and that file from one list in one pass, which is the
   only thing keeping them aligned. Never regenerate one without the others.

## Do not silently change

- **`1.fine-tune-GRPO-llm.py --reward shipped`.** It is deliberately buggy: only fenced
  ` ```json ` output is scored, the bare-JSON branch raises `TypeError` and lands at
  `-1`, so a perfect answer caps at 0.8. That cap *is* the plateau in
  `results/llm-fill-in/reward.csv`, and the published model card explains it at length.
  Fixing it in place would make the card wrong and the checkpoint unreproducible. The
  corrected implementation already exists as `--reward corrected`; point people there.

- **The flat fill-in prompt.** `dependencies/fill_eval_runner.build_prompt` and
  `1.model-push-to-huggingface.build_prompt` are character-for-character identical on
  purpose — the second one goes into the published model card and its usage snippet. The
  GRPO checkpoint is sensitive to the encoding (chat template → fenced output; flat
  prompt → bare JSON). Change one, change both, and update the card text.

- **`3.system-eval.py --retry_selection oracle`** (the default) scores each decomposition
  attempt against the ground-truth topic and keeps the best. That is label leakage: the
  composed-route metrics are an upper bound, not a blind-run score. It stays the default
  for continuity with published runs, prints a warning, and `--retry_selection first` is
  the honest alternative. If someone reports composed-route numbers, ask which mode.

- **Metric key names in `dependencies/SuperTripletEvaluator.py`.** The CSVs in
  `results/sentence-transformers/` are keyed by them.

- **`is_zero_value` in `2.perf-eval-result-eval.py`** excludes booleans from the
  zero test on purpose. Before that, `False == 0` made a genuinely-`false` property
  vanish from the required set and turned a correct prediction of it into a false
  positive. Fixing it raised every backend's scores (row EM +~2.6 points), so numbers
  from this version are not comparable with anything quoted before it — the README
  records the before/after. Do not revert it, and do not extend it to booleans again.

## Conventions

- **Resume is mandatory** for anything that makes per-record model calls. Two patterns:
  a `.done` file of completed 0-based indices (fill-in), or a `.ckpt` of SHA-256 hashes
  of input lines (`0.data-gen-fill.py`). Always append the marker *after* the payload is
  `fsync`'d, never before, or a crash between the two loses a record.
- **Durations are seconds** everywhere, in variables, in `*_seconds_per_sample` keys, in
  `progress-*.json`. An earlier revision timed in milliseconds under seconds-named keys;
  the Gemini stats files under `results/` still carry that error and the README documents
  it. Do not "correct" the stored files — they are recorded measurements.
- **Configuration is CLI flags with env-var defaults** (`OLLAMA_HOST`, `OLLAMA_MODEL`,
  `FAISS_INDEX_PATH`, `SENTENCE_TRANSFORMER_PATH`, `LOCAL_MODEL`, `OPENAI_API_KEY`, ...).
  Never hard-code a key or a host in a new script.
- **Failures skip, runs continue.** A bad record logs and moves on; only a missing input
  file or an unusable model is fatal. Long runs are expensive to restart.
- **Comments explain *why*.** The code is plain enough to read; the reasoning behind a
  threshold, a model choice or a prompt rule is not. Match that density.
- English only in code, comments and output. Some files were mixed-language and have been
  translated; don't reintroduce.

## Environment

Windows, PowerShell, Python 3.14 in `.venv`. Installed: torch 2.12, transformers 5.12,
sentence-transformers 5.6, datasets 5.0, faiss, sklearn, huggingface-hub. **Not
installed**: `trl`, `wandb`, `flask` — the GRPO and Flask stages cannot run here without
installing them first.

`transformers` 5.x renamed `torch_dtype` to `dtype`; `sentence_transformers` deprecated
`main_distance_function` in favour of `main_similarity_function`. Both are already
updated across the repo — don't regress them from older examples.

Set `PYTHONIOENCODING=utf-8` when piping script output through Bash; the console codepage
is GBK and the data contains non-ASCII characters.

## Verifying a change

There is no test suite, and the real inputs are large (27,770 fill-eval records, 35 MB of
interfaces) and mostly gated behind a remote Ollama server. Practical checks:

```bash
python -m py_compile *.py dependencies/*.py

# Scoring path, on real data, cheap:
python 2.perf-eval-result-eval.py --top_percent 1

# Retrieval path end to end, on a slice, into the scratch directory:
python 3.build-faiss-index.py --model all-MiniLM-L6-v2 --limit 40 \
  --index-out /tmp/i.index --embeddings-out /tmp/e.npy \
  --metadata-out /tmp/m.json --dataset-original-out /tmp/do.jsonl
```

For the fill-in runner, substitute a stub `extract` function and a 5-line slice of
`fill-eval.jsonl` rather than loading a model — that exercises resume, line alignment and
the stats files in under a second. Never point a smoke test at the real `data/` or
`results/` output paths; those files are the published record.

Do not run stage 0, 1 or 3 to completion to "check" an edit — they cost hours of GPU or
tens of thousands of remote model calls.
