# InterTwin: An Autonomous Framework for Interoperable Digital Twins Management and Composition

An end-to-end pipeline for **generating mass-scale DTDL v2 interfaces**, **synthesising
fill-in evaluation data**, **fine-tuning retrieval and extraction models**, **running
inference**, **scoring results**, **evaluating the whole system**, and **serving an
interactive agent UI**.

At a high level:

1. **Generate** digital-twin topics and DTDL v2 Interface objects (JSONL).
2. **Synthesize** "anchor" paragraphs plus the structured "answer" instance they imply.
3. **Train** a SentenceTransformer for retrieval, and a small LLM via GRPO for extraction.
4. **Index** the interface catalogue into FAISS.
5. **Run inference** to fill properties from anchors (local Transformers, or a hosted
   OpenAI-compatible endpoint).
6. **Evaluate** model outputs and full-system behaviour.
7. **Deploy** a Flask agent UI.

---

## Repository layout

Scripts are prefixed by pipeline stage. Everything is run from the repository root.

### 0) Data generation
- `0.data-gen-dtdl.py` — topic discovery + DTDL Interface generation via Ollama; writes
  `data/topics.jsonl`, `data/interfaces.jsonl`, and resumes from `data/interfaces_done.txt`.
- `0.data-gen-fill.py` — turns `interfaces.jsonl` into `fill-eval.jsonl`:
  - `anchor` — a paragraph describing only **Property** fields (never the interface id,
    the dockerImage path, or telemetry)
  - `answer` — the full instance JSON (interface id, all properties, telemetry zeroed)

  Checkpointed per input line in `<output>.ckpt`, so it resumes safely.
- `0.data-gen-interface-to-triplet.py` — builds `data/triplet.jsonl`
  (`query` / `positive` / `negative`) for retrieval training. Negatives are sampled from
  a different topic, filtered lexically, then verified by an LLM judge.
- `0.data-push-to-huggingface.py` — publishes every JSONL as a config of one HF dataset
  repo and generates the dataset card.

### 1) Fine-tuning
- `1.fine-tune-sentence-transformer.py` — trains the retrieval model on triplets
  (MultipleNegativesRankingLoss by default) with periodic evaluation.
- `1.fine-tune-GRPO-llm.py` — GRPO fine-tune of `Qwen/Qwen2-0.5B-Instruct` for JSON
  extraction. See **Reward functions** below before you run it.
- `1.model-push-to-huggingface.py` — uploads the GRPO checkpoint and generates a model
  card measured from the checkpoint itself and the logged reward curve.

### 2) Performance evaluation
- `2.perf-eval-fill-gen-local.py` — fill-in with a **local Transformers** model.
- `2.perf-eval-fill-gen-gemini.py` — fill-in through a Gemini/OpenAI-compatible endpoint.
- `2.perf-eval-result-eval.py` — scores prediction files against `fill-eval.jsonl`
  (precision / recall / F1 / exact match) and prints a comparison table.
- `2.perf-eval-sentence-transformer.py` — benchmarks embedding models on the held-out
  triplet split.

### 3) System-level evaluation
- `3.build-faiss-index.py` — encodes the interface catalogue and writes the index,
  embeddings, metadata and `dataset_original.jsonl`. **Required before stage 3.**
- `3.system-eval.py` — retrieval → decomposition → composition → fill-in → verification.

### 4) Deployment
- `4.deploy-agi-flask.py` — Flask agent loop with SSE streaming and per-step cards.

### Shared code
- `dependencies/fill_eval_runner.py` — the loop both fill-in backends share (schema
  lookup, prompt, coercion, resume log, timing). Keeps the two backends comparable.
- `dependencies/SuperTripletEvaluator.py` — triplet accuracy plus raw positive/negative
  cosines.
- `dependencies/ThresholdedTripletEvaluator.py` — triplet accuracy under a similarity
  threshold, matching how `3.system-eval.py --min_sim` actually uses the model.

---

## Data and outputs

Shipped under `data/`:

| file | contents |
| --- | --- |
| `topics.jsonl` | one topic per line: id / title / brief |
| `interfaces.jsonl` | one DTDL v2 Interface JSON object per line, no wrapper |
| `fill-eval.jsonl` | `{"anchor": "...", "answer": {...}}`, 27,770 records |
| `triplet.jsonl` | `{query, positive, negative}` for retrieval training |
| `dataset_small.jsonl` | 100 composed system queries: `{query, group_id}` |
| `dataset_mid.jsonl` | 100 composed queries with an interface / group reference |
| `Qwen2-0.5B-GRPO-Fill-In/` | the released fill-in checkpoint |

Produced by the pipeline:

- `models/faiss.index`, `models/embeddings.npy`, `models/metadata.json`,
  `data/dataset_original.jsonl` — from `3.build-faiss-index.py`, all four index-aligned.
- `results/<label>/filled-output-<label>.jsonl` plus `.done`, `progress-*.json` and
  `sample_time_stats-*.txt` — from the stage-2 fill-in scripts.
- `outputs/evaluation_results.jsonl`, `outputs/debug_results.jsonl` and their
  `.summary.json` companions — from `3.system-eval.py`.

`results/sentence-transformers/` holds the runs published with the project: the eight
`filled-output-*.jsonl` fill-in comparisons, and the CSV curves exported from W&B. (The
fill-in runs sit under that directory for historical reasons — they are LLM results, not
sentence-transformer results. `2.perf-eval-result-eval.py` finds them at any depth.)

> **Known issue in the shipped timing files.** The Gemini `sample_time_stats-*.txt` files
> report **milliseconds** under `*_seconds_per_sample` keys — the script that wrote them
> timed in ms but labelled in seconds. The values are right, the unit in the key is not;
> divide by 1000. The current scripts time and label in seconds throughout, and the
> `*_per_sample=...ms` files are correctly labelled.

---

## Prerequisites

Python 3.9+ (developed against 3.14). Install per stage:

```bash
pip install -r requirements.txt
```

Several scripts call an **Ollama-compatible server** (`/api/generate` and `/api/chat`).
Point them at yours with `--host` / `--model`, or the `OLLAMA_HOST` / `OLLAMA_MODEL`
environment variables, which every stage now honours.

---

## Pretrained model downloads

To evaluate without re-training:

**Sentence Transformers**
- `st-dt-MiniLM-L6` (v1.1): https://github.com/iodt-2/digital-twin-composition/releases/download/v1.1/st-dt-MiniLM-L6.zip
- `st-dt-deberta` (v1.0): https://github.com/iodt-2/digital-twin-composition/releases/download/v1.0/st-dt-deberta.zip

**LLM for fill-in**
- `Qwen2-0.5B-GRPO-Fill-In` (v1.0): https://github.com/iodt-2/digital-twin-composition/releases/download/v1.0/Qwen2-0.5B-GRPO-Fill-In.zip

Extract into `models/`. `3.system-eval.py` expects the retrieval model at
`models/MiniLM-L6-based-new-triplets-final` unless you set `SENTENCE_TRANSFORMER_PATH`.

To publish your own copy to the Hub with a generated model card:

```bash
python 1.model-push-to-huggingface.py \
  --model-dir data/Qwen2-0.5B-GRPO-Fill-In \
  --repo-id YOUR_USER/Qwen2-0.5B-GRPO-Fill-In \
  --smoke-test --dry-run
```

Drop `--dry-run` to push; the repo is private unless you pass `--public`.

---

## Quickstart

### Step 1 — Generate topics and interfaces

```bash
python 0.data-gen-dtdl.py \
  --target-topics 1000 \
  --topics-batch 50 \
  --host http://YOUR_OLLAMA_HOST:PORT \
  --model YOUR_MODEL
```

Writes `data/topics.jsonl`, `data/interfaces.jsonl` and `data/interfaces_done.txt`.
Resuming is the default; pass `--no-resume` to start over. Optionally anchor the topic
space with `--seed-file cls350_seed.txt`.

### Step 2 — Create the fill-eval dataset

```bash
python 0.data-gen-fill.py --input data/interfaces.jsonl --output data/fill-eval.jsonl
```

Also writes `data/fill-eval.jsonl.ckpt`; re-run to resume.

### Step 3a — Train the retrieval model

Build triplets, then fine-tune:

```bash
python 0.data-gen-interface-to-triplet.py --workers 4
python 1.fine-tune-sentence-transformer.py --run-name MiniLM-L6-based-v2
```

Saves to `models/<run-name>-final`. `--no-wandb` disables logging. Benchmark several
embedding models on the held-out split with:

```bash
python 2.perf-eval-sentence-transformer.py
```

### Step 3b — GRPO fine-tune the extraction model

```bash
python 1.fine-tune-GRPO-llm.py --dataset llm-fill-ft.ds --reward corrected
```

#### Reward functions

`--reward shipped` (the default) reproduces the run that produced the released
checkpoint, quirks intact: it only scores fenced ` ```json ` output, because the
bare-JSON branch indexes a list of chat messages as a dict, raises, and lands at `-1`. A
perfect answer is therefore worth 0.8 — which is exactly where the logged curve in
`results/llm-fill-in/reward.csv` plateaus, and what the published model card explains.

`--reward corrected` accepts both fenced and bare JSON, compares floats with a tolerance,
and tops out at 1.0. **Use it for new runs**; it will not reproduce the published curve.

### Step 4 — Run fill-in inference

```bash
# local checkpoint
python 2.perf-eval-fill-gen-local.py --model models/Qwen2-0.5B-GRPO-Fill-In

# hosted, OpenAI-compatible
export OPENAI_API_KEY=...
python 2.perf-eval-fill-gen-gemini.py --model models/gemini-2.5-pro
```

Both write `results/<label>/filled-output-<label>.jsonl`, one row per input line —
including `{}` for records they could not process, which keeps the file line-aligned with
`fill-eval.jsonl`. Both resume from the `.done` file.

### Step 5 — Score the predictions

```bash
python 2.perf-eval-result-eval.py                       # everything under results/
python 2.perf-eval-result-eval.py \
  --pred results/gemini-2.5-pro/filled-output-gemini-2.5-pro.jsonl \
  --tol 0.0 --top_percent 70
```

Rows whose prediction is `{}` are excluded rather than scored as total failures, so a
backend is measured on what it attempted; the count is reported separately.

> **Scoring correction — absolute numbers moved.** Keys whose gold value is `0` are
> skipped, because that is how the generator initialises telemetry. That test used to
> catch boolean `false` as well (in Python, `False == 0`), so a genuinely-`false`
> property was dropped from the required set *and* a model that predicted it correctly
> was charged a false positive for an "extra" key. Booleans are now excluded from the
> zero test. This affects 0.44% of fields but ~2.6 points of per-row exact match: on
> `gpt-oss-120b` at `--top_percent 20`, precision 0.9524 → 0.9604, F1 0.9578 → 0.9620,
> row EM 0.8423 → 0.8680. All backends gain similarly, so rankings are unchanged, but
> figures quoted from earlier runs are not comparable with figures from this version.

### Step 6 — Build the index, then run system evaluation

```bash
python 3.build-faiss-index.py --model models/MiniLM-L6-based-new-triplets-final
python 3.system-eval.py --k 1 --normalize
```

The index **must** be built with the same embedding model `3.system-eval.py` loads, and
`data/dataset_original.jsonl` must come from the same run — retrieval is scored by FAISS
id, and that file is what maps ids back to topics.

> **`--retry_selection` changes what the composed-route numbers mean.** On the default
> `oracle`, each decomposition attempt is scored against the ground-truth topic and the
> best one is kept (retrying until F1 clears `--retry_f1_target`). That makes the labels
> part of the system's own search, so composed-route figures are an **upper bound**. Pass
> `--retry_selection first` for a blind run that never consults the labels. The script
> prints a warning whenever `oracle` is in effect.

---

## Deployment demo

```bash
export OLLAMA_HOST="http://YOUR_OLLAMA_HOST:PORT"
export OLLAMA_MODEL="YOUR_MODEL"
python 4.deploy-agi-flask.py
# then open http://127.0.0.1:5000
```

> **The demo executes model-written Python in-process, unsandboxed.** That is how the
> agent searches FAISS and composes interfaces, but it means anyone who can reach the
> port — and any prompt that can steer the model — can run code as you. It binds
> `127.0.0.1` and will not bind anything else unless you set `ALLOW_REMOTE=1`. Set
> `ALLOW_CODE_EXEC=0` to watch the reasoning with execution switched off. `MAX_STEPS`
> (default 20) bounds a run that never says "Finished".

---

## License

MIT — as declared here and in the published model card. (There is no `LICENSE` file in
the repository yet; add one with your copyright line to make it enforceable.)
