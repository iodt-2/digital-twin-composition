# InterTwin: An Autonomous Framework for Interoperable Digital Twins Management and Composition

This repo contains a practical end-to-end pipeline for **generating mass-scale DTDL v2 interfaces**, **creating “fill” evaluation data**, **fine-tuning models for extraction**, **running inference**, **evaluating results**, **running system-level evaluation**, and **deploying an interactive agent UI**.

At a high level:

1. **Generate** many digital-twin topics and corresponding **DTDL v2 Interface** objects (JSONL).
2. **Synthesize** “anchor” paragraphs + structured “answer” instances for evaluation/training.
3. **Train**:
   - a SentenceTransformer for semantic retrieval (triplet training), or
   - a small LLM via GRPO for structured JSON extraction.
4. **Run inference** to fill properties from anchors (local Transformers or Gemini/OpenAI-compatible API).
5. **Evaluate** model outputs and full-system behavior.
6. **Deploy** an interactive Flask-based agent UI.

---

## Repository layout

The scripts are grouped by stage (prefix numbers reflect the pipeline order):

### 0) Data generation
- `0.data-gen-dtdl.py` — Mass-scale topic discovery + DTDL Interface generation via Ollama; writes `topics.jsonl` and `interfaces.jsonl`, and supports resume via a done log.
- `0.data-gen-fill.py` — Converts `interfaces.jsonl` into `fill-eval.jsonl` containing:
  - `anchor`: a natural-language paragraph describing only **Property** fields (excluding docker/telemetry)
  - `answer`: a structured instance JSON with the full field set (includes interface id and telemetry placeholders)  
  Includes checkpointing (`.ckpt`) to resume.
- `0.data-gen-interface-to-triplet.py` — Builds a triplet dataset (`triplet_database.jsonl`) from `interfaces.jsonl` for training retrieval models.

### 1) Fine-tuning
- `1.fine-tune-sentence-transformer.py` — Fine-tunes a SentenceTransformer using triplets and evaluates periodically via custom evaluators.
- `1.fine-tune-GRPO-llm.py` — Fine-tunes `Qwen/Qwen2-0.5B-Instruct` using TRL GRPO with a JSON-validity / key-match style reward function.

### 2) Performance evaluation
- `2.perf-eval-fill-gen-local.py` — Runs **local Transformers** extraction to fill properties from anchors; supports resume with `.done` indices and writes progress stats.
- `2.perf-eval-fill-gen-gemini.py` — Runs extraction through a Gemini/OpenAI-compatible chat-completions endpoint; supports resume and writes progress/time stats.
- `2.perf-eval-result-eval.py` — Compares filled outputs against `fill-eval.jsonl` answers and reports precision/recall/F1/EM and time stats; can auto-discover predictions under `finished/`.
- `2.perf-eval-sentence-transformer.py` — Benchmarks multiple embedding models on the triplet test set.

### 3) System-level evaluation
- `3.system-eval.py` — End-to-end retrieval/decomposition/composition/fill-in evaluation pipeline with FAISS + SentenceTransformer + local Qwen/Ollama steps.

### 4) Deployment / demos
- `4.deploy-agi-flask.py` — A Flask web UI agent loop with SSE streaming, step cards, and in-browser rendering.

---

## Outputs

Common artifacts you’ll see:

- `topics.jsonl` — one topic per line (id/title/brief).
- `interfaces.jsonl` — one **DTDL v2 Interface JSON object per line** (no wrapper).
- `interfaces_done.txt` — topic ids that have been processed for interfaces.
- `fill-eval.jsonl` — one record per interface line: `{ "anchor": "...", "answer": {...} }`, plus a `.ckpt` file for resume.
- `triplet_database.jsonl` — triplets for retrieval training: `{query, positive, negative}`.
- Trained/released models under `models/` (for example: `models/MiniLM-L6-based-new-triplets-final/` and `models/Qwen2-0.5B-GRPO-Fill-In/`).
- Filled inference outputs (per model folder), e.g.:
  - `checkpoint-5500/filled-output-checkpoint-5500.jsonl`
  - along with `.done`, `progress-*.json`, and `sample_time_stats-*.txt`

---

## Prerequisites

### Python
You’ll need Python 3.9+ (recommended) and typical ML tooling depending on which stage you run.

### Ollama (remote or local)
Several scripts call an Ollama-compatible server:
- DTDL generation uses `/api/generate`
- Fill-eval generation uses `/api/chat`
- Deploy scripts also talk to Ollama (`/api/chat`)

If your Ollama server differs, pass `--host/--model` (where supported) or set environment variables where supported.

---

## Pretrained model downloads

If you prefer evaluation/inference without re-training, download these released checkpoints:

### Sentence Transformers
- `st-dt-MiniLM-L6` (v1.1): https://github.com/iodt-2/digital-twin-composition/releases/download/v1.1/st-dt-MiniLM-L6.zip
- `st-dt-deberta` (v1.0): https://github.com/iodt-2/digital-twin-composition/releases/download/v1.0/st-dt-deberta.zip

### LLM for fill-in
- `Qwen2-0.5B-GRPO-Fill-In` (v1.0): https://github.com/iodt-2/digital-twin-composition/releases/download/v1.0/Qwen2-0.5B-GRPO-Fill-In.zip

Suggested extraction target: `models/`.

---

## Quickstart

### Step 1 — Generate topics + interfaces
Generates many unique digital twin topics and then produces multiple DTDL Interfaces for each topic.

```bash
python 0.data-gen-dtdl.py \
  --target-topics 1000 \
  --topics-batch 50 \
  --host http://YOUR_OLLAMA_HOST:PORT \
  --model YOUR_MODEL
````

This writes:

* `topics.jsonl`
* `interfaces.jsonl`
* `interfaces_done.txt`

and is designed to be resumable by re-running with `--resume`.

Optional seeding (to anchor the topic space from a text file):

```bash
python 0.data-gen-dtdl.py --seed-file cls350_seed.txt --target-topics 1000
```

---

### Step 2 — Create fill-eval dataset (anchor + answer)

Converts `interfaces.jsonl` into a dataset suitable for training/eval of extraction.

```bash
python 0.data-gen-fill.py \
  --input interfaces.jsonl \
  --output fill-eval.jsonl \
  --host http://YOUR_OLLAMA_HOST:PORT \
  --model YOUR_MODEL
```

It writes `fill-eval.jsonl` and a checkpoint file `fill-eval.jsonl.ckpt` so you can resume safely.

---

### Step 3a — Train a SentenceTransformer (retrieval)

First build triplets:

```bash
python 0.data-gen-interface-to-triplet.py
```

This reads `interfaces.jsonl` and writes `triplet_database.jsonl`.

Then fine-tune:

```bash
python 1.fine-tune-sentence-transformer.py
```

This trains using a triplet-style objective (MultipleNegativesRankingLoss by default), logs to Weights & Biases, and saves models under `models/...`.

Or use a released checkpoint from the links in **Pretrained model downloads**.

(Optional) benchmark multiple embedding models:

```bash
python 2.perf-eval-sentence-transformer.py
```

---

### Step 3b — Fine-tune a small LLM for JSON extraction (GRPO)

If you have a dataset on disk at `llm-fill-ft.ds`, you can run GRPO fine-tuning:

```bash
python 1.fine-tune-GRPO-llm.py
```

This uses TRL’s `GRPOTrainer` and saves to `Qwen2-0.5B-GRPO/`.

Or use the released fill-in model from **Pretrained model downloads**.

---

### Step 4 — Run local extraction inference (fill properties)

This script reads `fill-eval.jsonl`, looks up property schemas from `interfaces.jsonl`, and uses a local Transformers text-generation pipeline to extract only the specified properties.

```bash
# Optionally point to a local model checkpoint
export LOCAL_MODEL="Qwen2-0.5B-GRPO/checkpoint-5500"

python 2.perf-eval-fill-gen-local.py
```

Outputs are written under a folder named after your model path tail (e.g. `checkpoint-5500/`), with resume support via a `.done` file and progress stats.

To run hosted inference through a Gemini/OpenAI-compatible endpoint:

```bash
python 2.perf-eval-fill-gen-gemini.py
```

Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`) before running.

---

### Step 5 — Evaluate prediction quality

Compare one or more filled outputs against `fill-eval.jsonl`:

```bash
python 2.perf-eval-result-eval.py \
  --eval fill-eval.jsonl \
  --pred checkpoint-5500/filled-output-checkpoint-5500.jsonl \
  --tol 0.0 \
  --top_percent 70
```

If you omit `--pred`, it can scan `finished/` for `*.jsonl` outputs automatically and print a summary table.

---

### Step 6 — Run end-to-end system evaluation

```bash
python 3.system-eval.py --k 1 --normalize
```

`3.system-eval.py` supports additional controls (for example decomposition thresholds, model paths, and output files) via CLI flags.

---

## Deployment demos

### Flask UI agent demo

Starts a local web UI with step-by-step streaming (SSE), allowing you to enter a query, run the agent, and inspect assistant output + execution output per step. Defaults can be overridden with env vars `OLLAMA_HOST` and `OLLAMA_MODEL`.

```bash
export OLLAMA_HOST="http://YOUR_OLLAMA_HOST:PORT"
export OLLAMA_MODEL="YOUR_MODEL"
python 4.deploy-agi-flask.py
# then open http://127.0.0.1:5000
```

---

## License

MIT. See `LICENSE`.
