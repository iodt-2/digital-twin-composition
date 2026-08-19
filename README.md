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
- `0.data-gen-fill-ft-dataset.py` — turns `fill-eval.jsonl` + `interfaces.jsonl` into the
  `save_to_disk` dataset the stage-1 and stage-2 scripts read (`row`, `prompt`,
  `ground_truth`, `interface`, `n_fields`). `--prompt` picks the prompt format and
  `--like` mirrors an existing build's split row for row. See
  [Fill-in prompt formats](#fill-in-prompt-formats).
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
- `2.perf-eval-fill-gen-local.py` — fill-in over the fine-tuning dataset
  (`data/llm-fill-ft-80-20-iface.ds` by default), with a **local Transformers** model or
  `--ollama`.
- `2.perf-eval-fill-gen-gemini.py` — fill-in over the same kind of dataset
  (`data/llm-fill-ft-80-20.ds` by default), through **Gemini** on Vertex AI or
  `--gemini-api`.
  Both send the row's `prompt` as it is stored and recover the fields and DTDL types the
  answer is coerced to from that prompt's `- "name" (schema)` lines, so pointing them at
  the same `--dataset` and `--split` makes the two runs differ in nothing but the model.
- `2.perf-eval-result-eval.py` — scores prediction files against `fill-eval.jsonl`, or
  with `--dataset` against a split of `llm-fill-ft-80-20.ds` (precision / recall / F1 /
  exact match), and prints a comparison table.
- `2.perf-eval-sentence-transformer.py` — benchmarks embedding models on the held-out
  triplet split.

### 3) System-level evaluation
- `3.build-faiss-index.py` — encodes the interface catalogue and writes the index,
  embeddings, metadata and `dataset_original.jsonl`. **Required before stage 3.**
- `3.system-eval.py` — retrieval → decomposition → composition → fill-in → verification.

### 4) Deployment
- `4.deploy.py` — the whole pipeline as one command: query → decomposition → composition
  → fill-in → DTDL instance → `docker-compose.yaml` → confirm. The request describes
  independent components, so the decomposition also analyses how the parts would exchange
  data and the composition turns that into the wiring between the subsystem containers;
  declining the compose file restarts the lot. Dependency-free once the index and the
  models are in place.
- `4.deploy-test.py` — the deployment, tested and scored: runs `4.deploy.py`'s stages for
  real, retries until retrieval is complete (`-K`, default 10), and grades decomposition,
  composition, fill-in, the instance and the compose file against the docker-images
  ground truth. See [Testing the deployment](#testing-the-deployment).
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

- `data/llm-fill-ft-80-20.ds` and `data/llm-fill-ft-80-20-iface.ds` — the fill-in
  fine-tuning datasets, from `0.data-gen-fill-ft-dataset.py`. Same 22,216 / 5,554 split
  over the same rows in the same order, and identical `ground_truth`; they differ only in
  `prompt`. See [Fill-in prompt formats](#fill-in-prompt-formats).
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

### Fill-in prompt formats

`0.data-gen-fill-ft-dataset.py` writes two prompt formats over the same rows, so a
checkpoint trained on one can be compared against the other without any other variable
moving:

```bash
# `fields` — what data/llm-fill-ft-80-20.ds ships with
python 0.data-gen-fill-ft-dataset.py --prompt fields \
  --output data/llm-fill-ft-80-20.ds

# `interface` — the DTDL definition in place of the field list
python 0.data-gen-fill-ft-dataset.py --prompt interface \
  --like data/llm-fill-ft-80-20.ds --output data/llm-fill-ft-80-20-iface.ds
```

`fields` names the interface inline and lists the fields to fill as
`- "name" (schema)`. It is `dependencies/fill_eval_runner.build_prompt`
character-for-character, which is what the two jsonl-driven backends send and what the
published model card documents, so the two stay in step.

`interface` drops that list and hands over the interface itself, minified — the same way
`3.system-eval.py` and `4.deploy.py` give an interface to a fill-in prompt. The model has
to read the DTDL and work out which fields are fillable, rather than being handed the
answer's key set. `--interface-content` controls how much of it goes in: `full` (the
default) is the catalogue entry unchanged, telemetry and the `dockerImage` path included;
`no-docker` drops the image the anchor is forbidden to mention; `properties-only` also
drops telemetry, leaving the prompt's keys exactly the keys `ground_truth` is scored on.
The prompt roughly doubles in length — about 1,260 characters median for `fields` against
2,260 for `interface --interface-content full`.

`--like` is what keeps the builds comparable: each row lands in whichever split held its
`row` index in the named dataset, in the same order, so the same 5,554 rows stay held out
and prediction files under `results/` remain line-aligned with the new `test`.

> One row differs from the shipped `llm-fill-ft-80-20.ds` beyond the prompt text.
> `interfaces.jsonl` carries three `@id`s twice with different bodies, and the shipped
> build resolved the prompt by `@id`, so row 17066 was asked for an `updateInterval` that
> belongs to the *other* `dtmi:solar_farm_aero_drag_model:wind_sensor;1`. The builder
> pairs record N with catalogue line N instead — which is how `0.data-gen-fill.py`
> produced them — and asks for the two fields that row's answer actually has. Everything
> else, `ground_truth` and `n_fields` included, reproduces bit-for-bit.

### Step 4 — Run fill-in inference

```bash
# local checkpoint, over the held-out 20% of data/llm-fill-ft-80-20.ds
python 2.perf-eval-fill-gen-local.py --model models/Qwen2-0.5B-GRPO-Fill-In

# the same rows, generated on an Ollama server instead
OLLAMA_HOST=http://10.10.10.4:11434 \
python 2.perf-eval-fill-gen-local.py --ollama --model gpt-oss:120b

# the same rows again, hosted: Gemini on Vertex AI (gcloud ADC), or --gemini-api
python 2.perf-eval-fill-gen-gemini.py --model gemini-2.5-pro \
  --dataset data/llm-fill-ft-80-20.ds
GEMINI_API_KEY=... python 2.perf-eval-fill-gen-gemini.py --gemini-api \
  --model gemini-2.5-flash --thinking-budget 0
```

All of them read one split of a `save_to_disk` dataset — `--dataset`, `--split test` —
and write `results/<label>/filled-output-<label>.jsonl`, one row per input record,
including `{}` for records they could not process, which keeps the file line-aligned with
the input. All of them resume from the `.done` file. A refused connection, an HTTP
timeout, an expired credential or a rate limit that outlives `--retries` is *not*
recorded as a failed record: it aborts the run with nothing written for that row, because
an endpoint that is down would otherwise burn every remaining row as `{}`.

The Gemini backend needs `google-genai`. On Vertex AI it takes `--project` /
`--location` (`$GOOGLE_CLOUD_PROJECT`, `$GOOGLE_CLOUD_LOCATION`) and application-default
credentials from `gcloud auth application-default login`. Its `--max-output-tokens`
defaults to 2048 rather than the local backend's 512 because a 2.5 model draws its
thinking tokens from that same budget; `--thinking-budget 0` turns thinking off on the
models that allow it.

### Step 5 — Score the predictions

```bash
python 2.perf-eval-result-eval.py                       # everything under results/
python 2.perf-eval-result-eval.py \
  --pred results/gemini-2.5-pro/filled-output-gemini-2.5-pro.jsonl \
  --tol 0.0 --top_percent 70

# a run that is still going: score the rows it has reached
python 2.perf-eval-result-eval.py --partial \
  --pred results/gpt-oss-120b/filled-output-gpt-oss-120b.jsonl
```

A run is line-aligned with whatever it read, and two inputs are in play: the dataset split
both fill-in scripts read now (5,554 rows in `test`), and `fill-eval.jsonl` (27,770
lines), which the shipped runs under `results/sentence-transformers/` were produced
against by the earlier jsonl-driven revision of the Gemini script. The scorer keys on the
row count: it loads both inputs, matches each prediction file to the one whose length it
has, and prints them in one table with a `gold` column naming the input. So the bare
command scores everything under `results/`, old runs and new alike. `--split`,
`--test-size` and `--seed` apply only to the `fill-eval.jsonl` half; `--dataset` and
`--dataset-split` point at the other.

`--partial` additionally scores a file that stops short of the end, over the prefix it
reached: rows are appended in input order, so an unfinished run still yields a number for
what it did. Its length names no input, so the scorer says which gold it assumed — check
that line before quoting the number.

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

## Deployment

### From a query to running containers

`4.deploy.py` runs the whole pipeline and ends at a stack you can bring up:

```bash
python 4.deploy.py @../docker-images/sample-query.md -o outputs/deploy
cd outputs/deploy && docker compose up
```

It writes up to six files into `--out-dir` — the first is skipped by
`--from-interfaces` and the composition's two by a decomposition that came back with a
single interface — and prints the decomposition, the couplings, the retrieval hits and the
wiring as it goes:

| File | Stage |
| ---- | ----- |
| `decomposition.json` | the sub-queries, what each one retrieved, and the coupling analysis |
| `wiring.json` | the composition analysis: what was proposed, what was rejected and why |
| `composed-interface.json` | the new interface the composition creates |
| `composition.json` | the wiring between the subsystems, for the orchestrator |
| `instance.json` | the DTDL instance, flat or `{"subsystems": {...}}` |
| `docker-compose.yaml` | one service per subsystem, plus the orchestrator |

The query is decomposed, each sub-query retrieves its top-1 interface, and the hits
become one composed interface; a decomposition that yields a single interface is not a
composition, so it emits the flat instance and one service with no orchestrator. Per
service, `dockerImage` becomes the image, the remaining properties become
`UPPER_SNAKE_CASE` environment variables, and the interface id is kept as a
`dtdl.interface` label. Null properties are dropped, since nothing in the source text
stated them, and `interface` and `dockerImage` are copied from the catalogue rather than
asked of the model — the anchors never mention the image path, so a model asked for it
would be guessing.

#### The subsystems are independent, and the coupling is analysed

A request describes components, not a system diagram: each subsystem says what it holds,
what it accepts and what it produces, and nothing about which of the others it exchanges
data with. Working that out is two of the pipeline's stages.

**Decomposition** splits the request and then, in a second call, analyses how those parts
would have to exchange data — which part produces what, which part needs it, which needs
something no part produces. It is a second call because the split prompt is deliberately
told to keep each sub-query self-contained and to use no facts from another, which is
what makes each one a usable retrieval query; that prompt is character-identical to
`3.system-eval.py`'s and stays that way.

**Composition** resolves that intent onto the twins retrieval actually returned. The
model is given each member's `outputType` (what it serves on its data endpoint) and
`updateFields` (the exact telemetry keys its update endpoint accepts) and answers with
the connections between them: image 3 accepts a field called `buildingState`, image 1
serves a `BuildingState`, so `buildingData → energyForecast as buildingState`. It is not
shown the `relationships` an interface may declare — that would be reciting the answer.

An analysis is a claim about other interfaces, so every part of it is checked before it
reaches a compose file: both ends must be members, `as` must be a field the target really
accepts, one field takes one source, and an edge that closes a cycle is kept but demoted
to `feedback` — a mutual dependency can only be met by the previous cycle's value, and an
orchestrator handed a cycle aborts at start-up. Every accepted field no connection feeds
is then derived as an external input, routed from the caller, so the two halves stay
exhaustive however the model answered. `wiring.json` records what was proposed, what was
rejected and why. Where the members do declare relationships, the run also prints how the
analysis compared with them.

**Measured**, `gpt-oss:120b` on `../docker-images/sample-query.md` — a request that states
no couplings at all:

| | |
| --- | --- |
| Connections recovered | **8/8**, 0 missed, 0 invented; `inputs` and `output` identical to the declared reference |
| Twins retrieved | 5/5 |
| Property values | 71/71 against `sample-instance.json` |
| Containers that start | 6/6, bound through each image's own `load_properties()` |
| Cycle | the generated stack passes all 32 of `docker-images/test_stack.py`'s coherence and cycle tests |

Two things that took measuring to find. The analysis first recovered only **6/8** — it
missed `currentBuildingState` (image 5's name for what images 3 and 4 call
`buildingState`, so the field a `BuildingState` fills is not spelled like the type) and
`previousPlan` (a twin's own previous output, which reads as coming from whichever twin
*accepts* plans). Both are addressed by rules in `build_wiring_prompt` that name no
interface, type or field of this chain: check every accepted field against every member,
match a field by what it carries rather than how it is spelled, and treat a `previous` /
`prior` / `last` / `accepted` field as a feedback edge that may come from the target
itself. And it never invented an edge in any run — with a mis-retrieved member set it
wired *one* connection and left the rest as external inputs rather than sourcing a
building state from a weather station.

**The subsystems never call each other.** The generated `docker-compose.yaml` carries a
`ziren/composition-orchestrator` service bound to the analysed wiring, which reads each
subsystem's data endpoint and posts to the update endpoints of the subsystems that need
it. A composition where nothing a member serves fits anything another accepts still
deploys; it just has no internal wiring, and every field is an input.

#### The last stage is yours

The compose file is printed and confirmed before it is anything to run:

```
[y] accept and deploy this stack   [r] restart the process   [q] quit
```

`r` restarts the whole process — a new decomposition, a new coupling analysis, a new
retrieval, a new composition — and asks for one line of feedback first, which is appended
to every prompt of the next attempt. Decoding is greedy, so the seed moves on each
attempt too; a restart that changed nothing would otherwise return the stack that was
just declined. `--max-attempts` bounds it (5 by default) and `-y` skips the question,
as does a non-interactive stdin.

This stage is load-bearing, not decorative, and the reason is measured. **Retrieval on a
coupling-free request is not reliable on the first attempt**: four attempt-1 runs of the
sample query retrieved 3/5, 3/5, 3/5 and 5/5. When the split summarises a subsystem's
paragraph instead of carrying it over, the summary drifts toward generic building-IoT
vocabulary and lands on the wrong interface — measured at 0.4473 to the right interface
against 0.5675 to the wrong one, where the paragraph itself scores 0.7149. One line of
feedback — *"Do not summarise. Each sub-query must carry that subsystem's whole
description from the request, word for word"* — restores verbatim copying and 5/5, with
the sub-queries coming back at exactly the paragraph lengths and the paragraphs' scores.
Wording inside the request cannot substitute: text under `DESCRIPTION:` is read as
content, while feedback arrives after it as an instruction. That asymmetry is why
declining and restarting is part of the pipeline rather than advice in a README.

#### Without the models

`--from-interfaces` composes named interface files and skips decomposition and retrieval;
`--no-fill` emits the instance with null properties instead of calling the fill-in model,
and selects `--wiring declared` — the connections the catalogue's `relationships` imply,
which needs no model. Together they are an offline catalogue-to-compose generator, and
the way `orchestrator/composition.example.json` in the other repository is regenerated:

```bash
python 4.deploy.py --from-interfaces ../docker-images/image*/dtdl/interface.jsonl \
                   --no-fill -o outputs/wiring
```

`--wiring` takes `analysed`, `declared` or `auto` (the default: `declared` under
`--no-fill`, `analysed` otherwise), so `--from-interfaces --wiring analysed` composes a
named set of interfaces by analysis — which, against a catalogue that declares its
couplings, is how the analysis is scored.

### Testing the deployment

`4.deploy-test.py` runs the whole pipeline through `4.deploy.py`'s own `attempt()` and
scores every stage against the docker-images ground truth — no copy of the pipeline, no
mocks, four real model calls per attempt:

```bash
python 4.deploy-test.py                    # sample query, up to 10 attempts
python 4.deploy-test.py -K 3 --no-cycle    # fewer retries, skip the stack suite
python 4.deploy-test.py --score-only outputs/deploy-test/attempt-02   # offline
```

When retrieval does not return every expected twin, it retries — up to `-K` times
(default 10) — sending the restart feedback and moving the decoding seed exactly as a
user declining the stack would, and each attempt keeps its artefacts in its own
`attempt-NN/` directory. The accepted attempt is then scored:

| Stage | Scored against | Gate |
| ----- | -------------- | ---- |
| 1. Decomposition | one part per subsystem, retrieval complete; coupling flows vs the declared chain | retrieval (couplings informational) |
| 2. Composition | connections recovered / missed / invented vs `derive_wiring`, plus `inputs` and `output` | all recovered, none invented |
| 3. Fill-in | every property value vs `sample-instance.json` | 100% |
| 4. The instance | shape: catalogue property sets, no telemetry, wiring embedded unchanged | all checks |
| 5. Compose file | services, images, ports, environment, `COMPOSITION`; start-up through each image's own `load_properties()`; docker-images' full stack suite driven on the generated file | all checks |

It prints per-section PASS/FAIL, writes `report.json` with the metrics (recall,
precision, per-interface fill-in accuracy, per-attempt timings), and exits 0 only when
everything gated passes — so it can sit in a shell `&&`. The coupling analysis is
reported but not gated: it is intent for the composition stage to correct, and the
corrected wiring is what gates.

Note that for a synthesised twin the generated image is the placeholder
`registry.local/dtm/...` path the generator assigned — that file is a deployment
*manifest*, not a reference to images that exist. The
[docker-images](https://github.com/iodt-2/docker-images) catalogue entries are the
exception: they name images published on Docker Hub, so a stack composed from them
comes up.

### Agent UI demo

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
