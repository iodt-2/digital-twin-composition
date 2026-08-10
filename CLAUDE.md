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

`4.deploy-test.py` is the deployment's scored test: it drives `4.deploy.py`'s own
`attempt()` end to end, retries with the restart feedback until retrieval is complete
(`-K`, default 10), and scores every stage against the docker-images ground truth —
decomposition, composition, fill-in, the instance and the compose file, including
docker-images' full stack suite driven on the generated file. `--score-only DIR`
re-scores existing artefacts with no model calls, which is how to iterate on the
script itself. Exit 0 only when everything gated passes.

Two instance shapes come out of the pipeline, and anything consuming an instance has to
handle both: the direct route's flat `{"interface", "<property>": ...}` (also the shape
of `answer` in `fill-eval.jsonl`), and the composed route's
`{"interface", "subsystems": {"<name>": {...}}}`. `4.deploy.py` is the worked example —
it runs every stage and ends at a `docker-compose.yaml` the user is asked to confirm.
Its stages are the eval script's plus two the eval script does not have: the
decomposition also analyses how the parts would exchange data, and the composition
turns that into the connections between the retrieved twins (invariant 5). Declining
the compose file restarts all of them.

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

5. **How subsystems connect is analysed, then checked — never guessed from names.**
   A query describes independent components and says nothing about how they couple, so
   `4.deploy.py` works the coupling out: the decomposition stage analyses which part
   would hand data to which, and the composition stage asks the model to resolve that
   onto the twins retrieval returned, given only what each one states about itself —
   `outputType` (what it serves on its data endpoint) and `updateFields` (the telemetry
   keys its update endpoint accepts, which is *not* always the Telemetry names: a twin
   may model one observation and take a batch of them under one key).

   An analysis is a claim about other interfaces, so `validate_wiring` checks every
   part of it against the catalogue before it reaches a compose file: both ends must be
   members, `as` must be a field the target actually accepts, one field takes one
   source, and an edge that closes a cycle is demoted to `feedback` rather than handed
   to an orchestrator that would abort on it. `inputs` is then *derived* as the
   complement — every accepted field no connection feeds — so the two halves stay
   exhaustive however the model answered, and the model's own `inputs` is only
   cross-checked and reported. Never trust an edge you have not put through that
   function, and never fall back to matching field names: a field called
   `buildingState` in two unrelated topics would wire two twins that have never heard
   of each other.

   The rules in `build_wiring_prompt` are there because each one bought a measured
   connection, and none of them names an interface, type or field of any catalogue.
   *Check every accepted field against every member* and *match a field by what it
   carries, not how it is spelled* recovered `currentBuildingState` — image 5 calls a
   `BuildingState` by a name images 3 and 4 spell differently — and the generic
   `previous`/`prior`/`last`/`accepted` feedback rule, with its `alpha`/`Widget` example,
   recovered `previousPlan`, a twin's own previous output. 6/8 became 8/8. If you rewrite
   that prompt, re-measure with `--from-interfaces ... --wiring analysed --no-fill`,
   which scores one composition against the declared reference in a single model call.
   Never fix a miss by naming the missing edge, the chain, or its field names in the
   prompt: that scores the prompt, not the analysis.

   `relationships` (per field a twin cannot produce, a `target` interface and the
   `consumes` output type that feeds it) is still read by `derive_wiring`, but it is no
   longer how a deployment is wired. It is the offline route (`--wiring declared`, which
   `--no-fill` selects, and which needs no model), the way compositions published before
   the analysis are reproduced, and the ground truth `agreement` scores an analysis
   against. **The wiring analysis is never shown it** — `self_description` withholds it
   on purpose, because reciting a declaration is not composing. An interface declaring
   none of these keys still composes and deploys; it simply has no internal wiring.

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

- **The decomposition prompt.** `3.system-eval.py.build_decompose_prompt` and
  `4.deploy.py.build_decompose_prompt` are character-for-character identical, for the
  same reason in the other direction: a deployment that decomposes differently from the
  measured system is not the system the retrieval numbers describe. Change one, change
  both. `4.deploy.py`'s fill-in prompt is the eval script's composed one with
  `interface` and `dockerImage` removed from the required keys, per invariant 3 — both
  are copied from the catalogue instead.

  This is why the coupling analysis is a *second* call rather than extra keys in that
  prompt's output: the split prompt is told to keep every sub-query self-contained and
  to use no facts from another, which is exactly what makes each one a usable retrieval
  query, and widening it would change the sub-queries and with them every retrieval
  number published. `build_couplings_prompt` runs over the parts it produced. For the
  same reason a restart's feedback is *appended* to the prompt string by `guidance`
  rather than edited into `build_decompose_prompt`, so attempt 1 sends the prompt this
  script has always sent, byte for byte. Verify with the check in the stubbed test
  below, not by eye.

  Know what that pinning costs, because it is measured and it will look like a bug.
  `You MAY summarize wording` in that prompt is the single biggest influence on
  retrieval: on a request that states its own couplings the model carries each
  subsystem's paragraph over almost verbatim and retrieval is 5/5, but on a
  coupling-free request — which is what a request is now — it often summarises instead,
  and a summary of a paragraph that scores 0.7149 against the right interface can score
  0.4473 while scoring 0.5675 against a wrong one. Four attempt-1 runs of
  `sample-query.md` gave 3/5, 3/5, 3/5, 5/5. **Do not fix this in the prompt** — that
  invalidates every published retrieval number. The intended remedies are the restart
  (feedback asking for the description verbatim restores 5/5 reliably) and, if a
  one-attempt demo matters more than index continuity, re-encoding the five catalogue
  entries whose `description` fields were written for the older coupled query. Diagnose
  it by scoring texts against the index directly, never by re-running the pipeline: the
  paragraphs, the sub-queries and the catalogue descriptions can each be encoded and
  searched offline in seconds, which is how the above was pinned down.

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

There is no unit-test suite, and the real inputs are large (27,770 fill-eval records,
35 MB of interfaces) and mostly gated behind a remote Ollama server. With the Ollama
host and the FAISS index available, the one command that scores the whole deployment
is:

```bash
python 4.deploy-test.py          # retries retrieval up to -K times, exits 0 on PASS
python 4.deploy-test.py --score-only outputs/deploy-test/attempt-02   # offline re-score
```

Without them, practical checks:

```bash
python -m py_compile *.py dependencies/*.py

# Composition, wiring and compose generation, offline: no index, no model, no network.
# `--no-fill` selects the declared route and skips the confirmation, so this is also
# the regression check for it - the composition it writes must stay identical to the
# committed one, which is what docker-images' CatalogueWiringTests re-derives. The five
# docker-images interfaces are the only ones in the catalogue that declare
# relationships, so they are the fixture for anything touching `derive_wiring`.
python 4.deploy.py --from-interfaces ../docker-images/image*/dtdl/interface.jsonl \
                   --no-fill -o /tmp/wiring
python -c "import json,sys; a,b=[json.load(open(p)) for p in sys.argv[1:]]; \
  print('composition unchanged:', a==b)" \
  /tmp/wiring/composition.json ../docker-images/orchestrator/composition.example.json

# Scoring path, on real data, cheap:
python 2.perf-eval-result-eval.py --top_percent 1

# Retrieval path end to end, on a slice, into the scratch directory:
python 3.build-faiss-index.py --model all-MiniLM-L6-v2 --limit 40 \
  --index-out /tmp/i.index --embeddings-out /tmp/e.npy \
  --metadata-out /tmp/m.json --dataset-original-out /tmp/do.jsonl
```

`validate_wiring` is the one piece of new logic worth exercising directly, because it
is what stands between a model's answer and a stack that will not start. It is a pure
function over the catalogue, so it needs no model and no stub:

```bash
python - <<'PY'
import glob, importlib.util
spec = importlib.util.spec_from_file_location("deploy", "4.deploy.py")
deploy = importlib.util.module_from_spec(spec); spec.loader.exec_module(deploy)

members = {}
for path in sorted(glob.glob("../docker-images/image*/dtdl/interface.jsonl")):
    for interface in deploy.load_interfaces(path):
        members[deploy.subsystem_key(interface, members)] = interface

notes = []
wiring = deploy.validate_wiring(members, {"connections": [
    {"from": "buildingData",    "to": "energyForecast",  "as": "buildingState"},
    {"from": "contextData",     "to": "energyForecast",  "as": "buildingState"},    # taken
    {"from": "buildingData",    "to": "energyForecast",  "as": "weatherNow"},       # no field
    {"from": "weatherApi",      "to": "energyForecast",  "as": "contextForecast"},  # no member
    {"from": "energyForecast",  "to": "energyOptimizer", "as": "energyForecast"},
    {"from": "energyOptimizer", "to": "energyForecast",  "as": "historicalSeries"}, # cycle
], "output": "nowhere"}, notes.append)
print("\n".join(notes))
assert len(wiring["connections"]) == 3, wiring["connections"]
assert wiring["connections"][-1].get("feedback"), "a cycle is demoted, never dropped"
assert wiring["output"] in members, wiring["output"]
assert wiring["inputs"]["observations"] == ["buildingData"]
print("validate_wiring ok")
PY
```

To exercise the model-facing stages without a model, replace `deploy.ollama` with a
function returning canned JSON and call `deploy.attempt` with `--from-interfaces` (so no
FAISS is needed) or with a fake catalogue object exposing `top1`. Assert in particular
that `prompts[0] == deploy.build_decompose_prompt(query, 5)` on attempt 1, which is the
character-identity above, and that the wiring prompt contains neither `relationships`
nor `consumes`.

For the fill-in runner, substitute a stub `extract` function and a 5-line slice of
`fill-eval.jsonl` rather than loading a model — that exercises resume, line alignment and
the stats files in under a second. Never point a smoke test at the real `data/` or
`results/` output paths; those files are the published record.

Do not run stage 0, 1 or 3 to completion to "check" an edit — they cost hours of GPU or
tens of thousands of remote model calls.
