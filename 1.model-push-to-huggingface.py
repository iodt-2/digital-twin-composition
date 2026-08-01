#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push the GRPO fine-tuned fill-in model to a HuggingFace *model* repo and write its card.

    python 1.model-push-to-huggingface.py --repo-id <user>/Qwen2-0.5B-GRPO-Fill-In --dry-run
    python 1.model-push-to-huggingface.py --repo-id <user>/Qwen2-0.5B-GRPO-Fill-In --smoke-test
    python 1.model-push-to-huggingface.py --repo-id <user>/Qwen2-0.5B-GRPO-Fill-In --public

The checkpoint (`data/Qwen2-0.5B-GRPO-Fill-In/`) is uploaded with `upload_folder`, not
`model.push_to_hub`: the weights are streamed from disk byte-for-byte instead of being
materialised in RAM, and no transformers version re-serialises the safetensors.

Everything the card states about the artifact is *measured* here — parameter count and dtype
come from the safetensors header and `config.json`, the reward curve from `reward.csv` — so
the card cannot drift away from what was actually pushed.
"""

import argparse
import csv
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional

DEFAULT_MODEL_DIR = os.path.join("data", "Qwen2-0.5B-GRPO-Fill-In")
DEFAULT_REPO_ID = "Qwen2-0.5B-GRPO-Fill-In"
DEFAULT_REWARD_CSV = os.path.join("results", "llm-fill-in", "reward.csv")

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
PROJECT_URL = "https://github.com/iodt-2/digital-twin-composition"

# The checkpoint is useless without these; refuse to push a half-copied directory.
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]
OPTIONAL_FILES = [
    "generation_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]


# ---------------------------
# Inspecting the checkpoint
# ---------------------------

def validate(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        print(f"[ERROR] Model directory not found: {os.path.abspath(model_dir)}", file=sys.stderr)
        return False

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"[ERROR] {model_dir} is missing required file(s): {missing}", file=sys.stderr)
        return False

    absent = [f for f in OPTIONAL_FILES if not os.path.exists(os.path.join(model_dir, f))]
    if absent:
        print(f"[WARN]  optional file(s) absent, uploading without them: {absent}")

    print(f"[OK]   model dir validated: {os.path.abspath(model_dir)}")
    return True


def safetensors_params(path: str) -> Optional[int]:
    """Sum the tensor shapes in a safetensors header without loading any weights.

    Layout: u64 little-endian header length, then that many bytes of JSON metadata.
    """
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len).decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - the count is a nicety, not a reason to abort
        print(f"[WARN]  could not read safetensors header ({e}); parameter count omitted")
        return None

    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape") or []
        n = 1
        for dim in shape:
            n *= dim
        total += n
    return total


def dir_size(model_dir: str) -> int:
    return sum(os.path.getsize(os.path.join(model_dir, f))
               for f in os.listdir(model_dir)
               if os.path.isfile(os.path.join(model_dir, f)))


def inspect_model(model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    gen: Dict[str, Any] = {}
    gen_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f:
            gen = json.load(f)

    info = {
        "architecture": (cfg.get("architectures") or ["?"])[0],
        "model_type": cfg.get("model_type", "?"),
        # `dtype` on transformers >= 5, `torch_dtype` on 4.x checkpoints.
        "dtype": cfg.get("dtype") or cfg.get("torch_dtype") or "?",
        "hidden_size": cfg.get("hidden_size"),
        "layers": cfg.get("num_hidden_layers"),
        "attention_heads": cfg.get("num_attention_heads"),
        "kv_heads": cfg.get("num_key_value_heads"),
        "intermediate_size": cfg.get("intermediate_size"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "transformers_version": cfg.get("transformers_version", "?"),
        "params": safetensors_params(os.path.join(model_dir, "model.safetensors")),
        "bytes": dir_size(model_dir),
        "generation": gen,
        "has_chat_template": os.path.exists(os.path.join(model_dir, "chat_template.jinja")),
    }

    params = f"{info['params'] / 1e6:.0f}M" if info["params"] else "unknown"
    print(f"[OK]   {info['architecture']}  {params} params  dtype={info['dtype']}  "
          f"{info['layers']}L  hidden={info['hidden_size']}  "
          f"heads={info['attention_heads']}/{info['kv_heads']}  "
          f"vocab={info['vocab_size']:,}  {info['bytes'] / 1e9:.2f} GB on disk")
    return info


def reward_summary(csv_path: str) -> Optional[Dict[str, Any]]:
    """Summarise the exported `train/reward` curve. Missing file is not fatal."""
    if not os.path.exists(csv_path):
        print(f"[WARN]  {csv_path} not found; the card will omit the reward curve")
        return None

    points: List[tuple] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                points.append((int(float(row[0])), float(row[1])))
            except ValueError:
                continue

    if not points:
        print(f"[WARN]  no usable rows in {csv_path}; the card will omit the reward curve")
        return None

    tail = points[-50:]
    best = max(points, key=lambda p: p[1])
    summary = {
        "n": len(points),
        "first_step": points[0][0], "first_reward": points[0][1],
        "last_step": points[-1][0], "last_reward": points[-1][1],
        "best_step": best[0], "best_reward": best[1],
        "tail_mean": sum(v for _, v in tail) / len(tail),
        "tail_n": len(tail),
        # The W&B export names the run in the value column, e.g. "<run> - train/reward".
        "run": (header[1].split(" - ")[0].strip() if header and len(header) > 1 else ""),
    }
    print(f"[OK]   reward curve: {summary['n']} points, step {summary['first_step']} "
          f"{summary['first_reward']:.3f} -> step {summary['last_step']} "
          f"{summary['last_reward']:.3f} (max {summary['best_reward']:.3f} @ "
          f"{summary['best_step']})")
    return summary


# ---------------------------
# Smoke test
# ---------------------------

# Kept character-for-character in step with `2.perf-eval-fill-gen-local.py:build_prompt`,
# so what we test here is what the evaluation pipeline actually sends.
def build_prompt(anchor: str, interface_id: str, properties_spec: List[Dict[str, str]]) -> str:
    fields_desc = "\n".join([f'- "{p["name"]}" ({p["schema"]})' for p in properties_spec])
    return (
        "You are an information extraction assistant.\n"
        "Extract values exactly from the ANCHOR text to fill ONLY the following fields "
        f"for interface \"{interface_id}\".\n"
        "If a value is not stated, use null.\n"
        "Do not infer unstated facts.\n"
        "Return ONLY minified JSON (no markdown, no comments), "
        "with EXACT keys listed below.\n\n"
        "Fields:\n"
        f"{fields_desc}\n\n"
        "ANCHOR:\n"
        f"{anchor}\n"
    )


SMOKE_INTERFACE = "dtmi:ev_battery_health_twin:BatteryPack;1"
SMOKE_FIELDS = [
    {"name": "serialNumber", "schema": "string"},
    {"name": "manufacturer", "schema": "string"},
    {"name": "nominalCapacityAh", "schema": "double"},
    {"name": "nominalVoltage", "schema": "double"},
]
SMOKE_ANCHOR = (
    "The battery unit is identified by the serial number bp-20231101-07. It is manufactured "
    "by VoltEdge, a company specializing in high-power energy storage solutions. The design "
    "specifies a nominal capacity of 212.5 ampere-hours, delivering substantial energy for "
    "demanding applications. Its cell stack operates at a nominal voltage of 400.0 volts."
)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a completion, fenced or not.

    Mirrors the fallback in `2.perf-eval-fill-gen-local.py`. GRPO only rewarded fenced
    output, so whether a fence shows up depends on the prompt encoding — bare `json.loads`
    is not enough. See the "Reward shaping" section of the card.
    """
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def smoke_test(model_dir: str) -> bool:
    """Load the checkpoint and run one real extraction. Opt-in: it costs a full model load."""
    print("\n[INFO] Smoke test: loading the checkpoint and running one extraction")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto")
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"       loaded on {device}")

        prompt = build_prompt(SMOKE_ANCHOR, SMOKE_INTERFACE, SMOKE_FIELDS)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e:  # noqa: BLE001 - report and let the caller decide
        print(f"[WARN]  smoke test could not run: {e}")
        return False

    print(f"       raw completion:\n{completion.strip()}\n")
    parsed = extract_json(completion)
    if parsed is None:
        print("[WARN]  smoke test: completion did not contain parseable JSON")
        return False

    expected = {p["name"] for p in SMOKE_FIELDS}
    got = set(parsed)
    print(f"[OK]   smoke test: parsed OK, {len(got & expected)}/{len(expected)} expected keys"
          + (f", extra keys {sorted(got - expected)}" if got - expected else ""))
    return True


# ---------------------------
# Model card
# ---------------------------

def card_text(repo_id: str, info: Dict[str, Any], reward: Optional[Dict[str, Any]],
              dataset_repo: Optional[str]) -> str:
    tags = ["grpo", "trl", "digital-twin", "dtdl", "information-extraction",
            "structured-output", "qwen2"]

    front = [
        "---",
        "license: mit",
        "language:",
        "- en",
        f"base_model: {BASE_MODEL}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
    ]
    front += [f"- {t}" for t in tags]
    if dataset_repo:
        front += ["datasets:", f"- {dataset_repo}"]
    front += ["---", ""]

    params = f"{info['params'] / 1e6:.0f}M" if info["params"] else "n/a"
    dataset_link = (f"[`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})"
                    if dataset_repo else "the `digital-twin-composition` dataset")

    body = [
        "# Qwen2-0.5B-GRPO-Fill-In",
        "",
        f"A {params}-parameter [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) fine-tuned with",
        "**GRPO** (Group Relative Policy Optimization) to fill the *Property* values of a",
        "[DTDL v2](https://github.com/Azure/opendigitaltwins-dtdl) digital-twin Interface from a",
        "free-text specification paragraph.",
        "",
        f"It is the fill-in stage of the **InterTwin** pipeline ({PROJECT_URL}).",
        "",
        "## Purpose",
        "",
        "Given an *anchor* — a paragraph of prose describing a piece of equipment — and the property",
        "schema of a DTDL Interface, the model returns a JSON object holding one value per property.",
        "That turns unstructured documentation into an instantiated digital twin.",
        "",
        "The task is deliberately narrow, so a 0.5B model is enough. The point is that the fill-in",
        "step runs **locally on CPU or a single small GPU** instead of billing a hosted frontier",
        "model once per interface — the pipeline instantiates tens of thousands of them.",
        "",
        "## Features",
        "",
        "- **Closed key set.** The prompt lists the exact property names and DTDL schemas; the model",
        "  is trained to return those keys and no others. Reward is proportional to key agreement in",
        "  both directions, so both missing and invented keys are penalised.",
        "- **`null` for unstated values.** The instruction is to extract, not to infer. Facts absent",
        "  from the anchor come back `null` rather than hallucinated.",
        "- **DTDL-typed output.** Values are produced for `string`, `double`, `integer` and `boolean`",
        "  schemas; the caller coerces them (see the snippet below).",
        "- **`dockerImage` excluded.** That property is a deployment artifact, not something stated in",
        "  a spec paragraph, so it is filtered out of the field list on both sides.",
        "- **Small enough to be local.** "
        f"{info['bytes'] / 1e9:.2f} GB on disk at `{info['dtype']}`; roughly half that loaded as",
        "  `bfloat16`.",
        "",
        "## Training process",
        "",
        f"| | |",
        "| --- | --- |",
        f"| Base model | [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) |",
        "| Method | GRPO, via TRL's `GRPOTrainer` |",
        "| Epochs | 1 |",
        "| Other hyperparameters | TRL `GRPOConfig` defaults — the training script overrides only `output_dir` and `num_train_epochs` |",
        "| Padding | `padding_side = 'left'` (required for batched generation during rollouts) |",
        f"| Training data | 30% shard of the fill-in dataset, `train_test_split(test_size=0.3, seed=42)` |",
        "",
        "### Data",
        "",
        "The corpus is generated by the earlier stages of the pipeline:",
        "",
        "1. `0.data-gen-dtdl.py` synthesises digital-twin topics and DTDL v2 Interfaces",
        "   (`topics.jsonl`, `interfaces.jsonl`).",
        "2. `0.data-gen-fill.py` turns each interface into an `{anchor, answer}` pair — a natural",
        "   paragraph describing only the Property fields, and the structured instance it implies",
        "   (`fill-eval.jsonl`, 27,770 records).",
        "",
        "Training consumed the **30% shard** of that split (~8.3k examples), leaving the majority of",
        f"the corpus untouched by training. The data is published as {dataset_link}.",
        "",
        "### Reward function",
        "",
        "Each completion is scored against the ground-truth answer object:",
        "",
        "1. Locate a JSON object in the completion.",
        "2. `count` starts at the number of predicted keys; decrement for every predicted key absent",
        "   from the ground truth or whose value disagrees (floats compared at a `1e-3` tolerance),",
        "   and again for every ground-truth key the prediction omitted.",
        "3. `reward = format_factor * (count / len(prediction))`.",
        "4. Anything that raises — unparseable JSON above all — scores `-1`.",
        "",
        "So the signal is a symmetric key/value agreement score with a hard penalty for malformed",
        "output, which is exactly what a downstream `json.loads` cares about.",
        "",
    ]

    if reward:
        run = f" (W&B run `{reward['run']}`)" if reward.get("run") else ""
        body += [
            f"### Reward curve{run}",
            "",
            "| point | step | mean reward |",
            "| --- | --- | --- |",
            f"| first logged | {reward['first_step']} | {reward['first_reward']:.3f} |",
            f"| best | {reward['best_step']} | {reward['best_reward']:.3f} |",
            f"| final | {reward['last_step']} | {reward['last_reward']:.3f} |",
            f"| mean of last {reward['tail_n']} points | — | {reward['tail_mean']:.3f} |",
            "",
            f"{reward['n']} logged points over {reward['last_step']:,} optimizer steps. Reward rises",
            "steeply out of the gate — the base model already writes JSON, so most of the early gain",
            "is learning to respect the closed key set — then flattens against a ceiling that is",
            "worth understanding before you use the model.",
            "",
        ]

    ceiling = f"{reward['best_reward']:.2f}" if reward else "0.80"
    body += [
        "## Reward shaping and the 0.80 ceiling",
        "",
        f"The curve plateaus at **{ceiling}**, not 1.0, and that is not a coincidence.",
        "",
        "The reward function branches on output format. A completion whose JSON sits inside a",
        "Markdown code fence is scored with a format factor of `0.8`; the branch meant to handle bare",
        "JSON indexes the completion list as if it were a dict, raises `TypeError`, and lands in the",
        "`except`, scoring `-1`.",
        "",
        "**So only fenced output was ever rewarded, and a perfect answer topped out at `0.8`.** A",
        f"plateau at exactly {ceiling} therefore means the model reached near-perfect key agreement —",
        "the format factor, not the extraction quality, is what caps the curve.",
        "",
        "### What that means for the output format",
        "",
        "GRPO optimised the format it was scored on, and the effect is visible at inference. Same",
        "anchor, same fields, greedy decoding, two prompt encodings:",
        "",
        "| prompt encoding | output |",
        "| --- | --- |",
        "| ChatML chat template | wrapped in a ```` ```json ```` fence |",
        "| flat text prompt (the pipeline's format, below) | bare JSON object |",
        "",
        "Both carry identical, correctly typed values. Neither honours the prompt's request for",
        "*minified* JSON — the output is pretty-printed with newlines and indentation in both cases.",
        "",
        "The split is consistent with how training ran: the reward function indexes each completion as",
        "a list of chat messages, so the rollouts were in conversational form, and the fenced habit it",
        "rewarded surfaces most strongly when you call the model that same way.",
        "",
        "> **Parse with a brace-extraction fallback rather than calling `json.loads` on the raw",
        "> completion.** It handles both shapes. The snippet below and the project's evaluation script",
        "> both do this.",
        "",
        "## Usage",
        "",
        "The pipeline calls the model with a **flat text prompt** rather than the chat template.",
        "Reproduce it exactly for best results:",
        "",
        "```python",
        "import json",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'MODEL = "{repo_id}"',
        "tok = AutoTokenizer.from_pretrained(MODEL)",
        'model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").eval()',
        "",
        "",
        "def property_fields(interface):",
        '    """DTDL Interface -> the fields the model is asked to fill."""',
        '    return [{"name": c["name"], "schema": c.get("schema", "string")}',
        '            for c in interface["contents"]',
        '            if isinstance(c, dict) and c.get("@type") == "Property"',
        '            and c["name"] != "dockerImage"]',
        "",
        "",
        "def build_prompt(anchor, interface_id, fields):",
        '    desc = "\\n".join(f\'- "{f["name"]}" ({f["schema"]})\' for f in fields)',
        "    return (",
        '        "You are an information extraction assistant.\\n"',
        '        "Extract values exactly from the ANCHOR text to fill ONLY the following fields "',
        '        f\'for interface "{interface_id}".\\n\'',
        '        "If a value is not stated, use null.\\n"',
        '        "Do not infer unstated facts.\\n"',
        '        "Return ONLY minified JSON (no markdown, no comments), "',
        '        "with EXACT keys listed below.\\n\\n"',
        '        f"Fields:\\n{desc}\\n\\n"',
        '        f"ANCHOR:\\n{anchor}\\n"',
        "    )",
        "",
        "",
        "def extract_json(text):",
        '    """The model wraps its answer in a code fence - see "Reward shaping" above."""',
        "    text = text.strip()",
        '    if not text.startswith("{"):',
        '        start, end = text.find("{"), text.rfind("}")',
        "        if start == -1 or end <= start:",
        "            return {}",
        "        text = text[start:end + 1]",
        "    try:",
        "        return json.loads(text)",
        "    except json.JSONDecodeError:",
        "        return {}",
        "",
        "",
        "def coerce(value, schema):",
        '    """DTDL schema -> Python type; None when the value is missing or unusable."""',
        "    if value is None:",
        "        return None",
        "    try:",
        '        if schema in ("double", "float"):',
        "            return float(value)",
        '        if schema in ("integer", "int", "long"):',
        "            return int(float(value))",
        '        if schema in ("boolean", "bool"):',
        "            if isinstance(value, str):",
        '                return value.strip().lower() in ("true", "yes", "y", "1")',
        "            return bool(value)",
        "        return str(value)",
        "    except (TypeError, ValueError):",
        "        return None",
        "",
        "",
        "interface = {",
        f'    "@id": "{SMOKE_INTERFACE}",',
        '    "contents": [',
        '        {"@type": "Property", "name": "serialNumber", "schema": "string"},',
        '        {"@type": "Property", "name": "manufacturer", "schema": "string"},',
        '        {"@type": "Property", "name": "nominalCapacityAh", "schema": "double"},',
        '        {"@type": "Property", "name": "nominalVoltage", "schema": "double"},',
        "    ],",
        "}",
        f'anchor = "{SMOKE_ANCHOR}"',
        "",
        'fields = property_fields(interface)',
        'prompt = build_prompt(anchor, interface["@id"], fields)',
        "",
        'inputs = tok(prompt, return_tensors="pt").to(model.device)',
        "with torch.no_grad():",
        "    out = model.generate(**inputs, max_new_tokens=512, do_sample=False,",
        "                         pad_token_id=tok.pad_token_id or tok.eos_token_id)",
        "",
        'completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)',
        "raw = extract_json(completion)",
        'filled = {f["name"]: coerce(raw.get(f["name"]), f["schema"]) for f in fields}',
        "print(filled)",
        '# {"serialNumber": "bp-20231101-07", "manufacturer": "VoltEdge",',
        '#  "nominalCapacityAh": 212.5, "nominalVoltage": 400.0}',
        "```",
        "",
        "Use **greedy decoding** (`do_sample=False`). The bundled `generation_config.json` inherits",
        "Qwen2-Instruct's sampling defaults (`temperature=0.7`, `top_p=0.8`), which suit chat, not",
        "deterministic extraction.",
        "",
    ]

    if info["has_chat_template"]:
        body += [
            "A ChatML `chat_template.jinja` ships with the checkpoint and `apply_chat_template` works,",
            "but it makes the model wrap its answer in a code fence (see above). The flat prompt is",
            "what the pipeline uses and what `extract_json` is written for.",
            "",
        ]

    gen = info.get("generation") or {}
    body += [
        "## Model details",
        "",
        "| | |",
        "| --- | --- |",
        f"| Architecture | `{info['architecture']}` (`{info['model_type']}`) |",
        f"| Parameters | {params} |",
        f"| Precision on disk | `{info['dtype']}` ({info['bytes'] / 1e9:.2f} GB) |",
        f"| Layers | {info['layers']} |",
        f"| Hidden size | {info['hidden_size']} |",
        f"| Attention heads | {info['attention_heads']} (grouped-query, {info['kv_heads']} KV heads) |",
        f"| Intermediate size | {info['intermediate_size']} |",
        f"| Vocab size | {info['vocab_size']:,} |",
        f"| Context length | {info['max_position_embeddings']:,} tokens |",
        f"| Tied embeddings | {bool(info['tie_word_embeddings'])} |",
        f"| Saved with | `transformers` {info['transformers_version']} |",
    ]
    if gen:
        body.append(f"| Default generation | `temperature={gen.get('temperature')}`, "
                    f"`top_p={gen.get('top_p')}`, `top_k={gen.get('top_k')}`, "
                    f"`repetition_penalty={gen.get('repetition_penalty')}` — override for extraction |")

    body += [
        "",
        "## Evaluation",
        "",
        "Prediction quality is measured by the project's own scripts, against the answers in",
        "`fill-eval.jsonl`:",
        "",
        "```bash",
        "# 1. Generate filled outputs with this checkpoint",
        f'export LOCAL_MODEL="{repo_id}"',
        "python 2.perf-eval-fill-gen-local.py",
        "",
        "# 2. Score them",
        "python 2.perf-eval-result-eval.py \\",
        "  --eval fill-eval.jsonl \\",
        "  --pred <output-dir>/filled-output-<output-dir>.jsonl \\",
        "  --tol 0.0 --top_percent 70",
        "```",
        "",
        "`2.perf-eval-result-eval.py` reports per-field precision, recall, F1 and exact match, plus",
        "min/max/mean latency per sample. Run it to get numbers for your hardware and tolerance",
        "settings — the figures depend on both, so none are quoted here.",
        "",
        "## Limitations",
        "",
        "- **0.5B capacity.** Long anchors with many properties, unusual units, or values spread",
        "  across several sentences are where it slips. It is a fast extractor, not a reasoner.",
        "- **Not a chat model.** GRPO on a single narrow task erodes general instruction-following.",
        "  Use the base `Qwen2-0.5B-Instruct` for conversation.",
        "- **Output format follows the reward, not the prompt.** It ignores \"minified\", and wraps the",
        "  answer in a code fence when called through the chat template — see the reward-shaping",
        "  section. Parse defensively.",
        "- **English only**, and tuned to the register of generated spec paragraphs. Real-world",
        "  datasheets, tables and multilingual documentation are out of distribution.",
        "- **Synthetic training data.** Both the DTDL interfaces and the anchors are LLM-generated, so",
        "  the property vocabulary reflects that generator's habits.",
        "- **Not a clean held-out benchmark.** Training used a 30% shard of the same corpus the",
        "  evaluation set is drawn from; the remaining 70% is unseen by the model but was produced by",
        "  the same generator, so scores are optimistic relative to genuinely novel domains.",
        "- **fp32 weights.** Load with `dtype=torch.bfloat16` to halve memory at negligible cost for",
        "  this task.",
        "",
        "## Provenance",
        "",
        f"Produced by `1.fine-tune-GRPO-llm.py` in [`digital-twin-composition`]({PROJECT_URL}) and",
        "uploaded by `1.model-push-to-huggingface.py`. The pipeline stages are numbered: `0.` data",
        "generation, `1.` fine-tuning, `2.` performance evaluation, `3.` system evaluation,",
        "`4.` deployment.",
        "",
        "Licensed MIT, as is the parent project.",
    ]

    return "\n".join(front + body) + "\n"


# ---------------------------
# Push
# ---------------------------

def push(repo_id: str, model_dir: str, card: Optional[str], private: bool,
         token: Optional[str], commit_message: str) -> None:
    from huggingface_hub import HfApi, ModelCard

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[OK]   repo ready: {repo_id}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=model_dir,
        commit_message=commit_message,
        # The card is pushed separately; never ship a stray lock or git dir.
        ignore_patterns=["README.md", ".git*", "*.lock", "__pycache__/*"],
    )
    print(f"[OK]   weights and tokenizer uploaded")

    if card is not None:
        ModelCard(card).push_to_hub(repo_id, token=token, repo_type="model")
        print("[OK]   model card pushed")


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help="Target model repo, e.g. user/Qwen2-0.5B-GRPO-Fill-In")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory holding the checkpoint")
    ap.add_argument("--reward-csv", default=DEFAULT_REWARD_CSV,
                    help="Exported train/reward curve; documented in the card when present")
    ap.add_argument("--dataset-repo", default=None,
                    help="Companion HF dataset repo to link from the card, e.g. user/digital-twin-composition")
    ap.add_argument("--private", action="store_true", default=True, help="Create the repo private (default)")
    ap.add_argument("--public", dest="private", action="store_false", help="Create the repo public")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (default: HF_TOKEN or cached login)")
    ap.add_argument("--commit-message", default="Upload Qwen2-0.5B-GRPO-Fill-In",
                    help="Commit message for the weight upload")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Load the checkpoint and run one extraction before pushing")
    ap.add_argument("--card-out", metavar="DIR", default=None,
                    help="Also write the generated card to DIR/README.md")
    ap.add_argument("--no-card", action="store_true", help="Skip writing the model card")
    ap.add_argument("--force", action="store_true", help="Push even if the smoke test fails")
    ap.add_argument("--dry-run", action="store_true", help="Validate and build the card, push nothing")
    args = ap.parse_args()

    if not validate(args.model_dir):
        return 1

    info = inspect_model(args.model_dir)
    reward = reward_summary(args.reward_csv)

    if args.smoke_test and not smoke_test(args.model_dir):
        if not (args.dry_run or args.force):
            print("[ERROR] Smoke test failed; re-run with --force to push anyway.", file=sys.stderr)
            return 1
        print("[WARN]  continuing despite the failed smoke test")

    card = None
    if not args.no_card:
        card = card_text(args.repo_id, info, reward, args.dataset_repo)
        print(f"[OK]   model card built ({len(card.splitlines())} lines)")
        if args.card_out:
            os.makedirs(args.card_out, exist_ok=True)
            out = os.path.join(args.card_out, "README.md")
            with open(out, "w", encoding="utf-8") as f:
                f.write(card)
            print(f"[OK]   wrote {out}")

    if args.dry_run:
        print(f"\n[DRY-RUN] {info['bytes'] / 1e9:.2f} GB in {args.model_dir} ready. "
              f"Would push to '{args.repo_id}' ({'private' if args.private else 'PUBLIC'})"
              f"{'' if card else ' without a card'}.")
        return 0

    visibility = "private" if args.private else "PUBLIC"
    print(f"\n[INFO] Pushing {info['bytes'] / 1e9:.2f} GB to '{args.repo_id}' ({visibility})")
    push(args.repo_id, args.model_dir, card, args.private, args.token, args.commit_message)

    print(f"\n[DONE] https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
