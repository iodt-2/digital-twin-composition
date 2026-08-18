#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit train/evaluation leakage in the Digital Twin property fill-in dataset.

The released GRPO checkpoint was trained on the **30% shard** of
`train_test_split(test_size=0.3, seed=42)` over `data/fill-eval.jsonl`
(`1.fine-tune-GRPO-llm.py --split test`), so the held-out evaluation partition is
the other **70%** — the shard `datasets` names "train". The two names are inverted
with respect to their role, so this script never uses them bare: the shard the
model saw is the TRAINING partition, the shard it did not is the EVALUATION
partition, and every number below is reported as a fraction of the evaluation
partition, because that is the set whose scores a leak would inflate.

The audit has two independent halves, and the report keeps them apart on purpose:

  1. DETERMINISTIC evidence - shared row indices, exact duplicate records,
     identical normalised anchors, identical answers, identical distinctive
     property-value configurations, and shared interface identifiers. These are
     proofs. A count of zero here is a fact about the split.

  2. SEMANTIC evidence - cosine neighbourhood over sentence embeddings, and an
     LLM asked to judge the suspicious pairs. This is *supporting evidence only*.
     An LLM judge has no error bound: it can miss a paraphrase and it can call a
     coincidence a copy. Treat a clean LLM verdict as "nothing found by this
     method", never as a proof that the evaluation partition is uncontaminated.

The distinction that makes the semantic half worth running at all is that this
corpus is meant to be internally similar: every record is a digital twin
description, thousands share an interface, and property names like `serialNumber`
repeat across the whole catalogue. So a high cosine score is the *normal* state of
two unrelated records here, and neither a shared domain, a shared interface type,
a shared schema nor shared property names is leakage. Leakage is the evaluation
record being the same underlying twin instance, a paraphrase of a training
description, or the same distinctive property-value configuration - anything a
model could answer out of memory rather than by extracting from the anchor.

Outputs (under --out-dir):
    audit.jsonl            one line per evaluation sample: neighbours, cosine
                           similarities, deterministic flags, LLM judgements
    summary.json           the numbers printed in the final report
    llm-cache.jsonl        every LLM judgement, keyed by prompt hash; an
                           interrupted run resumes from it without re-asking
    nn-similarities.npy    (eval x k) cosine similarities, float32
    nn-indices.npy         (eval x k) training-partition row numbers
    embeddings-*.npy       encoder output, reused when the anchors are unchanged

Configuration is CLI flags with environment-variable defaults:
    OLLAMA_HOST, OLLAMA_MODEL       the semantic leakage judge
    LEAKAGE_EMBED_MODEL             the sentence encoder

Examples:
    # deterministic half only - no encoder, no network, seconds
    python 0.data-leakage-check.py --deterministic-only

    # everything except the judge: encode, index, and print the neighbourhood.
    # Run this first - the similarity distribution is what tells you where to put
    # --judge-threshold, and the embeddings it caches make the judge run cheap.
    python 0.data-leakage-check.py --judge-mode none

    # full audit at the default 0.85 floor (~10.5k pairs, ~7 h on one gpt-oss:20b)
    python 0.data-leakage-check.py

    # a first pass in an hour: the most suspicious pairs only, resumable
    python 0.data-leakage-check.py --judge-threshold 0.90 --judge-limit 1000

    # the stricter reading of "inspect every evaluation sample"
    python 0.data-leakage-check.py --judge-mode top1

Interrupting is safe and re-running resumes: judgements are cached by prompt hash
and embeddings by content, so only the unfinished requests are re-issued.
"""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- the split, which must match 1.fine-tune-GRPO-llm.py exactly -------------
SPLIT_TEST_SIZE = 0.3
SPLIT_SEED = 42
TRAINED_SHARD = "test"    # the shard `--split test` trained on: 30%, model-seen
HELDOUT_SHARD = "train"   # the remaining 70%: the evaluation partition

DEFAULT_DATA = os.path.join("data", "fill-eval.jsonl")
DEFAULT_OUT = os.path.join("results", "leakage-audit")

# `dockerImage` is a pure function of the interface id and telemetry is zeroed in
# every answer (invariant 3), so neither carries information about the instance.
# Leaving them in would make unrelated records look identically configured.
IGNORE_KEYS = {"interface", "dockerImage"}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.10.10.4:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Anchors the generator never really wrote. `0.data-gen-fill.py` stores whatever
# the model returned, so a refusal lands in the corpus as a valid-looking record: a
# 43-character anchor and a genuine answer. Those are byte-identical across
# unrelated topics, which trips the identical-anchor check without any model having
# memorised anything - they are a data-quality defect, so they are detected and
# reported on their own line rather than counted as leakage.
REFUSAL_MARKERS = ("i'm sorry", "i am sorry", "i cannot", "i can't", "i can not",
                   "as an ai", "unable to fulfill", "unable to comply",
                   "cannot fulfill", "can't fulfill", "cannot assist", "can't assist")
MIN_ANCHOR_CHARS = 120

# A general-purpose encoder by default, not the fine-tuned retrieval model: that
# one was trained on triplets mined from this same corpus, so it deliberately
# pulls same-topic records together and would report domain similarity as if it
# were duplication. Override with --embed-model when you want that view.
DEFAULT_EMBED_MODEL = os.getenv("LEAKAGE_EMBED_MODEL", "all-MiniLM-L6-v2")

# Bump when build_judge_prompt changes: the cache is keyed by prompt text, so an
# edited prompt must not silently reuse verdicts formed under the old wording.
PROMPT_VERSION = "leakage-judge-v3"

JUDGE_KEYS = ("same_underlying_instance", "near_duplicate_description",
              "same_property_value_configuration", "same_interface_semantics",
              "leakage", "confidence", "reason")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_underlying_instance": {"type": "boolean"},
        "near_duplicate_description": {"type": "boolean"},
        "same_property_value_configuration": {"type": "boolean"},
        "same_interface_semantics": {"type": "boolean"},
        "leakage": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": list(JUDGE_KEYS),
}


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse failed at {path}:{line_no}: {exc}") from exc
    return out


def split_indices(n_rows: int, test_size: float, seed: int) -> Dict[str, List[int]]:
    """Row numbers of each shard, reproduced with `datasets.train_test_split`.

    There is no numpy fallback here on purpose. An audit whose split does not
    match the one training consumed answers a question nobody asked, and a
    hand-rolled permutation does not reproduce `datasets`' shuffling - it would
    hand back a "held-out" set that silently overlaps training.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"--test-size must be in (0, 1), got {test_size}")
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] `datasets` is required: it is what reproduces the exact shards "
            "1.fine-tune-GRPO-llm.py trained on. Install it, or the audit is not "
            "about the released checkpoint."
        ) from exc

    shards = Dataset.from_dict({"row": list(range(n_rows))}).train_test_split(
        test_size=test_size, seed=seed)
    return {name: [int(i) for i in shards[name]["row"]] for name in ("train", "test")}


# ---------------------------------------------------------------------------
# Normalisation - what "identical" means for each deterministic check
# ---------------------------------------------------------------------------
def normalise_anchor(text: str) -> str:
    """Case, unicode and whitespace folded away.

    The generator emits non-breaking hyphens, curly quotes and the occasional
    double space, so two byte-different anchors can be the same sentence. NFKC
    folds those variants together; casefold and whitespace collapse do the rest.
    """
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(folded.split())


def loose_anchor(text: str) -> str:
    """As above, with every non-alphanumeric character dropped.

    Catches the pair that differs only in punctuation or in how a unit was
    spelled. Reported separately: it is a looser claim than `normalise_anchor`.
    """
    return "".join(ch for ch in normalise_anchor(text) if ch.isalnum() or ch == " ")


def is_degenerate_anchor(text: str, min_chars: int = MIN_ANCHOR_CHARS) -> bool:
    """True for an anchor that describes nothing: a refusal, or far too short.

    NFKC leaves the curly apostrophe alone, so it is folded here - the refusals in
    this corpus are written with U+2019.
    """
    normalised = normalise_anchor(text)
    if len(normalised) < min_chars:
        return True
    head = normalised[:200].replace("’", "'")
    return any(marker in head for marker in REFUSAL_MARKERS)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def is_zero_value(value: Any) -> bool:
    # Booleans stay in: `False == 0` in Python, and a genuinely-false property is
    # part of the configuration, not an unfilled telemetry slot.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def distinctive_config(answer: Dict[str, Any]) -> Dict[str, Any]:
    """The property-value pairs that identify an instance.

    Interface id, dockerImage and the zeroed telemetry slots are dropped: they are
    shared by construction across every record of a topic, so keeping them would
    make "same configuration" mean "same interface".
    """
    if not isinstance(answer, dict):
        return {}
    return {k: v for k, v in answer.items()
            if k not in IGNORE_KEYS and not is_zero_value(v)}


def anchor_of(record: Dict[str, Any]) -> str:
    return str(record.get("anchor", "") or "")


def interface_of(record: Dict[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("interface", "") or "")
    return ""


# ---------------------------------------------------------------------------
# Deterministic audit
# ---------------------------------------------------------------------------
def build_index_maps(records: Sequence[Dict[str, Any]],
                     rows: Sequence[int]) -> Dict[str, Dict[str, List[int]]]:
    """key -> training rows carrying it, for each notion of "identical"."""
    maps: Dict[str, Dict[str, List[int]]] = {
        name: defaultdict(list) for name in
        ("record", "anchor", "loose_anchor", "answer", "config", "interface")
    }
    for row in rows:
        record = records[row]
        answer = record.get("answer")
        maps["record"][canonical(record)].append(row)
        maps["anchor"][normalise_anchor(anchor_of(record))].append(row)
        maps["loose_anchor"][loose_anchor(anchor_of(record))].append(row)
        maps["answer"][canonical(answer)].append(row)
        config = distinctive_config(answer if isinstance(answer, dict) else {})
        # An empty configuration would match every other empty one, which is a
        # property of the record being uninformative, not of it being copied.
        maps["config"][canonical(config) if config else f"\x00empty:{row}"].append(row)
        maps["interface"][interface_of(record)].append(row)
    return {name: dict(mapping) for name, mapping in maps.items()}


def deterministic_audit(records: Sequence[Dict[str, Any]],
                        train_rows: Sequence[int],
                        eval_rows: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-evaluation-sample overlap flags, and the counts that summarise them."""
    maps = build_index_maps(records, train_rows)
    train_set = set(train_rows)
    flags: List[Dict[str, Any]] = []
    counters: Counter = Counter()

    for row in eval_rows:
        record = records[row]
        answer = record.get("answer")
        config = distinctive_config(answer if isinstance(answer, dict) else {})
        degenerate = is_degenerate_anchor(anchor_of(record))
        hit = {
            "row_also_in_training": row in train_set,
            "exact_duplicate_record": maps["record"].get(canonical(record), []),
            "same_normalised_anchor": maps["anchor"].get(normalise_anchor(anchor_of(record)), []),
            "same_loose_anchor": maps["loose_anchor"].get(loose_anchor(anchor_of(record)), []),
            "same_answer": maps["answer"].get(canonical(answer), []),
            "same_config": maps["config"].get(canonical(config), []) if config else [],
            "same_interface": maps["interface"].get(interface_of(record), []),
        }
        # Row lists are kept short in the audit file: the first few witnesses are
        # enough to check a claim by hand, the count is what the report needs.
        flags.append({
            "row_also_in_training": hit["row_also_in_training"],
            "degenerate_anchor": degenerate,
            "exact_duplicate_record": len(hit["exact_duplicate_record"]),
            "same_normalised_anchor": len(hit["same_normalised_anchor"]),
            "same_loose_anchor": len(hit["same_loose_anchor"]),
            "same_answer": len(hit["same_answer"]),
            "same_config": len(hit["same_config"]),
            "same_interface": len(hit["same_interface"]),
            "witnesses": {name: value[:5] for name, value in hit.items()
                          if isinstance(value, list) and value},
        })
        counters["row_also_in_training"] += int(hit["row_also_in_training"])
        counters["degenerate_anchor"] += int(degenerate)
        for name in ("exact_duplicate_record", "same_normalised_anchor", "same_loose_anchor",
                     "same_answer", "same_config", "same_interface"):
            counters[name] += int(bool(hit[name]))
        # The same check with the refusals taken out, which is the number that
        # actually speaks to memorisation.
        if hit["same_normalised_anchor"] and not degenerate:
            counters["same_normalised_anchor_excluding_degenerate"] += 1

    n_eval = max(len(eval_rows), 1)
    summary = {
        name: {"count": int(counters[name]), "percent": 100.0 * counters[name] / n_eval}
        for name in ("row_also_in_training", "exact_duplicate_record",
                     "same_normalised_anchor", "same_normalised_anchor_excluding_degenerate",
                     "same_loose_anchor", "same_answer", "same_config", "same_interface",
                     "degenerate_anchor")
    }
    return flags, summary


def whole_corpus_duplicates(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Duplication inside the file as a whole.

    Context for the cross-shard numbers: a random split can only put a duplicate
    on both sides if the file contains duplicates to begin with. If these are
    zero, cross-shard duplication is impossible and the checks above are a
    formality; if they are large, the split is doing the only work.
    """
    anchors: Counter = Counter()
    answers: Counter = Counter()
    degenerate = 0
    for record in records:
        anchors[normalise_anchor(anchor_of(record))] += 1
        answers[canonical(record.get("answer"))] += 1
        degenerate += int(is_degenerate_anchor(anchor_of(record)))
    return {
        "duplicate_anchor_rows": sum(c for c in anchors.values() if c > 1),
        "distinct_duplicated_anchors": sum(1 for c in anchors.values() if c > 1),
        "duplicate_answer_rows": sum(c for c in answers.values() if c > 1),
        "distinct_duplicated_answers": sum(1 for c in answers.values() if c > 1),
        "degenerate_anchor_rows": degenerate,
    }


# ---------------------------------------------------------------------------
# Semantic neighbourhood
# ---------------------------------------------------------------------------
def fingerprint(model_name: str, texts: Sequence[str]) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    digest.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()[:16]


def encode_anchors(texts: Sequence[str], model_name: str, out_dir: str,
                   batch_size: int, device: Optional[str], use_cache: bool):
    """Unit-normalised embeddings, cached per (model, exact text list).

    Encoding 27k paragraphs is the slowest offline step, and an interrupted run
    should not pay for it twice - so the cache key is the content, not the path.
    """
    import numpy as np

    cache_path = os.path.join(out_dir, f"embeddings-{fingerprint(model_name, texts)}.npy")
    if use_cache and os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            print(f"[INFO] Reusing cached embeddings: {cache_path} {embeddings.shape}")
            return embeddings, cache_path
        print(f"[WARN] Cached embeddings {cache_path} have {embeddings.shape[0]} rows, "
              f"expected {len(texts)}; re-encoding.")

    from sentence_transformers import SentenceTransformer

    print(f"[INFO] Encoding {len(texts)} anchors with '{model_name}' ...")
    started = time.time()
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(list(texts), batch_size=batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=True).astype("float32")
    print(f"[INFO] Encoded in {time.time() - started:.1f}s -> {embeddings.shape}")
    if use_cache:
        np.save(cache_path, embeddings)
    return embeddings, cache_path


def nearest_neighbours(train_emb, eval_emb, top_k: int):
    """Top-k training neighbours of every evaluation anchor, by cosine similarity.

    The vectors are unit-normalised, so FAISS's inner product *is* the cosine.
    The index is exact (IndexFlatIP): an approximate one could miss the single
    near-duplicate that the whole audit exists to find.
    """
    import faiss

    top_k = max(1, min(top_k, train_emb.shape[0]))
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)
    print(f"[INFO] FAISS IndexFlatIP over {index.ntotal} training vectors; "
          f"searching {eval_emb.shape[0]} evaluation vectors for top-{top_k} ...")
    started = time.time()
    similarities, indices = index.search(eval_emb, top_k)
    print(f"[INFO] Search finished in {time.time() - started:.1f}s")
    return similarities, indices


def similarity_stats(values: Sequence[float]) -> Dict[str, float]:
    import numpy as np

    if len(values) == 0:
        return {}
    array = np.asarray(values, dtype="float64")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "min": float(array.min()),
        "std": float(array.std()),
    }


# ---------------------------------------------------------------------------
# The LLM semantic leakage judge
# ---------------------------------------------------------------------------
def build_judge_prompt(train_record: Dict[str, Any], eval_record: Dict[str, Any]) -> str:
    """One pair, as the judge sees it.

    The cosine score is deliberately withheld. The pair was selected *because* it
    scored highly; telling the model so invites it to ratify the retrieval rather
    than read the two records, and the whole point of this stage is an opinion
    that is independent of the embedding.

    Every fixed word comes before the pair so that the whole instruction block is
    one shared prefix: the server reuses its KV cache across requests instead of
    re-reading a kilobyte of rules for each of thousands of pairs.
    """
    train_answer = train_record.get("answer") if isinstance(train_record.get("answer"), dict) else {}
    eval_answer = eval_record.get("answer") if isinstance(eval_record.get("answer"), dict) else {}
    pair = {
        "training_sample": {
            "description": anchor_of(train_record),
            "interface": train_answer.get("interface", ""),
            "properties": distinctive_config(train_answer),
        },
        "evaluation_sample": {
            "description": anchor_of(eval_record),
            "interface": eval_answer.get("interface", ""),
            "properties": distinctive_config(eval_answer),
        },
    }
    return (
        "You are auditing a machine-learning dataset for train/evaluation data leakage.\n"
        "\n"
        "The task the model performs is: read a natural-language description of a digital\n"
        "twin and emit its configuration - an interface identifier plus property-value\n"
        "pairs. You are given one TRAINING sample (the model saw it) and one EVALUATION\n"
        "sample (it is supposed to be unseen). Decide whether the evaluation sample could\n"
        "be answered largely from memory of the training sample.\n"
        "\n"
        "LEAKAGE means at least one of:\n"
        "  - the two records describe the SAME underlying digital twin instance (same\n"
        "    physical or logical thing: same identifier, serial number, asset tag, or an\n"
        "    unmistakable combination of concrete values);\n"
        "  - the evaluation description is a PARAPHRASE or a superficial edit of the\n"
        "    training description (reordered sentences, synonyms, reworded units, a\n"
        "    changed name with everything else intact);\n"
        "  - the two carry substantially the SAME DISTINCTIVE property-value\n"
        "    configuration, so that reproducing the training answer would score on the\n"
        "    evaluation sample. Distinctive means at least one shared value that\n"
        "    identifies this thing: an identifier, serial number, asset tag, hostname or\n"
        "    URL, a timestamp, or an unusual/precise number. A run of shared values that\n"
        "    are all round defaults is NOT distinctive.\n"
        "\n"
        "NOT LEAKAGE, on its own or in combination:\n"
        "  - the same application domain or topic (both are batteries, both are HVAC);\n"
        "  - the same interface identifier, the same schema, or the same property NAMES;\n"
        "  - the same writing style, sentence templates or boilerplate phrasing - the\n"
        "    whole corpus was generated by one generator and reads alike;\n"
        "  - the same units, ranges or value types with DIFFERENT concrete values;\n"
        "  - generic values that carry no identity (0, 1, true, false, 'default');\n"
        "  - round or conventional configuration numbers that any unrelated system might\n"
        "    have picked: batch size 5000, 30-day retention, port 8080, a 60-second\n"
        "    interval, 100 Hz, 24 hours. Two systems agreeing on such defaults is a\n"
        "    coincidence of engineering convention, not a copy.\n"
        "This dataset is intentionally dense: thousands of records share an interface and\n"
        "property names repeat across the entire catalogue. High surface similarity is the\n"
        "normal state here, not evidence. Judge the INSTANCE and its VALUES, not the type.\n"
        "\n"
        "Answer with a single JSON object, no prose, with exactly these keys:\n"
        '  "same_underlying_instance"          true if both describe the same instance\n'
        '  "near_duplicate_description"        true if one description is a paraphrase or\n'
        "                                      superficial modification of the other\n"
        '  "same_property_value_configuration" true if the distinctive property VALUES are\n'
        "                                      substantially the same\n"
        '  "same_interface_semantics"          true if the interfaces model the same kind\n'
        "                                      of thing (informational: true here alone is\n"
        "                                      NOT leakage)\n"
        '  "leakage"                           true only under the LEAKAGE rules above\n'
        '  "confidence"                        number 0.0-1.0 in your own verdict\n'
        '  "reason"                            one short sentence citing the deciding fact\n'
        "\n"
        "PAIR:\n"
        f"{json.dumps(pair, ensure_ascii=False, indent=2)}\n"
    )


def ollama_judge(prompt: str, host: str, model: str, timeout: int,
                 think: Optional[str]) -> Dict[str, Any]:
    """One structured verdict from Ollama.

    `/api/chat` with a JSON schema is the path that works with reasoning models
    such as gpt-oss, whose visible answer is empty when `/api/generate` is asked
    for `format: json`. The fallbacks below cover servers or models that reject
    the schema, the think level, or the chat endpoint - in that order, each one
    giving up a little structure rather than the whole verdict.
    """
    def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            host.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    options = {"temperature": 0.0, "num_predict": 1024}
    messages = [{"role": "user", "content": prompt}]
    attempts: List[Tuple[str, Dict[str, Any]]] = []
    base = {"model": model, "stream": False, "options": options, "messages": messages}
    if think:
        attempts.append(("/api/chat", {**base, "format": JUDGE_SCHEMA, "think": think}))
    attempts.append(("/api/chat", {**base, "format": JUDGE_SCHEMA}))
    attempts.append(("/api/chat", {**base, "format": "json"}))
    attempts.append(("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                       "options": options}))

    last_error: Optional[Exception] = None
    for path, payload in attempts:
        try:
            answer = post(path, payload)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue
        text = (answer.get("message", {}) or {}).get("content") if path == "/api/chat" \
            else answer.get("response")
        if not text or not str(text).strip():
            last_error = ValueError(f"empty reply from {path}")
            continue
        try:
            return parse_judgement(str(text))
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"judge failed: {last_error}")


def parse_judgement(text: str) -> Dict[str, Any]:
    """The first JSON object in a reply, coerced to the documented schema."""
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        raise ValueError(f"no JSON in reply: {stripped[:200]!r}")
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"unparseable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")

    def as_bool(key: str) -> Optional[bool]:
        raw = value.get(key)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            if raw.strip().lower() in {"true", "yes", "1"}:
                return True
            if raw.strip().lower() in {"false", "no", "0"}:
                return False
        return None

    verdict: Dict[str, Any] = {key: as_bool(key) for key in JUDGE_KEYS[:5]}
    if verdict["leakage"] is None:
        raise ValueError(f"reply has no usable 'leakage' key: {canonical(value)[:200]}")
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    verdict["confidence"] = None if confidence is None else max(0.0, min(1.0, confidence))
    verdict["reason"] = str(value.get("reason", ""))[:600]
    return verdict


class JudgementCache:
    """Append-only judgement store, keyed by the hash of the exact prompt.

    Keyed by prompt rather than by row pair so that editing the prompt (or
    switching model) invalidates the affected entries instead of resurrecting a
    verdict formed under different instructions. Each line is flushed and fsynced
    before it counts as done, so a crash loses at most the request in flight.
    """

    def __init__(self, path: str, model: str, enabled: bool = True):
        self.path = path
        self.model = model
        self.enabled = enabled
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._handle = None
        if not enabled:
            return
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from a killed run
                    if entry.get("key"):
                        self.entries[entry["key"]] = entry
            print(f"[INFO] Judgement cache: {len(self.entries)} completed pairs in {path}")
        self._handle = open(path, "a", encoding="utf-8")

    @staticmethod
    def key_for(model: str, prompt: str) -> str:
        digest = hashlib.sha256()
        digest.update(PROMPT_VERSION.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(model.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(prompt.encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(key)

    def put(self, key: str, eval_row: int, train_row: int, rank: int,
            similarity: float, judgement: Dict[str, Any]) -> None:
        entry = {"key": key, "model": self.model, "prompt_version": PROMPT_VERSION,
                 "eval_row": eval_row, "train_row": train_row, "rank": rank,
                 "similarity": similarity, "judgement": judgement}
        with self._lock:
            self.entries[key] = entry
            if self._handle is None:
                return
            self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def select_judge_pairs(mode: str, similarities, indices, eval_rows: Sequence[int],
                       train_rows: Sequence[int], threshold: float, limit: int,
                       degenerate_rows: Optional[set] = None) -> List[Dict[str, Any]]:
    """The (evaluation, training) pairs to put in front of the judge.

    top1       the rank-1 neighbour of every evaluation sample - the reading of
               "inspect every evaluation sample", and the most expensive one.
    threshold  every top-k neighbour at or above --judge-threshold. Cheaper, and
               it looks at more than one neighbour where the neighbourhood is
               dense, which is where a duplicate would hide.
    all        every top-k pair. Only sane on a --limit'ed slice.

    Pairs where either anchor is degenerate are dropped unless the caller passes
    no `degenerate_rows`: two identical refusals are a perfect textual match and
    the judge would rightly call them duplicates, which would put a data-quality
    defect into the leakage count. They are already counted in section 1.

    Sorted by similarity descending so that a --judge-limit cap spends the budget
    on the most suspicious pairs rather than on whatever came first.
    """
    skip = degenerate_rows or set()
    pairs: List[Dict[str, Any]] = []
    for position, eval_row in enumerate(eval_rows):
        if eval_row in skip:
            continue
        ranks = range(1) if mode == "top1" else range(indices.shape[1])
        for rank in ranks:
            similarity = float(similarities[position][rank])
            if mode == "threshold" and similarity < threshold:
                continue
            train_position = int(indices[position][rank])
            if train_position < 0:
                continue
            train_row = int(train_rows[train_position])
            if train_row in skip:
                continue
            pairs.append({"eval_position": position, "eval_row": int(eval_row),
                          "train_row": train_row,
                          "rank": rank + 1, "similarity": similarity})
    pairs.sort(key=lambda pair: pair["similarity"], reverse=True)
    if limit and len(pairs) > limit:
        print(f"[INFO] --judge-limit {limit}: judging the {limit} most similar of "
              f"{len(pairs)} candidate pairs.")
        pairs = pairs[:limit]
    return pairs


def run_judgements(pairs: List[Dict[str, Any]], records: Sequence[Dict[str, Any]],
                   cache: JudgementCache, host: str, model: str, timeout: int,
                   think: Optional[str], workers: int) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Judge every pair, resuming from cache. Failures are logged and skipped."""
    results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    todo: List[Dict[str, Any]] = []
    for pair in pairs:
        prompt = build_judge_prompt(records[pair["train_row"]], records[pair["eval_row"]])
        pair["prompt"] = prompt
        pair["key"] = JudgementCache.key_for(model, prompt)
        cached = cache.get(pair["key"])
        if cached is not None:
            results[(pair["eval_row"], pair["train_row"])] = cached["judgement"]
        else:
            todo.append(pair)

    print(f"[INFO] LLM judge: {len(pairs)} pairs, {len(pairs) - len(todo)} already cached, "
          f"{len(todo)} to request from {model} at {host} ({workers} workers).")
    if not todo:
        return results

    state = {"done": 0, "failed": 0, "started": time.time()}
    lock = threading.Lock()

    def judge(pair: Dict[str, Any]) -> None:
        try:
            judgement = ollama_judge(pair["prompt"], host, model, timeout, think)
        except (RuntimeError, ValueError) as exc:
            # Failures skip: a long audit is expensive to restart, and an
            # un-cached failure is simply re-asked on the next run.
            with lock:
                state["failed"] += 1
                if state["failed"] <= 10:
                    print(f"[WARN] judge failed for eval row {pair['eval_row']} vs train row "
                          f"{pair['train_row']}: {exc}")
            return
        cache.put(pair["key"], pair["eval_row"], pair["train_row"], pair["rank"],
                  pair["similarity"], judgement)
        with lock:
            results[(pair["eval_row"], pair["train_row"])] = judgement
            state["done"] += 1
            if state["done"] % 25 == 0 or state["done"] == len(todo):
                elapsed = time.time() - state["started"]
                rate = state["done"] / elapsed if elapsed else 0.0
                remaining = (len(todo) - state["done"]) / rate if rate else 0.0
                print(f"[INFO] judged {state['done']}/{len(todo)} "
                      f"({rate:.2f}/s, ETA {remaining / 60:.1f} min, "
                      f"{state['failed']} failed)")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(judge, todo))

    if state["failed"]:
        print(f"[WARN] {state['failed']} pairs could not be judged and are not cached; "
              f"re-run to retry them.")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def print_report(summary: Dict[str, Any]) -> None:
    def line(label: str, value: str) -> None:
        print(f"  {label:<44} {value}")

    sizes = summary["dataset"]
    det = summary["deterministic"]
    n_eval = sizes["evaluation_audited"]  # percentages are of what was actually audited

    print("\n" + "=" * 78)
    print("DATA LEAKAGE AUDIT - Digital Twin property fill-in dataset")
    print("=" * 78)
    print(f"\nDataset      {sizes['path']}")
    line("total records", f"{sizes['total']}")
    line(f"training partition (shard '{sizes['trained_shard']}', model-seen)",
         f"{sizes['training_partition']}")
    audited = "" if not sizes.get("evaluation_sampled") else f"  (audited: {sizes['evaluation_audited']})"
    line(f"evaluation partition (shard '{sizes['held_out_shard']}', held out)",
         f"{n_eval}{audited}")
    line("split", f"train_test_split(test_size={sizes['test_size']}, seed={sizes['seed']})")

    print("\n" + "-" * 78)
    print("1. DETERMINISTIC EVIDENCE  (proofs; percentages are of audited evaluation samples)")
    print("-" * 78)
    for key, label in (
            ("row_also_in_training", "shared row indices"),
            ("exact_duplicate_record", "exact duplicate records"),
            ("same_normalised_anchor", "identical normalised anchors"),
            ("same_normalised_anchor_excluding_degenerate",
             "  ... excluding degenerate anchors"),
            ("same_loose_anchor", "identical anchors (punctuation-insensitive)"),
            ("same_answer", "identical answers (full configuration)"),
            ("same_config", "identical distinctive property-values"),
            ("same_interface", "same interface identifier")):
        entry = det[key]
        line(label, f"{entry['count']:>7}  ({entry['percent']:.2f}%)")

    corpus = summary["corpus_duplication"]
    print("\n  Duplication inside the whole file (the only way a random split could")
    print("  place the same record on both sides):")
    line("rows sharing a normalised anchor", f"{corpus['duplicate_anchor_rows']:>7}")
    line("rows sharing a full answer", f"{corpus['duplicate_answer_rows']:>7}")
    line("rows with a degenerate anchor (whole file)", f"{corpus['degenerate_anchor_rows']:>7}")

    if det["degenerate_anchor"]["count"]:
        print(f"\n  {det['degenerate_anchor']['count']} audited evaluation anchors are degenerate - a "
              "generator refusal, or under")
        print(f"  {MIN_ANCHOR_CHARS} characters. Refusals are byte-identical across unrelated topics, so "
              "they")
        print("  match each other across the split while describing nothing. That is a data")
        print("  defect, not memorisation: the rows are unanswerable either way, and they are")
        print("  excluded from the identical-anchor line above and from the LLM judge.")

    per_interface = sizes["total"] / max(summary["interfaces"]["distinct_total"], 1)
    print(f"\n  Interfaces: {summary['interfaces']['distinct_total']} distinct over {sizes['total']} "
          f"records ({per_interface:.2f} records per interface).")
    if per_interface < 1.5:
        print("  A shared interface identifier is therefore rare here rather than routine, so")
        print("  the pairs on that line are worth reading individually - but a shared type is")
        print("  still not leakage on its own; only shared instances and values are.")
    else:
        print("  Sharing an interface identifier is expected by design and is NOT leakage; it")
        print("  is reported because it bounds how much of the answer is a type label.")

    semantic = summary.get("semantic")
    print("\n" + "-" * 78)
    print("2. SEMANTIC NEIGHBOURHOOD  (embedding geometry; descriptive, not a verdict)")
    print("-" * 78)
    if not semantic:
        print("  skipped (--deterministic-only)")
    else:
        line("encoder", semantic["embed_model"])
        line("index", f"FAISS IndexFlatIP, {semantic['index_size']} training vectors, "
                      f"top-{semantic['top_k']}")
        stats = semantic["nearest_neighbour_similarity"]
        print("\n  Cosine similarity to the NEAREST training anchor:")
        for key, label in (("mean", "mean"), ("median", "median"), ("p95", "95th percentile"),
                           ("p99", "99th percentile"), ("max", "maximum")):
            line(label, f"{stats[key]:.4f}")
        print("\n  Evaluation samples whose nearest training anchor is at or above:")
        for threshold, count in sorted(semantic["threshold_counts"].items(),
                                       key=lambda item: float(item[0])):
            line(f"cosine >= {float(threshold):.2f}",
                 f"{count:>7}  ({percent(count, semantic['evaluation_audited']):.2f}%)")

    judge = summary.get("llm_judge")
    print("\n" + "-" * 78)
    print("3. LLM-ASSISTED SEMANTIC JUDGEMENT  (supporting evidence, NOT proof)")
    print("-" * 78)
    if not judge:
        print("  skipped (--judge-mode none)")
    else:
        line("model", f"{judge['model']} @ {judge['host']}")
        line("selection", f"--judge-mode {judge['mode']}" + (
            f", threshold {judge['threshold']}" if judge["mode"] == "threshold" else ""))
        line("pairs judged", f"{judge['pairs_judged']:>7} of {judge['pairs_selected']} selected")
        line("evaluation samples covered",
             f"{judge['eval_covered']:>7}  ({percent(judge['eval_covered'], n_eval):.2f}% of eval)")
        print()
        line("pairs flagged as leakage", f"{judge['pairs_flagged']:>7}")
        line("evaluation samples flagged",
             f"{judge['eval_flagged']:>7}  ({percent(judge['eval_flagged'], judge['eval_covered']):.2f}% of covered, "
             f"{percent(judge['eval_flagged'], n_eval):.2f}% of eval)")
        for key, label in (("same_underlying_instance", "  ... same underlying instance"),
                           ("near_duplicate_description", "  ... near-duplicate description"),
                           ("same_property_value_configuration", "  ... same property-value config"),
                           ("same_interface_semantics", "  ... same interface semantics (not leakage)")):
            line(label, f"{judge['pair_flags'][key]:>7}")
        if judge.get("mean_confidence") is not None:
            line("mean confidence (flagged pairs)", f"{judge['mean_confidence']:.2f}")

    examples = summary.get("examples") or {}
    if examples.get("deterministic") or examples.get("llm_flagged"):
        print("\n" + "-" * 78)
        print("EVIDENCE TO READ BY HAND  (row numbers are 0-based lines of the dataset)")
        print("-" * 78)
        for item in examples.get("deterministic", []):
            note = "  [degenerate anchor]" if item.get("degenerate_anchor") else ""
            print(f"  eval {item['eval_row']:>6}  {item['category']:<34} "
                  f"training rows {item['train_rows']}{note}")
        if examples.get("deterministic") and examples.get("llm_flagged"):
            print()
        for item in examples.get("llm_flagged", []):
            print(f"  eval {item['eval_row']:>6} <- train {item['train_row']:>6}  "
                  f"cos {item['similarity']:.3f}  conf {item['confidence']}")
            print(f"          {item['reason'][:140]}")
        if examples.get("truncated"):
            print(f"\n  ... {examples['truncated']} more; the full set is in audit.jsonl "
                  "and llm-cache.jsonl.")

    print("\n" + "-" * 78)
    print("READING THIS REPORT")
    print("-" * 78)
    print("  Section 1 is decisive: those counts are computed, not estimated.")
    print("  Sections 2 and 3 can only ever say 'this method found nothing'. The judge")
    print("  is an LLM with no error bound - it was shown the pairs an embedding model")
    print("  called similar, so a duplicate phrased unlike its twin can be missed, and")
    print("  a flagged pair still has to be read by a human before it means anything.")
    print(f"\n  Per-sample detail: {summary['outputs']['audit']}")
    print(f"  Summary JSON:      {summary['outputs']['summary']}")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit train/evaluation leakage in the fill-in dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="JSONL of {anchor, answer} records")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Directory for audit outputs")
    parser.add_argument("--test-size", type=float, default=SPLIT_TEST_SIZE,
                        help="Fraction held out by train_test_split (must match training)")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Split seed")
    parser.add_argument("--trained-shard", choices=["train", "test"], default=TRAINED_SHARD,
                        help="Which shard the released model trained on")
    parser.add_argument("--limit", type=int, default=0,
                        help="Audit only the first N evaluation samples in shard order "
                             "(a uniform random subsample; 0 = all)")
    parser.add_argument("--deterministic-only", action="store_true",
                        help="Skip embeddings, FAISS and the LLM entirely")

    group = parser.add_argument_group("semantic neighbourhood")
    group.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                       help="SentenceTransformer name or path")
    group.add_argument("--batch-size", type=int, default=256, help="Encoder batch size")
    group.add_argument("--device", default=None, help="Encoder device (default: auto)")
    group.add_argument("--top-k", type=int, default=5, help="Neighbours retrieved per sample")
    group.add_argument("--no-embedding-cache", action="store_true",
                       help="Do not read or write the cached embeddings")

    group = parser.add_argument_group("LLM judge")
    group.add_argument("--judge-mode", choices=["threshold", "top1", "all", "none"],
                       default="threshold", help="Which pairs to send to the judge")
    # 0.85, not the 0.90 that the similarity distribution suggests: the first
    # confirmed same-instance pair found here (one `fan-01` written up under two
    # topics, same speed envelope, same maintenance date) sits at 0.809. Paraphrase
    # across topics scores lower than boilerplate agreement within one, so a high
    # floor filters out the leaks and keeps the coincidences. Below ~0.80 the
    # candidate count explodes into the tens of thousands with nothing to show for
    # it; --judge-mode top1 is the thorough option, not a lower floor.
    group.add_argument("--judge-threshold", type=float, default=0.85,
                       help="Cosine floor for --judge-mode threshold")
    group.add_argument("--judge-limit", type=int, default=0,
                       help="Cap on judged pairs, most similar first (0 = no cap)")
    # More is not faster. Measured against one gpt-oss:20b on this server: 2 and 4
    # workers both sustain ~0.44 pairs/s, 8 drops to 0.39, and 12 collapses to
    # ~0.1 - a single generation slot, so extra requests only queue, and enough of
    # them make the server thrash. Raise it only against a server you have measured.
    group.add_argument("--judge-workers", type=int, default=4, help="Concurrent requests")
    group.add_argument("--host", default=OLLAMA_HOST, help="Ollama host")
    group.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model")
    group.add_argument("--timeout", type=int, default=300, help="Per-request timeout, seconds")
    group.add_argument("--think", default="low", choices=["none", "low", "medium", "high"],
                       help="Reasoning effort for models that support it")
    group.add_argument("--no-judge-cache", action="store_true",
                       help="Do not read or write llm-cache.jsonl")
    group.add_argument("--judge-degenerate", action="store_true",
                       help="Also judge pairs whose anchor is a generator refusal "
                            "(they are excluded by default; see section 1 of the report)")
    group.add_argument("--neighbour-text-chars", type=int, default=240,
                       help="Neighbour anchor characters kept in audit.jsonl (0 = all)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset not found: {args.data}")
        return 2
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.data)
    if not records:
        print(f"[ERROR] {args.data} holds no records.")
        return 2
    shards = split_indices(len(records), args.test_size, args.seed)
    held_out = "train" if args.trained_shard == "test" else "test"
    train_rows = shards[args.trained_shard]
    eval_rows_all = shards[held_out]
    eval_rows = eval_rows_all[:args.limit] if args.limit else eval_rows_all

    print(f"[INFO] {len(records)} records from {args.data}")
    print(f"[INFO] train_test_split(test_size={args.test_size}, seed={args.seed}): "
          f"training partition = shard '{args.trained_shard}' ({len(train_rows)} rows, "
          f"model-seen), evaluation partition = shard '{held_out}' "
          f"({len(eval_rows_all)} rows, held out)")
    if args.limit:
        print(f"[INFO] --limit {args.limit}: auditing {len(eval_rows)} evaluation samples. "
              "A shard is in permutation order, so this is a uniform subsample - but the "
              "training side is never sub-sampled, or overlap would be understated.")

    # --- 1. deterministic --------------------------------------------------
    started = time.time()
    flags, det_summary = deterministic_audit(records, train_rows, eval_rows)
    corpus = whole_corpus_duplicates(records)
    interfaces = Counter(interface_of(record) for record in records)
    print(f"[INFO] Deterministic checks finished in {time.time() - started:.1f}s")

    summary: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "path": args.data,
            "total": len(records),
            "training_partition": len(train_rows),
            "evaluation_partition": len(eval_rows_all),
            "evaluation_audited": len(eval_rows),
            "evaluation_sampled": bool(args.limit),
            "test_size": args.test_size,
            "seed": args.seed,
            "trained_shard": args.trained_shard,
            "held_out_shard": held_out,
        },
        "deterministic": det_summary,
        "corpus_duplication": corpus,
        "interfaces": {
            "distinct_total": len(interfaces),
            "distinct_in_training": len({interface_of(records[r]) for r in train_rows}),
            "distinct_in_evaluation": len({interface_of(records[r]) for r in eval_rows}),
        },
        "outputs": {
            "audit": os.path.join(args.out_dir, "audit.jsonl"),
            "summary": os.path.join(args.out_dir, "summary.json"),
        },
    }

    # The rows a human has to look at, rather than only the counts. An audit that
    # says "13 samples share a configuration" without naming them cannot be checked.
    deterministic_examples: List[Dict[str, Any]] = []
    for position, eval_row in enumerate(eval_rows):
        flag = flags[position]
        for key, label in (("exact_duplicate_record", "exact duplicate record"),
                           ("same_answer", "identical answer"),
                           ("same_normalised_anchor", "identical normalised anchor"),
                           ("same_config", "identical property-value config"),
                           ("same_interface", "same interface identifier")):
            if flag[key]:
                deterministic_examples.append({
                    "eval_row": eval_row, "category": label,
                    "train_rows": flag["witnesses"].get(key, [])[:3],
                    "degenerate_anchor": flag["degenerate_anchor"],
                })
                break  # one line per sample: the strongest category it hit
    summary["examples"] = {"deterministic": deterministic_examples[:10],
                           "deterministic_total": len(deterministic_examples),
                           "truncated": max(0, len(deterministic_examples) - 10)}

    # --- 2. semantic neighbourhood ----------------------------------------
    similarities = indices = None
    if not args.deterministic_only:
        import numpy as np

        train_texts = [anchor_of(records[row]) for row in train_rows]
        eval_texts = [anchor_of(records[row]) for row in eval_rows]
        # Encoded one partition at a time, each cached under a fingerprint of its
        # own text list, so a re-run with a different --limit still reuses the
        # training side - which is the expensive half to keep whole.
        train_emb, _ = encode_anchors(train_texts, args.embed_model, args.out_dir,
                                      args.batch_size, args.device,
                                      not args.no_embedding_cache)
        eval_emb, _ = encode_anchors(eval_texts, args.embed_model, args.out_dir,
                                     args.batch_size, args.device,
                                     not args.no_embedding_cache)
        similarities, indices = nearest_neighbours(train_emb, eval_emb, args.top_k)
        np.save(os.path.join(args.out_dir, "nn-similarities.npy"), similarities)
        np.save(os.path.join(args.out_dir, "nn-indices.npy"),
                np.asarray([[train_rows[int(j)] for j in row] for row in indices], dtype="int64"))

        top1 = [float(row[0]) for row in similarities]
        thresholds = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99)
        summary["semantic"] = {
            "embed_model": args.embed_model,
            "index_size": int(train_emb.shape[0]),
            "top_k": int(indices.shape[1]),
            "evaluation_audited": len(eval_rows),
            "nearest_neighbour_similarity": similarity_stats(top1),
            "all_neighbour_similarity": similarity_stats(
                [float(value) for row in similarities for value in row]),
            "threshold_counts": {f"{t:.2f}": int(sum(1 for s in top1 if s >= t))
                                 for t in thresholds},
            "outputs": {
                "similarities": os.path.join(args.out_dir, "nn-similarities.npy"),
                "indices": os.path.join(args.out_dir, "nn-indices.npy"),
            },
        }

    # --- 3. LLM judge ------------------------------------------------------
    judgements: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not args.deterministic_only and args.judge_mode != "none":
        degenerate_rows = set()
        if not args.judge_degenerate:
            degenerate_rows = {row for row in list(train_rows) + list(eval_rows)
                               if is_degenerate_anchor(anchor_of(records[row]))}
        pairs = select_judge_pairs(args.judge_mode, similarities, indices, eval_rows,
                                   train_rows, args.judge_threshold, args.judge_limit,
                                   degenerate_rows)
        cache = JudgementCache(os.path.join(args.out_dir, "llm-cache.jsonl"), args.model,
                               enabled=not args.no_judge_cache)
        try:
            judgements = run_judgements(pairs, records, cache, args.host, args.model,
                                        args.timeout, None if args.think == "none" else args.think,
                                        args.judge_workers)
        finally:
            cache.close()

        covered = {pair["eval_row"] for pair in pairs
                   if (pair["eval_row"], pair["train_row"]) in judgements}
        flagged_pairs = [(key, verdict) for key, verdict in judgements.items()
                         if verdict.get("leakage")]
        confidences = [v["confidence"] for _, v in flagged_pairs if v.get("confidence") is not None]
        summary["llm_judge"] = {
            "host": args.host,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "mode": args.judge_mode,
            "threshold": args.judge_threshold,
            "pairs_selected": len(pairs),
            "pairs_judged": len(judgements),
            "eval_covered": len(covered),
            "pairs_flagged": len(flagged_pairs),
            "eval_flagged": len({eval_row for (eval_row, _), _ in flagged_pairs}),
            "eval_flagged_percent_of_evaluation": percent(
                len({eval_row for (eval_row, _), _ in flagged_pairs}), len(eval_rows)),
            "pair_flags": {key: sum(1 for verdict in judgements.values() if verdict.get(key))
                           for key in JUDGE_KEYS[:4]},
            "mean_confidence": statistics.fmean(confidences) if confidences else None,
            "degenerate_rows_excluded": len(degenerate_rows),
            "cache": os.path.join(args.out_dir, "llm-cache.jsonl"),
        }

        # Most confident first, then most similar: the order a reviewer should
        # read them in, since the judge's own doubt is the best triage signal.
        flagged = sorted(
            ({"eval_row": pair["eval_row"], "train_row": pair["train_row"],
              "similarity": pair["similarity"],
              "confidence": judgements[(pair["eval_row"], pair["train_row"])].get("confidence"),
              "reason": judgements[(pair["eval_row"], pair["train_row"])].get("reason", "")}
             for pair in pairs
             if judgements.get((pair["eval_row"], pair["train_row"]), {}).get("leakage")),
            key=lambda item: (-(item["confidence"] or 0.0), -item["similarity"]))
        summary["examples"]["llm_flagged"] = flagged[:10]
        summary["examples"]["llm_flagged_total"] = len(flagged)
        summary["examples"]["truncated"] += max(0, len(flagged) - 10)

    # --- audit file --------------------------------------------------------
    audit_path = summary["outputs"]["audit"]
    keep = args.neighbour_text_chars
    with open(audit_path, "w", encoding="utf-8") as handle:
        for position, eval_row in enumerate(eval_rows):
            record = records[eval_row]
            neighbours = []
            if similarities is not None:
                for rank in range(indices.shape[1]):
                    train_position = int(indices[position][rank])
                    if train_position < 0:
                        continue
                    train_row = int(train_rows[train_position])
                    train_record = records[train_row]
                    text = anchor_of(train_record)
                    neighbours.append({
                        "rank": rank + 1,
                        "train_row": train_row,
                        "cosine_similarity": round(float(similarities[position][rank]), 6),
                        "interface": interface_of(train_record),
                        "anchor": text if keep <= 0 else text[:keep],
                        "properties": distinctive_config(train_record.get("answer") or {}),
                        "llm_judgement": judgements.get((eval_row, train_row)),
                    })
            handle.write(json.dumps({
                "eval_row": eval_row,
                "interface": interface_of(record),
                "anchor": anchor_of(record),
                "answer": record.get("answer"),
                "deterministic": flags[position],
                "max_cosine_similarity": neighbours[0]["cosine_similarity"] if neighbours else None,
                "neighbours": neighbours,
                "llm_leakage": (None if not neighbours or all(
                    n["llm_judgement"] is None for n in neighbours)
                    else any(bool((n["llm_judgement"] or {}).get("leakage")) for n in neighbours)),
            }, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    with open(summary["outputs"]["summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    print_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
