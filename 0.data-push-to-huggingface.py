#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform the local JSONL datasets into HuggingFace format and push them to a SINGLE
dataset repo, one `config` per dataset.

    python 0.data-push-to-huggingface.py --repo-id <user>/digital-twin-composition --dry-run
    python 0.data-push-to-huggingface.py --repo-id <user>/digital-twin-composition

Configs produced (split name in brackets):

    triplets    [train]  query / positive / negative      <- triplet.jsonl
    interfaces  [train]  DTDL interface catalogue          <- interfaces.jsonl
    fill_eval   [train]  anchor text + filled answer       <- fill-eval.jsonl
    topics      [train]  topic id / title / brief          <- topics.jsonl
    eval_small  [test]   composed end-to-end queries       <- dataset_small.jsonl
    eval_mid    [test]   composed queries + interface      <- dataset_mid.jsonl

Nested DTDL is stored as JSON *strings* (`contents`, `answer`, `interface`). That is
deliberate: `contents` mixes dicts with bare strings, `schema` is sometimes a dict, and
every `fill-eval` answer has a different key set, so Arrow cannot infer one struct type
for them. Every config stays lossless — `json.loads` the column to get the original value.
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional

from datasets import Dataset, Features, Value

DEFAULT_DATA_DIR = "data"
DEFAULT_REPO_ID = "digital-twin-composition"


# ---------------------------
# Helpers
# ---------------------------

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno} skipped, invalid JSON: {e}", file=sys.stderr)


def dumps(value: Any) -> str:
    """Serialize a nested value to a compact JSON string ('' for missing)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def topic_of(interface_id: str) -> str:
    """dtmi:<topic>:<Name>;1 -> <topic>"""
    parts = (interface_id or "").split(":")
    return parts[1] if len(parts) > 2 else ""


def s(value: Any) -> str:
    return "" if value is None else str(value)


# ---------------------------
# Per-dataset row transforms
# ---------------------------

def row_triplet(r: Dict[str, Any]) -> Dict[str, str]:
    # `positive` / `negative` are already JSON strings produced by the triplet generator.
    return {
        "query": s(r.get("query")),
        "positive": dumps(r.get("positive")),
        "negative": dumps(r.get("negative")),
    }


def row_interface(r: Dict[str, Any]) -> Dict[str, str]:
    iid = s(r.get("@id"))
    return {
        "id": iid,
        "topic": topic_of(iid),
        "display_name": s(r.get("displayName")),
        "description": s(r.get("description")),
        "context": s(r.get("@context")),
        "type": s(r.get("@type")),
        "contents": dumps(r.get("contents")),
    }


def row_fill_eval(r: Dict[str, Any]) -> Dict[str, str]:
    answer = r.get("answer") or {}
    iid = s(answer.get("interface")) if isinstance(answer, dict) else ""
    return {
        "anchor": s(r.get("anchor")),
        "interface_id": iid,
        "topic": topic_of(iid),
        "answer": dumps(answer),
    }


def row_topic(r: Dict[str, Any]) -> Dict[str, str]:
    return {"id": s(r.get("id")), "title": s(r.get("title")), "brief": s(r.get("brief"))}


def row_eval_small(r: Dict[str, Any]) -> Dict[str, str]:
    return {"query": s(r.get("query")), "group_id": s(r.get("group_id"))}


def row_eval_mid(r: Dict[str, Any]) -> Dict[str, str]:
    iface = r.get("interface") or {}
    # Half the rows carry {"group_id": ...}, half a full DTDL interface.
    group_id = iface.get("group_id") if isinstance(iface, dict) else None
    if not group_id and isinstance(iface, dict):
        group_id = topic_of(s(iface.get("@id")))
    return {
        "query": s(r.get("query")),
        "group_id": s(group_id),
        "interface": dumps(iface),
        "expected_output": s(r.get("expected_output")),
    }


class Config:
    def __init__(self, name: str, source: str, split: str,
                 transform: Callable[[Dict[str, Any]], Dict[str, str]],
                 columns: List[str], description: str):
        self.name = name
        self.source = source
        self.split = split
        self.transform = transform
        self.columns = columns
        self.description = description

    @property
    def features(self) -> Features:
        # Everything is a string: the nested columns are JSON text, and uniform typing
        # keeps Arrow from guessing at the heterogeneous DTDL payloads.
        return Features({c: Value("string") for c in self.columns})


CONFIGS: List[Config] = [
    Config("triplets", "triplet.jsonl", "train", row_triplet,
           ["query", "positive", "negative"],
           "Retrieval triplets: natural-language query, correct interface, different-topic negative."),
    Config("interfaces", "interfaces.jsonl", "train", row_interface,
           ["id", "topic", "display_name", "description", "context", "type", "contents"],
           "The DTDL interface catalogue every other config refers to."),
    Config("fill_eval", "fill-eval.jsonl", "train", row_fill_eval,
           ["anchor", "interface_id", "topic", "answer"],
           "Property-filling evaluation: a spec paragraph and the filled interface it implies."),
    Config("topics", "topics.jsonl", "train", row_topic,
           ["id", "title", "brief"],
           "Digital-twin topics; `id` is the middle segment of every dtmi interface id."),
    Config("eval_small", "dataset_small.jsonl", "test", row_eval_small,
           ["query", "group_id"],
           "Composed end-to-end system queries, grouped by topic."),
    Config("eval_mid", "dataset_mid.jsonl", "test", row_eval_mid,
           ["query", "group_id", "interface", "expected_output"],
           "Composed queries paired with a topic reference or a full interface."),
]


def build(cfg: Config, data_dir: str, limit: int = 0) -> Optional[Dataset]:
    path = os.path.join(data_dir, cfg.source)
    if not os.path.exists(path):
        print(f"[WARN] {cfg.name}: {path} not found, skipping")
        return None

    rows: List[Dict[str, str]] = []
    for r in read_jsonl(path):
        rows.append(cfg.transform(r))
        if limit and len(rows) >= limit:
            break
    if not rows:
        print(f"[WARN] {cfg.name}: no rows in {path}, skipping")
        return None

    ds = Dataset.from_list(rows, features=cfg.features, split=cfg.split)
    print(f"[OK]   {cfg.name:<11} {len(ds):>6} rows  split={cfg.split}  columns={cfg.columns}")
    return ds


def verify(cfg: Config, ds: Dataset) -> None:
    """Confirm the JSON-string columns still parse back into their original values."""
    json_cols = [c for c in ("positive", "negative", "contents", "answer", "interface")
                 if c in ds.column_names]
    if not json_cols:
        return
    checked = 0
    for row in ds.select(range(min(200, len(ds)))):
        for col in json_cols:
            value = row[col]
            if value:
                json.loads(value)  # raises if we mangled it
                checked += 1
    print(f"       round-trip ok ({checked} JSON values parsed across {json_cols})")


def report_integrity(built: List[tuple]) -> Dict[str, int]:
    """Print how well the configs cross-reference each other, so surprises show up here.

    The numbers are returned so the dataset card documents what was actually pushed
    instead of hard-coded figures that drift as the sources are regenerated.
    """
    by_name = {cfg.name: ds for cfg, ds in built}
    stats: Dict[str, int] = {}
    ifs, tops = by_name.get("interfaces"), by_name.get("topics")
    if ifs is None:
        return stats

    print("\n[INFO] Cross-config integrity")
    iface_ids = set(ifs["id"])
    if tops is not None:
        topic_ids = set(tops["id"])
        iface_topics = set(ifs["topic"])
        orphan = iface_topics - topic_ids
        stats["orphan_topics"] = len(orphan)
        stats["orphan_topic_rows"] = sum(1 for t in ifs["topic"] if t in orphan)
        stats["topics_without_interface"] = len(topic_ids - iface_topics)
        if orphan:
            print(f"       {len(orphan)} interface topic(s) absent from topics "
                  f"({stats['orphan_topic_rows']} rows): {sorted(orphan)[:5]}")
        print(f"       topics with no interface: {stats['topics_without_interface']}")

    tri = by_name.get("triplets")
    if tri is not None:
        stats["triplet_positive_orphans"] = sum(
            1 for p in tri["positive"] if json.loads(p).get("@id") not in iface_ids)
        stats["triplet_negative_orphans"] = sum(
            1 for n in tri["negative"] if json.loads(n).get("@id") not in iface_ids)
        print(f"       triplet positives not in interfaces: {stats['triplet_positive_orphans']}")
        print(f"       triplet negatives not in interfaces: {stats['triplet_negative_orphans']}"
              " (negatives are stored inline, so this is a lookup gap, not missing data)")

    fe = by_name.get("fill_eval")
    if fe is not None:
        stats["fill_eval_orphans"] = len({i for i in fe["interface_id"] if i and i not in iface_ids})
        print(f"       fill_eval interface_ids not in interfaces: {stats['fill_eval_orphans']}")
    return stats


# ---------------------------
# Dataset card
# ---------------------------

def card_text(repo_id: str, built: List[tuple]) -> str:
    """Describe what the dataset contains. Integrity findings stay on the console."""
    lines = [
        "# Digital Twin Composition",
        "",
        "Datasets for retrieving and filling [DTDL](https://github.com/Azure/opendigitaltwins-dtdl)",
        "digital-twin interfaces from natural-language requests. All parts live in this one repo",
        "as separate configs.",
        "",
        "## Configs",
        "",
        "| config | split | rows | description |",
        "| --- | --- | --- | --- |",
    ]
    for cfg, ds in built:
        lines.append(f"| `{cfg.name}` | {cfg.split} | {len(ds):,} | {cfg.description} |")
    lines += [
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'triplets = load_dataset("{repo_id}", "triplets", split="train")',
        f'interfaces = load_dataset("{repo_id}", "interfaces", split="train")',
        "```",
        "",
        "## Nested fields are JSON strings",
        "",
        "`positive`, `negative`, `contents`, `answer` and `interface` hold JSON **text**, because",
        "DTDL payloads are not uniformly typed — `contents` mixes objects with bare strings, a",
        "`schema` may be a string or an object, and every `fill_eval` answer has its own key set.",
        "Storing them as strings keeps the data lossless; parse a column to get the original value:",
        "",
        "```python",
        "import json",
        'interface = json.loads(triplets[0]["positive"])',
        'print(interface["@id"])  # dtmi:<topic>:<Name>;1',
        "```",
        "",
        "## How the parts join",
        "",
        "Every interface id has the form `dtmi:<topic>:<Name>;1`, and `<topic>` is normally a",
        "`topics.id`. `fill_eval.interface_id` and the `id` in a parsed `triplets.positive` both",
        "resolve to `interfaces.id`. The `topic` columns are precomputed from the id for convenience.",
        "",
        "`interfaces`, `fill_eval` and `triplets` are aligned one-to-one: same row count, and the",
        "same multiset of interface ids. Every `triplets` positive and negative resolves to a row",
        "in `interfaces`.",
        "",
        "## Provenance",
        "",
        "Generated by the scripts in the `digital-twin-composition` project: interfaces and topics",
        "are LLM-synthesised, triplet negatives are sampled from a different topic and verified by",
        "an LLM judge plus a lexical near-duplicate filter.",
    ]
    return "\n".join(lines) + "\n"


def push_card(repo_id: str, built: List[tuple], token: Optional[str]) -> None:
    """Write the prose card while preserving the YAML config block push_to_hub generated."""
    from huggingface_hub import DatasetCard

    try:
        card = DatasetCard.load(repo_id, token=token, repo_type="dataset")
    except Exception as e:  # noqa: BLE001 - a missing card is fine, we make one
        print(f"[WARN] Could not load existing card ({e}); creating a fresh one")
        card = DatasetCard("")
    card.text = card_text(repo_id, built)
    card.push_to_hub(repo_id, token=token, repo_type="dataset")
    print(f"[OK]   dataset card pushed ({len(built)} configs documented)")


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help="Target dataset repo, e.g. user/digital-twin-composition")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory holding the source JSONL files")
    ap.add_argument("--configs", nargs="*", default=None,
                    help=f"Subset to process (default: all) from: {[c.name for c in CONFIGS]}")
    ap.add_argument("--limit", type=int, default=0, help="Only take the first N rows per config (smoke tests)")
    ap.add_argument("--private", action="store_true", default=True, help="Create the repo private (default)")
    ap.add_argument("--public", dest="private", action="store_false", help="Create the repo public")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (default: HF_TOKEN or cached login)")
    ap.add_argument("--save-to-disk", metavar="DIR", default=None,
                    help="Also write each config locally as parquet")
    ap.add_argument("--dry-run", action="store_true", help="Build and verify locally, push nothing")
    ap.add_argument("--no-card", action="store_true", help="Skip writing the dataset card")
    args = ap.parse_args()

    selected = CONFIGS
    if args.configs:
        unknown = set(args.configs) - {c.name for c in CONFIGS}
        if unknown:
            print(f"[ERROR] Unknown config(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = [c for c in CONFIGS if c.name in args.configs]

    print(f"[INFO] Source directory: {os.path.abspath(args.data_dir)}")
    print(f"[INFO] Building {len(selected)} config(s)\n")

    built = []
    for cfg in selected:
        ds = build(cfg, args.data_dir, args.limit)
        if ds is None:
            continue
        verify(cfg, ds)
        built.append((cfg, ds))

    if not built:
        print("[ERROR] Nothing to push - no source files were readable.", file=sys.stderr)
        return 1

    report_integrity(built)  # console-only sanity check; the card documents contents, not findings

    if args.save_to_disk:
        os.makedirs(args.save_to_disk, exist_ok=True)
        for cfg, ds in built:
            out = os.path.join(args.save_to_disk, f"{cfg.name}-{cfg.split}.parquet")
            ds.to_parquet(out)
            print(f"[OK]   wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    total = sum(len(ds) for _, ds in built)
    if args.dry_run:
        print(f"\n[DRY-RUN] {len(built)} config(s), {total:,} rows ready. "
              f"Would push to '{args.repo_id}' ({'private' if args.private else 'PUBLIC'}).")
        return 0

    visibility = "private" if args.private else "PUBLIC"
    print(f"\n[INFO] Pushing {len(built)} config(s), {total:,} rows to '{args.repo_id}' ({visibility})")
    for cfg, ds in built:
        print(f"[INFO] -> {cfg.name} ({len(ds):,} rows)")
        ds.push_to_hub(
            args.repo_id,
            config_name=cfg.name,
            split=cfg.split,
            private=args.private,
            token=args.token,
        )

    if not args.no_card:
        push_card(args.repo_id, built, args.token)

    print(f"\n[DONE] https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
