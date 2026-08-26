#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the fill-in fine-tuning dataset from `fill-eval.jsonl` + `interfaces.jsonl`.

    python 0.data-gen-fill-ft-dataset.py --output data/llm-fill-ft-80-20-iface.ds \
        --like data/llm-fill-ft-80-20.ds

One row per `fill-eval.jsonl` record, in the `save_to_disk` layout the stage-1 and
stage-2 scripts read:

    row           0-based line index in fill-eval.jsonl - the join key back to the source
    prompt        the flat extraction prompt handed to the model
    ground_truth  the expected answer object, minified JSON
    interface     the DTDL interface @id
    n_fields      number of keys in ground_truth

Two prompt formats
------------------
`fields` is what `data/llm-fill-ft-80-20.ds` shipped with, and is
`dependencies/fill_eval_runner.build_prompt` character-for-character: the interface id
inline plus a flat `- "name" (schema)` list of the fields to fill.

`interface` (default) replaces that list with the interface itself, so the model reads
the DTDL definition rather than a pre-digested field list - the same way
`3.system-eval.py` and `4.deploy.py` hand an interface to a fill-in prompt. What counts
as "the interface" is `--interface-content`:

    full             the interfaces.jsonl object unchanged - dockerImage and Telemetry
                     included, so the model has to work out which fields are fillable
    no-docker        the same, minus the dockerImage Property, whose value the anchor is
                     forbidden to mention
    properties-only  minus dockerImage and minus every Telemetry, so the keys in the
                     prompt are exactly the keys the answer is scored on

Ground truth
------------
A Property is expected of the model when it is not `dockerImage`, is present in the
record's `answer`, and its value is not a numeric zero. That last clause is not a
heuristic bolted on here - it is what the shipped dataset does, and it matters because
`0.data-gen-fill.py` zeroes Telemetry into the answer *after* writing the properties, so
a name declared as both Property and Telemetry arrives as a 0 the anchor never states.
Asking for it would only measure guessing. Reproduces the shipped `ground_truth` and
`n_fields` on all 27,770 rows.

Splitting
---------
`--like <dataset dir>` reuses the split of an existing build: each row keeps whichever
split held its `row` index, in the same order. That is what makes a rebuild comparable
with the run it replaces - the same 5,554 rows stay held out, so prediction files under
`results/` stay line-aligned with the new `test`. Without it, a fresh shuffled split is
drawn from `--test-size` and `--seed`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

FIELDS_PROMPT = "fields"
INTERFACE_PROMPT = "interface"
INTERFACE_CONTENTS = ("full", "no-docker", "properties-only")


def minified(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_interfaces(catalogue: List[Dict[str, Any]],
                       records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """The interface each record was generated from, aligned with `records`.

    Paired by line, not by `@id`: `0.data-gen-fill.py` walks interfaces.jsonl top to
    bottom writing one record per line, and the catalogue carries three ids twice with
    different bodies, so an `@id` lookup would hand one of each pair the other's
    definition. When the two files are no longer aligned - a filtered or partial
    fill-eval.jsonl - it falls back to an `@id` lookup, which is exact for every id but
    those three.
    """
    wanted = [(r.get("answer") or {}).get("interface") for r in records]
    if len(catalogue) == len(records) and all(
            iface.get("@id") == want for iface, want in zip(catalogue, wanted)):
        return catalogue, "line order"

    by_id: Dict[str, Dict[str, Any]] = {}
    for iface in catalogue:
        interface_id = iface.get("@id")
        if interface_id:
            by_id.setdefault(interface_id, iface)
    missing = sorted({w for w in wanted if w not in by_id}, key=str)
    if missing:
        raise LookupError(f"{len(missing)} interface id(s) referenced by the records are "
                          f"not in the catalogue, e.g. {missing[:3]}")
    return [by_id[w] for w in wanted], "@id lookup"


def property_spec(interface: Dict[str, Any]) -> List[Dict[str, str]]:
    """The Property fields a model is asked to fill: everything but `dockerImage`."""
    spec = []
    for content in interface.get("contents", []) or []:
        if not isinstance(content, dict) or content.get("@type") != "Property":
            continue
        name = content.get("name")
        if not name or name == "dockerImage":
            continue
        spec.append({"name": name, "schema": content.get("schema", "string")})
    return spec


def interface_for_prompt(interface: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "full":
        return interface
    drop_telemetry = mode == "properties-only"
    contents = []
    for content in interface.get("contents", []) or []:
        if not isinstance(content, dict):
            contents.append(content)
            continue
        if content.get("@type") == "Property" and content.get("name") == "dockerImage":
            continue
        if drop_telemetry and content.get("@type") == "Telemetry":
            continue
        contents.append(content)
    return {**interface, "contents": contents}


def build_prompt_fields(anchor: str, interface_id: str, spec: List[Dict[str, str]]) -> str:
    """`dependencies/fill_eval_runner.build_prompt`, kept identical on purpose."""
    fields_desc = "\n".join(f'- "{p["name"]}" ({p["schema"]})' for p in spec)
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


def build_prompt_interface(anchor: str, interface_text: str) -> str:
    return (
        "You are an information extraction assistant.\n"
        "Extract values exactly from the ANCHOR text to fill ONLY the fields "
        "for the interface.\n"
        "If a value is not stated, use null.\n"
        "Do not infer unstated facts.\n"
        "Return ONLY minified JSON (no markdown, no comments), "
        "with EXACT keys listed below.\n\n"
        "Interface:\n"
        f"{interface_text}\n\n"
        "ANCHOR:\n"
        f"{anchor}\n"
    )


def expected_answer(answer: Dict[str, Any], spec: List[Dict[str, str]]) -> Dict[str, Any]:
    """The keys the model is scored on. See "Ground truth" in the module docstring."""
    expected = {}
    for field in spec:
        name = field["name"]
        if name not in answer:
            continue
        value = answer[name]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
            continue
        expected[name] = value
    return expected


def split_assignment(path: Path) -> Dict[str, List[int]]:
    """Split name -> its `row` indices, in the order that build laid them down."""
    from datasets import load_from_disk

    existing = load_from_disk(str(path))
    if not hasattr(existing, "items"):  # a single-split save is a bare Dataset
        raise SystemExit(f"[ERROR] {path} is a single Dataset, not a DatasetDict; "
                         f"--like needs the splits to mirror")
    order: Dict[str, List[int]] = {}
    for name, ds in existing.items():
        if "row" not in ds.column_names:
            raise SystemExit(f"[ERROR] {path}[{name}] has no `row` column, so its split "
                             f"cannot be mirrored: {ds.column_names}")
        order[name] = list(ds["row"])
    return order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fill-eval", default=str(Path("data") / "fill-eval.jsonl"),
                    help='Source records, one {"anchor": ..., "answer": {...}} per line')
    ap.add_argument("--interfaces", default=str(Path("data") / "interfaces.jsonl"),
                    help="DTDL interfaces the field spec and prompt body are read from")
    ap.add_argument("--output", required=True, help="save_to_disk directory to write")
    ap.add_argument("--prompt", choices=(FIELDS_PROMPT, INTERFACE_PROMPT),
                    default=INTERFACE_PROMPT, help="Prompt format (default: interface)")
    ap.add_argument("--interface-content", choices=INTERFACE_CONTENTS, default="full",
                    help="How much of the interface the `interface` prompt carries "
                         "(default: full)")
    ap.add_argument("--like", default=None,
                    help="Existing dataset dir whose splits are mirrored row for row; "
                         "without it a fresh split is drawn")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Held-out fraction for a fresh split (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Fresh-split shuffle seed")
    args = ap.parse_args()

    fill_eval_path, interfaces_path = Path(args.fill_eval), Path(args.interfaces)
    for path in (fill_eval_path, interfaces_path):
        if not path.exists():
            print(f"[ERROR] Not found: {path}", file=sys.stderr, flush=True)
            return 1

    records = read_jsonl(fill_eval_path)
    catalogue = read_jsonl(interfaces_path)

    # Every interface resolved before a row is built: a record pointing at one the
    # catalogue does not carry is a mismatched pair of inputs, not a bad sample.
    try:
        interfaces, how = resolve_interfaces(catalogue, records)
    except LookupError as e:
        print(f"[ERROR] {fill_eval_path} and {interfaces_path} do not match: {e}",
              file=sys.stderr, flush=True)
        return 1
    print(f"[INFO] {len(records)} records, {len(catalogue)} interfaces, paired by {how}",
          flush=True)

    rows: Dict[int, Dict[str, Any]] = {}
    for index, (record, interface) in enumerate(zip(records, interfaces)):
        answer = record.get("answer") or {}
        interface_id = interface["@id"]
        spec = property_spec(interface)
        anchor = record.get("anchor", "")
        prompt = (build_prompt_fields(anchor, interface_id, spec)
                  if args.prompt == FIELDS_PROMPT
                  else build_prompt_interface(
                      anchor,
                      minified(interface_for_prompt(interface, args.interface_content))))
        expected = expected_answer(answer, spec)
        rows[index] = {
            "row": index,
            "prompt": prompt,
            "ground_truth": minified(expected),
            "interface": interface_id,
            "n_fields": len(expected),
        }

    from datasets import Dataset, DatasetDict

    if args.like:
        order = split_assignment(Path(args.like))
        unknown = sorted({r for rs in order.values() for r in rs} - set(rows))
        if unknown:
            print(f"[ERROR] {args.like} references {len(unknown)} row index(es) that "
                  f"{fill_eval_path} does not have, e.g. {unknown[:3]}; the two were "
                  f"built from different sources", file=sys.stderr, flush=True)
            return 1
        dropped = len(rows) - sum(len(rs) for rs in order.values())
        if dropped:
            print(f"[WARN] {dropped} record(s) are absent from {args.like} and are left "
                  f"out too, so the two builds stay row-for-row comparable", flush=True)
        splits = {name: [rows[r] for r in row_order] for name, row_order in order.items()}
    else:
        full = Dataset.from_list([rows[i] for i in sorted(rows)])
        pair = full.train_test_split(test_size=args.test_size, seed=args.seed)
        splits = {name: list(ds) for name, ds in pair.items()}

    dataset = DatasetDict({name: Dataset.from_list(items) for name, items in splits.items()})
    dataset.save_to_disk(args.output)
    for name, ds in dataset.items():
        print(f"[INFO] {name}: {len(ds)} rows", flush=True)
    print(f"[COMPLETE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
