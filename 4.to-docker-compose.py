#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn an instantiated DTDL instance into a `docker-compose.yaml`.

    python 4.to-docker-compose.py data/fill-eval.jsonl --line 1
    python 4.to-docker-compose.py instance.json -o docker-compose.yaml

Accepts either shape the pipeline produces:

    flat      {"interface": "dtmi:...", "<property>": value, ...}     -> one service
    composed  {"interface": "...", "subsystems": {"<name>": {...}}}   -> one per subsystem

Each interface becomes one service: `dockerImage` is the image, the remaining properties
become environment variables (UPPER_SNAKE_CASE), and the interface id is kept as a label
so a running container maps back to the twin it came from. Null properties are dropped —
they were never stated in the source text, so there is nothing to configure with.

YAML is written directly rather than through PyYAML: the output is a fixed two-level
shape, and this keeps the script dependency-free.
"""

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Tuple

# Not configuration: one identifies the image, the other the twin. Both are emitted
# elsewhere in the service, so neither belongs in `environment`.
SKIP_KEYS = {"interface", "dockerImage"}


def env_name(prop: str) -> str:
    """nominalCapacityAh -> NOMINAL_CAPACITY_AH"""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", prop)
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()


def service_name(interface_id: str, fallback: str) -> str:
    """dtmi:ev_battery_health_twin:BatteryPack;1 -> battery_pack"""
    tail = (interface_id or "").split(":")[-1].split(";")[0]
    name = env_name(tail).lower() if tail else ""
    return name or fallback


def quote(value: Any) -> str:
    """Every scalar is emitted as a quoted string; Compose treats env values as text."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def instances(instance: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """-> [(fallback_name, flat property dict), ...] for flat and composed instances."""
    subsystems = instance.get("subsystems")
    if isinstance(subsystems, dict):
        return [(name, props) for name, props in subsystems.items() if isinstance(props, dict)]
    return [("twin", instance)]


def to_compose(instance: Dict[str, Any]) -> str:
    lines = ["services:"]
    used = set()

    for fallback, props in instances(instance):
        name = service_name(props.get("interface") or "", env_name(fallback).lower())
        # Composed instances can pull two components with the same display name.
        base, n = name, 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)

        image = props.get("dockerImage") or instance.get("dockerImage")
        interface_id = props.get("interface") or instance.get("interface") or ""

        lines.append(f"  {name}:")
        lines.append(f"    image: {quote(image)}" if image
                     else f"    image: {quote('registry.local/dtm/' + name + ':latest')}"
                          "  # no dockerImage property in the instance")
        lines.append(f"    container_name: {name}")
        lines.append("    restart: unless-stopped")
        if interface_id:
            lines.append("    labels:")
            lines.append(f"      dtdl.interface: {quote(interface_id)}")

        env = {k: v for k, v in props.items() if k not in SKIP_KEYS and v is not None}
        if env:
            lines.append("    environment:")
            for key, value in env.items():
                lines.append(f"      {env_name(key)}: {quote(value)}")

    return "\n".join(lines) + "\n"


def load_instance(path: str, line: int) -> Dict[str, Any]:
    text = sys.stdin.read() if path == "-" else open(path, "r", encoding="utf-8").read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        rows = [ln for ln in text.splitlines() if ln.strip()]
        if not 1 <= line <= len(rows):
            raise SystemExit(f"[ERROR] --line {line} is out of range (1..{len(rows)})")
        obj = json.loads(rows[line - 1])

    # Unwrap the containers the pipeline stores instances in.
    for key in ("answer", "predicted_instance"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise SystemExit("[ERROR] Expected a JSON object holding a DTDL instance")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance", help="JSON or JSONL file holding the instance ('-' for stdin)")
    ap.add_argument("--line", type=int, default=1, help="Which line to use if the input is JSONL")
    ap.add_argument("-o", "--out", default="-", help="Output file (default: stdout)")
    args = ap.parse_args()

    compose = to_compose(load_instance(args.instance, args.line))
    if args.out == "-":
        sys.stdout.write(compose)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(compose)
        print(f"[DONE] wrote {args.out}")


if __name__ == "__main__":
    main()
