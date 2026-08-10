#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the deployment pipeline end to end, and score every stage against the catalogue.

    python 4.deploy-test.py                      # sample query, up to 10 attempts
    python 4.deploy-test.py -K 3 -o outputs/deploy-test
    python 4.deploy-test.py --score-only outputs/deploy-test/attempt-02

It runs `4.deploy.py`'s own `attempt()` - a real decomposition, coupling analysis,
retrieval, composition analysis and fill-in, four model calls per attempt - and when
retrieval does not return every expected twin it retries, up to `-K` times (default
10). Each retry sends the pipeline's restart feedback (`--feedback`, default the line
measured to make the decomposer carry each description over verbatim) and moves the
decoding seed exactly as a user declining the stack would. Every attempt keeps its
artefacts in its own `attempt-NN/` directory, so a failed attempt can be compared
with the one that succeeded.

The accepted attempt is then scored, stage by stage, against what the catalogue
declares (`relationships` -> `derive_wiring`, the reference answer) and against
`sample-instance.json` (the reference fill-in):

    1. decomposition   one part per subsystem, retrieval hits, and the coupling
                       analysis' data flows vs the declared chain (informational -
                       the composition stage is the gate, because it corrects the
                       intent against the real interfaces)
    2. composition     the analysed connections between the twins vs the declared
                       ones - recovered / missed / invented - plus inputs and output
    3. fill-in         every property value vs the reference instance
    4. the instance    shape: one bound subsystem per interface, catalogue property
                       sets, no telemetry, the wiring embedded unchanged
    5. docker-compose  services, images, ports, labels, environment, the embedded
                       COMPOSITION document, start-up through each image's own
                       `load_properties()`, and docker-images' full stack suite
                       (coherence + a driven cycle) against the generated file

`--score-only DIR` re-scores an existing run's artefacts without a single model
call, which is the cheap way to iterate on this script itself.

Needs the FAISS index, the sentence transformer and an Ollama host, like a real
deployment; `--score-only` needs none of them. Exit code 0 only when every gated
check passes, so it can sit in a shell `&&`.
"""

import argparse
import glob
import importlib.util
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))

# The feedback a retry sends. Measured (see README "The last stage is yours"): the
# pinned decomposition prompt may summarise a coupling-free request's paragraphs, and
# a summary can score 0.45 against the right interface while the paragraph scores
# 0.71; this line restores verbatim copying and with it complete retrieval. Override
# with --feedback; an empty string retries on the moved seed alone.
RETRY_FEEDBACK = ("Do not summarise. Each sub-query must carry that subsystem's whole "
                  "description from the request, word for word, including every "
                  "setting it lists.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The pipeline under test. Its `attempt()` is driven directly so this script cannot
# drift from what a user runs - there is no copy of the pipeline in here.
deploy = load_module(os.path.join(HERE, "4.deploy.py"), "deploy")
say = deploy.say


def head(title):
    say(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


class Section:
    """One scored stage: named checks, each pass/fail, plus free-form metrics."""

    def __init__(self, name):
        self.name, self.checks, self.metrics = name, [], {}

    def check(self, label, ok, detail=""):
        self.checks.append({"label": label, "pass": bool(ok), "detail": detail or None})
        say(f"  {'ok  ' if ok else 'FAIL'} {label}"
            + (f"  ({detail})" if detail and not ok else ""))
        return bool(ok)

    @property
    def passed(self):
        return all(entry["pass"] for entry in self.checks)

    def report(self):
        return {"passed": self.passed, "checks": self.checks, "metrics": self.metrics}


# -------------------------
# Ground truth
# -------------------------
def catalogue_truth(images):
    """What the catalogue says the answer is.

    `members` are the interfaces themselves, keyed the way the pipeline keys them, so
    every comparison below speaks the same vocabulary as the artefacts. `declared` is
    the reference wiring, re-derived from the `relationships` the interfaces carry -
    the analysis never sees those, which is exactly what makes them usable as ground
    truth. `reference` is the reference fill-in, keyed by interface id so a scored
    instance does not have to agree on subsystem key spelling.
    """
    members, folders = {}, {}
    for path in sorted(glob.glob(os.path.join(images, "image*", "dtdl",
                                              "interface.jsonl"))):
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        for interface in deploy.load_interfaces(path):
            key = deploy.subsystem_key(interface, members)
            members[key] = interface
            folders[key] = folder
    if not members:
        raise SystemExit(f"no image*/dtdl/interface.jsonl under {images}")

    declared = deploy.derive_wiring(members, note=lambda message: None)

    reference, reference_path = None, os.path.join(images, "sample-instance.json")
    if os.path.exists(reference_path):
        with open(reference_path, encoding="utf-8") as handle:
            document = json.load(handle)
        reference = {value["interface"]: value
                     for value in document.get("subsystems", {}).values()
                     if isinstance(value, dict) and value.get("interface")}
    return members, folders, declared, reference


def telemetry_of(interface):
    return {content["name"] for content in interface.get("contents", [])
            if content.get("@type") == "Telemetry"}


def read_json(out_dir, name):
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def retrieved_in(out_dir):
    document = read_json(out_dir, "decomposition.json")
    if document is None:
        return set()
    return {part.get("interface") for part in document.get("parts", [])} - {None}


# -------------------------
# The attempts
# -------------------------
def run_attempts(arguments, query, members):
    """Run the pipeline until retrieval is complete, or the attempts run out.

    The retry is the pipeline's own restart, not a fabrication of this script:
    `attempt(number, feedback)` moves the decoding seed through `decoding_for` and
    appends the feedback to every prompt through `guidance`, exactly as declining
    the stack interactively does. An attempt that raises is logged and retried too -
    a model returning prose instead of JSON should cost one attempt, not the run.
    """
    expected = {interface["@id"] for interface in members.values()}
    loaded = []

    def catalogue():
        # FAISS and the encoder load once, however many attempts follow.
        if not loaded:
            loaded.append(deploy.Catalogue())
        return loaded[0]

    log, best = [], None
    for number in range(1, arguments.max_attempts + 1):
        out_dir = os.path.join(arguments.out_dir, f"attempt-{number:02d}")
        namespace = argparse.Namespace(
            out_dir=out_dir, from_interfaces=None, no_fill=False, wiring="analysed",
            max_subsystems=len(expected), base_port=arguments.base_port,
            timeout=arguments.timeout)
        feedback = None if number == 1 else arguments.feedback
        head(f"attempt {number}/{arguments.max_attempts}"
             + (f"  (feedback: {feedback[:50]}...)" if feedback else ""))

        started, error = time.monotonic(), None
        try:
            deploy.attempt(namespace, query, catalogue, number, feedback)
        except SystemExit as exc:        # the pipeline's own aborts: bad JSON, no hits
            error = str(exc)[:300]
        except Exception as exc:         # a network hiccup costs an attempt, not the run
            error = f"{type(exc).__name__}: {exc}"[:300]
        seconds = round(time.monotonic() - started, 1)

        retrieved = retrieved_in(out_dir)
        hit = len(retrieved & expected)
        log.append({"attempt": number, "retrieved": f"{hit}/{len(expected)}",
                    "missed": sorted(expected - retrieved), "seconds": seconds,
                    "feedback": feedback, "error": error})
        say(f"\nattempt {number}: retrieval {hit}/{len(expected)}, {seconds}s"
            + (f"\n  error: {error}" if error else ""))

        if error is None and expected <= retrieved:
            return out_dir, number, log, True
        if error is None and (best is None or hit > best[0]):
            best = (hit, out_dir, number)

    if best is None:
        raise SystemExit("every attempt failed before producing artefacts; "
                         "see the errors above")
    say(f"\nretrieval never completed in {arguments.max_attempts} attempts; "
        f"scoring attempt {best[2]}, the best ({best[0]}/{len(expected)})")
    return best[1], best[2], log, False


# -------------------------
# 1. Decomposition
# -------------------------
def score_decomposition(out_dir, members, declared):
    head("1. DECOMPOSITION - the parts, retrieval, and the coupling analysis")
    section = Section("decomposition")
    document = read_json(out_dir, "decomposition.json")
    if document is None:
        section.check("decomposition.json exists", False)
        return section

    expected = {interface["@id"] for interface in members.values()}
    parts = document.get("parts", [])
    for part in parts:
        mark = "ok  " if part.get("interface") in expected else "MISS"
        say(f"  part {part['part']}: {str(part.get('sub_query'))[:70]}")
        say(f"     {mark} -> {part.get('subsystem')}  {part.get('interface')}")
    retrieved = {part.get("interface") for part in parts} - {None}

    section.check(f"one part per subsystem ({len(parts)} parts, "
                  f"{len(expected)} interfaces)", len(parts) == len(expected))
    section.check(f"retrieval complete ({len(retrieved & expected)}/{len(expected)})",
                  expected <= retrieved,
                  detail=f"missed {sorted(expected - retrieved)}")

    # The coupling analysis speaks in data, not field names, so it is scored at
    # (from, to) granularity. It is informational rather than gated: its whole design
    # is to be an intent the composition stage corrects against the interfaces, and
    # the corrected result is scored - and gated - in section 2.
    declared_pairs = {(c["from"], c["to"]) for c in declared["connections"]}
    flows = document.get("flows_on_subsystems") or []
    flow_pairs = {(flow.get("from"), flow.get("to")) for flow in flows}
    recovered = flow_pairs & declared_pairs
    say("\n  coupling analysis (informational; section 2 is the gate):")
    for flow in flows:
        mark = "ok   " if (flow.get("from"), flow.get("to")) in declared_pairs else "extra"
        say(f"    {mark} {flow.get('from')} -> {flow.get('to')}"
            + ("  (feedback)" if flow.get("feedback") else "")
            + f": {str(flow.get('data'))[:44]}")
    for pair in sorted(declared_pairs - flow_pairs):
        say(f"    miss  {pair[0]} -> {pair[1]}")
    section.metrics["couplings"] = {
        "recovered": len(recovered), "declared": len(declared_pairs),
        "extra": len(flow_pairs - declared_pairs),
        "recall": round(len(recovered) / len(declared_pairs), 3) if declared_pairs else None,
        "precision": round(len(recovered) / len(flow_pairs), 3) if flow_pairs else None}
    say(f"    -> {len(recovered)}/{len(declared_pairs)} declared pairs recovered, "
        f"{len(flow_pairs - declared_pairs)} extra")
    return section


# -------------------------
# 2. Composition
# -------------------------
def score_composition(out_dir, declared):
    head("2. COMPOSITION - the connections between the digital twins")
    section = Section("composition")
    wiring = read_json(out_dir, "wiring.json")
    if wiring is None:
        section.check("wiring.json exists", False)
        return section

    accepted = wiring.get("accepted") or {}
    declared_edges = deploy.edges(declared["connections"])
    got = deploy.edges(accepted.get("connections") or [])
    for connection in accepted.get("connections") or []:
        edge = (connection["from"], connection["to"], connection["as"],
                bool(connection.get("feedback")))
        mark = "ok   " if edge in declared_edges else "EXTRA"
        arrow = "<<" if connection.get("feedback") else "<-"
        say(f"  {mark} {connection['to']:<17} {arrow} {connection['from']:<17} "
            f"as {connection['as']}")
    for edge in sorted(declared_edges - got):
        say(f"  MISS  {edge[1]:<17} {'<<' if edge[3] else '<-'} {edge[0]:<17} as {edge[2]}")
    if wiring.get("notes"):
        say("\n  rejected or corrected by validate_wiring:")
        for note in wiring["notes"]:
            say(f"    - {note}")
    say("")

    recovered = got & declared_edges
    section.metrics["connections"] = {
        "recovered": len(recovered), "declared": len(declared_edges),
        "missed": sorted(f"{e[0]} -> {e[1]} as {e[2]}" for e in declared_edges - got),
        "extra": sorted(f"{e[0]} -> {e[1]} as {e[2]}" for e in got - declared_edges),
        "recall": round(len(recovered) / len(declared_edges), 3) if declared_edges else None,
        "precision": round(len(recovered) / len(got), 3) if got else None}
    section.check(f"every declared connection recovered "
                  f"({len(recovered)}/{len(declared_edges)})", declared_edges <= got)
    section.check(f"no invented connections ({len(got - declared_edges)} extra)",
                  not (got - declared_edges))
    section.check("external inputs equal the declared routing",
                  accepted.get("inputs") == declared["inputs"])
    section.check(f"composed output is {declared['output']}",
                  accepted.get("output") == declared["output"])
    section.check("the wiring came from the analysis, not the declarations",
                  wiring.get("source") == "analysed",
                  detail=f"source={wiring.get('source')!r}")
    return section


# -------------------------
# 3. Fill-in
# -------------------------
def score_fill_in(out_dir, members, reference):
    head("3. FILL-IN - property values against the reference instance")
    section = Section("fill-in")
    instance = read_json(out_dir, "instance.json")
    if instance is None:
        section.check("instance.json exists", False)
        return section
    if reference is None:
        say("  no sample-instance.json beside the catalogue; values cannot be scored")
        return section

    telemetry_by_id = {interface["@id"]: telemetry_of(interface)
                       for interface in members.values()}
    bound_by_id = {value.get("interface"): value
                   for value in (instance.get("subsystems") or {}).values()
                   if isinstance(value, dict)}
    correct = total = 0
    per_interface = {}
    for interface_id, expected in reference.items():
        bound = bound_by_id.get(interface_id) or {}
        skip = deploy.NOT_EXTRACTED | telemetry_by_id.get(interface_id, set())
        mismatched, ok, count = [], 0, 0
        for prop, value in expected.items():
            if prop in skip:
                continue
            count += 1
            if bound.get(prop) == value:
                ok += 1
            else:
                mismatched.append({"property": prop, "expected": value,
                                   "got": bound.get(prop)})
        name = interface_id.split(":")[-1].split(";")[0]
        say(f"  {name:<22} {ok}/{count}")
        for entry in mismatched:
            say(f"      {entry['property']}: expected {entry['expected']!r}, "
                f"got {entry['got']!r}")
        correct, total = correct + ok, total + count
        per_interface[interface_id] = {"correct": ok, "total": count,
                                       "mismatched": mismatched}
    say("")
    section.metrics["values"] = {
        "correct": correct, "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "per_interface": per_interface}
    section.check(f"every stated property value extracted ({correct}/{total})",
                  correct == total)
    return section


# -------------------------
# 4. The DTDL instance
# -------------------------
def score_instance(out_dir, members):
    head("4. THE DTDL INSTANCE - shape and embedded wiring")
    section = Section("instance")
    instance = read_json(out_dir, "instance.json")
    wiring = read_json(out_dir, "wiring.json")
    composition = read_json(out_dir, "composition.json")
    if instance is None or wiring is None or composition is None:
        section.check("instance.json, wiring.json and composition.json exist", False)
        return section

    by_id = {interface["@id"]: interface for interface in members.values()}
    subsystems = instance.get("subsystems") or {}
    section.check("one bound subsystem per retrieved interface",
                  {value.get("interface") for value in subsystems.values()}
                  == set(by_id))
    section.check("the composed instance carries the orchestrator image",
                  instance.get("dockerImage") == deploy.ORCHESTRATOR_IMAGE)
    section.check("the composed instance serves the two-endpoint contract",
                  instance.get("dataEndpoint") == deploy.DATA_ENDPOINT
                  and instance.get("updateEndpoint") == deploy.UPDATE_ENDPOINT)

    keys_ok, images_ok, telemetry_leaks = True, True, []
    for key, bound in subsystems.items():
        interface = by_id.get(bound.get("interface"))
        if interface is None:
            keys_ok = False
            continue
        # properties_of() emits `interface` plus exactly the catalogue's property
        # names (dockerImage included, copied not extracted); anything else - a
        # telemetry key especially - means the fill-in leaked past its spec.
        if set(bound) != {"interface"} | set(deploy.property_names(interface)):
            keys_ok = False
        if bound.get("dockerImage") != deploy.declared_value(interface, "dockerImage"):
            images_ok = False
        telemetry_leaks += [f"{key}.{name}" for name in telemetry_of(interface)
                            if name in bound]
    section.check("each subsystem carries exactly the catalogue's property set", keys_ok)
    section.check("each dockerImage is copied from the catalogue, not extracted",
                  images_ok)
    section.check("no telemetry key appears in any bound subsystem",
                  not telemetry_leaks, detail=", ".join(telemetry_leaks[:5]))

    accepted = wiring.get("accepted") or {}
    section.check("the instance embeds the accepted wiring unchanged",
                  instance.get("connections") == accepted.get("connections")
                  and instance.get("inputs") == accepted.get("inputs")
                  and instance.get("output") == accepted.get("output"))
    section.check("the instance records how the wiring was found",
                  instance.get("wiringSource") == wiring.get("source"))
    section.check("composition.json names the same interface and subsystems",
                  composition.get("interface") == instance.get("interface")
                  and set(composition.get("subsystems") or {}) == set(subsystems))
    return section


# -------------------------
# 5. docker-compose.yaml
# -------------------------
def bind_module(path, name, environment):
    """Import an image's app.py under the given environment and bind it, the way
    the container would at start-up. Returns an error string, or None."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, environment, clear=False):
            spec.loader.exec_module(module)
            if hasattr(module, "load_composition"):
                module.load_composition()
            else:
                module.load_properties()
        return None
    except BaseException as exc:                        # SystemExit is the common case
        return f"{type(exc).__name__}: {exc}"[:200]


def score_compose(out_dir, images, members, folders, run_cycle):
    head("5. docker-compose.yaml - the stack, bound and driven")
    section = Section("compose")
    instance = read_json(out_dir, "instance.json")
    composition = read_json(out_dir, "composition.json")
    compose_path = os.path.join(out_dir, "docker-compose.yaml")
    if instance is None or composition is None or not os.path.exists(compose_path):
        section.check("instance.json, composition.json and docker-compose.yaml exist",
                      False)
        return section

    # docker-images owns the compose reader and the stack suite; reusing them keeps
    # this script from re-stating what a valid stack is.
    sys.path.insert(0, images)
    import test_stack
    try:
        services = test_stack.read_compose(Path(compose_path))
    except ValueError as exc:
        section.check("the compose file parses", False, detail=str(exc))
        return section
    section.check("the compose file parses", True)

    by_id = {interface["@id"]: key for key, interface in members.items()}
    twin_services = {service["labels"].get("dtdl.interface"): (name, service)
                     for name, service in services.items()
                     if service["labels"].get("dtdl.interface") in by_id}
    orchestrator = next(((name, service) for name, service in services.items()
                         if service["image"] == instance.get("dockerImage")), None)
    section.check(f"one service per twin plus the orchestrator "
                  f"({len(services)} services)",
                  len(twin_services) == len(members) and orchestrator is not None
                  and len(services) == len(members) + 1)

    ports = [entry.split(":")[0] for service in services.values()
             for entry in service["ports"]]
    section.check("host ports do not collide", len(ports) == len(set(ports)))

    images_ok, environment_ok, bad_env = True, True, []
    for key, interface in members.items():
        found = twin_services.get(interface["@id"])
        if found is None:
            images_ok = False
            continue
        name, service = found
        if service["image"] != deploy.declared_value(interface, "dockerImage"):
            images_ok = False
        bound = next((value for value in (instance.get("subsystems") or {}).values()
                      if value.get("interface") == interface["@id"]), {})
        expected_env = {deploy.env_name(prop): deploy.env_value(value)
                        for prop, value in bound.items()
                        if prop not in deploy.NOT_EXTRACTED and value is not None}
        if service["environment"] != expected_env:
            environment_ok = False
            bad_env.append(name)
    section.check("every twin runs its catalogue image", images_ok)
    section.check("every environment is exactly the instance's non-null properties",
                  environment_ok, detail=", ".join(bad_env))
    if orchestrator is not None:
        raw = orchestrator[1]["environment"].get("COMPOSITION")
        section.check("the orchestrator's COMPOSITION equals composition.json",
                      raw is not None and json.loads(raw) == composition)
        section.check("the orchestrator is labelled with the composed interface",
                      orchestrator[1]["labels"].get("dtdl.interface")
                      == instance.get("interface"))

    say("\n  start-up, through each image's own binding:")
    starts = {}
    for key, interface in members.items():
        found = twin_services.get(interface["@id"])
        error = ("no service" if found is None else
                 bind_module(os.path.join(images, folders[key], "app.py"),
                             f"bind_{folders[key]}", found[1]["environment"]))
        starts[key] = error
        say(f"    {key:<18} {'starts' if error is None else 'ABORTS: ' + error}")
    orchestrator_error = "no service" if orchestrator is None else bind_module(
        os.path.join(images, "orchestrator", "app.py"), "bind_orchestrator",
        {"COMPOSITION": orchestrator[1]["environment"].get("COMPOSITION", "")})
    say(f"    {'orchestrator':<18} "
        f"{'starts' if orchestrator_error is None else 'ABORTS: ' + orchestrator_error}")
    section.check("all six containers bind their configuration at start-up",
                  orchestrator_error is None
                  and not any(error for error in starts.values()))

    if not run_cycle:
        say("\n  stack suite skipped (--no-cycle)")
        return section

    # The strongest statement available: docker-images' own coherence checks and a
    # driven two-cycle run, pointed at the generated file instead of the sample.
    test_stack.COMPOSE = Path(compose_path)
    stream = io.StringIO()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(test_stack.StackCoherenceTests),
        loader.loadTestsFromTestCase(test_stack.StackCycleTests)])
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    failed = [test.id().split(".")[-1] for test, _ in result.failures + result.errors]
    say(f"\n  stack suite on the generated file: {result.testsRun} tests, "
        f"{len(result.failures)} failures, {len(result.errors)} errors")
    for name in failed:
        say(f"    FAIL {name}")
    section.metrics["stack_suite"] = {"ran": result.testsRun, "failed": failed}
    section.check("the generated stack passes the full coherence and cycle suite",
                  result.wasSuccessful())
    return section


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("query", nargs="?", default=None,
                        help="the request; @FILE reads a file. Default: the sample "
                             "query beside the catalogue")
    parser.add_argument("-K", "--max-attempts", type=int, default=10,
                        help="attempts before giving up on complete retrieval "
                             "(default 10)")
    parser.add_argument("-o", "--out-dir",
                        default=os.path.join("outputs", "deploy-test"))
    parser.add_argument("--images", default=os.path.join(HERE, "..", "docker-images"),
                        help="the catalogue images repository (ground truth)")
    parser.add_argument("--feedback", default=RETRY_FEEDBACK,
                        help="restart feedback for retries; '' retries on the moved "
                             "seed alone")
    parser.add_argument("--timeout", type=int, default=300,
                        help="model timeout per call, seconds")
    parser.add_argument("--base-port", type=int, default=8081)
    parser.add_argument("--no-cycle", action="store_true",
                        help="skip docker-images' stack suite")
    parser.add_argument("--score-only", metavar="DIR",
                        help="score an existing attempt directory; no model calls")
    arguments = parser.parse_args()

    images = os.path.abspath(arguments.images)
    members, folders, declared, reference = catalogue_truth(images)
    expected = {interface["@id"] for interface in members.values()}

    if arguments.score_only:
        out_dir, number, log = arguments.score_only, None, []
        complete = expected <= retrieved_in(out_dir)
    else:
        query = arguments.query
        if query is None:
            query = "@" + os.path.join(images, "sample-query.md")
        if query.startswith("@"):
            with open(query[1:], encoding="utf-8") as handle:
                query = handle.read()
        query = query.strip()
        out_dir, number, log, complete = run_attempts(arguments, query, members)

    sections = [
        score_decomposition(out_dir, members, declared),
        score_composition(out_dir, declared),
        score_fill_in(out_dir, members, reference),
        score_instance(out_dir, members),
        score_compose(out_dir, images, members, folders, not arguments.no_cycle),
    ]

    head("SUMMARY")
    if log:
        for entry in log:
            say(f"  attempt {entry['attempt']}: retrieval {entry['retrieved']}, "
                f"{entry['seconds']}s"
                + (f", error: {entry['error']}" if entry["error"] else ""))
        say("")
    for section in sections:
        done = sum(1 for check in section.checks if check["pass"])
        say(f"  {'PASS' if section.passed else 'FAIL'}  {section.name:<15} "
            f"{done}/{len(section.checks)} checks")
    verdict = complete and all(section.passed for section in sections)
    say(f"\n  verdict: {'PASS' if verdict else 'FAIL'}"
        + ("" if complete else "  (retrieval never completed)"))

    report = {
        "what": "4.deploy-test.py: the deployment pipeline scored stage by stage "
                "against the catalogue's declared wiring and reference instance",
        "model": deploy.OLLAMA_MODEL, "host": deploy.OLLAMA_HOST,
        "max_attempts": arguments.max_attempts, "attempts": log,
        "scored_attempt": number, "scored_dir": out_dir,
        "retrieval_complete": complete,
        "sections": {section.name: section.report() for section in sections},
        "verdict": "PASS" if verdict else "FAIL"}
    report_path = os.path.join(
        arguments.out_dir if not arguments.score_only else out_dir, "report.json")
    deploy.write(report_path, report)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
