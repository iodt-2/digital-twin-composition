#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end deployment: a natural-language request in, a deployable stack out.

    python 4.deploy.py @../docker-images/sample-query.md -o outputs/deploy
    cd outputs/deploy && docker compose up

    user query -> decomposition -> composition -> fill-in -> instance -> compose file
                  parts+couplings   wiring                              -> confirm

The request describes independent components and says nothing about how they fit
together: a subsystem states what it holds, what it accepts and what it produces,
never who it talks to. Working that out is the pipeline's job, and it happens twice.

`decomposition` splits the request and then, in a second call, analyses how the
parts would have to exchange data - which part produces what, which part needs it.
That analysis is in the request's own vocabulary, because no interface has been
retrieved yet. `composition` resolves it onto the twins retrieval actually returned:
a third call reads each member's `outputType` and `updateFields` and answers with
the connections between them, which are then checked against the catalogue - a
field a twin does not accept, a producer that is not a member, two sources for one
field and a cycle in the graph are all caught here rather than at `docker compose
up`. What survives is written into the DTDL instance and the composed interface.

Twins still never call each other. The composed interface is realised by an
orchestrator container that reads each subsystem's `dataEndpoint` and posts what it
finds to the `updateEndpoint` of the subsystems the analysed connections name.

Stages 1-4 are the ones `3.system-eval.py` measures, with the scoring replaced by a
deployment; the prompt that splits the request is character-identical to that
script's so the two decompose alike. The coupling and wiring analyses are stages
that script does not have and does not measure.

The last stage is the user's. The compose file is printed and confirmed before it
is anything to run; declining restarts the whole process - a new decomposition, a
new retrieval, a new analysis - optionally with a line of feedback that is appended
to every prompt of the next attempt.

Prerequisites (env-overridable, same names as the rest of the pipeline):
    FAISS_INDEX_PATH, METADATA_PATH   built by 3.build-faiss-index.py
    SENTENCE_TRANSFORMER_PATH         the model the index was encoded with
    OLLAMA_HOST / OLLAMA_MODEL        decomposition, wiring analysis and fill-in

`--from-interfaces` (compose named interfaces, skip decomposition and retrieval) and
`--no-fill` (leave every property null, and take the wiring from what the catalogue
declares) need none of them, and together turn this into an offline
catalogue-to-compose generator.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.10.10.4:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./models/faiss.index")
METADATA_PATH = os.getenv("METADATA_PATH", "./models/metadata.json")
SENTENCE_TRANSFORMER_PATH = os.getenv(
    "SENTENCE_TRANSFORMER_PATH", "./models/MiniLM-L6-based-new-triplets-final")
ORCHESTRATOR_IMAGE = os.getenv("ORCHESTRATOR_IMAGE", "ziren/composition-orchestrator:latest")

DATA_ENDPOINT = "/v1/data/"
UPDATE_ENDPOINT = "/v1/update/"
CONTAINER_PORT = 8080

# Never asked of the fill-in model: `interface` is known, and no anchor may mention
# `dockerImage`, so a model asked for it would be guessing. Both are copied from the
# catalogue after the model has answered.
NOT_EXTRACTED = {"interface", "dockerImage"}


def say(message=""):
    print(message, flush=True)


# -------------------------
# The model
# -------------------------
# Greedy decoding with a fixed seed. Sub-query length decides retrieval - the longer
# the decomposition, the more of each subsystem's description survives to be embedded -
# and sampling varies that length run to run. Left to the server's default, a deployment
# averaged 2.6 of 5 twins over seven runs. Override with DECODING to experiment.
#
# Pinning bounds that variance but does not remove it. On the query that stated its own
# couplings, three consecutive runs retrieved 5/5 with identical scores. On a query that
# states none - which is what a request now looks like - three runs produced three
# different decompositions, retrieving 3/5 each with the same two misses but scores
# differing by up to 0.04, and part 1 landing on a different wrong interface in one of
# them. So treat a retrieval count as a sample, not a constant: gpt-oss:120b on Ollama
# is not bit-reproducible at temperature 0, and `--wiring declared` is the only route
# here that is reproducible without a model at all.
DECODING = json.loads(os.getenv("DECODING", '{"temperature": 0, "seed": 1}'))


def decoding_for(attempt):
    """`DECODING` for attempt 1, and a different seed for each restart after it.

    Greedy decoding is deterministic, so a restart that changed nothing would
    return the same stack the user just declined. The seed moves instead; feedback,
    where the user gave any, changes the prompts as well.
    """
    if attempt <= 1 or "seed" not in DECODING:
        return DECODING
    return {**DECODING, "seed": DECODING["seed"] + attempt - 1}


def ollama(prompt, timeout, options=None):
    """One completion, /api/generate with /api/chat as the fallback."""
    options = DECODING if options is None else options

    def post(path, payload):
        request = urllib.request.Request(
            OLLAMA_HOST.rstrip("/") + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        answer = post("/api/generate", {"model": OLLAMA_MODEL, "prompt": prompt,
                                        "stream": False, "options": options})
        return str(answer["response"])
    except (urllib.error.HTTPError, KeyError):
        answer = post("/api/chat", {"model": OLLAMA_MODEL, "stream": False,
                                    "options": options,
                                    "messages": [{"role": "user", "content": prompt}]})
        return str(answer["message"]["content"])


def parse_json(text):
    """The first JSON value in a model's reply.

    Models fence, prefix, explain, and sometimes close one brace too many, so the
    value is decoded from the first bracket with `raw_decode`, which stops at the end
    of the first complete value and ignores whatever follows. Slicing to the last
    bracket instead would swallow that trailing brace and fail on a reply whose JSON
    was perfectly good.
    """
    stripped = text.strip()
    starts = [position for position in (stripped.find("{"), stripped.find("["))
              if position != -1]
    if not starts:
        raise SystemExit(f"model returned no JSON:\n{text}")
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped[min(starts):])
        return value
    except json.JSONDecodeError as exc:
        raise SystemExit(f"model returned unparseable JSON: {exc}\n{text}") from exc


def guidance(feedback):
    """The user's feedback, as a block appended to a prompt on a restart.

    It is a suffix rather than an edit so that the first attempt's prompts are the
    prompts this script has always sent - in particular the split prompt stays
    character-identical to `3.system-eval.py`'s until the user asks for something
    different.
    """
    if not feedback:
        return ""
    return ("\nThe user rejected the previous attempt and asks for this to be "
            f"different:\n{feedback}\n")


# -------------------------
# The prompts
# -------------------------
def build_decompose_prompt(description_text: str, max_parts: int) -> str:
    """Character-identical to 3.system-eval.py's build_decompose_prompt: a deployment
    that decomposes differently is not the system the retrieval numbers describe.
    Change one, change both."""
    return (
        "You are a decomposition assistant.\n"
        "Given a DESCRIPTION of a desired composed a large digital twin system, split it into sub-queries for subsystem "
        "digital twin interfaces used to search sub-components, reformat and extract information for the subsystem"
        "so the sub-query can be used in the FAISS search.\n"
        "Important constraints:\n"
        f"- Return ONLY minified JSON array of strings.\n"
        f"- Array length must be between 1 and {max_parts}.\n"
        f"- Split into reasonable number of sub-queries.\n"
        "- Each sub-query must be self-contained and target ONE subsystem/interface.\n"
        "- Do NOT use description, data or facts from other sub-queries.\n"
        "- Using similar language structure in the main query to construct sub-queries.\n"
        "- You MAY summarize wording, BUT you MUST preserve ALL stated facts and constraints.\n"
        "- Do NOT drop or generalize any information.\n"
        "- Do NOT simplify information.\n"
        "- Keep such values verbatim when present.\n"
        "- Do not add new explanations.\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n"
    )


def build_couplings_prompt(description_text, sub_queries):
    """The other half of the decomposition: how the parts would exchange data.

    The split prompt is deliberately blind to this - it is told to keep every
    sub-query self-contained and to use no facts from another, which is what makes
    each one a usable retrieval query. So the couplings are analysed in a second
    call, over the parts that prompt produced. Nothing is known about interfaces
    yet, so the answer is in the request's own words; `build_wiring_prompt` resolves
    it onto the twins that were retrieved.
    """
    listing = "\n".join(f"{number}. {text}"
                        for number, text in enumerate(sub_queries, 1))
    return (
        "You are a data-flow analyst for digital twin systems.\n"
        "A DESCRIPTION of one composed system has been split into the PARTS below. Each "
        "part is an independent component: it states what it holds, what it accepts and "
        "what it produces, and says nothing about the other parts.\n"
        "Work out how data would have to move between the parts for the composed system "
        "to work end to end.\n"
        "Rules:\n"
        "- Return ONLY minified JSON, in EXACTLY this shape:\n"
        '  {"parts":[{"part":1,"produces":"<what this part serves to others, or null>",'
        '"needs":["<data it must be given>"]}],'
        '"flows":[{"from":1,"to":2,"data":"<what moves>","feedback":false}]}\n'
        "- `part`, `from` and `to` are the 1-based numbers of the parts listed below.\n"
        "- One flow per distinct piece of data that moves; name the data in the words "
        "the parts themselves use.\n"
        "- A part may need something no other part produces. Say so in `needs` and emit "
        "no flow for it: it comes from outside the composed system.\n"
        "- Set `feedback` true only where a part needs a value produced by a LATER part, "
        "or by itself, and can therefore only be given the previous cycle's value.\n"
        "- Do NOT invent a flow to make the system look connected. Only claim one where "
        "what one part produces is what another part accepts.\n"
        "- No prose outside the JSON.\n\n"
        "DESCRIPTION:\n"
        f"{description_text}\n\n"
        "PARTS:\n"
        f"{listing}\n"
    )


def self_description(interface):
    """What a member says about itself, and nothing about who it connects to.

    `relationships` is deliberately withheld. Where a catalogue declares them they
    are the answer to the question being asked, so an analysis that read them would
    be reciting rather than composing - see `--wiring declared` for the route that
    does read them, on purpose.
    """
    return {"interface": interface.get("@id"),
            "displayName": interface.get("displayName"),
            "description": interface.get("description"),
            "serves": interface.get("outputType"),
            "accepts": update_fields(interface)}


def build_wiring_prompt(query, members, flows):
    """The composition itself: which member feeds which, and under what field.

    Both sides are given as the members state them - what each serves on its data
    endpoint, and the exact field names each accepts on its update endpoint - so a
    connection is answerable without guessing at names. The decomposition's coupling
    analysis comes along as the intent to resolve, not as the answer: retrieval may
    have returned a twin that splits or merges what a part described.
    """
    spec = {key: self_description(interface) for key, interface in members.items()}
    intent = ("\n\nWHAT THE DECOMPOSITION EXPECTED TO MOVE (intent, in the request's "
              "own words - resolve it onto the subsystems above, and correct it where "
              "they do not match):\n"
              f"{json.dumps(flows, separators=(',', ':'))}" if flows else "")
    return (
        "You are a composition assistant for digital twin systems.\n"
        "The SUBSYSTEMS below were retrieved independently and know nothing about each "
        "other. Compose them: decide how data moves between them so that the composed "
        "system works end to end.\n"
        "Each subsystem serves one output object on its data endpoint (`serves`) and "
        "accepts a fixed set of telemetry fields on its update endpoint (`accepts`). An "
        "orchestrator reads a subsystem's output and posts it to another subsystem as one "
        "of its accepted fields. That is the only way data moves; the subsystems never "
        "call each other.\n"
        "Rules:\n"
        "- Return ONLY minified JSON, in EXACTLY this shape:\n"
        '  {"connections":[{"from":"<subsystem>","to":"<subsystem>","as":"<accepted '
        'field>","feedback":false,"why":"<one short clause>"}],'
        '"inputs":{"<accepted field>":["<subsystem>"]},"output":"<subsystem>"}\n'
        f"- `from`, `to` and the values in `inputs` MUST be exactly these keys: {list(spec)}\n"
        "- `as` MUST be one of the `accepts` field names of the `to` subsystem, spelled "
        "exactly as listed. Never invent a field.\n"
        "- Only connect where what `from` serves is what `to` accepts under that field. "
        "If nothing a subsystem serves fits a field, that field is not connected.\n"
        "- Go through EVERY field in EVERY subsystem's `accepts` list and ask which "
        "member, if any, serves what that field is for. A field is left unconnected "
        "only after you have checked it against all of them.\n"
        "- Match a field by what it carries, not by how it is spelled: a field may be "
        "named differently from the type that fills it, so a field called "
        "`currentSomething`, `latestSomething` or `somethingIn` accepts the `Something` "
        "a member serves.\n"
        "- At most one connection per (`to`, `as`) pair: a field is fed by one source.\n"
        "- The non-feedback connections must form no cycle. Where a subsystem needs a "
        "value that only becomes available later in the cycle, or its own previous "
        "output, set `feedback` true - that delivers what the source held before this "
        "cycle. `from` and `to` may be the SAME subsystem: a twin whose field wants the "
        "previous version of what that twin itself serves is connected to itself, with "
        "`feedback` true.\n"
        "- A field whose name says `previous`, `prior`, `last` or `accepted` wants an "
        "earlier version of something, so find the member that serves that thing and "
        "connect it with `feedback` true - even when that member is the `to` subsystem "
        "itself. A subsystem `alpha` that serves a `Widget` and also accepts a field "
        "`priorWidget` is taking its own earlier output: "
        '{"from":"alpha","to":"alpha","as":"priorWidget","feedback":true,'
        '"why":"its own last widget"}. Do not leave such a field unconnected because no '
        "*other* member serves it.\n"
        "- `inputs` lists every accepted field left unconnected, mapped to the "
        "subsystems that accept it: those are supplied by the caller of the composed "
        "system.\n"
        "- `output` is the subsystem whose output is the composed system's output - the "
        "end of the chain, the one nothing else reads.\n"
        "- `why` is for the reader, one clause, no more.\n"
        "- No prose outside the JSON.\n\n"
        "SUBSYSTEMS:\n"
        f"{json.dumps(spec, separators=(',', ':'))}"
        f"{intent}\n\n"
        "DESCRIPTION OF THE COMPOSED SYSTEM:\n"
        f"{query or '(none given; compose from the subsystems above)'}\n"
    )


def build_fillin_prompt(query, composed, members):
    """3.system-eval.py's composed prompt, without the keys nothing may extract."""
    spec = {key: [n for n in property_names(iface) if n not in NOT_EXTRACTED]
            for key, iface in members.items()}
    return (
        "You are an information extraction assistant.\n"
        "Return a FULL initiated instance for a COMPOSED Interface with multiple subsystems.\n"
        "Rules:\n"
        "- Use ONLY values explicitly stated in DESCRIPTION. Do not infer unstated facts.\n"
        "- If a value is not stated, use null.\n"
        "- Do NOT output any telemetry fields.\n"
        "- Return ONLY minified JSON (no markdown, no comments).\n"
        "- Output must follow EXACTLY this JSON shape:\n"
        '  {"interface": "<interface_id>", "subsystems": { "<subsystem_name>": { "<property>": value_or_null, ... }, ... }}\n'
        f"- The 'interface' field MUST be exactly: {composed['@id']!r}\n"
        f"- The 'subsystems' keys MUST be exactly: {list(spec)}\n"
        "- For each subsystem, include EXACTLY the listed property keys (and no others), in this spec:\n"
        f"{json.dumps(spec, separators=(',', ':'))}\n\n"
        "COMPOSED INTERFACE (for context):\n"
        f"{json.dumps(composed, separators=(',', ':'))}\n\n"
        "DESCRIPTION:\n"
        f"{query}\n"
    )


# -------------------------
# Reading the catalogue
# -------------------------
def property_names(interface):
    return [c["name"] for c in interface.get("contents", [])
            if c.get("@type") == "Property" and "name" in c]


def declared_value(interface, name):
    for content in interface.get("contents", []):
        if content.get("name") == name:
            return content.get("value")
    return None


def update_fields(interface):
    """The telemetry keys this twin's update endpoint accepts.

    Declared, not inferred: an interface may model one observation while its endpoint
    takes a batch of them under a single key, and only the interface knows which.
    Falls back to the Telemetry names for a catalogue that predates the field.
    """
    return interface.get("updateFields") or [
        c["name"] for c in interface.get("contents", [])
        if c.get("@type") == "Telemetry" and "name" in c]


def subsystem_key(interface, taken):
    """`dtmi:building_energy_optimisation:BuildingDataTwin;1` -> `buildingData`.

    This is the composition's own vocabulary: it names the subsystems, both ends of
    every connection and the input routing, so it has to be short and collision-free.
    """
    tail = str(interface.get("@id", "")).split(":")[-1].split(";")[0]
    words = re.split(r"[^0-9a-zA-Z]+", re.sub(r"(Twin|Interface)$", "", tail) or tail)
    words = [w for w in words if w] or ["subsystem"]
    name = words[0][:1].lower() + words[0][1:] + "".join(w.title() for w in words[1:])
    while name in taken:
        name += "2"
    return name


def load_interfaces(path):
    interfaces = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                interfaces.append(row["interface"] if isinstance(row.get("interface"), dict)
                                  else row)
    return interfaces


class Catalogue:
    """FAISS over the interface catalogue. Imported lazily so --from-interfaces needs
    neither faiss nor torch."""

    def __init__(self):
        import faiss
        import numpy
        from sentence_transformers import SentenceTransformer

        for path in (FAISS_INDEX_PATH, METADATA_PATH, SENTENCE_TRANSFORMER_PATH):
            if not os.path.exists(path):
                raise SystemExit(f"not found: {path}\n"
                                 f"build the index first: python 3.build-faiss-index.py")
        self.numpy = numpy
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)

    def top1(self, text):
        """(interface, cosine similarity) for the nearest catalogue entry.

        The index is inner product over normalised vectors, so the query is
        normalised too and the score is a cosine - comparable with the thresholds
        3.system-eval.py reports.
        """
        vector = self.model.encode([text], convert_to_numpy=True).astype("float32")
        vector /= self.numpy.linalg.norm(vector, axis=1, keepdims=True) + 1e-12
        scores, ids = self.index.search(vector, 1)
        doc_id = int(ids[0, 0])
        if doc_id < 0:
            return None, 0.0
        return self.metadata[doc_id].get("interface"), float(scores[0, 0])


# -------------------------
# The wiring between containers
# -------------------------
def edges(connections):
    """A connection set, comparable between two derivations of the same wiring."""
    return {(c["from"], c["to"], c["as"], bool(c.get("feedback")))
            for c in connections}


def route_inputs(members, connections):
    """Every accepted field no connection feeds, mapped to the members that accept it.

    Derived rather than taken from the model: a field is either fed by a connection
    or supplied by the caller, and making the two halves exhaustive by construction
    is what stops a field silently reaching nothing. The model is asked for `inputs`
    anyway, and `validate_wiring` reports where its answer disagreed with this one.
    """
    wired = {key: {c["as"] for c in connections if c["to"] == key} for key in members}
    inputs = {}
    for key, interface in members.items():
        for field in update_fields(interface):
            if field not in wired[key]:
                inputs.setdefault(field, []).append(key)
    return inputs


def output_subsystem(members, connections):
    """The composed output: the member nothing downstream reads."""
    consumed = {c["from"] for c in connections if not c.get("feedback")}
    sinks = [key for key in members if key not in consumed]
    return (sinks or list(members))[-1]


def reaches(connections, source, target):
    """Whether `target` is reachable from `source` over non-feedback connections."""
    seen, stack = set(), [source]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack += [c["to"] for c in connections
                  if c["from"] == node and not c.get("feedback")]
    return False


def validate_wiring(members, proposed, note):
    """The model's composition, checked against what the members actually accept.

    An analysed wiring is a claim about other people's interfaces, so every part of
    it is checked before it reaches a compose file: both ends must be members, the
    field must be one the target's update endpoint accepts, and a field may have one
    source. A connection that closes a cycle is kept but demoted to `feedback` -
    the orchestrator visits sources before targets and aborts start-up on a cycle,
    and a mutual dependency can only be satisfied by the previous cycle's value
    anyway. `inputs` and `output` are then derived, so they cannot contradict the
    connections that survived.
    """
    connections, seen = [], set()
    for candidate in proposed.get("connections") or []:
        if not isinstance(candidate, dict):
            note(f"dropped a connection that is not an object: {candidate!r}")
            continue
        source, target, field = (str(candidate.get("from") or ""),
                                 str(candidate.get("to") or ""),
                                 str(candidate.get("as") or ""))
        if source not in members or target not in members:
            note(f"dropped {source or '?'} -> {target or '?'}: "
                 f"not both of them are subsystems here")
            continue
        accepted = update_fields(members[target])
        if field not in accepted:
            note(f"dropped {source} -> {target} as {field or '?'}: "
                 f"{target} accepts {accepted}")
            continue
        if (target, field) in seen:
            note(f"dropped {source} -> {target} as {field}: {field} already has a "
                 f"source, and one field takes one")
            continue
        seen.add((target, field))
        connection = {"from": source, "to": target, "as": field}
        if candidate.get("feedback") or source == target:
            connection["feedback"] = True
        elif reaches(connections, target, source):
            connection["feedback"] = True
            note(f"{source} -> {target} as {field} closes a cycle, so it is a feedback "
                 f"connection: {target} is given what {source} held before the cycle")
        connections.append(connection)

    inputs = route_inputs(members, connections)
    claimed = proposed.get("inputs")
    if isinstance(claimed, dict) and claimed != inputs:
        note(f"the analysis routed {sorted(claimed)} as external inputs; the fields no "
             f"connection feeds are {sorted(inputs)}, and those are what is routed")

    output = proposed.get("output")
    if output not in members:
        derived = output_subsystem(members, connections)
        if output is not None:
            note(f"the analysis named {output!r} as the composed output, which is not a "
                 f"subsystem here; using {derived}, which nothing downstream reads")
        output = derived
    return {"connections": connections, "inputs": inputs, "output": output}


def derive_wiring(members, note=say):
    """The wiring the catalogue declares, where it declares any.

    The route `--wiring declared` takes, and the one that predates the analysis: an
    interface says what it serves on its data endpoint (`outputType`) and, per
    telemetry field it cannot produce itself, which output feeds it
    (`relationships`, with `target` naming an interface and `consumes` naming an
    output type). Matching the two gives one connection per edge: image 3 needs a
    BuildingState, image 1 serves one, so `buildingData -> energyForecast as
    buildingState`.

    `target` is preferred because it is exact; `consumes` is the fallback that lets a
    different but equivalent twin take a member's place.

    It is still here for three reasons: it needs no model, so the offline
    catalogue-to-compose generator keeps working; it reproduces the compositions
    published before the analysis existed; and where a catalogue declares
    relationships it is the ground truth the analysis is scored against.
    """
    by_id = {iface.get("@id"): key for key, iface in members.items()}
    by_output = {}
    for key, iface in members.items():
        by_output.setdefault(iface.get("outputType"), key)

    connections = []
    for key, iface in members.items():
        for relationship in iface.get("relationships", []):
            name = relationship.get("name")
            source = (by_id.get(relationship.get("target"))
                      or by_output.get(relationship.get("consumes")))
            if source is None or name not in update_fields(iface):
                note(f"{key}.{name}: nothing in the composition produces it")
                continue
            connection = {"from": source, "to": key, "as": name}
            # A twin reading its own output reads what it held before this cycle.
            # Marking that keeps the dependency graph acyclic, and is how the
            # previous plan reaches the optimiser.
            if relationship.get("feedback") or source == key:
                connection["feedback"] = True
            connections.append(connection)

    return {"connections": connections,
            "inputs": route_inputs(members, connections),
            "output": output_subsystem(members, connections)}


def agreement(members, analysed):
    """How an analysed wiring compares with the one the catalogue declares.

    Only meaningful where the members declare relationships at all - in this
    catalogue the five docker-images twins are the ones that do. It changes nothing
    about what is deployed; it is what makes the analysis a measured stage rather
    than an opaque one, and it is worth a glance before confirming a stack.
    """
    if not any(iface.get("relationships") for iface in members.values()):
        return None
    declared = edges(derive_wiring(members, note=lambda message: None)["connections"])
    found = edges(analysed["connections"])
    return {"declared": sorted(map(list, declared)),
            "recovered": sorted(map(list, declared & found)),
            "missed": sorted(map(list, declared - found)),
            "extra": sorted(map(list, found - declared))}


def compose(members, wiring, wiring_source):
    """The new interface a composition creates.

    The `<name>_properties_and_telemetries` blocks are the shape 3.system-eval.py
    composes and its fill-in prompt reads. The endpoint and wiring keys are what makes
    the composed interface deployable: it serves the same two endpoints as any twin,
    which is what lets a composed system be composed again.
    """
    topics = {str(iface.get("@id", "")).split(":")[1] for iface in members.values()
              if str(iface.get("@id", "")).count(":") >= 2}
    words = re.split(r"[_\s]+", topics.pop()) if len(topics) == 1 else []
    name = " ".join(w.capitalize() for w in words) if words else "Composed System"
    identifier = (f"dtmi:{'_'.join(w.lower() for w in words)}:"
                  f"{''.join(w.capitalize() for w in words)};1" if words
                  else "dtmi:composed_system:ComposedSystem;1")

    composed = {"@context": "dtmi:dtdl:context;2", "@id": identifier, "@type": "Interface",
                "displayName": name, "dataEndpoint": DATA_ENDPOINT,
                "updateEndpoint": UPDATE_ENDPOINT,
                "outputType": members[wiring["output"]].get("outputType"),
                "subsystemInterfaces": {key: iface.get("@id")
                                        for key, iface in members.items()},
                "wiringSource": wiring_source,
                **{key: value for key, value in wiring.items()}}
    for key, iface in members.items():
        composed[f"{key}_properties_and_telemetries"] = iface.get("contents", [])
    return composed


def properties_of(interface, filled):
    """One subsystem's bound configuration, with what the model must not extract put
    back from the catalogue."""
    bound = {"interface": interface.get("@id"),
             "dockerImage": declared_value(interface, "dockerImage")}
    for name in property_names(interface):
        if name not in NOT_EXTRACTED:
            bound[name] = filled.get(name)
    return bound


# -------------------------
# docker-compose.yaml
# -------------------------
def env_name(prop):
    """buildingId -> BUILDING_ID, the convention every generated twin reads."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", prop).upper()


def env_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(value)                 # scalar lists read better as CSV
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def scalar(text):
    """A YAML double-quoted scalar. JSON string escaping is a subset of YAML's, so
    json.dumps emits a valid one - which keeps this dependency-free and safe for the
    JSON that object properties and the COMPOSITION document carry."""
    return json.dumps(str(text))


def service_names(members):
    """`ziren/building-data-twin:latest` -> `building-data-twin`, plus the
    orchestrator. These name the containers the composition's urls point at."""
    names, taken = {}, set()
    for key, interface in members.items():
        image = str(declared_value(interface, "dockerImage") or key)
        name = re.sub(r"[^a-z0-9._-]+", "-", image.split("/")[-1].split(":")[0].lower())
        name = name.strip("-") or key.lower()
        while name in taken:
            name += "-2"
        names[key] = name
        taken.add(name)
    names["#orchestrator"] = "composition-orchestrator"
    return names


def render_compose(services, base_port):
    """services: [(name, image, interface id, {env})] in port order."""
    lines = ["# Generated by 4.deploy.py.",
             "# Properties are bound here, at instantiation. Telemetry is posted per",
             "# cycle to each service's update endpoint and never appears below.",
             "services:"]
    for offset, (name, image, interface, environment) in enumerate(services):
        lines += [f"  {name}:", f"    image: {scalar(image)}"]
        if interface:
            lines += ["    labels:", f"      dtdl.interface: {scalar(interface)}"]
        lines += ["    ports:",
                  f"      - {scalar(f'{base_port + offset}:{CONTAINER_PORT}')}"]
        if environment:
            lines.append("    environment:")
            lines += [f"      {key}: {scalar(value)}" for key, value in environment.items()]
    return "\n".join(lines) + "\n"


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(obj if isinstance(obj, str)
                     else json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    say(f"  {path}")


# -------------------------
# One attempt
# -------------------------
def attempt(arguments, query, catalogue, number, feedback):
    """One pass of the whole process, from the query to the files on disk.

    Returns the compose file's text, which is what the user is asked to confirm. A
    restart calls this again from the top: the decomposition, the retrieval, the
    analysis and the fill-in are all redone, because a wiring the user rejected may
    be the consequence of any of them.
    """
    options = decoding_for(number)
    out_dir = arguments.out_dir
    members, origins, couplings, sub_queries, flows = {}, {}, {}, [], []

    # ---- decomposition: the parts, and how they would exchange data ----
    if arguments.from_interfaces:
        say(f"composing {len(arguments.from_interfaces)} interface file(s)")
        for path in arguments.from_interfaces:
            for interface in load_interfaces(path):
                members[subsystem_key(interface, members)] = interface
    else:
        say(f"decomposing {len(query)} characters")
        sub_queries = parse_json(ollama(build_decompose_prompt(query,
                                                              arguments.max_subsystems)
                                        + guidance(feedback),
                                        arguments.timeout, options))
        if not isinstance(sub_queries, list) or not sub_queries:
            raise SystemExit("decomposition did not return a non-empty JSON array")
        sub_queries = [str(s).strip() for s in sub_queries][:arguments.max_subsystems]
        for position, text in enumerate(sub_queries, 1):
            say(f"  part {position}: {text[:66]}")

        say("\nanalysing how the parts exchange data")
        analysis = parse_json(ollama(build_couplings_prompt(query, sub_queries)
                                     + guidance(feedback), arguments.timeout, options))
        couplings = analysis if isinstance(analysis, dict) else {}
        for flow in couplings.get("flows") or []:
            if isinstance(flow, dict):
                say(f"  part {flow.get('from')} -> part {flow.get('to')}"
                    f"{' (feedback)' if flow.get('feedback') else ''}: "
                    f"{str(flow.get('data'))[:50]}")
        if not couplings.get("flows"):
            say("  [warn] no data flow between the parts was found; the composition "
                "will have no internal wiring unless the analysis below finds one")

        say("\nretrieving")
        for index, sub_query in enumerate(sub_queries, 1):
            interface, score = catalogue().top1(sub_query)
            if interface is None:
                say(f"  [warn] no hit for: {sub_query[:70]}")
                continue
            key = subsystem_key(interface, members)
            members[key] = interface
            origins[key] = index
            say(f"  {key:<18} sim={score:.4f} {interface.get('@id')}")
        # The coupling analysis speaks in part numbers, the wiring analysis in
        # subsystem keys, so the flows are translated onto whatever retrieval
        # returned. A flow whose end went unretrieved is dropped: there is no
        # subsystem for it to reach. `couplings` keeps the untranslated answer, which
        # is what `decomposition.json` records.
        by_part = {index: key for key, index in origins.items()}
        flows = [{**flow, "from": by_part[flow["from"]], "to": by_part[flow["to"]]}
                 for flow in couplings.get("flows") or []
                 if isinstance(flow, dict) and flow.get("from") in by_part
                 and flow.get("to") in by_part]

        # Written here rather than with the other files at the end: this stage costs
        # two model calls and a retrieval, and a later stage failing must not take
        # the record of them with it. A missed retrieval is diagnosed from this file.
        write(os.path.join(out_dir, "decomposition.json"),
              {"attempt": number, "feedback": feedback or None,
               "parts": [{"part": index, "sub_query": text,
                          "subsystem": by_part.get(index),
                          "interface": (members[by_part[index]].get("@id")
                                        if index in by_part else None)}
                         for index, text in enumerate(sub_queries, 1)],
               "couplings": couplings, "flows_on_subsystems": flows})
    if not members:
        raise SystemExit("no subsystems to deploy")

    # ---- composition: the wiring, analysed or declared ----
    mode = arguments.wiring
    if mode == "auto":
        mode = "declared" if arguments.no_fill else "analysed"
    proposed, notes = {}, []
    if len(members) == 1:
        # One interface is not a composition: there is nothing to wire it to.
        wiring, mode = {"connections": [], "inputs": route_inputs(members, []),
                        "output": next(iter(members))}, "single"
    elif mode == "declared":
        say("\nwiring from what the catalogue declares")
        wiring = derive_wiring(members, note=lambda message: say(f"  [warn] {message}"))
    else:
        say(f"\ncomposing the subsystems with {OLLAMA_MODEL}")
        answer = parse_json(ollama(build_wiring_prompt(query, members, flows)
                                   + guidance(feedback), arguments.timeout, options))
        proposed = answer if isinstance(answer, dict) else {}
        if not isinstance(answer, dict):
            say("  [warn] the wiring analysis did not return an object; nothing is wired")
        wiring = validate_wiring(members, proposed, notes.append)
        for message in notes:
            say(f"  [warn] {message}")

    composed = compose(members, wiring, mode)
    say(f"\ncomposed {composed['@id']}")
    for connection in wiring["connections"]:
        why = next((str(c.get("why")) for c in proposed.get("connections") or []
                    if isinstance(c, dict) and c.get("from") == connection["from"]
                    and c.get("to") == connection["to"]
                    and c.get("as") == connection["as"] and c.get("why")), "")
        say(f"  {connection['to']:<18} {'<<' if connection.get('feedback') else '<-'} "
            f"{connection['from']:<18} as {connection['as']}"
            f"{'   # ' + why[:60] if why else ''}")
    for field, targets in wiring["inputs"].items():
        say(f"  external input {field} -> {', '.join(targets)}")

    scored = agreement(members, wiring)
    if scored and mode == "analysed":
        say(f"\nagainst the {len(scored['declared'])} connections these interfaces "
            f"declare: {len(scored['recovered'])} recovered, "
            f"{len(scored['missed'])} missed, {len(scored['extra'])} extra")
        for edge in scored["missed"]:
            say(f"  missed {edge[0]} -> {edge[1]} as {edge[2]}")
        for edge in scored["extra"]:
            say(f"  extra  {edge[0]} -> {edge[1]} as {edge[2]}")

    # ---- fill-in ----
    filled = {key: {} for key in members}
    if arguments.no_fill:
        say("\nfill-in skipped; every property will be null")
    else:
        say(f"\nfilling from {OLLAMA_MODEL}")
        answer = parse_json(ollama(build_fillin_prompt(query, composed, members)
                                   + guidance(feedback), arguments.timeout, options))
        returned = answer.get("subsystems") if isinstance(answer, dict) else None
        if not isinstance(returned, dict):
            raise SystemExit("fill-in did not return a subsystems object")
        for key in members:
            filled[key] = returned.get(key) if isinstance(returned.get(key), dict) else {}
            stated = sum(1 for value in filled[key].values() if value is not None)
            say(f"  {key:<18} {stated} properties stated")

    # ---- instance and compose file ----
    names = service_names(members)
    bound = {key: properties_of(iface, filled[key]) for key, iface in members.items()}
    services = [(names[key], values["dockerImage"], values["interface"],
                 {env_name(prop): env_value(value) for prop, value in values.items()
                  if prop not in NOT_EXTRACTED and value is not None})
                for key, values in bound.items()]

    say("")
    if len(members) == 1:
        # Nothing to orchestrate: an orchestrator with a single member would only
        # forward the caller's update.
        instance = next(iter(bound.values()))
    else:
        instance = {"interface": composed["@id"], "name": composed["displayName"],
                    "dockerImage": ORCHESTRATOR_IMAGE, "dataEndpoint": DATA_ENDPOINT,
                    "updateEndpoint": UPDATE_ENDPOINT, "subsystems": bound,
                    "wiringSource": mode, **wiring}
        composition = {
            "interface": composed["@id"], "name": composed["displayName"],
            "subsystems": {key: {"url": f"http://{names[key]}:{CONTAINER_PORT}",
                                 "dataEndpoint": iface.get("dataEndpoint") or DATA_ENDPOINT,
                                 "updateEndpoint": iface.get("updateEndpoint")
                                 or UPDATE_ENDPOINT}
                           for key, iface in members.items()},
            **wiring}
        services.append((names["#orchestrator"], ORCHESTRATOR_IMAGE, composed["@id"],
                         {"COMPOSITION": json.dumps(composition, separators=(",", ":"))}))
        write(os.path.join(out_dir, "composed-interface.json"), composed)
        write(os.path.join(out_dir, "composition.json"), composition)

    write(os.path.join(out_dir, "wiring.json"),
          {"source": mode, "attempt": number, "proposed": proposed or None,
           "notes": notes, "accepted": wiring, "agreement": scored})
    write(os.path.join(out_dir, "instance.json"), instance)
    text = render_compose(services, arguments.base_port)
    write(os.path.join(out_dir, "docker-compose.yaml"), text)
    return text


# -------------------------
# The user's stage
# -------------------------
def confirm(text, attempts_left):
    """Show the stack and ask. `True` accepts, `False` stops, a string restarts.

    The compose file is the thing the user actually gets, so it is the thing they
    are shown - not a summary of it. Declining restarts the process rather than
    patching the file, because what produced a wrong stack is upstream of it: the
    string is the feedback for the next attempt, and it may be empty.
    """
    say("\n" + "-" * 72)
    say(text.rstrip())
    say("-" * 72)
    # A user out of restarts is still asked: accepting is always theirs to do, and
    # only the restart runs out.
    while True:
        say("\n[y] accept and deploy this stack   "
            + ("[r] restart the process   " if attempts_left else "")
            + "[q] quit")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            say("no answer")
            return False
        if choice in ("y", "yes", ""):
            return True
        if choice in ("q", "quit", "n", "no"):
            return False
        if choice in ("r", "restart"):
            if not attempts_left:
                say("  no attempts left; --max-attempts allows more")
                continue
            say("what should be different? (one line, or blank to just try again)")
            try:
                # "" restarts without changing a prompt; only the seed moves.
                return input("> ").strip()
            except EOFError:
                return ""
        say(f"  {choice!r} is not one of "
            + ("y, r, q" if attempts_left else "y, q"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("query", nargs="?", default="",
                        help="the request; @FILE reads a file, - reads stdin")
    parser.add_argument("-o", "--out-dir", default=os.path.join("outputs", "deploy"))
    parser.add_argument("--from-interfaces", nargs="+", metavar="PATH",
                        help="compose these interfaces; skip decomposition and retrieval")
    parser.add_argument("--no-fill", action="store_true",
                        help="leave every property null instead of calling the model")
    parser.add_argument("--wiring", choices=("auto", "analysed", "declared"),
                        default="auto",
                        help="how the connections are found: analysed by the model from "
                             "what each member serves and accepts, or read from the "
                             "relationships the catalogue declares. auto is declared "
                             "with --no-fill (which is offline), analysed otherwise")
    parser.add_argument("--max-subsystems", type=int, default=5)
    parser.add_argument("--base-port", type=int, default=8081)
    parser.add_argument("--timeout", type=int, default=120, help="model timeout, seconds")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="how many times the process may be restarted, in total")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="do not ask; keep the first stack generated")
    arguments = parser.parse_args()

    query = arguments.query
    if query == "-":
        query = sys.stdin.read()
    elif query.startswith("@"):
        with open(query[1:], encoding="utf-8") as handle:
            query = handle.read()
    query = query.strip()
    if not query and not arguments.from_interfaces:
        parser.error("a query is required unless --from-interfaces is given")

    # Loading FAISS and the encoder costs more than a model call, so it happens once
    # however many attempts the user takes.
    loaded = []

    def catalogue():
        if not loaded:
            loaded.append(Catalogue())
        return loaded[0]

    # Nothing to confirm without someone to ask: piping a query in through stdin
    # closes it, and so does a cron job. `--no-fill` is not asked about either -
    # every property is null, so no twin in that stack would survive start-up and
    # there is nothing to accept.
    interactive = not (arguments.yes or arguments.no_fill) and sys.stdin.isatty()
    feedback, number = None, 1
    while True:
        if number > 1:
            say(f"\n{'=' * 72}\nattempt {number}"
                f"{': ' + feedback.strip() if feedback.strip() else ''}\n{'=' * 72}")
        text = attempt(arguments, query, catalogue, number, feedback)
        if not interactive:
            say(f"\n  cd {arguments.out_dir} && docker compose up")
            return 0
        answer = confirm(text, arguments.max_attempts - number)
        if answer is True:
            say(f"\n  cd {arguments.out_dir} && docker compose up")
            return 0
        if answer is False:
            # Not an error, but not a stack to bring up either: exiting non-zero is
            # what stops a `4.deploy.py ... && docker compose up` at the `&&`.
            say(f"\nnot accepted. {arguments.out_dir} holds attempt {number}; "
                f"run again to compose it afresh.")
            return 1
        feedback = answer
        number += 1


if __name__ == "__main__":
    sys.exit(main())
