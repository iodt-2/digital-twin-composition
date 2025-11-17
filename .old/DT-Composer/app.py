# app.py
# Digital Twin Interface Composer
# Standardized for DTDL interface objects (each corpus row is a single DTDL interface JSON object),
# e.g. {"@id": "...", "@type":"Interface", "displayName":"...", "description":"...", "contents":[...]}
#
# Features:
# - Two-column (left=INPUT, right=OUTPUT) per step, with code highlighting traces
# - Step 1: LLM decomposition (OpenAI-compatible chat endpoint)
# - Step 2: FAISS retrieval (user must confirm one hit per component)
# - Step 3: LLM property filling (nullable values preserved)
# - Step 4: Web backfill (Google PSE + page content fetch via BeautifulSoup)
# - Step 5: Final DTDL instances
# - Embeddings providers: openai | gemini | local (sentence-transformers)
# - FAISS auto rebuilds if embedding provider/model/dimension changes
#
# Corpus format is NOT "compatible" or mixed—it's ONLY the DTDL object format shown above.

import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, render_template, request, jsonify, url_for, send_file, flash
from dotenv import load_dotenv
import requests
import numpy as np
import faiss  # pip install faiss-cpu
from bs4 import BeautifulSoup  # pip install beautifulsoup4

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# --------------------- Config ---------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()  # openai | gemini | local
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")

TOP_K = int(os.getenv("TOP_K", 5))
WEB_FETCH_MAX = int(os.getenv("WEB_FETCH_MAX", 2))  # how many links to fetch per query

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "faiss_index.bin"
INDEX_META_PATH = DATA_DIR / "faiss_index.meta.json"
CORPUS_PATH = DATA_DIR / "interfaces.jsonl"

# In-memory store (list of DTDL objects)
CORPUS: List[Dict[str, Any]] = []
# mapping: row index -> DTDL object
ID_TO_ROW: Dict[int, Dict[str, Any]] = {}
INDEX = None  # faiss index handle

# --------------------- Utilities ---------------------
def redact(s: str) -> str:
    if not s: return s
    if len(s) <= 8: return "****"
    return s[:4] + "..." + s[-4:]

def trunc(x: Any, limit: int = 2000) -> Any:
    if isinstance(x, str):
        return x if len(x) <= limit else x[:limit] + f"... [truncated {len(x)-limit} chars]"
    if isinstance(x, list):
        return [trunc(i, limit) for i in x]
    if isinstance(x, dict):
        return {k: trunc(v, limit) for k, v in x.items()}
    return x

def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        import re
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if m:
            return json.loads(m.group(1))
        raise

def dtdl_label(dtdl: Dict[str, Any]) -> str:
    if dtdl.get("displayName"): return str(dtdl["displayName"])
    rid = (dtdl.get("@id") or "unknown").strip()
    if ":" in rid:
        tail = rid.split(":")[-1]
        return tail.split(";")[0]
    return rid

def dtdl_text_for_embedding(dtdl: Dict[str, Any]) -> str:
    """Build a descriptive string from a DTDL interface for embedding."""
    desc = dtdl.get("description") or ""
    dn = dtdl.get("displayName") or dtdl.get("@id") or ""
    props, tels, cmds = [], [], []
    for c in dtdl.get("contents", []):
        t = c.get("@type")
        nm = c.get("name")
        if t == "Property": props.append(nm)
        elif t == "Telemetry": tels.append(nm)
        elif t == "Command": cmds.append(nm)
    bits = [f"{dn}: {desc}"]
    if props: bits.append("Properties: " + ", ".join([p for p in props if p]))
    if tels:  bits.append("Telemetry: " + ", ".join([t for t in tels if t]))
    if cmds:  bits.append("Commands: " + ", ".join([c for c in cmds if c]))
    return " | ".join([b for b in bits if b])

def writable_properties(dtdl: Dict[str, Any]) -> List[str]:
    names = []
    for c in dtdl.get("contents", []):
        if c.get("@type") == "Property":
            # In your standard, "writable" is not required; we treat all properties as fillable
            names.append(c.get("name"))
    return [n for n in names if n]

# --------------------- LLM chat with trace ---------------------
def chat_json_with_trace(messages: List[Dict[str, Any]], response_schema_hint: str | None = None
                         ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type":"json_object"},
    }
    if response_schema_hint:
        payload["messages"] = [
            {"role":"system","content": f"Return strictly valid minified JSON following this structure: {response_schema_hint}"}
        ] + messages

    input_trace = {
        "endpoint": f"{OPENAI_API_BASE}/chat/completions",
        "model": OPENAI_MODEL,
        "headers": {"Authorization": f"Bearer {redact(OPENAI_API_KEY)}", "Content-Type":"application/json"},
        "payload": trunc(payload)
    }
    resp = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    js = resp.json()
    content = js["choices"][0]["message"]["content"]
    out = safe_json_loads(content)
    input_trace["response_excerpt"] = trunc(js, 2000)
    return out, input_trace

# --------------------- Embeddings (multi-provider) ---------------------
_local_st_model = None

def _embed_local(texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    global _local_st_model
    try:
        if _local_st_model is None:
            from sentence_transformers import SentenceTransformer  # optional
            # _local_st_model = SentenceTransformer("all-MiniLM-L6-v2")
            _local_st_model = SentenceTransformer("./dt-v7-deploy-final")
        vecs = _local_st_model.encode(texts, normalize_embeddings=False)
        vecs = np.asarray(vecs, dtype="float32")
        # trace = {"provider":"local","model":"all-MiniLM-L6-v2","queries": texts, "dim": int(vecs.shape[1])}
        trace = {"provider":"local","model":"dt-v7-deploy-final","queries": texts, "dim": int(vecs.shape[1])}
        return vecs, trunc(trace)
    except Exception as e:
        raise RuntimeError("Local embeddings require sentence-transformers==2.7.0") from e

def _embed_openai(texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": texts}
    trace = {
        "provider":"openai","endpoint": f"{OPENAI_API_BASE}/embeddings",
        "headers": {"Authorization": f"Bearer {redact(OPENAI_API_KEY)}","Content-Type":"application/json"},
        "payload": trunc(payload)
    }
    resp = requests.post(f"{OPENAI_API_BASE}/embeddings", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()["data"]
    vecs = np.array([row["embedding"] for row in data], dtype="float32")
    trace["dim"] = int(vecs.shape[1])
    return vecs, trunc(trace)

def _embed_gemini(texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:batchEmbedContents"
    params = {"key": GEMINI_API_KEY}
    requests_payload = [{"model": f"models/{GEMINI_EMBED_MODEL}", "content": {"parts":[{"text": t}]}} for t in texts]
    trace = {"provider":"gemini","endpoint": url, "(query-param) key": redact(GEMINI_API_KEY), "payload": trunc({"requests": requests_payload})}
    resp = requests.post(url, params=params, json={"requests": requests_payload}, timeout=120)
    resp.raise_for_status()
    js = resp.json()
    vecs = [np.array(item["values"], dtype="float32") for item in js.get("embeddings", [])]
    if not vecs:
        arr = np.zeros((0,768), dtype="float32"); trace["dim"] = 768; return arr, trunc(trace)
    arr = np.vstack(vecs); trace["dim"] = int(arr.shape[1]); return arr, trunc(trace)

def embed_with_trace(texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if EMBEDDINGS_PROVIDER == "gemini": return _embed_gemini(texts)
    if EMBEDDINGS_PROVIDER == "local":  return _embed_local(texts)
    return _embed_openai(texts)

# --------------------- Google PSE + page fetch ---------------------
def fetch_page_text(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = " ".join(soup.get_text("\n").split())
        return text
    except Exception as e:
        return f"[fetch-error] {e}"

def google_pse_search_with_trace(q: str, num: int = 3) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return [], {"provider":"google-pse","note":"API key or CSE ID missing","query": q}
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": num}
    trace = {"provider":"google-pse","endpoint":"https://www.googleapis.com/customsearch/v1",
             "params": {"key": redact(GOOGLE_API_KEY), "cx": GOOGLE_CSE_ID, "q": q, "num": num}}
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    items = js.get("items", []) or []

    hits = []
    for i, it in enumerate(items):
        link = it.get("link")
        body = None
        if link and i < WEB_FETCH_MAX:
            full = fetch_page_text(link)
            body = full[:2000] + (f"... [truncated {len(full)-2000} chars]" if len(full) > 2000 else "")
        hits.append({"title": it.get("title"), "snippet": it.get("snippet"), "link": link, "content": body})
    trace["response_excerpt"] = trunc(js, 1500)
    return hits, trace

# --------------------- Corpus / FAISS (DTDL-only) ---------------------
def load_corpus() -> List[Dict[str, Any]]:
    """
    Load a corpus where each entry is a pure DTDL Interface object:
      {"@id": "...", "@type":"Interface", "displayName":"...", "description":"...", "contents":[...]}
    File can be JSONL (preferred) or a JSON array.
    """
    if not CORPUS_PATH.exists(): return []
    txt = CORPUS_PATH.read_text(encoding="utf-8").strip()
    if not txt: return []
    try:
        obj = json.loads(txt)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict): return [obj]
    except json.JSONDecodeError:
        pass
    # JSONL
    recs = []
    for i, line in enumerate(txt.splitlines(), start=1):
        line = line.strip()
        if not line: continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON on line {i} of {CORPUS_PATH.name}: {e}") from e
    return recs

def current_embed_signature() -> Dict[str, Any]:
    vec, trace = embed_with_trace(["__dim_probe__"])
    if vec.ndim == 1: vec = vec.reshape(1, -1)
    dim = int(vec.shape[1]) if vec.size else trace.get("dim", 768)
    model_name = EMBED_MODEL if EMBEDDINGS_PROVIDER == "openai" else (
        GEMINI_EMBED_MODEL if EMBEDDINGS_PROVIDER == "gemini" else "local/all-MiniLM-L6-v2"
    )
    return {"provider": EMBEDDINGS_PROVIDER, "model": model_name, "dim": dim}

def read_index_meta() -> dict | None:
    if INDEX_META_PATH.exists():
        try: return json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        except Exception: return None
    return None

def write_index_meta(meta: dict) -> None:
    INDEX_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def ensure_faiss_index():
    """Build or load FAISS index from a corpus of DTDL interfaces (standardized)."""
    global CORPUS, ID_TO_ROW
    CORPUS = load_corpus()
    if not CORPUS:
        raise RuntimeError(f"{CORPUS_PATH} is empty. Add DTDL interface lines (your example format).")

    # Normalize: compute 'label' and 'text' for each DTDL interface
    normed: List[Dict[str, Any]] = []
    for dtdl in CORPUS:
        if not isinstance(dtdl, dict) or dtdl.get("@type") != "Interface":
            raise RuntimeError("All corpus entries must be DTDL Interface objects.")
        dtdl["_label"] = dtdl_label(dtdl)
        dtdl["_text"]  = dtdl_text_for_embedding(dtdl)
        normed.append(dtdl)
    CORPUS = normed

    sig = current_embed_signature()
    rebuild = True
    if INDEX_PATH.exists() and INDEX_META_PATH.exists():
        stored = read_index_meta() or {}
        rebuild = any([stored.get("provider") != sig["provider"],
                       stored.get("model")    != sig["model"],
                       stored.get("dim")      != sig["dim"]])

    if not rebuild:
        index = faiss.read_index(str(INDEX_PATH))
        if getattr(index, "d", None) != sig["dim"]:
            rebuild = True

    if rebuild:
        texts = [row["_text"] for row in CORPUS]
        vecs, _ = embed_with_trace(texts)
        vecs = vecs.astype("float32")
        if vecs.ndim == 1: vecs = vecs.reshape(1, -1)
        d = int(vecs.shape[1])
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        faiss.write_index(index, str(INDEX_PATH))
        write_index_meta(sig)

    ID_TO_ROW = {i: row for i, row in enumerate(CORPUS)}
    return index

# --------------------- Domain logic ---------------------
def decompose_query_with_trace(user_query: str) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    messages = [
        {"role":"system","content": "You are a systems architect. Break a product request into DT components. "
                                    "Return JSON with 'components': [{name, description, key_requirements[]}]."},
        {"role":"user","content": user_query}
    ]
    js, trace = chat_json_with_trace(
        messages, response_schema_hint='{"components":[{"name":"...","description":"...","key_requirements":["..."]}]}'
    )
    # Normalize to strict shape
    components: List[Dict[str, Any]] = []
    for it in (js.get("components") or []):
        if not isinstance(it, dict): continue
        components.append({
            "name": it.get("name") or "component",
            "description": it.get("description", ""),
            "key_requirements": it.get("key_requirements", []) or []
        })
    return {"components": components}, trace

def fill_properties_with_llm_with_trace(user_query: str, dtdl_iface: Dict[str, Any]) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    props = writable_properties(dtdl_iface)
    schema_hint = {
        "dtmi": dtdl_iface.get("@id", ""),
        "interface_display_name": dtdl_iface.get("displayName", ""),
        "properties": [{"property": p, "value": None, "source":"user", "note": ""} for p in props]
    }
    sys = ("Given a user's product description, infer values for the properties of the provided DTDL interface. "
           "Only use facts or information stated in the user's input; if unknown, set value to null. Use numeric types "
           "where appropriate. Return strictly the JSON shape I provided. Use original dockerImage provided in the"
           "interface.")
    messages = [
        {"role":"system","content": sys},
        {"role":"user","content": "User provided information:\n" + user_query + "\n\nDTDL interface:\n" + json.dumps(dtdl_iface)}
    ]
    raw, trace = chat_json_with_trace(messages, response_schema_hint=json.dumps(schema_hint))

    # If model returned a flat map, coerce
    if isinstance(raw, dict) and "properties" not in raw and all(isinstance(k, str) for k in raw.keys()):
        raw = {
            "dtmi": dtdl_iface.get("@id",""),
            "interface_display_name": dtdl_iface.get("displayName",""),
            "properties": [{"property":k,"value":v,"source":"inferred"} for k,v in raw.items()]
        }

    raw.setdefault("dtmi", dtdl_iface.get("@id",""))
    raw.setdefault("interface_display_name", dtdl_iface.get("displayName",""))
    raw.setdefault("properties", [{"property": p, "value": None, "source":"user"} for p in props])

    # Ensure all properties present
    have = {p.get("property") for p in raw.get("properties", []) if isinstance(p, dict)}
    for p in props:
        if p not in have:
            raw["properties"].append({"property": p, "value": None, "source":"user"})
    return raw, trace

def backfill_missing_with_web_with_trace(instance: Dict[str,Any], component_name: str
                                         ) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    web_traces: List[Dict[str,Any]] = []
    for p in instance.get("properties", []):
        if p.get("value") is None:
            q = f"{component_name} {instance.get('dtmi','')} property {p.get('property','')} typical value"
            hits, trace = google_pse_search_with_trace(q, num=3)
            if hits:
                best = hits[0]
                snippet = best.get("content") or best.get("snippet") or ""
                snippet_short = snippet[:300] + (f"... [truncated]" if len(snippet) > 300 else "")
                p["note"] = f"Suggestion: {snippet_short} (source: {best.get('link')})"
                p["source"] = "web"
            web_traces.append(trace)
    return instance, web_traces

def generate_final_instance(instance: Dict[str,Any]) -> Dict[str,Any]:
    props = {p["property"]: p.get("value") for p in instance.get("properties", []) if isinstance(p, dict) and "property" in p}
    return {"$dtId": f"inst:{instance.get('dtmi','unknown')}:{uuid.uuid4().hex[:8]}",
            "$metadata": {"$model": instance.get("dtmi","")}, **props}

# --------------------- Routes ---------------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/download")
def download_file():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        flash("File not found.","danger")
        return render_template("index.html")
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

# Step 1
@app.post("/api/decompose")
def api_decompose():
    global INDEX
    if INDEX is None: INDEX = ensure_faiss_index()
    data = request.get_json(force=True) or {}
    q = (data.get("query") or "").strip()
    if not q: return jsonify({"error":"missing query"}), 400
    out, in_trace = decompose_query_with_trace(q)
    return jsonify({"input": in_trace, "output": out})

# Step 2
@app.post("/api/retrieve")
def api_retrieve():
    global INDEX
    if INDEX is None: INDEX = ensure_faiss_index()
    data = request.get_json(force=True) or {}
    comps = data.get("components", [])
    if not isinstance(comps, list): return jsonify({"error":"components must be a list"}), 400

    # Build queries from components
    queries = []
    for c in comps:
        reqs = ", ".join(c.get("key_requirements", []))
        q = f"{c.get('name','component')}. Requirements: {reqs}. {c.get('description','')}"
        queries.append(q)

    qv, emb_trace = embed_with_trace(queries)
    if qv.ndim == 1: qv = qv.reshape(1, -1)
    faiss.normalize_L2(qv)

    retrieval: Dict[str, List[Dict[str,Any]]] = {}
    try:
        for qi, c in enumerate(comps):
            D, I = INDEX.search(qv[qi:qi+1], TOP_K)
            hits = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1: continue
                dtdl = ID_TO_ROW[idx]
                hits.append({"score": float(score),
                             "id": dtdl.get("@id"),
                             "label": dtdl.get("_label"),
                             "text": dtdl.get("_text"),
                             "dtdl": dtdl})
            retrieval[c.get("name","component")] = hits
    except AssertionError:
        INDEX = ensure_faiss_index()
        for qi, c in enumerate(comps):
            D, I = INDEX.search(qv[qi:qi+1], TOP_K)
            hits = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1: continue
                dtdl = ID_TO_ROW[idx]
                hits.append({"score": float(score),
                             "id": dtdl.get("@id"),
                             "label": dtdl.get("_label"),
                             "text": dtdl.get("_text"),
                             "dtdl": dtdl})
            retrieval[c.get("name","component")] = hits

    return jsonify({"input": emb_trace, "output": retrieval})

# Step 2.5 confirm
@app.post("/api/confirm")
def api_confirm():
    data = request.get_json(force=True) or {}
    selection = data.get("selection", {})
    if not isinstance(selection, dict):
        return jsonify({"error": "selection must be an object"}), 400
    # selection may contain {_action:"delete"} entries — just echo them
    return jsonify({"input": {"selection": trunc(selection)}, "output": selection})


# Step 3
@app.post("/api/fill")
def api_fill():
    data = request.get_json(force=True) or {}
    q = data.get("query", "")
    selection = data.get("selection", {})
    if not isinstance(selection, dict):
        return jsonify({"error": "selection must be an object"}), 400

    outputs: Dict[str, Any] = {}
    inputs: Dict[str, Any] = {}
    for comp, hit in selection.items():
        # Skip deleted components
        if isinstance(hit, dict) and hit.get("_action") == "delete":
            outputs[comp] = {"chosen": {"_action": "delete"}, "filled": None}
            continue

        iface = (hit or {}).get("dtdl", {})
        filled, trace = fill_properties_with_llm_with_trace(q, iface)
        outputs[comp] = {"chosen": hit, "filled": filled}
        inputs[comp] = trace

    return jsonify({"input": trunc(inputs), "output": outputs})


# Step 4
@app.post("/api/webfill")
def api_webfill():
    data = request.get_json(force=True) or {}
    fills = data.get("fills", {})
    if not isinstance(fills, dict):
        return jsonify({"error": "fills must be an object"}), 400

    out: Dict[str, Any] = {}
    in_traces: Dict[str, Any] = {}

    for comp, bundle in fills.items():
        # Deleted components stay deleted
        if isinstance(bundle, dict) and (bundle.get("chosen", {}) or {}).get("_action") == "delete":
            out[comp] = None
            in_traces[comp] = [{"note": "deleted component"}]
            continue

        filled = bundle["filled"] if (isinstance(bundle, dict) and "filled" in bundle) else bundle
        if filled is None:
            out[comp] = None
            in_traces[comp] = [{"note": "no data (deleted?)"}]
            continue

        enriched, traces = backfill_missing_with_web_with_trace(filled, comp)
        out[comp] = enriched
        in_traces[comp] = traces

    return jsonify({"input": trunc(in_traces), "output": out})


# Step 5
@app.post("/api/instances")
def api_instances():
    data = request.get_json(force=True) or {}
    webfilled = data.get("webfilled", {})
    if not isinstance(webfilled, dict):
        return jsonify({"error": "webfilled must be an object"}), 400

    final_instances = {}
    for comp, inst in webfilled.items():
        if inst is None:   # deleted component
            continue
        final_instances[comp] = generate_final_instance(inst)

    out_dir = Path("/tmp") / f"dtc_{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = out_dir / "instances.json"
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(final_instances, f, ensure_ascii=False, indent=2)

    return jsonify({
        "input": {"webfilled_excerpt": trunc(webfilled)},
        "output": {"instances": final_instances, "download": url_for("download_file", path=str(pack_path))}
    })


# --------------------- Entrypoint ---------------------
if __name__ == "__main__":
    try:
        if "INDEX" not in globals() or INDEX is None:
            local_files = [INDEX_PATH, INDEX_META_PATH]
            for local_file in local_files:
                if os.path.exists(local_file):
                    os.remove(local_file)
                    print(f"[startup] warmup: file {local_file} deleted successfully.")
            print("[startup] warmup: loading/building FAISS ...")
            INDEX = ensure_faiss_index()
            sig = read_index_meta() or current_embed_signature()
            print(f"[startup] FAISS ready | provider={sig.get('provider')} | model={sig.get('model')} | dim={sig.get('dim')} | corpus={len(CORPUS)}")
    except Exception as e:
        print(f"[startup] Warning: {e}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
