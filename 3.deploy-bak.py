import contextlib
import io
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
from flask import Flask, Response, jsonify, render_template_string, request

# ----------------------------
# Config (env overridable)
# ----------------------------
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://10.1.1.1:60002")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

# Original prompt idea preserved from your script, but now the user's query is injected dynamically. :contentReference[oaicite:1]{index=1}
INITIAL_PROMPT_TEMPLATE = """
You are an agentic AI system designed to generate a specific Digital Twin instance from the interface in the repository.
Each time, you need to choose the most suitable action based on the current situation to achieve the goal.
Each step should analyse the previous results and generate a Python code block (except for Finished).
Python code runs independently.

Available actions:
1 Search
  - Use Sentence Transformer at `./MiniLM-L6-fine-tuned`
  - Search in the FAISS
    - FAISS_INDEX_PATH = "dt_faiss.index"
    - EMBEDDINGS_PATH = "dt_embeddings.npy"
    - METADATA_PATH = "dt_metadata.json"
  - Results must match domain semantics and with high similarity
  - Indicate clearly if the results match the query
2 Decompose
  - Only if no match domain semantics are found in a Search
  - Break a domain into multiple subsystems
  - Generate brief descriptions for each subsystem
  - Perform a 1 Search for each subsystem
3 Compose
  - Compose subsystems' interfaces into the domain before breaking down
  - Synthetic digital twin interface from scratch is not allowed
4 Fill in
  - For each property, fill in a dummy value
  - Don't change the value of the dockerImage property
5 Construct a database
  - Construct a FAISS database from `interfaces.jsonl`
  - Contains DTDL v2 digital twin interface definitions (one per line)
  - Used for semantic search
6 Finished
  - Output 'Finished' without any other text if you have successfully generated the correct instance.

User's query: {user_query}
"""

# ----------------------------
# Utilities
# ----------------------------
def execute_python(code: str) -> str:
    """
    Execute dynamically generated Python code and capture printed output.
    Matches your original approach. :contentReference[oaicite:2]{index=2}
    """
    buffer = io.StringIO()
    local_vars = {}
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, local_vars)
    except Exception as e:
        return f"[Error executing code] {e}"

    printed_output = buffer.getvalue().strip()
    result = local_vars.get("result")
    if result is not None:
        combined = f"{printed_output}\n[result] {result}".strip()
        return combined if combined else "[No output]"
    return printed_output or "[No output]"


def extract_python_block(text: str) -> Optional[str]:
    if "```python" not in text:
        return None
    try:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    except Exception:
        return None


def ollama_chat_stream(messages: List[dict], host: str, model: str):
    """
    Stream tokens from Ollama /api/chat. Similar to your query_ollama_chat but yields chunks. :contentReference[oaicite:3]{index=3}
    """
    payload = {"model": model, "messages": messages, "stream": True}
    with requests.post(f"{host}/api/chat", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            if "message" in data and "content" in data["message"]:
                yield data["message"]["content"]

            if data.get("done"):
                break


# ----------------------------
# Run state
# ----------------------------
@dataclass
class StepRecord:
    step: int
    assistant_text: str = ""
    exec_output: str = ""
    status: str = "running"  # running|done|finished|error


@dataclass
class RunState:
    run_id: str
    created_at: float = field(default_factory=time.time)
    user_query: str = ""
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    messages: List[dict] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    event_q: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    stop_flag: bool = False
    error: Optional[str] = None


RUNS: Dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Runner (Flask GUI)</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body {
    background: #f4f6fb;
    color: #1f2937;
  }

  .card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
  }

  .muted {
    color: #6b7280;
  }

  pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 0;
    color: #111827;
  }

  textarea, input {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
  }

  textarea::placeholder,
  input::placeholder {
    color: #9ca3af;
  }

  .badge-soft {
    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
    font-weight: 500;
  }

  .step-title {
    letter-spacing: .2px;
    color: #111827;
  }

  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                 "Liberation Mono", "Courier New", monospace;
    font-size: 0.95rem;
  }

  .glow {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  }

  /* Code/output panels */
  .card pre {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 10px;
    border-radius: 8px;
  }

  /* Buttons */
  .btn-primary {
    background-color: #2563eb;
    border-color: #2563eb;
  }

  .btn-primary:hover {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
  }

  .btn-outline-light {
    color: #1f2937;
    border-color: #d1d5db;
  }

  .btn-outline-light:hover {
    background: #e5e7eb;
  }
</style>

</head>
<body>
<div class="container py-4">
  <div class="row g-3 align-items-stretch">
    <div class="col-12">
      <div class="card glow p-3">
        <div class="d-flex flex-wrap gap-2 align-items-center justify-content-between">
          <div>
            <div class="h4 mb-1">Agent Runner</div>
            <div class="muted">Type a query, run step-by-step, and see outputs for every step.</div>
          </div>
          <div class="d-flex gap-2">
            <span class="badge badge-soft p-2">Host: <span id="hostLabel" class="mono"></span></span>
            <span class="badge badge-soft p-2">Model: <span id="modelLabel" class="mono"></span></span>
          </div>
        </div>

        <div class="mt-3">
          <label class="form-label muted">User Query</label>
          <textarea id="query" class="form-control mono" rows="2"
            placeholder="e.g., Generate an instance for Porsche Taycan 4S 2022."></textarea>
        </div>

        <div class="mt-3 row g-2">
          <div class="col-md-5">
            <label class="form-label muted">Ollama Host (optional)</label>
            <input id="host" class="form-control mono" placeholder="http://10.1.1.1:60002">
          </div>
          <div class="col-md-5">
            <label class="form-label muted">Model (optional)</label>
            <input id="model" class="form-control mono" placeholder="gpt-oss:120b">
          </div>
          <div class="col-md-2 d-grid">
            <label class="form-label muted">&nbsp;</label>
            <button id="startBtn" class="btn btn-primary">Start</button>
          </div>
        </div>

        <div class="mt-3 d-flex gap-2 align-items-center">
          <button id="stopBtn" class="btn btn-outline-light btn-sm" disabled>Stop</button>
          <div class="muted small">Status: <span id="statusText">idle</span></div>
        </div>
      </div>
    </div>

    <div class="col-12">
      <div id="steps"></div>
    </div>
  </div>
</div>

<script>
let runId = null;
let es = null;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function stepCard(step) {
  const id = `step-${step}`;
  return `
  <div class="card p-3 mb-3" id="${id}">
    <div class="d-flex flex-wrap gap-2 align-items-center justify-content-between mb-2">
      <div class="h6 mb-0 step-title">Step ${step}</div>
      <span class="badge badge-soft" id="${id}-badge">running</span>
    </div>

    <div class="row g-3">
      <div class="col-lg-6">
        <div class="muted small mb-1">Assistant output (streamed)</div>
        <div class="card p-2 mono" style="background:#0b1630;">
          <pre id="${id}-assistant"></pre>
        </div>
      </div>
      <div class="col-lg-6">
        <div class="muted small mb-1">Execution output</div>
        <div class="card p-2 mono" style="background:#0b1630;">
          <pre id="${id}-exec"></pre>
        </div>
      </div>
    </div>
  </div>`;
}

function setStatus(text) {
  document.getElementById("statusText").textContent = text;
}

function setHostModelLabels(host, model) {
  document.getElementById("hostLabel").textContent = host || "";
  document.getElementById("modelLabel").textContent = model || "";
}

function ensureStep(step) {
  const stepsDiv = document.getElementById("steps");
  if (!document.getElementById(`step-${step}`)) {
    stepsDiv.insertAdjacentHTML("beforeend", stepCard(step));
  }
}

function appendAssistant(step, chunk) {
  ensureStep(step);
  const pre = document.getElementById(`step-${step}-assistant`);
  pre.textContent += chunk;
}

function setExec(step, text) {
  ensureStep(step);
  document.getElementById(`step-${step}-exec`).textContent = text;
}

function setBadge(step, status) {
  ensureStep(step);
  const badge = document.getElementById(`step-${step}-badge`);
  badge.textContent = status;
}

async function startRun() {
  const q = document.getElementById("query").value.trim();
  if (!q) {
    alert("Please enter a query.");
    return;
  }

  const host = document.getElementById("host").value.trim();
  const model = document.getElementById("model").value.trim();

  document.getElementById("steps").innerHTML = "";
  setStatus("starting...");
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;

  const resp = await fetch("/start", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ user_query: q, host: host, model: model })
  });

  if (!resp.ok) {
    const t = await resp.text();
    alert("Failed to start: " + t);
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    setStatus("idle");
    return;
  }

  const data = await resp.json();
  runId = data.run_id;
  setHostModelLabels(data.host, data.model);
  setStatus("running");

  es = new EventSource(`/events/${runId}`);
  es.addEventListener("meta", (e) => {
    // optional hook
  });

  es.addEventListener("assistant_chunk", (e) => {
    const msg = JSON.parse(e.data);
    appendAssistant(msg.step, msg.chunk);
  });

  es.addEventListener("exec_output", (e) => {
    const msg = JSON.parse(e.data);
    setExec(msg.step, msg.output);
  });

  es.addEventListener("step_status", (e) => {
    const msg = JSON.parse(e.data);
    setBadge(msg.step, msg.status);
  });

  es.addEventListener("done", (e) => {
    setStatus("done");
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    es.close();
  });

  es.addEventListener("error", (e) => {
    setStatus("error");
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    try { es.close(); } catch {}
  });
}

async function stopRun() {
  if (!runId) return;
  await fetch(`/stop/${runId}`, { method: "POST" });
  setStatus("stopping...");
  document.getElementById("stopBtn").disabled = true;
}

document.getElementById("startBtn").addEventListener("click", startRun);
document.getElementById("stopBtn").addEventListener("click", stopRun);
</script>
</body>
</html>
"""

# ----------------------------
# Background worker
# ----------------------------
def run_agent(state: RunState):
    try:
        system_prompt = INITIAL_PROMPT_TEMPLATE.format(user_query=state.user_query)

        state.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state.user_query},
        ]

        step = 0
        while not state.stop_flag:
            rec = StepRecord(step=step, status="running")
            state.steps.append(rec)
            state.event_q.put({"type": "step_status", "step": step, "status": "running"})

            # Stream assistant response into rec.assistant_text
            assistant_text_parts = []
            for chunk in ollama_chat_stream(state.messages, state.host, state.model):
                if state.stop_flag:
                    break
                assistant_text_parts.append(chunk)
                state.event_q.put({"type": "assistant_chunk", "step": step, "chunk": chunk})

            rec.assistant_text = "".join(assistant_text_parts).strip()

            if state.stop_flag:
                rec.status = "error"
                rec.exec_output = "[Stopped]"
                state.event_q.put({"type": "exec_output", "step": step, "output": rec.exec_output})
                state.event_q.put({"type": "step_status", "step": step, "status": "stopped"})
                break

            # Finished?
            if rec.assistant_text == "Finished":
                rec.status = "finished"
                rec.exec_output = "✅ Finished"
                state.event_q.put({"type": "exec_output", "step": step, "output": rec.exec_output})
                state.event_q.put({"type": "step_status", "step": step, "status": "finished"})
                break

            # Execute python block if present
            code = extract_python_block(rec.assistant_text)
            if code:
                out = execute_python(code)
                rec.exec_output = out
            else:
                rec.exec_output = rec.assistant_text or "[No output]"

            rec.status = "done"
            state.event_q.put({"type": "exec_output", "step": step, "output": rec.exec_output})
            state.event_q.put({"type": "step_status", "step": step, "status": "done"})

            # Memory update
            state.messages.append({"role": "assistant", "content": rec.assistant_text})
            state.messages.append(
                {"role": "user", "content": f"Step {step} output:\n{rec.exec_output}\nNext step?"}
            )
            step += 1

    except Exception as e:
        state.error = str(e)
        state.event_q.put({"type": "fatal", "message": state.error})
    finally:
        state.event_q.put({"type": "done"})


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/start")
def start():
    data = request.get_json(force=True)
    user_query = (data.get("user_query") or "").strip()
    if not user_query:
        return ("Missing user_query", 400)

    host = (data.get("host") or "").strip() or DEFAULT_HOST
    model = (data.get("model") or "").strip() or DEFAULT_MODEL

    run_id = uuid.uuid4().hex[:10]
    state = RunState(run_id=run_id, user_query=user_query, host=host, model=model)

    with RUNS_LOCK:
        RUNS[run_id] = state

    t = threading.Thread(target=run_agent, args=(state,), daemon=True)
    t.start()

    return jsonify({"run_id": run_id, "host": host, "model": model})


@app.post("/stop/<run_id>")
def stop(run_id):
    with RUNS_LOCK:
        state = RUNS.get(run_id)
    if not state:
        return ("Not found", 404)
    state.stop_flag = True
    return jsonify({"ok": True})


@app.get("/events/<run_id>")
def events(run_id):
    with RUNS_LOCK:
        state = RUNS.get(run_id)
    if not state:
        return ("Not found", 404)

    def sse():
        # A tiny "meta" event first (optional)
        yield "event: meta\ndata: {}\n\n"
        while True:
            try:
                evt = state.event_q.get(timeout=0.5)
            except queue.Empty:
                # keep-alive
                yield ": ping\n\n"
                continue

            etype = evt.get("type")
            if etype == "assistant_chunk":
                payload = json.dumps({"step": evt["step"], "chunk": evt["chunk"]})
                yield f"event: assistant_chunk\ndata: {payload}\n\n"
            elif etype == "exec_output":
                payload = json.dumps({"step": evt["step"], "output": evt["output"]})
                yield f"event: exec_output\ndata: {payload}\n\n"
            elif etype == "step_status":
                payload = json.dumps({"step": evt["step"], "status": evt["status"]})
                yield f"event: step_status\ndata: {payload}\n\n"
            elif etype == "fatal":
                payload = json.dumps({"message": evt.get("message", "Unknown error")})
                yield f"event: error\ndata: {payload}\n\n"
                break
            elif etype == "done":
                yield "event: done\ndata: {}\n\n"
                break

    return Response(sse(), mimetype="text/event-stream")


if __name__ == "__main__":
    # Run locally: http://127.0.0.1:5000
    # If exposing on a LAN, consider using host="0.0.0.0"
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
