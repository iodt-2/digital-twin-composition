from flask import Flask, render_template, Response, request, jsonify
import requests
import json
import io
import contextlib

app = Flask(__name__)

# Default values (can be overridden via frontend config)
DEFAULT_HOST = "http://10.1.1.1:60002"
DEFAULT_MODEL = "gpt-oss:120b"

BASE_PROMPT = """
You are an agentic AI system designed to generate a specific Digital Twin instance from the interface in the repository. Each time, you need to choose the most suitable action based on the current situation to achieve the goal. Each step should analyse the previous results and generate a Python code block (except for Finished). Python code runs independently.

Available actions:
1 Search
  - Use Sentence Transformer at `./dt-triplet-v3-MiniLM-L6-all-final`
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

stop_flags = {}

def query_ollama_chat(messages, stop_id, host, model):
    """Stream response from Ollama API, respecting stop signal."""
    payload = {"model": model, "messages": messages, "stream": True}
    with requests.post(f"{host}/api/chat", json=payload, stream=True) as r:
        r.raise_for_status()
        full_reply = ""
        for line in r.iter_lines():
            if stop_flags.get(stop_id):
                yield f"data: [Process stopped by user]\n\n"
                break
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "message" in data:
                    content = data["message"]["content"]
                    full_reply += content
                    yield f"data: {content}\n\n"
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue
        yield f"data: \n--- End of step ---\n\n"
        yield f"data: \n{full_reply}\n\n"


def execute_python(code):
    """Executes generated Python code and returns printed output."""
    buffer = io.StringIO()
    local_vars = {}
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, local_vars)
    except Exception as e:
        return f"[Error executing code] {e}"
    printed = buffer.getvalue().strip()
    result = local_vars.get("result")
    if result is not None:
        return f"{printed}\n[result] {result}".strip()
    return printed or "[No output]"


def iterative_session(user_query, stop_id, host, model):
    """Loop conversation until Finished, executing code automatically."""
    messages = [
        {"role": "system", "content": BASE_PROMPT.format(user_query=user_query)},
        {"role": "user", "content": user_query}
    ]

    step = 0
    while True:
        if stop_flags.get(stop_id):
            yield f"data: [Process stopped]\n\n"
            break

        yield f"data: --- Step {step} ---\n\n"
        reply = ""
        for chunk in query_ollama_chat(messages, stop_id, host, model):
            yield chunk
            reply += chunk.replace("data: ", "")

        if "Finished" in reply:
            yield f"data: ✅ Finished\n\n"
            break

        if "```python" in reply:
            code = reply.split("```python")[1].split("```")[0].strip()
            yield f"data: \nExecuting code...\n\n"
            result = execute_python(code)
            yield f"data: \nExecution result:\n{result}\n\n"
        else:
            result = reply

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Step {step} output:\n{result}\nNext step?"})
        step += 1


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/stream", methods=["POST"])
def stream():
    user_query = request.form.get("query", "")
    stop_id = request.form.get("id", "")
    host = request.form.get("host") or DEFAULT_HOST
    model = request.form.get("model") or DEFAULT_MODEL
    stop_flags[stop_id] = False
    return Response(iterative_session(user_query, stop_id, host, model), mimetype="text/event-stream")


@app.route("/stop", methods=["POST"])
def stop():
    stop_id = request.form.get("id", "")
    stop_flags[stop_id] = True
    return "Stopped"


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
