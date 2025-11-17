import contextlib
import io

import requests
import json

DEFAULT_HOST = "http://10.1.1.1:60002"
DEFAULT_MODEL = "gpt-oss:120b"

INITIAL_PROMPT = """
You are an agentic AI system designed to generate a specific Digital Twin instance from the interface in the repository. Each time, you need to choose the most suitable action based on the current situation to achieve the goal. Each step should analyse the previous results and generate a Python code block (except for Finished). Python code runs independently.

Available actions:
1 Search
  - Use Sentence Transformer at `./dt-triplet-v3-MiniLM-L6-all-final`
  - Search in the FAISS
    - FAISS_INDEX_PATH = "dt_faiss.index"           # Where to store the FAISS index
    - EMBEDDINGS_PATH = "dt_embeddings.npy"         # Numpy file with the embeddings
    - METADATA_PATH = "dt_metadata.json"            # JSON list of metadata (original lines)
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

User's query: Generate an instance for Porsche Taycan 4S 2022.
"""

def query_ollama_chat(messages):
    """
    Send chat-style messages to Ollama’s /api/chat endpoint and stream the response.
    """
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "stream": True
    }

    result = ""
    with requests.post(f"{DEFAULT_HOST}/api/chat", json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "message" in data:
                    content = data["message"]["content"]
                    print(content, end="", flush=True)  # live stream to console
                    result += content
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue

    print("\n")  # spacing
    return result.strip()

def execute_python(code):
    """
    Execute dynamically generated Python code and capture all printed output.
    Returns the combined printed output + any returned result variable.
    """
    # Capture stdout
    buffer = io.StringIO()
    local_vars = {}

    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, local_vars)
    except Exception as e:
        return f"[Error executing code] {e}"

    # Get all printed output
    printed_output = buffer.getvalue().strip()

    # Get the result variable if present
    result = local_vars.get("result")

    if result is not None:
        return f"{printed_output}\n[result] {result}".strip()
    else:
        return printed_output or "[No output]"

def main():
    messages = [
        {"role": "system", "content": INITIAL_PROMPT},
        {"role": "user", "content": "Generate an instance for Porsche Taycan 4S 2022."}
    ]

    step = 0
    while True:
        print(f"\n--- Step {step} ---")
        reply = query_ollama_chat(messages)

        if "Finished" in reply:
            print("\n✅ Finished")
            break

        # Extract and execute any code
        if "```python" in reply:
            code = reply.split("```python")[1].split("```")[0].strip()
            print("\nExecuting model-generated code...\n")
            result = execute_python(code)
            print("Execution result:", result)
        else:
            result = reply

        # Maintain conversational memory
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Step {step} output:\n{result}\nNext step?"})
        step += 1

if __name__ == "__main__":
    main()
