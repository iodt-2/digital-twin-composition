# Digital Twin DTDL Generator

This Python tool automatically generates **Digital Twin Definition Language (DTDL v2)** interface files for various domains using the **Ollama API**.  
It discovers new digital-twin topics, generates detailed DTDL interface definitions for each topic, and writes the results into a JSONL file — flushing each interface immediately for robustness.

## 🧰 Requirements

- Python ≥ 3.8  
- Running [Ollama](https://ollama.ai) instance accessible via HTTP  
- Installed models (default: `gpt-oss:120b`)

---

## ⚙️ Configuration

By default, the script connects to:

```
Host:  http://10.1.1.1:60002
Model: gpt-oss:120b
```

You can override these via command-line flags:

```bash
python dtdl_generator.py \
  --host http://localhost:11434 \
  --model llama3:70b \
  --num-topics 50 \
  --output results.jsonl
```

---

## 🚀 Usage

### 1️⃣ Run the Generator

```bash
python dtdl_generator.py
```

The script will:

1. Discover a list of relevant digital twin topics.
2. For each topic, prompt the model to generate multiple DTDL v2 interfaces.
3. Immediately write each valid interface as a JSON object to `output.jsonl`.

Each line in the file represents **one DTDL interface**, e.g.:

```json
{"@context":"dtmi:dtdl:context;2","@id":"dtmi:car:engine;1","@type":"Interface","displayName":"EngineModel",...}
```

### 2️⃣ Monitor the Process

During runtime, you’ll see debug messages such as:

```
[STEP] Discovering topics...
[DEBUG] 35 topics discovered.
[STEP] Generating interfaces for topic 'car' ...
[DEBUG] Valid interfaces parsed: 7
```

### 3️⃣ Output Example

```bash
cat output.jsonl | head -n 2
```

```json
{"@context":"dtmi:dtdl:context;2","@id":"dtmi:car:engine;1","@type":"Interface","displayName":"EngineModel",...}
{"@context":"dtmi:dtdl:context;2","@id":"dtmi:car:chassis;1","@type":"Interface","displayName":"ChassisModel",...}
```

---

## 📄 Output Format

* **File type:** JSON Lines (`.jsonl`)
* **One interface per line**
* **Encoding:** UTF-8

### Interface Example

```json
{
  "@context": "dtmi:dtdl:context;2",
  "@id": "dtmi:vehicle:engine;1",
  "@type": "Interface",
  "displayName": "EngineModel",
  "description": "Represents a V6 diesel engine ...",
  "contents": [
    {"@type":"Property","name":"dockerImage","schema":"string","value":"registry.local/dtm/vehicle/engine:v1.0.0"},
    {"@type":"Property","name":"updateRateHz","schema":"integer","writable":true},
    {"@type":"Property","name":"version","schema":"string"}
  ]
}
```

---

## 🧪 Debugging

Set `--debug` for verbose trace output:

```bash
python dtdl_generator.py --debug
```

You can inspect intermediate text responses and parsed objects to troubleshoot model output.