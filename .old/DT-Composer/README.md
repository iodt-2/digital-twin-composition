# 🧠 Digital Twin Interface Composer

This project is an interactive **web application** for composing and parameterizing **Digital Twin (DT)** interfaces using **LLM reasoning**, **FAISS retrieval**, and **Google Programmable Search** for context enrichment.

---

## 🚀 Features

### 🧩 Step-by-Step Interactive Pipeline
The app guides you through four major steps:

1. **Semantic Query Understanding (Step 1)**  
   - Sends the user’s natural-language query to an LLM (e.g., Gemini).  
   - Displays the **raw prompt** and **LLM response** side-by-side (input/output).  

2. **Component Retrieval (Step 2)**  
   - Uses **FAISS** to retrieve relevant DT interfaces from a local vector index.  
   - Allows users to **manually confirm or delete** each component before proceeding.  

3. **Property Filling (Step 3)**  
   - Automatically fills in the selected interfaces with parameter values using the LLM.  
   - Shows both the **complete filled interface JSON** and a **table of successfully filled properties**.  
   - Removes unnecessary metadata (e.g., `source`) for clarity.  

4. **Final Composition (Step 4)**  
   - Combines all validated components into a complete **Digital Twin model** (DTDL v2).  
   - Provides one-click download of the generated model JSON.

---

## 🧠 System Architecture

```

┌────────────────────────────────────────────────────────────┐
│                        Web Interface                       │
│        (Flask + Bootstrap + Highlight.js + jQuery)         │
└────────────────────────────────────────────────────────────┘
│
┌────────────┴────────────┐
│                         │
┌───────────────┐         ┌────────────────┐
│  Gemini / LLM │         │  FAISS Vector  │
│  (Reasoning)  │         │  Store         │
└───────────────┘         └────────────────┘
│                         │
└────────────┬────────────┘
│
┌────────────────┐
│ Google PSE API │
│ (Web Retrieval)│
└────────────────┘

````

---

## 🛠️ Installation

### 1️⃣ Prerequisites

- Python **3.9+**
- API keys:
  - **Gemini (Google Generative Language API)**
  - **Google Programmable Search Engine (PSE)**

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/yourname/dt-interface-composer.git
cd dt-interface-composer
````

### 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Before running the app, set your API keys in environment variables or in a `.env` file:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
export GOOGLE_PSE_API_KEY="your_google_pse_api_key"
export GOOGLE_PSE_CX="your_search_engine_cx"
```

Or create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_PSE_API_KEY=your_google_pse_api_key
GOOGLE_PSE_CX=your_search_engine_cx
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 💡 Usage Guide

1. Enter your **task description or research question** (e.g.,
   *"Generate a Digital Twin model for a hybrid-electric vehicle"*)
   and click **Run Step 1**.

2. Review the **semantic interpretation** from the LLM.

3. Click **Run Step 2** to retrieve components; you may confirm or remove suggested interfaces.

4. Proceed to **Run Step 3** to let the LLM fill in property values.

   * You’ll see:

     * The full filled JSON (right top)
     * The successfully filled properties (right bottom table)

5. Click **Run Step 4** to generate the **final DTDL model** and download it.

---

## 🧰 Project Structure

```
DT-Composer/
├── app.py                  # Main Flask application
├── static/
│   └── style.css           # custom styles
├── templates/
│   ├── base.html           # Main HTML layout
│   └── index.html          # Multi-step UI
├── data/
├── requirements.txt
└── README.md
```

---

## 🧪 Example Output

**Filled Property Table (Step 3):**

| Property      | Value        | Note                   |
| ------------- | ------------ | ---------------------- |
| `maxPowerKW`  | `190`        | Derived from LLM query |
| `massKg`      | `1620`       | From retrieved data    |
| `dockerImage` | `"repo/img"` | Inferred by model      |

---

## 🧩 Example DT Interface (FAISS entry)

```json
{
  "@id": "dtmi:car:autonomy;1",
  "@type": "Interface",
  "displayName": "AutonomousDrivingModel",
  "description": "Decision and planning outputs for AD stack.",
  "contents": [
    {
      "@type": "Telemetry",
      "name": "target",
      "schema": {
        "@type": "Object",
        "fields": [
          {"name": "speedMps", "schema": "double"},
          {"name": "steerDeg", "schema": "double"}
        ]
      }
    },
    {"@type": "Command", "name": "ingestPerception", "request": {"name": "msg", "schema": "string"}},
    {"@type": "Property", "name": "dockerImage", "schema": "string"}
  ]
}
```


## 📚 Citation

If you use this tool in your academic work, please cite:

> Ziren Xiao, *TBD*, 2025.