# 🏥 ASHA-Sahayak (आशा सहायक)

**AI-Powered Multilingual Maternal Health Assistant for ASHA Workers**

ASHA-Sahayak helps India's 1 million+ ASHA (Accredited Social Health Activist) workers deliver better maternal healthcare to pregnant women in rural areas. It combines voice-first multilingual AI, electronic health records, automated risk detection, and personalized nutrition planning — all running on Databricks.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Gradio UI (Databricks App)    │
│  Dashboard | Patients | Chat | EHR     │
├──────────┬──────────────────────────────┤
│ Services │  patient · ehr · chat · dash │
├──────────┼──────────────────────────────┤
│ Pipeline │  RAG · Risk · Ration · Sched │
├──────────┼──────────────────────────────┤
│ API Layer│  Sarvam AI · HuggingFace     │
├──────────┼──────────────────────────────┤
│ Storage  │  Delta Lake · FAISS · MLflow │
└──────────┴──────────────────────────────┘
```

**Key Technologies:**
- **PySpark + Delta Lake** — Versioned data storage (7 tables)
- **FAISS** — Vector search for medical guideline retrieval
- **Sarvam AI** — Indian-language LLM, STT, OCR, Translation
- **HuggingFace** — Multilingual embeddings (e5-small)
- **Databricks Foundation Model API** — Llama 3 fallback LLM
- **MLflow** — Experiment tracking
- **Gradio** — Web UI deployed as Databricks App

---

## 📂 Project Structure

```
Asha_Sahayak/
├── app/
│   └── main.py                  # Gradio frontend application
├── src/
│   ├── api/
│   │   ├── sarvam_client.py     # Sarvam AI (LLM, STT, Vision, TTS)
│   │   ├── sarvam_translate_client.py  # Translation (22 languages)
│   │   └── embeddings_client.py # HuggingFace embeddings
│   ├── pipeline/
│   │   ├── rag_pipeline.py      # RAG: retrieve → context → generate
│   │   ├── language_pipeline.py # Multi-modal language processing
│   │   ├── risk_engine.py       # Rule-based risk assessment
│   │   ├── ration_engine.py     # Nutrition plan generator
│   │   └── schedule_engine.py   # ANC checkup scheduler
│   ├── services/
│   │   ├── patient_service.py   # Patient CRUD
│   │   ├── ehr_service.py       # EHR upload + OCR parsing
│   │   ├── chat_service.py      # Conversation manager
│   │   └── dashboard_service.py # Village dashboard aggregation
│   └── utils/
│       └── delta_utils.py       # Delta Lake helpers & schemas
├── notebooks/
│   ├── 01_setup_delta_tables.py # Create Delta tables
│   ├── 02_seed_data.py          # Seed sample patients & EHRs
│   ├── 03_ingest_knowledge_base.py  # Build FAISS index
│   ├── 04_setup_mlflow.py       # Configure MLflow experiment
│   └── 05_demo_scenarios.py     # End-to-end demo walkthrough
├── data/
│   ├── knowledge_base/          # Medical guideline .txt files
│   │   ├── pmsma_guidelines.txt
│   │   ├── safe_motherhood_booklet.txt
│   │   ├── saksham_nutrition_norms.txt
│   │   └── who_anc_recommendations.txt
│   └── seed/                    # Sample data CSVs
│       ├── patients.csv
│       ├── ehr_samples.csv
│       └── facilities.csv
├── requirements.txt
└── README.md
```

---

## 🔑 Prerequisites

1. **Databricks Workspace** (Community Edition or higher)
2. **API Keys** (get free-tier keys):
   - [Sarvam AI](https://www.sarvam.ai/) — `SARVAM_API_KEY`
   - [HuggingFace](https://huggingface.co/settings/tokens) — `HF_API_KEY`
3. **Python 3.10+**

---

## 🚀 Step-by-Step Databricks Deployment

### Step 1: Set Up Databricks Workspace

1. Go to [Databricks Community Edition](https://community.cloud.databricks.com/) and sign up (free)
2. Or use your organization's Databricks workspace
3. Make sure you have access to **Repos**, **Notebooks**, and **Compute**

### Step 2: Clone the Repository

**Option A — Using Databricks Repos (Recommended):**
1. In your Databricks workspace, go to **Repos** in the left sidebar
2. Click **Add Repo**
3. Paste your Git URL or upload the project
4. The repo will appear at `/Workspace/Repos/<your-username>/asha-sahayak/`

**Option B — Upload Files Manually:**
1. In **Workspace**, create a folder: `/Workspace/Users/<your-email>/Asha_Sahayak/`
2. Upload all project files maintaining the directory structure
3. Upload notebooks (`.py` files in `notebooks/`) — Databricks auto-detects `# Databricks notebook source`

### Step 3: Create a Compute Cluster

1. Go to **Compute** → **Create Cluster**
2. Configuration:
   - **Name:** `asha-sahayak-cluster`
   - **Runtime:** Databricks Runtime **14.3 LTS** or later (includes Spark 3.5+)
   - **Node type:** Single Node (for Community Edition) or smallest available
   - **Auto-termination:** 60 minutes
3. Click **Create Cluster** and wait for it to start

### Step 4: Configure API Keys (Secrets)

**Option A — Databricks Secrets (Recommended for production):**

Open a notebook and run:

```python
# 1. Create a secret scope
# Run this from Databricks CLI or REST API:
# databricks secrets create-scope --scope asha-sahayak

# 2. Store keys
# databricks secrets put --scope asha-sahayak --key sarvam-api-key
# databricks secrets put --scope asha-sahayak --key hf-api-key
```

Or via the Secrets API in a notebook:
```python
# Verify secrets are accessible
try:
    key = dbutils.secrets.get("asha-sahayak", "sarvam-api-key")
    print("✅ Sarvam API key configured")
except:
    print("⚠️ Set up Databricks Secrets or use environment variables")
```

**Option B — Environment Variables (Quick start):**

At the top of each notebook, add:
```python
import os
os.environ["SARVAM_API_KEY"] = "your-sarvam-api-key-here"
os.environ["HF_API_KEY"] = "your-huggingface-api-key-here"
```

Or set cluster environment variables:
1. Go to **Compute** → your cluster → **Edit**
2. Under **Advanced Options** → **Spark** → **Environment Variables**
3. Add:
   ```
   SARVAM_API_KEY=your-key-here
   HF_API_KEY=your-key-here
   ```

### Step 5: Install Dependencies

Open a notebook and run:

```python
%pip install faiss-cpu httpx pdfplumber PyPDF2 sentence-transformers gradio mlflow
```

Or attach the `requirements.txt` to your cluster:
1. Go to **Compute** → your cluster → **Libraries**
2. Click **Install New** → **PyPI**
3. Install each package, or upload `requirements.txt`

### Step 6: Run Setup Notebooks (In Order!)

**IMPORTANT: Run these notebooks in sequence — each depends on the previous one.**

| Order | Notebook | What It Does | Time |
|-------|----------|-------------|------|
| 1️⃣ | `01_setup_delta_tables.py` | Creates 7 Delta Lake tables at `/dbfs/asha_sahayak/delta` | ~1 min |
| 2️⃣ | `02_seed_data.py` | Inserts 12 sample patients + 5 EHR records | ~2 min |
| 3️⃣ | `03_ingest_knowledge_base.py` | Chunks medical guidelines → embeds → builds FAISS index | ~5 min |
| 4️⃣ | `04_setup_mlflow.py` | Creates MLflow experiment + logs project params | ~1 min |

**To run a notebook:**
1. Open the notebook file from Repos/Workspace
2. Attach to your cluster (top-right dropdown)
3. Click **Run All** or run cells one by one

**Verify after each notebook:**
```python
# After 01: Check tables exist
display(spark.sql("SHOW TABLES IN default LIKE 'asha_*'"))

# After 02: Check patients
display(spark.read.format("delta").load("/dbfs/asha_sahayak/delta/patients_profiles"))

# After 03: Check FAISS index
import os
print(os.listdir("/dbfs/asha_sahayak/faiss/"))

# After 04: Check MLflow
import mlflow
exp = mlflow.get_experiment_by_name("/Shared/asha-sahayak-experiment")
print(f"Experiment: {exp.name}, ID: {exp.experiment_id}")
```

### Step 7: Run the Demo Notebook

1. Open `notebooks/05_demo_scenarios.py`
2. Attach to your cluster
3. Run all cells

This demonstrates three complete scenarios:
- **Scenario 1:** Patient registration + Hindi chat
- **Scenario 2:** EHR entry + automatic risk detection (RED alert)
- **Scenario 3:** Village dashboard + ration distribution

### Step 8: Deploy the Gradio App

**Option A — Databricks App (Recommended):**

1. In your workspace, go to **Apps** (or **Compute** → **Apps**)
2. Click **Create App**
3. Configure:
   - **Name:** `asha-sahayak`
   - **Source:** Point to `app/main.py`
   - **Framework:** Gradio
   - **Cluster:** Your compute cluster
4. Add environment variables:
   ```
   SARVAM_API_KEY=your-key
   HF_API_KEY=your-key
   ASHA_DELTA_BASE=/dbfs/asha_sahayak/delta
   ASHA_FAISS_PATH=/dbfs/asha_sahayak/faiss
   ```
5. Click **Deploy**
6. Access your app at: `https://<workspace-url>/apps/asha-sahayak`

**Option B — Run from Notebook:**

Create a new notebook and run:
```python
import sys
sys.path.insert(0, "/Workspace/Repos/<your-username>/asha-sahayak/Asha_Sahayak")

from app.main import build_app

app = build_app()
app.launch(share=True, server_port=8080)
```

**Option C — Driver Proxy (Community Edition):**

```python
# In a notebook cell
import os, sys
sys.path.insert(0, "/Workspace/Repos/<your-username>/asha-sahayak/Asha_Sahayak")

os.environ["SARVAM_API_KEY"] = "your-key-here"
os.environ["HF_API_KEY"] = "your-key-here"

from app.main import build_app
app = build_app()

# Launch with driver proxy for Community Edition
app.launch(
    server_name="0.0.0.0",
    server_port=8080,
    inline=False,
    share=True,  # Creates a public URL
)
```

### Step 9: Verify Everything Works

Open the Gradio app and test:

1. **🏠 Dashboard** — Click "Refresh Dashboard" → see village stats, alerts
2. **👤 Patients** — Click "Refresh List" → see 12 seeded patients
3. **➕ Register** — Fill form → register a new patient
4. **👩 Profile** → Select a patient → Load Profile
5. **💬 Chat** — Type "What should I eat?" → get response in Hindi
6. **📋 EHR** — Enter Hb=6.5, BP=165/110 → save → see RED risk alert
7. **📅 Schedule** — Generate ANC schedule
8. **🍚 Ration** — Generate personalized ration plan

---

## 📋 Configuration Reference

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SARVAM_API_KEY` | Sarvam AI API key | Yes |
| `HF_API_KEY` | HuggingFace API token | Yes |
| `DATABRICKS_TOKEN` | Databricks PAT (auto-set in notebooks) | Auto |
| `DATABRICKS_HOST` | Workspace URL (auto-set in notebooks) | Auto |
| `ASHA_DELTA_BASE` | Delta table storage path | Default: `/dbfs/asha_sahayak/delta` |
| `ASHA_FAISS_PATH` | FAISS index storage path | Default: `/dbfs/asha_sahayak/faiss` |

### Delta Lake Tables

| Table | Content |
|-------|---------|
| `patients_profiles` | Patient demographics, LMP, EDD, risk status |
| `ehr_records` | Lab results (Hb, BP, sugar, etc.) |
| `conversations` | Chat history (all languages) |
| `checkup_schedules` | ANC visit schedule per PMSMA |
| `ration_plans` | Weekly nutrition plans |
| `risk_assessments` | Risk evaluation results |
| `appointments` | Emergency/follow-up appointments |

### API Endpoints Used

| API | Model | Purpose |
|-----|-------|---------|
| Sarvam `sarvam-m` | LLM | Chat & health advice |
| Sarvam `saarika:v2` | STT | Voice → text |
| Sarvam `mayura` | Vision | Lab report OCR |
| Sarvam `mayura:v1` | Translation | 22 Indian languages |
| HuggingFace `multilingual-e5-small` | Embeddings | RAG retrieval |
| Databricks `meta-llama-3.1-70b` | LLM | Fallback generation |

---

## 🔧 Troubleshooting

### Common Issues

**"SparkSession not initialized"**
- Make sure you're running on a Databricks cluster, not locally
- Check cluster is in Running state

**"API key not found"**
- Verify `SARVAM_API_KEY` and `HF_API_KEY` are set
- Check Databricks Secrets scope exists: `dbutils.secrets.listScopes()`

**"Delta table not found"**
- Run `01_setup_delta_tables.py` first
- Verify path: `dbutils.fs.ls("/dbfs/asha_sahayak/delta/")`

**"FAISS index not found"**
- Run `03_ingest_knowledge_base.py` first
- Verify: `dbutils.fs.ls("/dbfs/asha_sahayak/faiss/")`

**Gradio app not loading**
- Check all imports resolve: run `import src.services.patient_service` in a notebook
- Verify `sys.path` includes the project root
- Check port 8080 is not in use

**LLM returns fallback response**
- Sarvam API may be rate-limited; the system auto-falls back to Databricks Foundation Model API
- Check `DATABRICKS_TOKEN` is set (auto-set inside Databricks)

---

## 📜 License

Built for the Databricks Hackathon. For demonstration and educational purposes.

---

## 🙏 Acknowledgments

- **Sarvam AI** — India-first multilingual AI models
- **Databricks** — Unified data + AI platform
- **PMSMA** (Pradhan Mantri Surakshit Matritva Abhiyan) — National ANC guidelines
- **WHO** — ANC recommendation framework
- **ICDS/SAKSHAM** — Nutrition program norms
