"""
RAG Pipeline for ASHA-Sahayak.
Retrieves relevant medical guidelines from FAISS vector store,
assembles patient context from Delta Lake, and calls LLM for response.
"""

import os
import json
import time
import numpy as np
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# FAISS Index Management
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = os.environ.get("ASHA_FAISS_PATH", "/dbfs/asha_sahayak/faiss")
_faiss_index = None
_chunk_metadata = None


def _load_faiss_index():
    """Load FAISS index and chunk metadata from disk."""
    global _faiss_index, _chunk_metadata

    if _faiss_index is not None:
        return _faiss_index, _chunk_metadata

    try:
        import faiss

        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        meta_file = os.path.join(FAISS_INDEX_PATH, "chunks_metadata.json")

        if os.path.exists(index_file):
            _faiss_index = faiss.read_index(index_file)
            with open(meta_file, "r") as f:
                _chunk_metadata = json.load(f)
            print(f"Loaded FAISS index with {_faiss_index.ntotal} vectors")
        else:
            print(f"FAISS index not found at {index_file}. RAG will work without guidelines context.")
            _faiss_index = None
            _chunk_metadata = []
    except ImportError:
        print("faiss-cpu not installed. RAG will work without vector search.")
        _faiss_index = None
        _chunk_metadata = []

    return _faiss_index, _chunk_metadata


def search_guidelines(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the FAISS vector store for relevant medical guidelines.

    Args:
        query: Search query (in English)
        top_k: Number of results to return

    Returns:
        List of {"text": ..., "source": ..., "score": ...} dicts
    """
    from src.api.embeddings_client import get_embeddings

    index, metadata = _load_faiss_index()

    if index is None or not metadata:
        return []

    query_vector = get_embeddings([query])

    distances, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata) and idx >= 0:
            results.append(
                {
                    "text": metadata[idx].get("text", ""),
                    "source": metadata[idx].get("source", "unknown"),
                    "score": float(dist),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Context Assembly
# ---------------------------------------------------------------------------

def assemble_patient_context(spark, patient_id: str) -> str:
    """
    Assemble comprehensive patient context from CSV tables.
    """
    from src.utils.delta_utils import read_table
    import pandas as pd

    context_parts = []

    # 1. Patient Profile
    try:
        patients_df = read_table(spark, "patients_profiles")
        row = patients_df[patients_df["patient_id"] == patient_id]
        if not row.empty:
            patient = row.iloc[0]
            from datetime import date

            today = date.today()
            lmp = patient["lmp_date"]
            if lmp and pd.notna(lmp):
                gestational_days = (today - lmp).days
                gestational_weeks = gestational_days // 7
                trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
            else:
                gestational_weeks = 0
                trimester = 0

            context_parts.append(
                f"## Patient Profile\n"
                f"Name: {patient['name']}\n"
                f"Age: {patient['age']} years\n"
                f"Village: {patient['village']}\n"
                f"LMP: {patient['lmp_date']}\n"
                f"EDD: {patient['edd']}\n"
                f"Gestational Age: {gestational_weeks} weeks (Trimester {trimester})\n"
                f"Blood Group: {patient['blood_group']}\n"
                f"Height: {patient['height_cm']} cm\n"
                f"Pre-pregnancy Weight: {patient['pre_pregnancy_weight_kg']} kg\n"
                f"Risk Status: {patient['risk_status']}\n"
                f"Language: {patient['language_preference']}\n"
            )
    except Exception as e:
        context_parts.append(f"[Patient profile unavailable: {e}]")

    # 2. Recent EHR Records (last 3)
    try:
        ehr_df = read_table(spark, "ehr_records")
        recent_ehrs = ehr_df[ehr_df["patient_id"] == patient_id].copy()
        recent_ehrs = recent_ehrs.sort_values("visit_date", ascending=False).head(3)
        if not recent_ehrs.empty:
            context_parts.append("\n## Recent Health Records")
            for _, ehr in recent_ehrs.iterrows():
                context_parts.append(
                    f"\nVisit Date: {ehr['visit_date']} (Week {ehr['gestational_weeks']}, T{ehr['trimester']})\n"
                    f"  Hemoglobin: {ehr['hemoglobin']} g/dL\n"
                    f"  BP: {ehr['bp_systolic']}/{ehr['bp_diastolic']} mmHg\n"
                    f"  Weight: {ehr['weight_kg']} kg\n"
                    f"  Urine Albumin: {ehr['urine_albumin']}\n"
                    f"  Urine Sugar: {ehr['urine_sugar']}\n"
                    f"  Fasting Blood Sugar: {ehr['blood_sugar_fasting']} mg/dL\n"
                    f"  PP Blood Sugar: {ehr['blood_sugar_pp']} mg/dL\n"
                    f"  Medicines: {ehr.get('prescribed_medicines', '')}\n"
                )
    except Exception:
        pass

    # 3. Recent Conversations (last 3)
    try:
        conv_df = read_table(spark, "conversations")
        recent_convs = conv_df[conv_df["patient_id"] == patient_id].copy()
        recent_convs = recent_convs.sort_values("timestamp", ascending=False).head(3)
        if not recent_convs.empty:
            context_parts.append("\n## Recent Conversations")
            for _, conv in recent_convs.iterrows():
                context_parts.append(
                    f"\n[{conv['timestamp']}] ASHA: {conv['translated_input']}\n"
                    f"AI: {conv['ai_response']}\n"
                )
    except Exception:
        pass

    # 4. Risk Assessments
    try:
        risk_df = read_table(spark, "risk_assessments")
        patient_risks = risk_df[risk_df["patient_id"] == patient_id].copy()
        patient_risks = patient_risks.sort_values("assessment_date", ascending=False).head(1)
        if not patient_risks.empty:
            r = patient_risks.iloc[0]
            context_parts.append(
                f"\n## Latest Risk Assessment\n"
                f"Level: {r['risk_level']}\n"
                f"Factors: {r['risk_factors']}\n"
                f"Action: {r['recommended_action']}\n"
                f"Emergency: {r['emergency_flag']}\n"
            )
    except Exception:
        pass

    return "\n".join(context_parts)


# ---------------------------------------------------------------------------
# RAG Pipeline — Main Function
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are ASHA-Sahayak, an AI assistant for ASHA (Accredited Social Health Activist) workers in India who manage maternal health care for pregnant women in rural areas.

Your role is to:
1. Help ASHA workers manage pregnant women's health based on their medical records
2. Provide evidence-based recommendations grounded in PMSMA, WHO, and ICDS guidelines
3. Detect potential health risks and flag emergencies
4. Recommend appropriate nutrition and ration plans
5. Advise on checkup schedules and ANC visits

IMPORTANT GUIDELINES:
- Always reference the patient context provided when answering
- Ground your responses in the medical guidelines retrieved from the knowledge base
- Flag any emergency signs immediately (severe anemia Hb<7, BP>160/110, bleeding, convulsions, reduced fetal movement)
- Be clear, concise, and actionable in your recommendations
- When recommending rations, map to Anganwadi-available items
- Use the PMSMA Green/Red sticker classification system
- Extract and return any health updates mentioned in the conversation as structured JSON

Respond in English. The response will be translated to the ASHA worker's preferred language separately.

If you detect health updates in the ASHA worker's message (e.g., new symptoms, test results), extract them as JSON at the end of your response in this format:
```health_updates
{"hemoglobin": null, "bp_systolic": null, "bp_diastolic": null, "weight_kg": null, "symptoms": [], "risk_flags": []}
```"""


def run_rag_pipeline(
    spark,
    patient_id: str,
    user_query: str,
    input_type: str = "TEXT",
) -> dict:
    """
    Run the full RAG pipeline:
    1. Search guidelines from FAISS
    2. Assemble patient context from Delta Lake
    3. Build prompt with context
    4. Call LLM
    5. Parse response for health updates

    Args:
        spark: SparkSession
        patient_id: Patient UUID
        user_query: User's query (already translated to English)
        input_type: TEXT, AUDIO, IMAGE

    Returns:
        {
            "response": str,           # LLM response
            "health_updates": dict,    # Extracted health data
            "guidelines_used": list,   # Retrieved guideline chunks
            "latency_ms": float,       # Total pipeline latency
        }
    """
    from src.api.sarvam_client import chat_completion

    start_time = time.time()

    # Step 1: Search guidelines
    guidelines = search_guidelines(user_query, top_k=5)
    guidelines_text = ""
    if guidelines:
        guidelines_text = "\n\n## Relevant Medical Guidelines\n"
        for g in guidelines:
            guidelines_text += f"\n[Source: {g['source']}]\n{g['text']}\n"

    # Step 2: Assemble patient context
    patient_context = assemble_patient_context(spark, patient_id)

    # Step 3: Build messages
    full_context = f"{patient_context}\n{guidelines_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Patient Context:\n{full_context}\n\nASHA Worker's Query:\n{user_query}",
        },
    ]

    # Step 4: Call LLM
    response_text = chat_completion(messages)

    # Step 5: Parse health updates from response
    health_updates = _extract_health_updates(response_text)

    latency_ms = (time.time() - start_time) * 1000

    # Step 6: Log to MLflow (best effort, disabled by default)
    _log_to_mlflow(user_query, response_text, guidelines, latency_ms)

    return {
        "response": _clean_response(response_text),
        "health_updates": health_updates,
        "guidelines_used": guidelines,
        "latency_ms": latency_ms,
    }


def _extract_health_updates(response_text: str) -> dict:
    """Extract structured health updates from LLM response."""
    default = {
        "hemoglobin": None,
        "bp_systolic": None,
        "bp_diastolic": None,
        "weight_kg": None,
        "symptoms": [],
        "risk_flags": [],
    }

    try:
        # Look for the health_updates JSON block
        if "```health_updates" in response_text:
            json_str = response_text.split("```health_updates")[1].split("```")[0].strip()
            parsed = json.loads(json_str)
            return {**default, **parsed}
    except (json.JSONDecodeError, IndexError):
        pass

    return default


def _clean_response(response_text: str) -> str:
    """Remove the health_updates JSON block from the visible response."""
    if "```health_updates" in response_text:
        return response_text.split("```health_updates")[0].strip()
    return response_text


def _log_to_mlflow(query: str, response: str, guidelines: list, latency_ms: float):
    """
    Best-effort logging to MLflow.

    Disabled by default to avoid Spark Connect / Databricks MLflow config noise.
    Enable only by setting:
        os.environ["ASHA_ENABLE_MLFLOW_LOGGING"] = "true"
    """
    enabled = os.environ.get("ASHA_ENABLE_MLFLOW_LOGGING", "false").strip().lower() == "true"
    if not enabled:
        return

    try:
        import mlflow

        with mlflow.start_run(run_name="rag_inference", nested=True):
            mlflow.log_param("query", query[:250])
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("num_guidelines_retrieved", len(guidelines))
            mlflow.log_param("response_length", len(response))
    except Exception as e:
        print(f"MLflow logging skipped: {e}")