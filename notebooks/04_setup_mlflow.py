# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Setup MLflow Experiment
# MAGIC Creates and configures the MLflow experiment for ASHA-Sahayak.

# COMMAND ----------

import mlflow

# COMMAND ----------

# Set experiment name
EXPERIMENT_NAME = "/Shared/asha-sahayak-experiment"

try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "asha-sahayak",
                "hackathon": "bharat-bricks-hacks-2026",
                "track": "swatantra",
            }
        )
        print(f"✅ Created MLflow experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
except Exception as e:
    # Fallback for local/free edition
    experiment_id = mlflow.create_experiment("asha-sahayak") if not mlflow.get_experiment_by_name("asha-sahayak") else mlflow.get_experiment_by_name("asha-sahayak").experiment_id
    EXPERIMENT_NAME = "asha-sahayak"
    print(f"Using fallback experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Initial Setup Parameters

# COMMAND ----------

with mlflow.start_run(run_name="project_setup") as run:
    # Project parameters
    mlflow.log_param("project_name", "ASHA-Sahayak")
    mlflow.log_param("primary_llm", "sarvam-m")
    mlflow.log_param("fallback_llm", "databricks-meta-llama-3.1-70b-instruct")
    mlflow.log_param("stt_model", "sarvam-saarika-v2")
    mlflow.log_param("vision_model", "sarvam-mayura")
    mlflow.log_param("translation_model", "sarvam-translate-mayura-v1")
    mlflow.log_param("embedding_model", "multilingual-e5-small")
    mlflow.log_param("vector_store", "FAISS")
    mlflow.log_param("data_store", "Delta Lake")
    mlflow.log_param("num_delta_tables", 7)
    
    # Architecture metrics
    mlflow.log_metric("num_knowledge_base_docs", 4)
    mlflow.log_metric("supported_languages", 11)
    mlflow.log_metric("risk_rules_count", 13)
    
    print(f"✅ Setup run logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Custom Logging Functions

# COMMAND ----------

def log_rag_inference(query, response, num_chunks, latency_ms):
    """Log a RAG inference call to MLflow."""
    with mlflow.start_run(run_name="rag_inference", nested=True):
        mlflow.log_param("query", query[:250])
        mlflow.log_param("response_preview", response[:250])
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("num_chunks_retrieved", num_chunks)
        mlflow.log_metric("response_length", len(response))


def log_risk_assessment(patient_id, risk_level, num_factors, emergency):
    """Log a risk assessment to MLflow."""
    with mlflow.start_run(run_name="risk_assessment", nested=True):
        mlflow.log_param("patient_id", patient_id[:8])
        mlflow.log_param("risk_level", risk_level)
        mlflow.log_metric("num_risk_factors", num_factors)
        mlflow.log_metric("emergency", int(emergency))


def log_translation(source_lang, target_lang, input_length, output_length, latency_ms):
    """Log a translation call."""
    with mlflow.start_run(run_name="translation", nested=True):
        mlflow.log_param("source_lang", source_lang)
        mlflow.log_param("target_lang", target_lang)
        mlflow.log_metric("input_length", input_length)
        mlflow.log_metric("output_length", output_length)
        mlflow.log_metric("latency_ms", latency_ms)

print("✅ MLflow experiment setup complete!")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"To view: mlflow.search_runs(experiment_ids=['{experiment_id}'])")
