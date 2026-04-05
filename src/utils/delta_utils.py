"""
Data utility functions for ASHA-Sahayak.
Provides read/write helpers using Pandas + CSV files.
Works without Spark/Java — suitable for Databricks Apps and local deployment.
"""

import os
import threading
import pandas as pd
from datetime import datetime, date


# ---------------------------------------------------------------------------
# Data directory setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.environ.get("ASHA_DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "tables"))
_SEED_DIR = os.path.join(_PROJECT_ROOT, "data", "seed")
_lock = threading.Lock()


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def _csv_path(table: str) -> str:
    return os.path.join(_DATA_DIR, f"{table}.csv")


# ---------------------------------------------------------------------------
# Spark-compatible shim (returns None; kept so callers don't break)
# ---------------------------------------------------------------------------

def get_spark():
    """Return None — Spark is not required for CSV-backed storage."""
    return None


def table_name(table: str) -> str:
    """Return the table name (identity; kept for backward compat)."""
    return table


# ---------------------------------------------------------------------------
# Column definitions for each table (replaces PySpark StructType schemas)
# ---------------------------------------------------------------------------

TABLE_COLUMNS = {
    "patients_profiles": [
        "patient_id", "asha_id", "name", "age", "village", "contact",
        "lmp_date", "edd", "blood_group", "height_cm", "pre_pregnancy_weight_kg",
        "risk_status", "language_preference", "registration_date", "last_updated",
    ],
    "ehr_records": [
        "record_id", "patient_id", "visit_date", "trimester", "gestational_weeks",
        "hemoglobin", "bp_systolic", "bp_diastolic", "weight_kg",
        "urine_albumin", "urine_sugar", "blood_sugar_fasting", "blood_sugar_pp",
        "hiv_status", "vdrl_status", "malaria_status",
        "usg_findings", "prescribed_medicines",
        "raw_document_path", "extracted_text", "created_at",
    ],
    "checkup_schedules": [
        "schedule_id", "patient_id", "visit_number", "scheduled_date",
        "actual_date", "visit_type", "tests_due", "status",
    ],
    "conversations": [
        "conversation_id", "patient_id", "asha_id", "timestamp",
        "input_type", "original_input", "translated_input",
        "ai_response", "translated_response", "extracted_health_updates",
    ],
    "risk_assessments": [
        "assessment_id", "patient_id", "assessment_date", "risk_level",
        "risk_factors", "recommended_action", "emergency_flag",
        "auto_appointment_created",
    ],
    "ration_plans": [
        "plan_id", "patient_id", "week_start_date", "week_end_date",
        "trimester", "daily_calorie_target", "protein_target_g",
        "ration_items", "supplements", "special_notes", "generated_by_model",
    ],
    "appointments": [
        "appointment_id", "patient_id", "facility_name", "facility_type",
        "scheduled_datetime", "appointment_type", "status", "notes",
    ],
    "ifa_logs": [
        "log_id", "patient_id", "dispensed_date", "quantity",
        "reported_adherence", "side_effects", "created_at",
    ],
    "anc_visits": [
        "visit_id", "patient_id", "visit_number", "visit_date",
        "gestational_weeks", "weight_kg", "bp_systolic", "bp_diastolic",
        "fundal_height_cm", "fetal_heart_rate", "fetal_presentation",
        "edema", "danger_signs", "tt_dose_given", "next_visit_date", "created_at",
    ],
    "nutrition_logs": [
        "log_id", "patient_id", "log_date", "meal_type", "food_items",
        "calories_est", "protein_est_g", "iron_est_mg", "calcium_est_mg", "created_at",
    ],
    "risk_scores": [
        "score_id", "patient_id", "score_date", "model_version",
        "risk_score", "risk_category", "anemia_risk", "preeclampsia_risk",
        "gdm_risk", "preterm_risk", "top_factors", "recommendations", "created_at",
    ],
    "notifications": [
        "notification_id", "patient_id", "asha_id", "notification_type",
        "message", "priority", "status", "scheduled_at", "sent_at", "created_at",
    ],
}

# Date columns that need parsing
_DATE_COLS = {
    "patients_profiles": ["lmp_date", "edd"],
    "ehr_records": ["visit_date"],
    "checkup_schedules": ["scheduled_date", "actual_date"],
    "conversations": ["timestamp"],
    "risk_assessments": ["assessment_date"],
    "ration_plans": ["week_start_date", "week_end_date"],
    "appointments": ["scheduled_datetime"],
}

# Seed file mapping: table_name → (seed_csv, column_rename_map)
_SEED_MAP = {
    "patients_profiles": (
        "patients.csv",
        {
            "created_at": "registration_date",
        },
    ),
    "ehr_records": (
        "ehr_samples.csv",
        {
            "ehr_id": "record_id",
        },
    ),
}


# ---------------------------------------------------------------------------
# Initialisation: ensure tables exist, seed from CSV if empty
# ---------------------------------------------------------------------------

def _init_table(table: str):
    """Create empty CSV with headers if it doesn't exist, seed if available."""
    _ensure_data_dir()
    path = _csv_path(table)
    if os.path.exists(path):
        return

    cols = TABLE_COLUMNS.get(table)
    if not cols:
        return

    # Try to seed from data/seed/ files
    if table in _SEED_MAP:
        seed_file, rename_map = _SEED_MAP[table]
        seed_path = os.path.join(_SEED_DIR, seed_file)
        if os.path.exists(seed_path):
            df = pd.read_csv(seed_path, dtype=str)
            if rename_map:
                df = df.rename(columns=rename_map)
            # Ensure all expected columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(path, index=False)
            print(f"✅ Seeded table '{table}' from {seed_file} ({len(df)} rows)")
            return

    # Create empty
    pd.DataFrame(columns=cols).to_csv(path, index=False)
    print(f"✅ Created empty table '{table}'")


def init_all_tables():
    """Ensure all table CSVs exist."""
    for table in TABLE_COLUMNS:
        _init_table(table)


# ---------------------------------------------------------------------------
# Core CRUD helpers
# ---------------------------------------------------------------------------

def read_table(spark, table: str) -> pd.DataFrame:
    """Read a table as a Pandas DataFrame."""
    _init_table(table)
    path = _csv_path(table)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Parse date columns
    for col in _DATE_COLS.get(table, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # Convert pure-date columns to date objects for compatibility
            if col in ("lmp_date", "edd", "visit_date", "scheduled_date",
                       "actual_date", "week_start_date", "week_end_date"):
                df[col] = df[col].dt.date
    # Parse numeric columns
    df = _coerce_numeric(table, df)
    return df


def _coerce_numeric(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns to proper types."""
    int_cols = set()
    float_cols = set()
    bool_cols = set()

    if table == "patients_profiles":
        int_cols = {"age"}
        float_cols = {"height_cm", "pre_pregnancy_weight_kg"}
    elif table == "ehr_records":
        int_cols = {"trimester", "gestational_weeks", "bp_systolic", "bp_diastolic"}
        float_cols = {"hemoglobin", "weight_kg", "blood_sugar_fasting", "blood_sugar_pp"}
    elif table == "checkup_schedules":
        int_cols = {"visit_number"}
    elif table == "risk_assessments":
        bool_cols = {"emergency_flag", "auto_appointment_created"}
    elif table == "ration_plans":
        int_cols = {"trimester", "daily_calorie_target", "protein_target_g"}

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map({"True": True, "true": True, "1": True, "False": False, "false": False, "0": False, "": False})

    return df


def append_rows(spark, table: str, rows: list[dict]) -> None:
    """Append rows (list of dicts) to a CSV table."""
    if not rows:
        return
    _init_table(table)
    cols = TABLE_COLUMNS.get(table, list(rows[0].keys()))

    new_df = pd.DataFrame(rows)
    # Keep only known columns, add missing ones
    for c in cols:
        if c not in new_df.columns:
            new_df[c] = ""
    new_df = new_df[[c for c in cols if c in new_df.columns]]

    with _lock:
        path = _csv_path(table)
        new_df.to_csv(path, mode="a", header=not os.path.exists(path) or os.path.getsize(path) == 0, index=False)


def upsert_row(spark, table: str, row: dict, key_col: str) -> None:
    """Upsert a single row: update if key exists, else insert."""
    _init_table(table)
    with _lock:
        df = read_table(None, table)
        key_val = str(row[key_col])
        mask = df[key_col].astype(str) == key_val
        if mask.any():
            for k, v in row.items():
                if k in df.columns:
                    df.loc[mask, k] = v
        else:
            new_row = pd.DataFrame([row])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(_csv_path(table), index=False)


def delete_rows(spark, table: str, key_col: str, key_val: str) -> None:
    """Delete all rows matching key_col == key_val."""
    _init_table(table)
    with _lock:
        df = read_table(None, table)
        df = df[df[key_col].astype(str) != str(key_val)]
        df.to_csv(_csv_path(table), index=False)


# Keep old name as alias
delete_row = delete_rows


def update_rows(spark, table: str, key_col: str, key_val: str, updates: dict) -> None:
    """Update fields for rows matching key_col == key_val."""
    _init_table(table)
    with _lock:
        df = read_table(None, table)
        mask = df[key_col].astype(str) == str(key_val)
        for k, v in updates.items():
            if k in df.columns:
                df.loc[mask, k] = v
        df.to_csv(_csv_path(table), index=False)


def read_table_pandas(spark, table: str) -> pd.DataFrame:
    """Read a table as a Pandas DataFrame (same as read_table now)."""
    return read_table(spark, table)