"""
Patient Service for ASHA-Sahayak.
CRUD operations for patient profiles stored in Delta Lake.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Optional

from src.utils.delta_utils import (
    get_spark, read_table, append_rows, upsert_row, delete_row, read_table_pandas
)
from pyspark.sql import functions as F


def register_patient(
    spark,
    name: str,
    age: int,
    lmp_date: date,
    village: str,
    contact: str,
    language_preference: str = "hi",
    blood_group: str = "",
    height_cm: float = 0.0,
    pre_pregnancy_weight_kg: float = 0.0,
    asha_id: str = "ASHA001",
) -> dict:
    """
    Register a new pregnant woman.
    Auto-calculates EDD (LMP + 280 days), gestational age, and trimester.
    """
    patient_id = str(uuid.uuid4())
    edd = lmp_date + timedelta(days=280)
    now = datetime.now()
    
    patient = {
        "patient_id": patient_id,
        "asha_id": asha_id,
        "name": name,
        "age": age,
        "village": village,
        "contact": contact,
        "lmp_date": lmp_date,
        "edd": edd,
        "blood_group": blood_group,
        "height_cm": float(height_cm) if height_cm else 0.0,
        "pre_pregnancy_weight_kg": float(pre_pregnancy_weight_kg) if pre_pregnancy_weight_kg else 0.0,
        "risk_status": "GREEN",
        "language_preference": language_preference,
        "registration_date": now,
        "last_updated": now,
    }
    
    append_rows(spark, "patients_profiles", [patient])
    
    # Calculate derived fields for return
    gestational_days = (date.today() - lmp_date).days
    gestational_weeks = gestational_days // 7
    trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
    
    # Auto-assess risk for age-based factors
    risk_status = "GREEN"
    if age < 18:
        risk_status = "RED"
    elif age > 35:
        risk_status = "RED"
    
    if risk_status != "GREEN":
        _update_risk_status(spark, patient_id, risk_status)
    
    return {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "village": village,
        "lmp_date": str(lmp_date),
        "edd": str(edd),
        "gestational_weeks": gestational_weeks,
        "trimester": trimester,
        "risk_status": risk_status,
        "message": f"✅ {name} registered successfully. EDD: {edd}, Currently {gestational_weeks} weeks (T{trimester})",
    }


def get_patient(spark, patient_id: str) -> Optional[dict]:
    """Get a patient profile by ID."""
    df = read_table(spark, "patients_profiles")
    row = df.filter(F.col("patient_id") == patient_id).first()
    
    if not row:
        return None
    
    today = date.today()
    lmp = row["lmp_date"]
    gestational_weeks = (today - lmp).days // 7 if lmp else 0
    trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
    
    return {
        "patient_id": row["patient_id"],
        "name": row["name"],
        "age": row["age"],
        "village": row["village"],
        "contact": row["contact"],
        "lmp_date": str(row["lmp_date"]),
        "edd": str(row["edd"]),
        "blood_group": row["blood_group"],
        "height_cm": row["height_cm"],
        "pre_pregnancy_weight_kg": row["pre_pregnancy_weight_kg"],
        "risk_status": row["risk_status"],
        "language_preference": row["language_preference"],
        "gestational_weeks": gestational_weeks,
        "trimester": trimester,
        "weeks_remaining": max(0, 40 - gestational_weeks),
        "asha_id": row["asha_id"],
    }


def list_patients(spark, asha_id: str = None) -> list:
    """List all patients, optionally filtered by ASHA worker."""
    df = read_table(spark, "patients_profiles")
    
    if asha_id:
        df = df.filter(F.col("asha_id") == asha_id)
    
    rows = df.orderBy(
        F.when(F.col("risk_status") == "RED", 0)
        .when(F.col("risk_status") == "YELLOW", 1)
        .otherwise(2),
        "name"
    ).collect()
    
    today = date.today()
    patients = []
    for row in rows:
        lmp = row["lmp_date"]
        gestational_weeks = (today - lmp).days // 7 if lmp else 0
        trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
        
        patients.append({
            "patient_id": row["patient_id"],
            "name": row["name"],
            "age": row["age"],
            "village": row["village"],
            "risk_status": row["risk_status"],
            "gestational_weeks": gestational_weeks,
            "trimester": trimester,
            "edd": str(row["edd"]),
            "language_preference": row["language_preference"],
        })
    
    return patients


def search_patients(spark, query: str, asha_id: str = None) -> list:
    """Search patients by name or village."""
    df = read_table(spark, "patients_profiles")
    
    if asha_id:
        df = df.filter(F.col("asha_id") == asha_id)
    
    df = df.filter(
        F.lower(F.col("name")).contains(query.lower()) |
        F.lower(F.col("village")).contains(query.lower())
    )
    
    rows = df.collect()
    today = date.today()
    
    return [
        {
            "patient_id": r["patient_id"],
            "name": r["name"],
            "age": r["age"],
            "village": r["village"],
            "risk_status": r["risk_status"],
            "gestational_weeks": (today - r["lmp_date"]).days // 7 if r["lmp_date"] else 0,
        }
        for r in rows
    ]


def update_patient(spark, patient_id: str, **updates) -> dict:
    """Update patient profile fields."""
    from src.utils.delta_utils import table_name
    from delta.tables import DeltaTable
    
    updates["last_updated"] = datetime.now()
    
    # Recalculate EDD if LMP changed
    if "lmp_date" in updates:
        updates["edd"] = updates["lmp_date"] + timedelta(days=280)
    
    path = table_path("patients_profiles")
    delta_table = DeltaTable.forPath(spark, path)
    
    set_clause = {k: F.lit(v) for k, v in updates.items()}
    delta_table.update(
        condition=F.col("patient_id") == patient_id,
        set=set_clause,
    )
    
    return {"message": f"Patient {patient_id} updated successfully", "updates": {k: str(v) for k, v in updates.items()}}


def _update_risk_status(spark, patient_id: str, risk_status: str):
    """Internal: update patient risk status."""
    try:
        from src.utils.delta_utils import table_name
        from delta.tables import DeltaTable
        
        delta_table = DeltaTable.forName(spark, table_name("patients_profiles"))
        delta_table.update(
            condition=F.col("patient_id") == patient_id,
            set={"risk_status": F.lit(risk_status), "last_updated": F.lit(datetime.now())},
        )
    except Exception as e:
        print(f"Error updating risk status: {e}")


def get_patients_dataframe(spark, asha_id: str = None):
    """Return patients as Pandas DataFrame for Gradio display."""
    patients = list_patients(spark, asha_id)
    import pandas as pd
    
    if not patients:
        return pd.DataFrame(columns=["Name", "Age", "Village", "Trimester", "Weeks", "Risk", "EDD"])
    
    df = pd.DataFrame(patients)
    df = df.rename(columns={
        "name": "Name",
        "age": "Age",
        "village": "Village",
        "trimester": "Trimester",
        "gestational_weeks": "Weeks",
        "risk_status": "Risk",
        "edd": "EDD",
    })
    
    return df[["Name", "Age", "Village", "Trimester", "Weeks", "Risk", "EDD", "patient_id"]]
