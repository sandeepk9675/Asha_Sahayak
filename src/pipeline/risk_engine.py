"""
Risk Assessment Engine for ASHA-Sahayak.
Rule-based deterministic risk classification per PMSMA/WHO guidelines.
Runs after every conversation and EHR upload.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.delta_utils import read_table, append_rows, update_rows


# ---------------------------------------------------------------------------
# Risk Rules (deterministic thresholds from PRD)
# ---------------------------------------------------------------------------

def assess_risk(
    spark,
    patient_id: str,
    ehr_data: Optional[dict] = None,
    conversation_symptoms: Optional[list] = None,
) -> dict:
    """
    Run risk assessment on a patient using latest EHR data and conversation inputs.
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        ehr_data: Optional dict with latest EHR values (overrides DB lookup)
        conversation_symptoms: Optional list of symptoms from conversation
        
    Returns:
        {
            "risk_level": "GREEN" | "YELLOW" | "RED",
            "risk_factors": [str],
            "recommended_action": str,
            "emergency_flag": bool,
            "auto_appointment_created": bool,
        }
    """
    risk_factors = []
    emergency = False
    risk_level = "GREEN"
    actions = []
    
    # ---- Get patient profile ----
    patients_df = read_table(spark, "patients_profiles")
    patient_row = patients_df[patients_df["patient_id"] == patient_id]
    
    if patient_row.empty:
        return {
            "risk_level": "GREEN",
            "risk_factors": ["Patient not found"],
            "recommended_action": "Register patient first",
            "emergency_flag": False,
            "auto_appointment_created": False,
        }
    
    age = patient_row.iloc[0]["age"]
    
    # ---- Get latest EHR (if not provided) ----
    if ehr_data is None:
        ehr_df = read_table(spark, "ehr_records")
        patient_ehrs = ehr_df[ehr_df["patient_id"] == patient_id].copy()
        patient_ehrs = patient_ehrs.sort_values("visit_date", ascending=False).head(1)
        if not patient_ehrs.empty:
            ehr = patient_ehrs.iloc[0]
            ehr_data = {
                "hemoglobin": ehr["hemoglobin"],
                "bp_systolic": ehr["bp_systolic"],
                "bp_diastolic": ehr["bp_diastolic"],
                "weight_kg": ehr["weight_kg"],
                "blood_sugar_fasting": ehr["blood_sugar_fasting"],
                "blood_sugar_pp": ehr["blood_sugar_pp"],
                "urine_albumin": ehr["urine_albumin"],
            }
        else:
            ehr_data = {}
    
    symptoms = conversation_symptoms or []
    
    # ====================================================================
    # RULE-BASED RISK ASSESSMENT
    # ====================================================================
    
    # --- EMERGENCY (RED - Immediate Referral) ---
    
    # Severe Anemia
    hb = ehr_data.get("hemoglobin")
    if hb is not None and hb < 7:
        risk_factors.append(f"Severe anemia (Hb={hb} g/dL < 7)")
        emergency = True
        actions.append("IMMEDIATE referral to facility for blood transfusion")
    
    # Severe Pre-eclampsia
    bp_sys = ehr_data.get("bp_systolic")
    bp_dia = ehr_data.get("bp_diastolic")
    if bp_sys is not None and bp_dia is not None:
        if bp_sys > 160 or bp_dia > 110:
            risk_factors.append(f"Severe pre-eclampsia (BP={bp_sys}/{bp_dia})")
            emergency = True
            actions.append("IMMEDIATE referral - severe pre-eclampsia")
    
    # Conversation-based emergencies
    emergency_symptoms = [
        "vaginal bleeding", "bleeding", "blood loss",
        "convulsions", "seizures", "fits",
        "reduced fetal movement", "baby not moving",
        "severe headache", "blurred vision",
        "loss of consciousness", "fainting",
        "high fever", "water breaking",
    ]
    
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for es in emergency_symptoms:
            if es in symptom_lower:
                risk_factors.append(f"Emergency symptom reported: {symptom}")
                emergency = True
                actions.append(f"IMMEDIATE referral - {symptom}")
                break
    
    # --- HIGH RISK (RED - Red Sticker) ---
    
    # Pregnancy-Induced Hypertension
    if bp_sys is not None and bp_dia is not None:
        if (bp_sys > 140 or bp_dia > 90) and not emergency:
            risk_factors.append(f"Pregnancy-induced hypertension (BP={bp_sys}/{bp_dia})")
            risk_level = "RED"
            actions.append("Red sticker. Increase visit frequency. Monitor BP weekly.")
    
    # Gestational Diabetes
    fasting_sugar = ehr_data.get("blood_sugar_fasting")
    if fasting_sugar is not None and fasting_sugar > 126:
        risk_factors.append(f"Gestational diabetes (Fasting glucose={fasting_sugar} mg/dL)")
        risk_level = "RED"
        actions.append("Red sticker. Dietary plan needed. Monitor blood sugar.")
    
    # Adolescent Pregnancy
    if age is not None and age < 18:
        risk_factors.append(f"Adolescent pregnancy (Age={age})")
        risk_level = "RED"
        actions.append("Red sticker. Specialist referral required.")
    
    # Elderly Primigravida
    if age is not None and age > 35:
        risk_factors.append(f"Elderly primigravida (Age={age})")
        risk_level = "RED"
        actions.append("Red sticker. Specialist referral required.")
    
    # Proteinuria
    urine_alb = ehr_data.get("urine_albumin", "Normal")
    if urine_alb and urine_alb not in ("Normal", "Trace"):
        risk_factors.append(f"Proteinuria (Albumin={urine_alb})")
        risk_level = "RED"
        actions.append("Monitor for pre-eclampsia. Refer if BP elevated.")
    
    # --- ELEVATED (YELLOW) ---
    
    # Moderate Anemia
    if hb is not None and 7 <= hb < 10:
        risk_factors.append(f"Moderate anemia (Hb={hb} g/dL)")
        if risk_level == "GREEN":
            risk_level = "YELLOW"
        actions.append("Increase IFA dosage. Iron-rich diet plan.")
    
    # Mild anemia
    if hb is not None and 10 <= hb < 11:
        risk_factors.append(f"Mild anemia (Hb={hb} g/dL)")
        if risk_level == "GREEN":
            risk_level = "YELLOW"
        actions.append("Continue IFA. Monitor Hb next visit.")
    
    # Borderline BP
    if bp_sys is not None and bp_dia is not None:
        if (130 <= bp_sys <= 140) or (85 <= bp_dia <= 90):
            if risk_level == "GREEN":
                risk_factors.append(f"Borderline BP ({bp_sys}/{bp_dia})")
                risk_level = "YELLOW"
                actions.append("Monitor BP closely. Reduce salt intake.")
    
    # Borderline blood sugar
    if fasting_sugar is not None and 100 < fasting_sugar <= 126:
        risk_factors.append(f"Borderline blood sugar (Fasting={fasting_sugar})")
        if risk_level == "GREEN":
            risk_level = "YELLOW"
        actions.append("Diet modification. Recheck in 2 weeks.")
    
    # Emergency overrides risk level
    if emergency:
        risk_level = "RED"
    
    # Default action
    if not actions:
        actions.append("Green sticker. Continue routine ANC schedule.")
    
    recommended_action = " | ".join(actions)
    
    # ---- Store assessment ----
    assessment = {
        "assessment_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "assessment_date": datetime.now(),
        "risk_level": risk_level,
        "risk_factors": json.dumps(risk_factors),
        "recommended_action": recommended_action,
        "emergency_flag": emergency,
        "auto_appointment_created": False,
    }
    
    try:
        append_rows(spark, "risk_assessments", [assessment])
    except Exception as e:
        print(f"Error storing risk assessment: {e}")
    
    # ---- Update patient risk status ----
    try:
        update_rows(spark, "patients_profiles", "patient_id", patient_id, {
            "risk_status": risk_level,
            "last_updated": datetime.now(),
        })
    except Exception as e:
        print(f"Error updating patient risk status: {e}")
    
    # ---- Auto-create appointment if emergency ----
    if emergency:
        try:
            _create_emergency_appointment(spark, patient_id, recommended_action)
            assessment["auto_appointment_created"] = True
        except Exception as e:
            print(f"Error creating emergency appointment: {e}")
    
    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommended_action": recommended_action,
        "emergency_flag": emergency,
        "auto_appointment_created": assessment["auto_appointment_created"],
    }


def _create_emergency_appointment(spark, patient_id: str, notes: str):
    """Auto-create an emergency appointment at nearest facility."""
    from datetime import timedelta
    
    appointment = {
        "appointment_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "facility_name": "District Hospital (DH)",
        "facility_type": "DH",
        "scheduled_datetime": datetime.now() + timedelta(hours=2),
        "appointment_type": "EMERGENCY",
        "status": "SCHEDULED",
        "notes": f"Auto-created emergency appointment. {notes}",
    }
    
    append_rows(spark, "appointments", [appointment])
    print(f"🚨 Emergency appointment created for patient {patient_id}")


def get_patient_risk_summary(spark, patient_id: str) -> dict:
    """Get the latest risk assessment for a patient."""
    risk_df = read_table(spark, "risk_assessments")
    patient_risks = risk_df[risk_df["patient_id"] == patient_id].copy()
    patient_risks = patient_risks.sort_values("assessment_date", ascending=False).head(1)
    
    if not patient_risks.empty:
        r = patient_risks.iloc[0]
        return {
            "risk_level": r["risk_level"],
            "risk_factors": json.loads(r["risk_factors"]) if r["risk_factors"] else [],
            "recommended_action": r["recommended_action"],
            "emergency_flag": bool(r["emergency_flag"]),
            "assessment_date": str(r["assessment_date"]),
        }
    
    return {
        "risk_level": "GREEN",
        "risk_factors": [],
        "recommended_action": "No assessment yet. Run first assessment.",
        "emergency_flag": False,
        "assessment_date": None,
    }
