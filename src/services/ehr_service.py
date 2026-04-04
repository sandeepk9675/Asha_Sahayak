"""
EHR (Electronic Health Record) Service for ASHA-Sahayak.
Handles EHR upload → OCR → parsing → storage in Delta Lake.
"""

import json
import uuid
import re
from datetime import datetime, date
from typing import Optional

from src.utils.delta_utils import get_spark, read_table, append_rows
from src.api.sarvam_client import extract_text_from_image, chat_completion
from pyspark.sql import functions as F


def upload_ehr_image(
    spark,
    patient_id: str,
    image_path: str,
) -> dict:
    """
    Process an uploaded EHR image:
    1. OCR via Sarvam Mayura Vision API
    2. LLM parsing to structured fields
    3. Store in Delta Lake
    4. Run risk assessment
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        image_path: Path to uploaded image
        
    Returns:
        Parsed EHR data dict
    """
    # Step 1: Extract text from image
    extracted_text = extract_text_from_image(
        image_path,
        prompt=(
            "Extract all medical information from this health report/lab result image. "
            "Include: Hemoglobin (Hb), Blood Pressure (BP systolic/diastolic), Weight, "
            "Blood Sugar (fasting/PP), Urine tests (albumin, sugar), HIV, VDRL, "
            "Malaria status, USG findings, prescribed medicines. "
            "Return the values clearly labeled."
        ),
    )
    
    # Step 2: Parse to structured data using LLM
    parsed = _parse_ehr_text(extracted_text)
    
    # Step 3: Calculate gestational info
    patients_df = read_table(spark, "patients_profiles")
    patient = patients_df.filter(F.col("patient_id") == patient_id).first()
    
    gestational_weeks = 0
    trimester = 1
    if patient and patient["lmp_date"]:
        gestational_days = (date.today() - patient["lmp_date"]).days
        gestational_weeks = gestational_days // 7
        trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
    
    # Step 4: Store in Delta Lake
    record_id = str(uuid.uuid4())
    ehr_record = {
        "record_id": record_id,
        "patient_id": patient_id,
        "visit_date": date.today(),
        "trimester": trimester,
        "gestational_weeks": gestational_weeks,
        "hemoglobin": parsed.get("hemoglobin"),
        "bp_systolic": parsed.get("bp_systolic"),
        "bp_diastolic": parsed.get("bp_diastolic"),
        "weight_kg": parsed.get("weight_kg"),
        "urine_albumin": parsed.get("urine_albumin", "Normal"),
        "urine_sugar": parsed.get("urine_sugar", "Normal"),
        "blood_sugar_fasting": parsed.get("blood_sugar_fasting"),
        "blood_sugar_pp": parsed.get("blood_sugar_pp"),
        "hiv_status": parsed.get("hiv_status", "Non-reactive"),
        "vdrl_status": parsed.get("vdrl_status", "Non-reactive"),
        "malaria_status": parsed.get("malaria_status", "Negative"),
        "usg_findings": parsed.get("usg_findings"),
        "prescribed_medicines": parsed.get("prescribed_medicines"),
        "raw_document_path": image_path,
        "extracted_text": extracted_text[:2000],  # Truncate for storage
        "created_at": datetime.now(),
    }
    
    append_rows(spark, "ehr_records", [ehr_record])
    
    # Step 5: Run risk assessment
    from src.pipeline.risk_engine import assess_risk
    risk_result = assess_risk(spark, patient_id, ehr_data=parsed)
    
    return {
        "record_id": record_id,
        "extracted_text": extracted_text,
        "parsed_data": parsed,
        "risk_assessment": risk_result,
        "message": f"✅ EHR uploaded and processed. Risk: {risk_result['risk_level']}",
    }


def add_ehr_manual(
    spark,
    patient_id: str,
    hemoglobin: float = None,
    bp_systolic: int = None,
    bp_diastolic: int = None,
    weight_kg: float = None,
    urine_albumin: str = "Normal",
    urine_sugar: str = "Normal",
    blood_sugar_fasting: float = None,
    blood_sugar_pp: float = None,
    usg_findings: str = "",
    prescribed_medicines: str = "",
) -> dict:
    """Add EHR record manually (when image upload isn't available)."""
    patients_df = read_table(spark, "patients_profiles")
    patient = patients_df.filter(F.col("patient_id") == patient_id).first()
    
    gestational_weeks = 0
    trimester = 1
    if patient and patient["lmp_date"]:
        gestational_days = (date.today() - patient["lmp_date"]).days
        gestational_weeks = gestational_days // 7
        trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
    
    record_id = str(uuid.uuid4())
    ehr_record = {
        "record_id": record_id,
        "patient_id": patient_id,
        "visit_date": date.today(),
        "trimester": trimester,
        "gestational_weeks": gestational_weeks,
        "hemoglobin": float(hemoglobin) if hemoglobin else None,
        "bp_systolic": int(bp_systolic) if bp_systolic else None,
        "bp_diastolic": int(bp_diastolic) if bp_diastolic else None,
        "weight_kg": float(weight_kg) if weight_kg else None,
        "urine_albumin": urine_albumin,
        "urine_sugar": urine_sugar,
        "blood_sugar_fasting": float(blood_sugar_fasting) if blood_sugar_fasting else None,
        "blood_sugar_pp": float(blood_sugar_pp) if blood_sugar_pp else None,
        "hiv_status": "Non-reactive",
        "vdrl_status": "Non-reactive",
        "malaria_status": "Negative",
        "usg_findings": usg_findings,
        "prescribed_medicines": prescribed_medicines,
        "raw_document_path": None,
        "extracted_text": None,
        "created_at": datetime.now(),
    }
    
    append_rows(spark, "ehr_records", [ehr_record])
    
    # Run risk assessment
    from src.pipeline.risk_engine import assess_risk
    ehr_data = {k: v for k, v in ehr_record.items() if v is not None}
    risk_result = assess_risk(spark, patient_id, ehr_data=ehr_data)
    
    return {
        "record_id": record_id,
        "risk_assessment": risk_result,
        "message": f"✅ EHR recorded. Risk: {risk_result['risk_level']}",
    }


def get_patient_ehrs(spark, patient_id: str) -> list:
    """Get all EHR records for a patient, most recent first."""
    ehr_df = read_table(spark, "ehr_records")
    rows = (
        ehr_df.filter(F.col("patient_id") == patient_id)
        .orderBy(F.col("visit_date").desc())
        .collect()
    )
    
    return [
        {
            "record_id": r["record_id"],
            "visit_date": str(r["visit_date"]),
            "trimester": r["trimester"],
            "gestational_weeks": r["gestational_weeks"],
            "hemoglobin": r["hemoglobin"],
            "bp": f"{r['bp_systolic']}/{r['bp_diastolic']}" if r['bp_systolic'] else "N/A",
            "weight_kg": r["weight_kg"],
            "urine_albumin": r["urine_albumin"],
            "urine_sugar": r["urine_sugar"],
            "blood_sugar_fasting": r["blood_sugar_fasting"],
            "blood_sugar_pp": r["blood_sugar_pp"],
            "usg_findings": r["usg_findings"],
            "prescribed_medicines": r["prescribed_medicines"],
        }
        for r in rows
    ]


def get_ehrs_dataframe(spark, patient_id: str):
    """Return EHRs as Pandas DataFrame for display."""
    import pandas as pd
    ehrs = get_patient_ehrs(spark, patient_id)
    
    if not ehrs:
        return pd.DataFrame(columns=["Date", "T", "Wk", "Hb", "BP", "Weight", "Sugar(F)", "Urine Alb"])
    
    df = pd.DataFrame(ehrs)
    df = df.rename(columns={
        "visit_date": "Date",
        "trimester": "T",
        "gestational_weeks": "Wk",
        "hemoglobin": "Hb",
        "bp": "BP",
        "weight_kg": "Weight",
        "blood_sugar_fasting": "Sugar(F)",
        "urine_albumin": "Urine Alb",
    })
    
    return df[["Date", "T", "Wk", "Hb", "BP", "Weight", "Sugar(F)", "Urine Alb"]]


def _parse_ehr_text(extracted_text: str) -> dict:
    """
    Parse OCR-extracted text into structured EHR fields using LLM.
    """
    if not extracted_text or extracted_text.startswith("["):
        return {}
    
    prompt = f"""Parse this medical report text into structured data. Extract the following fields:
- hemoglobin (float, in g/dL)
- bp_systolic (int, in mmHg)
- bp_diastolic (int, in mmHg)
- weight_kg (float)
- urine_albumin (string: Normal/Trace/+/++/+++)
- urine_sugar (string: Normal/+/++/+++/++++)
- blood_sugar_fasting (float, mg/dL)
- blood_sugar_pp (float, mg/dL)
- hiv_status (string: Reactive/Non-reactive)
- vdrl_status (string: Reactive/Non-reactive)
- malaria_status (string: Positive/Negative)
- usg_findings (string)
- prescribed_medicines (string)

Return ONLY valid JSON. Use null for values not found.

Medical Report Text:
{extracted_text}"""

    try:
        response = chat_completion(
            [
                {"role": "system", "content": "You are a medical data parser. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        return json.loads(json_str)
    except Exception as e:
        print(f"EHR parsing error: {e}")
        return _regex_parse_ehr(extracted_text)


def _regex_parse_ehr(text: str) -> dict:
    """Fallback: use regex patterns to extract common medical values."""
    result = {}
    
    # Hemoglobin
    hb_match = re.search(r'(?:hb|hemoglobin|haemoglobin)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if hb_match:
        result["hemoglobin"] = float(hb_match.group(1))
    
    # Blood Pressure
    bp_match = re.search(r'(?:bp|blood pressure)[:\s]*(\d+)[/](\d+)', text, re.IGNORECASE)
    if bp_match:
        result["bp_systolic"] = int(bp_match.group(1))
        result["bp_diastolic"] = int(bp_match.group(2))
    
    # Weight
    wt_match = re.search(r'(?:weight|wt)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if wt_match:
        result["weight_kg"] = float(wt_match.group(1))
    
    # Blood sugar
    sugar_match = re.search(r'(?:fasting|FBS)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if sugar_match:
        result["blood_sugar_fasting"] = float(sugar_match.group(1))
    
    return result
