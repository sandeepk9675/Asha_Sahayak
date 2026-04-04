"""
Ration Recommendation Engine for ASHA-Sahayak.
Uses RAG + LLM to generate personalized weekly ration plans
based on trimester, health conditions, and Anganwadi availability.
"""

import json
import uuid
from datetime import datetime, date, timedelta
from typing import Optional

from src.utils.delta_utils import get_spark, read_table, append_rows
from src.api.sarvam_client import chat_completion
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Nutrition Constants (WHO / ICDS / Saksham norms)
# ---------------------------------------------------------------------------

TRIMESTER_NUTRITION = {
    1: {
        "extra_kcal": 0,
        "protein_g": 55,
        "key_nutrients": ["Folic acid (400mcg)", "Iron (30mg)"],
        "description": "First trimester - focus on folate and iron",
    },
    2: {
        "extra_kcal": 340,
        "protein_g": 70,
        "key_nutrients": ["Iron (60mg)", "Calcium (1000mg)", "Protein"],
        "description": "Second trimester - increased calorie and protein needs",
    },
    3: {
        "extra_kcal": 450,
        "protein_g": 70,
        "key_nutrients": ["Iron (60mg)", "Calcium (1000mg)", "Protein", "DHA"],
        "description": "Third trimester - highest nutritional demands",
    },
}

# Anganwadi-available items (Saksham scheme mapped)
ANGANWADI_ITEMS = [
    {"item": "Rice", "calories_per_100g": 130, "protein_per_100g": 2.7},
    {"item": "Wheat flour (atta)", "calories_per_100g": 340, "protein_per_100g": 12},
    {"item": "Dal (lentils)", "calories_per_100g": 116, "protein_per_100g": 9},
    {"item": "Eggs", "calories_per_100g": 155, "protein_per_100g": 13},
    {"item": "Milk", "calories_per_100g": 42, "protein_per_100g": 3.4},
    {"item": "Groundnuts", "calories_per_100g": 567, "protein_per_100g": 26},
    {"item": "Jaggery (gur)", "calories_per_100g": 383, "protein_per_100g": 0.4},
    {"item": "Green leafy vegetables", "calories_per_100g": 23, "protein_per_100g": 2.9},
    {"item": "Seasonal fruits", "calories_per_100g": 50, "protein_per_100g": 0.5},
    {"item": "Soy chunks", "calories_per_100g": 345, "protein_per_100g": 52},
    {"item": "Fortified flour", "calories_per_100g": 350, "protein_per_100g": 11},
]

STANDARD_SUPPLEMENTS = {
    "ifa": {"name": "IFA tablet", "dosage": "100mg iron + 500mcg folic acid", "frequency": "1 tablet daily"},
    "ifa_double": {"name": "IFA tablet", "dosage": "100mg iron + 500mcg folic acid", "frequency": "2 tablets daily"},
    "calcium": {"name": "Calcium supplement", "dosage": "500mg", "frequency": "2 tablets daily"},
    "folic_acid": {"name": "Folic acid", "dosage": "5mg", "frequency": "1 tablet daily"},
}


def generate_ration_plan(
    spark,
    patient_id: str,
    use_llm: bool = True,
) -> dict:
    """
    Generate a personalized weekly ration plan for a patient.
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        use_llm: Whether to use LLM for personalized recommendations
        
    Returns:
        {
            "plan_id": str,
            "patient_name": str,
            "trimester": int,
            "daily_calorie_target": int,
            "protein_target_g": int,
            "ration_items": list,
            "supplements": list,
            "special_notes": str,
        }
    """
    # Get patient data
    patients_df = read_table(spark, "patients_profiles")
    patient = patients_df.filter(F.col("patient_id") == patient_id).first()
    
    if not patient:
        return {"error": "Patient not found"}
    
    # Calculate trimester
    lmp = patient["lmp_date"]
    today = date.today()
    if lmp:
        gestational_weeks = (today - lmp).days // 7
        trimester = 1 if gestational_weeks <= 12 else (2 if gestational_weeks <= 27 else 3)
    else:
        trimester = 1
        gestational_weeks = 0
    
    # Get latest EHR
    ehr_df = read_table(spark, "ehr_records")
    latest_ehr = (
        ehr_df.filter(F.col("patient_id") == patient_id)
        .orderBy(F.col("visit_date").desc())
        .limit(1)
        .collect()
    )
    
    hb = None
    bmi = None
    weight = patient["pre_pregnancy_weight_kg"]
    height_m = (patient["height_cm"] or 155) / 100
    conditions = []
    
    if latest_ehr:
        ehr = latest_ehr[0]
        hb = ehr["hemoglobin"]
        weight = ehr["weight_kg"] or weight
        fasting_sugar = ehr["blood_sugar_fasting"]
        
        if hb and hb < 11:
            conditions.append("anemia")
        if fasting_sugar and fasting_sugar > 126:
            conditions.append("gestational_diabetes")
    
    if weight and height_m:
        bmi = weight / (height_m ** 2)
        if bmi < 18.5:
            conditions.append("underweight")
        elif bmi > 25:
            conditions.append("overweight")
    
    # Base nutrition needs
    nutrition = TRIMESTER_NUTRITION.get(trimester, TRIMESTER_NUTRITION[1])
    base_calories = 1800  # Base for Indian women (sedentary)
    daily_calorie_target = base_calories + nutrition["extra_kcal"]
    protein_target = nutrition["protein_g"]
    
    # Build ration and supplements
    if use_llm:
        ration_items, supplements, special_notes = _llm_ration_plan(
            patient, trimester, gestational_weeks, hb, bmi, conditions, nutrition
        )
    else:
        ration_items, supplements, special_notes = _rule_based_ration(
            trimester, conditions, hb, bmi
        )
    
    # Store in Delta Lake
    plan_id = str(uuid.uuid4())
    today = date.today()
    plan = {
        "plan_id": plan_id,
        "patient_id": patient_id,
        "week_start_date": today,
        "week_end_date": today + timedelta(days=7),
        "trimester": trimester,
        "daily_calorie_target": daily_calorie_target,
        "protein_target_g": protein_target,
        "ration_items": json.dumps(ration_items),
        "supplements": json.dumps(supplements),
        "special_notes": special_notes,
        "generated_by_model": "sarvam-m" if use_llm else "rule-based",
    }
    
    try:
        append_rows(spark, "ration_plans", [plan])
    except Exception as e:
        print(f"Error storing ration plan: {e}")
    
    return {
        "plan_id": plan_id,
        "patient_name": patient["name"],
        "trimester": trimester,
        "gestational_weeks": gestational_weeks,
        "daily_calorie_target": daily_calorie_target,
        "protein_target_g": protein_target,
        "ration_items": ration_items,
        "supplements": supplements,
        "special_notes": special_notes,
        "conditions": conditions,
    }


def _rule_based_ration(
    trimester: int,
    conditions: list,
    hb: Optional[float],
    bmi: Optional[float],
) -> tuple:
    """Generate ration plan using rules (no LLM)."""
    
    ration_items = [
        {"item": "Rice", "quantity_g": 200, "frequency": "daily"},
        {"item": "Wheat flour (atta)", "quantity_g": 150, "frequency": "daily"},
        {"item": "Dal (lentils)", "quantity_g": 100, "frequency": "daily"},
        {"item": "Milk", "quantity_g": 500, "frequency": "daily (2 glasses)"},
        {"item": "Green leafy vegetables", "quantity_g": 150, "frequency": "daily"},
        {"item": "Seasonal fruits", "quantity_g": 100, "frequency": "daily"},
        {"item": "Eggs", "quantity_g": 100, "frequency": "daily (1-2 eggs)"},
    ]
    
    supplements = [STANDARD_SUPPLEMENTS["ifa"].copy(), STANDARD_SUPPLEMENTS["calcium"].copy()]
    special_notes = []
    
    # Condition-based adjustments
    if "anemia" in conditions:
        # Double IFA, add iron-rich foods
        supplements[0] = STANDARD_SUPPLEMENTS["ifa_double"].copy()
        ration_items.append({"item": "Jaggery (gur)", "quantity_g": 30, "frequency": "daily"})
        ration_items.append({"item": "Groundnuts", "quantity_g": 30, "frequency": "daily"})
        special_notes.append(f"Anemic (Hb={hb}). Doubled IFA. Added iron-rich foods: jaggery, dates, beetroot, green leafy vegetables.")
    
    if "gestational_diabetes" in conditions:
        # Reduce rice, increase protein
        for item in ration_items:
            if item["item"] == "Rice":
                item["quantity_g"] = 100
                item["frequency"] = "limited, prefer brown rice"
        ration_items.append({"item": "Soy chunks", "quantity_g": 50, "frequency": "daily"})
        special_notes.append("GDM detected. Low glycemic index diet. Reduce rice/sugar. Increase dal/vegetables/protein.")
    
    if "underweight" in conditions:
        ration_items.append({"item": "Groundnuts", "quantity_g": 50, "frequency": "daily"})
        ration_items.append({"item": "Fortified flour", "quantity_g": 100, "frequency": "daily"})
        special_notes.append(f"Underweight (BMI={bmi:.1f}). Extra THR from Anganwadi. High-calorie supplementation needed.")
    
    if "overweight" in conditions:
        for item in ration_items:
            if item["item"] == "Rice":
                item["quantity_g"] = 100
        special_notes.append(f"Overweight (BMI={bmi:.1f}). Reduce carbs, maintain protein. Encourage walking.")
    
    # Add trimester-specific notes
    if trimester == 1:
        supplements.append(STANDARD_SUPPLEMENTS["folic_acid"].copy())
        special_notes.append("T1: Folic acid critical for neural tube development.")
    elif trimester >= 2:
        special_notes.append(f"T{trimester}: Increased calorie (+{TRIMESTER_NUTRITION[trimester]['extra_kcal']}kcal) and protein needs.")
    
    return ration_items, supplements, " | ".join(special_notes) if special_notes else "Normal pregnancy diet."


def _llm_ration_plan(patient, trimester, weeks, hb, bmi, conditions, nutrition) -> tuple:
    """Use LLM to generate a personalized ration plan."""
    
    prompt = f"""Generate a personalized weekly ration plan for this pregnant woman:

Patient: {patient['name']}, Age {patient['age']}
Trimester: {trimester} (Week {weeks})
Hemoglobin: {hb} g/dL
BMI: {bmi:.1f if bmi else 'Unknown'}
Conditions: {', '.join(conditions) if conditions else 'None'}
Nutrition needs: +{nutrition['extra_kcal']}kcal/day, {nutrition['protein_g']}g protein/day

Available items from Anganwadi: Rice, wheat flour, dal, eggs, milk, groundnuts, jaggery, 
green leafy vegetables, seasonal fruits, soy chunks, fortified flour. 
Supplements: IFA tablets (100mg iron + 500mcg folic acid), Calcium (500mg).

Return ONLY a JSON object with this exact structure:
{{
  "ration_items": [{{"item": "...", "quantity_g": 123, "frequency": "daily"}}],
  "supplements": [{{"name": "...", "dosage": "...", "frequency": "..."}}],
  "special_notes": "..."
}}"""

    try:
        response = chat_completion(
            [
                {"role": "system", "content": "You are a maternal nutrition expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        
        # Parse JSON from response
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        data = json.loads(json_str)
        return (
            data.get("ration_items", []),
            data.get("supplements", []),
            data.get("special_notes", ""),
        )
    except Exception as e:
        print(f"LLM ration plan error: {e}. Falling back to rules.")
        return _rule_based_ration(trimester, conditions, hb, bmi)


def get_village_ration_summary(spark, asha_id: str = None) -> list:
    """
    Get ration distribution summary for all patients (village-level view).
    
    Returns list of {patient_name, trimester, ration_items, supplements, special_notes}
    """
    patients_df = read_table(spark, "patients_profiles")
    plans_df = read_table(spark, "ration_plans")
    
    if asha_id:
        patients_df = patients_df.filter(F.col("asha_id") == asha_id)
    
    # Get latest plan per patient
    from pyspark.sql.window import Window
    w = Window.partitionBy("patient_id").orderBy(F.col("week_start_date").desc())
    latest_plans = plans_df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")
    
    joined = patients_df.join(latest_plans, "patient_id", "left")
    results = joined.select(
        "name", "patient_id", "risk_status",
        F.coalesce("trimester", F.lit(0)).alias("trimester"),
        "ration_items", "supplements", "special_notes"
    ).collect()
    
    summary = []
    for r in results:
        summary.append({
            "patient_name": r["name"],
            "patient_id": r["patient_id"],
            "risk_status": r["risk_status"],
            "trimester": r["trimester"],
            "ration_items": json.loads(r["ration_items"]) if r["ration_items"] else [],
            "supplements": json.loads(r["supplements"]) if r["supplements"] else [],
            "special_notes": r["special_notes"] or "",
        })
    
    return summary
