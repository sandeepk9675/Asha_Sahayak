"""
ANC Schedule Engine for ASHA-Sahayak.
Auto-generates checkup schedules per PMSMA guidelines.
"""

import json
import uuid
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.utils.delta_utils import read_table, append_rows, delete_rows


# ---------------------------------------------------------------------------
# PMSMA ANC Schedule Guidelines
# ---------------------------------------------------------------------------

# Standard minimum 4 ANC visits + PMSMA 9th-of-month visits
# High-risk patients get more frequent visits

STANDARD_ANC_SCHEDULE = [
    {
        "visit_number": 1,
        "trimester": 1,
        "week_range": (8, 12),
        "visit_type": "ROUTINE_ANC",
        "tests_due": [
            "Registration & history",
            "Height/Weight/BMI",
            "Blood group & Rh typing",
            "Hemoglobin",
            "Urine routine (albumin, sugar)",
            "Blood sugar (fasting)",
            "HIV/VDRL",
            "USG (dating scan, if available)",
        ],
    },
    {
        "visit_number": 2,
        "trimester": 2,
        "week_range": (14, 26),
        "visit_type": "ROUTINE_ANC",
        "tests_due": [
            "Weight check",
            "BP measurement",
            "Hemoglobin",
            "Urine routine",
            "Fundal height",
            "USG (anomaly scan 18-20 weeks)",
            "OGTT (24-28 weeks for GDM screening)",
        ],
    },
    {
        "visit_number": 3,
        "trimester": 3,
        "week_range": (28, 34),
        "visit_type": "ROUTINE_ANC",
        "tests_due": [
            "Weight check",
            "BP measurement",
            "Hemoglobin",
            "Urine routine",
            "Fundal height",
            "Fetal heart rate",
            "USG (growth scan)",
            "Birth preparedness counseling",
        ],
    },
    {
        "visit_number": 4,
        "trimester": 3,
        "week_range": (36, 40),
        "visit_type": "ROUTINE_ANC",
        "tests_due": [
            "Weight check",
            "BP measurement",
            "Hemoglobin",
            "Urine routine",
            "Fundal height",
            "Fetal presentation",
            "Birth plan finalization",
            "Facility delivery planning",
        ],
    },
]


def generate_schedule(spark, patient_id: str) -> list:
    """
    Generate ANC checkup schedule for a patient based on LMP and PMSMA guidelines.
    
    Args:
        spark: SparkSession
        patient_id: Patient UUID
        
    Returns:
        List of schedule dicts with dates and tests
    """
    patients_df = read_table(spark, "patients_profiles")
    patient_row = patients_df[patients_df["patient_id"] == patient_id]
    
    if patient_row.empty:
        return []
    
    patient = patient_row.iloc[0]
    lmp = patient["lmp_date"]
    if not lmp or pd.isna(lmp):
        return []
    
    risk_status = patient["risk_status"] or "GREEN"
    today = date.today()
    gestational_days = (today - lmp).days
    current_week = gestational_days // 7
    
    schedules = []
    
    for visit in STANDARD_ANC_SCHEDULE:
        week_start, week_end = visit["week_range"]
        
        # Calculate scheduled date: LMP + midpoint of week range
        midpoint_week = (week_start + week_end) // 2
        scheduled_date = lmp + timedelta(weeks=midpoint_week)
        
        # Determine status
        if current_week > week_end + 2:
            status = "OVERDUE"
        elif current_week >= week_start:
            status = "PENDING"
        else:
            status = "PENDING"
        
        # Check if already completed
        existing = _check_existing_schedule(spark, patient_id, visit["visit_number"])
        if existing and existing["status"] == "COMPLETED":
            status = "COMPLETED"
            scheduled_date = existing.get("actual_date", scheduled_date)
        
        schedule = {
            "schedule_id": existing["schedule_id"] if existing else str(uuid.uuid4()),
            "patient_id": patient_id,
            "visit_number": visit["visit_number"],
            "scheduled_date": scheduled_date,
            "actual_date": existing.get("actual_date") if existing else None,
            "visit_type": visit["visit_type"],
            "tests_due": json.dumps(visit["tests_due"]),
            "status": status,
        }
        schedules.append(schedule)
    
    # Add PMSMA 9th-of-month visits
    pmsma_schedules = _generate_pmsma_visits(lmp, current_week, patient_id)
    schedules.extend(pmsma_schedules)
    
    # High-risk patients: add extra visits
    if risk_status == "RED":
        extra_visits = _generate_high_risk_visits(lmp, current_week, patient_id, len(schedules))
        schedules.extend(extra_visits)
    
    # Store schedules in Delta Lake
    _store_schedules(spark, patient_id, schedules)
    
    return schedules


def _check_existing_schedule(spark, patient_id: str, visit_number: int) -> Optional[dict]:
    """Check if a schedule entry already exists."""
    try:
        sched_df = read_table(spark, "checkup_schedules")
        existing = sched_df[
            (sched_df["patient_id"] == patient_id) &
            (sched_df["visit_number"] == visit_number)
        ]
        if not existing.empty:
            row = existing.iloc[0]
            return {
                "schedule_id": row["schedule_id"],
                "status": row["status"],
                "actual_date": row["actual_date"],
            }
    except Exception:
        pass
    return None


def _generate_pmsma_visits(lmp: date, current_week: int, patient_id: str) -> list:
    """Generate PMSMA 9th-of-month visits during pregnancy."""
    schedules = []
    edd = lmp + timedelta(days=280)
    
    # Find all 9th-of-month dates during pregnancy
    current_date = lmp
    visit_num = 10  # Start numbering after standard visits
    
    while current_date <= edd:
        # Find the 9th of the current or next month
        if current_date.day <= 9:
            pmsma_date = current_date.replace(day=9)
        else:
            # Move to next month
            if current_date.month == 12:
                pmsma_date = current_date.replace(year=current_date.year + 1, month=1, day=9)
            else:
                pmsma_date = current_date.replace(month=current_date.month + 1, day=9)
        
        if lmp < pmsma_date <= edd:
            weeks_at_visit = (pmsma_date - lmp).days // 7
            status = "COMPLETED" if pmsma_date < date.today() - timedelta(days=7) else "PENDING"
            if pmsma_date < date.today() and status == "PENDING":
                status = "OVERDUE"
            
            schedules.append({
                "schedule_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "visit_number": visit_num,
                "scheduled_date": pmsma_date,
                "actual_date": None,
                "visit_type": "PMSMA_9TH",
                "tests_due": json.dumps([
                    "PMSMA checkup",
                    "BP measurement",
                    "Weight",
                    "Specialist consultation (if available)",
                ]),
                "status": status,
            })
            visit_num += 1
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1, day=1)
    
    return schedules


def _generate_high_risk_visits(lmp: date, current_week: int, patient_id: str, start_num: int) -> list:
    """Generate additional visits for high-risk pregnancies (every 2 weeks)."""
    schedules = []
    edd = lmp + timedelta(days=280)
    visit_num = start_num + 1
    
    # Add bi-weekly visits from week 28 onwards
    for week in range(28, 40, 2):
        visit_date = lmp + timedelta(weeks=week)
        if visit_date > date.today() and visit_date <= edd:
            schedules.append({
                "schedule_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "visit_number": visit_num,
                "scheduled_date": visit_date,
                "actual_date": None,
                "visit_type": "HIGH_RISK_FOLLOWUP",
                "tests_due": json.dumps([
                    "BP monitoring",
                    "Weight",
                    "Hemoglobin (if anemic)",
                    "Fetal heart rate",
                    "Fundal height",
                ]),
                "status": "PENDING",
            })
            visit_num += 1
    
    return schedules


def _store_schedules(spark, patient_id: str, schedules: list):
    """Store schedule entries (replace existing for patient)."""
    if not schedules:
        return
    
    try:
        # Delete existing schedules for this patient, then insert new
        try:
            delete_rows(spark, "checkup_schedules", "patient_id", patient_id)
        except Exception:
            pass
        
        append_rows(spark, "checkup_schedules", schedules)
    except Exception as e:
        print(f"Error storing schedules: {e}")


def get_today_schedule(spark, asha_id: str = None) -> list:
    """Get today's scheduled visits across all patients."""
    today = date.today()
    
    sched_df = read_table(spark, "checkup_schedules")
    patients_df = read_table(spark, "patients_profiles")
    
    if asha_id:
        patients_df = patients_df[patients_df["asha_id"] == asha_id]
    
    today_visits = sched_df[
        (sched_df["scheduled_date"] == today) &
        (sched_df["status"].isin(["PENDING", "OVERDUE"]))
    ]
    
    merged = today_visits.merge(
        patients_df[["patient_id", "name", "village", "risk_status"]],
        on="patient_id",
        how="inner",
    )
    
    return [
        {
            "patient_name": row["name"],
            "patient_id": row["patient_id"],
            "village": row["village"],
            "risk_status": row["risk_status"],
            "visit_type": row["visit_type"],
            "tests_due": json.loads(row["tests_due"]) if row["tests_due"] else [],
            "status": row["status"],
        }
        for _, row in merged.iterrows()
    ]


def get_overdue_checkups(spark, asha_id: str = None) -> list:
    """Get all overdue checkups."""
    sched_df = read_table(spark, "checkup_schedules")
    patients_df = read_table(spark, "patients_profiles")
    
    if asha_id:
        patients_df = patients_df[patients_df["asha_id"] == asha_id]
    
    today = date.today()
    overdue_df = sched_df[
        (sched_df["scheduled_date"] < today) &
        (sched_df["status"] == "PENDING")
    ]
    
    merged = overdue_df.merge(
        patients_df[["patient_id", "name", "village", "risk_status"]],
        on="patient_id",
        how="inner",
    )
    merged = merged.sort_values("scheduled_date")
    
    return [
        {
            "patient_name": row["name"],
            "patient_id": row["patient_id"],
            "village": row["village"],
            "risk_status": row["risk_status"],
            "visit_type": row["visit_type"],
            "scheduled_date": str(row["scheduled_date"]),
            "days_overdue": (today - row["scheduled_date"]).days if row["scheduled_date"] else 0,
        }
        for _, row in merged.iterrows()
    ]