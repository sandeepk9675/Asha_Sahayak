"""
Dashboard Service for ASHA-Sahayak.
Pandas aggregations for the village-level dashboard.
"""

import json
from datetime import date, datetime
from typing import Optional

import pandas as pd

from src.utils.delta_utils import read_table


def get_dashboard_data(spark, asha_id: str = None) -> dict:
    """
    Get comprehensive dashboard data for the ASHA worker's village view.
    """
    today = date.today()
    
    # Load all relevant tables
    patients_df = read_table(spark, "patients_profiles")
    if asha_id:
        patients_df = patients_df[patients_df["asha_id"] == asha_id]
    
    result = {
        "today_date": str(today),
        "alerts": _get_alerts(spark, patients_df),
        "today_schedule": _get_today_schedule(spark, patients_df, today),
        "village_stats": _get_village_stats(spark, patients_df, today),
        "risk_distribution": _get_risk_distribution(patients_df),
        "trimester_distribution": _get_trimester_distribution(patients_df, today),
        "overdue_checkups": _get_overdue(spark, patients_df, today),
        "upcoming_deliveries": _get_upcoming_deliveries(patients_df, today),
    }
    
    return result


def _get_alerts(spark, patients_df) -> list:
    """Get active emergency/high-risk alerts."""
    try:
        risk_df = read_table(spark, "risk_assessments")
        if risk_df.empty:
            return []
        
        # Get latest risk per patient
        risk_df = risk_df.sort_values("assessment_date", ascending=False)
        latest_risks = risk_df.drop_duplicates(subset=["patient_id"], keep="first")
        
        # Filter RED or emergency
        alerts_df = latest_risks[
            (latest_risks["risk_level"] == "RED") | (latest_risks["emergency_flag"] == True)
        ]
        
        if alerts_df.empty:
            return []
        
        # Join with patients
        merged = alerts_df.merge(
            patients_df[["patient_id", "name", "village"]],
            on="patient_id",
            how="inner",
        )
        
        # Sort: emergency first, then by date
        merged = merged.sort_values(
            ["emergency_flag", "assessment_date"],
            ascending=[False, False],
        )
        
        return [
            {
                "patient_name": row["name"],
                "patient_id": row["patient_id"],
                "village": row["village"],
                "risk_level": row["risk_level"],
                "emergency": bool(row["emergency_flag"]),
                "risk_factors": json.loads(row["risk_factors"]) if row["risk_factors"] else [],
                "action": row["recommended_action"],
            }
            for _, row in merged.iterrows()
        ]
    except Exception:
        return []


def _get_today_schedule(spark, patients_df, today) -> list:
    """Get today's visit schedule."""
    try:
        sched_df = read_table(spark, "checkup_schedules")
        if sched_df.empty:
            return []
        
        # Filter today's pending/overdue
        visits_df = sched_df[
            (sched_df["scheduled_date"] == today) &
            (sched_df["status"].isin(["PENDING", "OVERDUE"]))
        ]
        
        if visits_df.empty:
            return []
        
        merged = visits_df.merge(
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
                "visit_number": row.get("visit_number", 0),
                "tests_due": json.loads(row["tests_due"]) if row["tests_due"] else [],
            }
            for _, row in merged.iterrows()
        ]
    except Exception:
        return []


def _get_village_stats(spark, patients_df, today) -> dict:
    """Get aggregate statistics."""
    total = len(patients_df)
    
    risk_counts = patients_df["risk_status"].value_counts().to_dict()
    
    # Overdue count
    overdue_count = 0
    try:
        sched_df = read_table(spark, "checkup_schedules")
        if not sched_df.empty:
            patient_ids = set(patients_df["patient_id"])
            overdue = sched_df[
                (sched_df["patient_id"].isin(patient_ids)) &
                (sched_df["scheduled_date"] < today) &
                (sched_df["status"] == "PENDING")
            ]
            overdue_count = overdue["patient_id"].nunique()
    except Exception:
        pass
    
    return {
        "total_patients": total,
        "green_count": risk_counts.get("GREEN", 0),
        "yellow_count": risk_counts.get("YELLOW", 0),
        "red_count": risk_counts.get("RED", 0),
        "overdue_checkups": overdue_count,
    }


def _get_risk_distribution(patients_df) -> dict:
    """Get risk level distribution."""
    return patients_df["risk_status"].value_counts().to_dict()


def _get_trimester_distribution(patients_df, today) -> dict:
    """Get distribution by trimester."""
    df = patients_df.copy()
    df["gestational_weeks"] = df["lmp_date"].apply(
        lambda lmp: (today - lmp).days / 7 if lmp and pd.notna(lmp) else 0
    )
    df["trimester"] = df["gestational_weeks"].apply(
        lambda w: 1 if w <= 12 else (2 if w <= 27 else 3)
    )
    counts = df["trimester"].value_counts().to_dict()
    return {f"T{int(k)}": v for k, v in counts.items()}


def _get_overdue(spark, patients_df, today) -> list:
    """Get patients with overdue checkups."""
    try:
        sched_df = read_table(spark, "checkup_schedules")
        if sched_df.empty:
            return []
        
        overdue_df = sched_df[
            (sched_df["scheduled_date"] < today) &
            (sched_df["status"] == "PENDING")
        ]
        
        if overdue_df.empty:
            return []
        
        merged = overdue_df.merge(
            patients_df[["patient_id", "name", "village", "risk_status"]],
            on="patient_id",
            how="inner",
        )
        merged = merged.sort_values("scheduled_date").head(10)
        
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
    except Exception:
        return []


def _get_upcoming_deliveries(patients_df, today) -> list:
    """Get patients with EDD in next 30 days."""
    from datetime import timedelta
    
    df = patients_df.copy()
    df = df[df["edd"].notna()]
    df = df[
        (df["edd"] >= today) &
        (df["edd"] <= today + timedelta(days=30))
    ]
    df = df.sort_values("edd")
    
    return [
        {
            "patient_name": row["name"],
            "patient_id": row["patient_id"],
            "village": row["village"],
            "risk_status": row["risk_status"],
            "edd": str(row["edd"]),
            "days_until_edd": (row["edd"] - today).days,
        }
        for _, row in df.iterrows()
    ]


def get_dashboard_summary_text(spark, asha_id: str = None) -> str:
    """Get a formatted text summary for the dashboard."""
    data = get_dashboard_data(spark, asha_id)
    
    lines = []
    lines.append(f"📅 Date: {data['today_date']}")
    lines.append("")
    
    # Alerts
    alerts = data["alerts"]
    if alerts:
        lines.append(f"⚠️ ALERTS ({len(alerts)})")
        for a in alerts:
            emoji = "🚨" if a["emergency"] else "🔴"
            factors = ", ".join(a["risk_factors"][:2]) if a["risk_factors"] else "High risk"
            lines.append(f"  {emoji} {a['patient_name']} — {factors}")
        lines.append("")
    
    # Today's schedule
    schedule = data["today_schedule"]
    lines.append(f"📋 TODAY'S SCHEDULE ({len(schedule)} visits)")
    if schedule:
        for v in schedule:
            lines.append(f"  • {v['patient_name']} ({v['village']}) — {v['visit_type']}")
    else:
        lines.append("  No visits scheduled for today.")
    lines.append("")
    
    # Stats
    stats = data["village_stats"]
    lines.append("📊 VILLAGE OVERVIEW")
    lines.append(f"  Total: {stats['total_patients']}  🟢 {stats['green_count']}  🟡 {stats['yellow_count']}  🔴 {stats['red_count']}")
    
    tri = data["trimester_distribution"]
    tri_str = " | ".join([f"{k}: {v}" for k, v in sorted(tri.items())])
    lines.append(f"  Trimesters: {tri_str}")
    lines.append(f"  Overdue checkups: {stats['overdue_checkups']}")
    lines.append("")
    
    # Upcoming deliveries
    upcoming = data["upcoming_deliveries"]
    if upcoming:
        lines.append(f"🏥 UPCOMING DELIVERIES (next 30 days): {len(upcoming)}")
        for u in upcoming:
            lines.append(f"  • {u['patient_name']} — EDD: {u['edd']} ({u['days_until_edd']} days)")
    
    return "\n".join(lines)
