"""
Dashboard Service for ASHA-Sahayak.
Spark SQL aggregations for the village-level dashboard.
"""

import json
from datetime import date, datetime
from typing import Optional

from src.utils.delta_utils import read_table
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def get_dashboard_data(spark, asha_id: str = None) -> dict:
    """
    Get comprehensive dashboard data for the ASHA worker's village view.
    
    Returns:
        {
            "today_date": str,
            "alerts": list,
            "today_schedule": list,
            "village_stats": dict,
            "risk_distribution": dict,
            "trimester_distribution": dict,
            "overdue_checkups": list,
            "upcoming_deliveries": list,
        }
    """
    today = date.today()
    
    # Load all relevant tables
    patients_df = read_table(spark, "patients_profiles")
    if asha_id:
        patients_df = patients_df.filter(F.col("asha_id") == asha_id)
    
    # Note: cache() is not supported on serverless compute
    
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
        
        # Get latest risk per patient
        w = Window.partitionBy("patient_id").orderBy(F.col("assessment_date").desc())
        latest_risks = risk_df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")
        
        alerts = (
            latest_risks
            .filter(
                (F.col("risk_level") == "RED") | (F.col("emergency_flag") == True)
            )
            .join(patients_df.select("patient_id", "name", "village"), "patient_id")
            .orderBy(F.col("emergency_flag").desc(), F.col("assessment_date").desc())
            .collect()
        )
        
        return [
            {
                "patient_name": a["name"],
                "patient_id": a["patient_id"],
                "village": a["village"],
                "risk_level": a["risk_level"],
                "emergency": a["emergency_flag"],
                "risk_factors": json.loads(a["risk_factors"]) if a["risk_factors"] else [],
                "action": a["recommended_action"],
            }
            for a in alerts
        ]
    except Exception:
        return []


def _get_today_schedule(spark, patients_df, today) -> list:
    """Get today's visit schedule."""
    try:
        sched_df = read_table(spark, "checkup_schedules")
        
        visits = (
            sched_df
            .filter(
                (F.col("scheduled_date") == today) &
                (F.col("status").isin("PENDING", "OVERDUE"))
            )
            .join(patients_df.select("patient_id", "name", "village", "risk_status"), "patient_id")
            .orderBy("scheduled_date")
            .collect()
        )
        
        return [
            {
                "patient_name": v["name"],
                "patient_id": v["patient_id"],
                "village": v["village"],
                "risk_status": v["risk_status"],
                "visit_type": v["visit_type"],
                "visit_number": v.get("visit_number", 0),
                "tests_due": json.loads(v["tests_due"]) if v["tests_due"] else [],
            }
            for v in visits
        ]
    except Exception:
        return []


def _get_village_stats(spark, patients_df, today) -> dict:
    """Get aggregate statistics."""
    total = patients_df.count()
    
    # Risk counts
    risk_counts = (
        patients_df
        .groupBy("risk_status")
        .agg(F.count("*").alias("count"))
        .collect()
    )
    risk_map = {r["risk_status"]: r["count"] for r in risk_counts}
    
    # Overdue count
    overdue_count = 0
    try:
        sched_df = read_table(spark, "checkup_schedules")
        patient_ids = [r["patient_id"] for r in patients_df.select("patient_id").collect()]
        overdue_count = (
            sched_df
            .filter(
                (F.col("patient_id").isin(patient_ids)) &
                (F.col("scheduled_date") < today) &
                (F.col("status") == "PENDING")
            )
            .select("patient_id")
            .distinct()
            .count()
        )
    except Exception:
        pass
    
    return {
        "total_patients": total,
        "green_count": risk_map.get("GREEN", 0),
        "yellow_count": risk_map.get("YELLOW", 0),
        "red_count": risk_map.get("RED", 0),
        "overdue_checkups": overdue_count,
    }


def _get_risk_distribution(patients_df) -> dict:
    """Get risk level distribution."""
    counts = (
        patients_df
        .groupBy("risk_status")
        .agg(F.count("*").alias("count"))
        .collect()
    )
    return {r["risk_status"]: r["count"] for r in counts}


def _get_trimester_distribution(patients_df, today) -> dict:
    """Get distribution by trimester."""
    patients_with_weeks = patients_df.withColumn(
        "gestational_weeks",
        F.datediff(F.lit(today), F.col("lmp_date")) / 7
    ).withColumn(
        "trimester",
        F.when(F.col("gestational_weeks") <= 12, 1)
        .when(F.col("gestational_weeks") <= 27, 2)
        .otherwise(3)
    )
    
    counts = (
        patients_with_weeks
        .groupBy("trimester")
        .agg(F.count("*").alias("count"))
        .collect()
    )
    
    return {f"T{int(r['trimester'])}": r["count"] for r in counts}


def _get_overdue(spark, patients_df, today) -> list:
    """Get patients with overdue checkups."""
    try:
        sched_df = read_table(spark, "checkup_schedules")
        
        overdue = (
            sched_df
            .filter(
                (F.col("scheduled_date") < today) &
                (F.col("status") == "PENDING")
            )
            .join(patients_df.select("patient_id", "name", "village", "risk_status"), "patient_id")
            .orderBy("scheduled_date")
            .limit(10)
            .collect()
        )
        
        return [
            {
                "patient_name": r["name"],
                "patient_id": r["patient_id"],
                "village": r["village"],
                "risk_status": r["risk_status"],
                "visit_type": r["visit_type"],
                "scheduled_date": str(r["scheduled_date"]),
                "days_overdue": (today - r["scheduled_date"]).days,
            }
            for r in overdue
        ]
    except Exception:
        return []


def _get_upcoming_deliveries(patients_df, today) -> list:
    """Get patients with EDD in next 30 days."""
    from datetime import timedelta
    
    upcoming = (
        patients_df
        .filter(
            (F.col("edd") >= today) &
            (F.col("edd") <= today + timedelta(days=30))
        )
        .orderBy("edd")
        .collect()
    )
    
    return [
        {
            "patient_name": r["name"],
            "patient_id": r["patient_id"],
            "village": r["village"],
            "risk_status": r["risk_status"],
            "edd": str(r["edd"]),
            "days_until_edd": (r["edd"] - today).days,
        }
        for r in upcoming
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
