# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Demo Scenarios: End-to-End Walk-through
# MAGIC
# MAGIC Three demo scenarios that exercise the full ASHA-Sahayak pipeline:
# MAGIC 1. New patient registration + Hindi voice chat
# MAGIC 2. EHR upload + automated risk detection
# MAGIC 3. Village dashboard + ration distribution

# COMMAND ----------

# MAGIC %pip install faiss-cpu httpx pdfplumber PyPDF2 --quiet

# COMMAND ----------

import os, sys, json
from datetime import date, timedelta

# Project root
project_root = "/Workspace/Repos/asha-sahayak/Asha_Sahayak"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("ASHA_DELTA_BASE", "/dbfs/asha_sahayak/delta")
os.environ.setdefault("ASHA_FAISS_PATH", "/dbfs/asha_sahayak/faiss")

from src.utils.delta_utils import get_spark
from src.services.patient_service import register_patient, get_patient, list_patients
from src.services.ehr_service import add_ehr_manual, get_patient_ehrs
from src.services.chat_service import chat, get_chat_history
from src.services.dashboard_service import get_dashboard_data, get_dashboard_summary_text
from src.pipeline.schedule_engine import generate_schedule, get_overdue_checkups
from src.pipeline.ration_engine import generate_ration_plan, get_village_ration_summary
from src.pipeline.risk_engine import assess_risk, get_patient_risk_summary
from src.pipeline.language_pipeline import process_text_input

spark = get_spark()
ASHA_ID = "ASHA001"

print("✅ All modules loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 1: New Patient Registration + Hindi Chat
# MAGIC
# MAGIC An ASHA worker registers a new pregnant woman and asks health questions in Hindi.

# COMMAND ----------

# 1a. Register a new patient
demo_lmp = date.today() - timedelta(weeks=20)  # 20 weeks pregnant

result = register_patient(
    spark,
    name="Sita Devi",
    age=23,
    lmp_date=demo_lmp,
    village="Rampur",
    contact="9876599999",
    language_preference="hi",
    blood_group="B+",
    height_cm=155,
    pre_pregnancy_weight_kg=50,
    asha_id=ASHA_ID,
)

print(f"Registration: {result['message']}")
demo_patient_id = result["patient_id"]
print(f"Patient ID: {demo_patient_id}")

# COMMAND ----------

# 1b. View patient profile
patient = get_patient(spark, demo_patient_id)
print(json.dumps(patient, indent=2, default=str))

# COMMAND ----------

# 1c. Generate ANC schedule
schedules = generate_schedule(spark, demo_patient_id)
print(f"\n📅 Generated {len(schedules)} ANC visits:")
for s in schedules:
    print(f"  Visit {s['visit_number']}: {s['scheduled_date']} ({s['visit_type']}) — {s['status']}")

# COMMAND ----------

# 1d. Chat in Hindi — ask about diet
chat_result = chat(
    spark,
    patient_id=demo_patient_id,
    message="गर्भावस्था में क्या खाना चाहिए?",  # "What should I eat during pregnancy?"
    language="hi",
    asha_id=ASHA_ID,
)

print(f"🤖 Response (Hindi):\n{chat_result['response']}\n")
print(f"📝 English version:\n{chat_result.get('english_response', 'N/A')}")

# COMMAND ----------

# 1e. Follow-up question about iron tablets
chat_result2 = chat(
    spark,
    patient_id=demo_patient_id,
    message="आयरन की गोली कब लेनी चाहिए?",  # "When should I take iron tablets?"
    language="hi",
    asha_id=ASHA_ID,
)

print(f"🤖 Response:\n{chat_result2['response']}")

# COMMAND ----------

# 1f. View conversation history
history = get_chat_history(spark, demo_patient_id)
print(f"\n💬 Conversation ({len(history)} messages):")
for h in history:
    role_icon = "👩" if h["role"] == "user" else "🤖"
    print(f"  {role_icon} [{h['language']}] {h['content'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2: EHR Upload + Automatic Risk Detection
# MAGIC
# MAGIC An ASHA worker enters lab results. The system detects severe anemia and high BP → RED risk.

# COMMAND ----------

# 2a. Add a concerning EHR record — low hemoglobin + high BP
ehr_result = add_ehr_manual(
    spark,
    patient_id=demo_patient_id,
    hemoglobin=6.5,          # Severe anemia (< 7)
    bp_systolic=165,          # Severe hypertension
    bp_diastolic=112,
    weight_kg=52,
    blood_sugar_fasting=82,
    urine_albumin="++",       # Proteinuria → pre-eclampsia risk
    prescribed_medicines="Methyldopa 250mg TDS, Iron sucrose IV",
)

print(f"EHR Result: {ehr_result['message']}")

# COMMAND ----------

# 2b. Check risk assessment — should be RED
risk = get_patient_risk_summary(spark, demo_patient_id)
print(f"\n🚨 RISK ASSESSMENT:")
print(f"  Level: {risk['risk_level']}")
print(f"  Factors: {risk['risk_factors']}")
print(f"  Action: {risk['recommended_action']}")

# Verify patient status updated
patient_updated = get_patient(spark, demo_patient_id)
print(f"\n  Patient risk_status in DB: {patient_updated['risk_status']}")

# COMMAND ----------

# 2c. View all EHRs
ehrs = get_patient_ehrs(spark, demo_patient_id)
print(f"\n📋 EHR Records ({len(ehrs)}):")
for e in ehrs:
    print(f"  {e['visit_date']}: Hb={e['hemoglobin']}, BP={e['bp']}, Weight={e['weight_kg']}kg, Urine={e['urine_albumin']}")

# COMMAND ----------

# 2d. Chat after EHR — system should reference the alert
chat_result3 = chat(
    spark,
    patient_id=demo_patient_id,
    message="What is the patient's current health condition?",
    language="en",
    asha_id=ASHA_ID,
)

print(f"🤖 Response:\n{chat_result3['response']}")
if chat_result3.get("risk_alert"):
    print(f"\n🚨 Risk Alert: {chat_result3['risk_alert']}")

# COMMAND ----------

# 2e. Generate ration plan (should show anemia-specific diet)
ration = generate_ration_plan(spark, demo_patient_id, use_llm=True)
print(f"\n🍚 Ration Plan for {ration['patient_name']}:")
print(f"  Trimester: T{ration['trimester']}")
print(f"  Calories: {ration['daily_calorie_target']} kcal/day")
print(f"  Protein: {ration['protein_target_g']}g/day")
print(f"  Conditions: {ration['conditions']}")

print("\n  Items:")
for item in ration.get("ration_items", []):
    print(f"    - {item['item']}: {item['quantity_g']}g ({item['frequency']})")

print("\n  Supplements:")
for s in ration.get("supplements", []):
    print(f"    - {s['name']}: {s['dosage']} ({s['frequency']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 3: Village Dashboard + Ration Distribution
# MAGIC
# MAGIC ASHA worker views village-level overview — all patients, alerts, upcoming deliveries.

# COMMAND ----------

# 3a. Full dashboard
dashboard = get_dashboard_data(spark, ASHA_ID)

print("📊 VILLAGE DASHBOARD")
print("=" * 60)

stats = dashboard.get("village_stats", {})
print(f"\n👥 Total Patients: {stats.get('total_patients', 0)}")
print(f"  🟢 Normal: {stats.get('green_count', 0)}")
print(f"  🟡 Watch: {stats.get('yellow_count', 0)}")
print(f"  🔴 High Risk: {stats.get('red_count', 0)}")
print(f"  ⚠️ Overdue Checkups: {stats.get('overdue_checkups', 0)}")

tri = dashboard.get("trimester_distribution", {})
print(f"\n📊 By Trimester: {tri}")

# COMMAND ----------

# 3b. Alerts
alerts = dashboard.get("alerts", [])
print(f"\n🚨 ALERTS ({len(alerts)}):")
for a in alerts:
    factors = ", ".join(a.get("risk_factors", []))
    emergency = "🆘 EMERGENCY" if a.get("emergency") else "⚠️"
    print(f"  {emergency} {a['patient_name']} — {factors}")

# COMMAND ----------

# 3c. Today's schedule
today_sched = dashboard.get("today_schedule", [])
print(f"\n📅 TODAY'S SCHEDULE ({len(today_sched)} visits):")
for t in today_sched:
    print(f"  → {t['patient_name']}: {t['visit_type']} (Visit #{t['visit_number']})")

if not today_sched:
    print("  No visits scheduled for today")

# COMMAND ----------

# 3d. Upcoming deliveries
deliveries = dashboard.get("upcoming_deliveries", [])
print(f"\n🏥 UPCOMING DELIVERIES ({len(deliveries)}):")
for d in deliveries:
    risk_icon = "🟢" if d["risk_status"] == "GREEN" else "🟡" if d["risk_status"] == "YELLOW" else "🔴"
    print(f"  {risk_icon} {d['patient_name']} — EDD: {d['edd']} ({d['days_until_edd']} days away)")

# COMMAND ----------

# 3e. Village-wide ration summary
print("\n🍚 VILLAGE RATION DISTRIBUTION:")
print("=" * 60)

ration_summary = get_village_ration_summary(spark, ASHA_ID)
for r in ration_summary:
    items = ", ".join([f"{i['item']}({i['quantity_g']}g)" for i in r["ration_items"][:3]]) if r["ration_items"] else "Not yet generated"
    supps = ", ".join([s["name"] for s in r["supplements"]]) if r["supplements"] else "-"
    print(f"  {r['patient_name']} (T{r['trimester']}, {r['risk_status']}): {items} | Supplements: {supps}")

# COMMAND ----------

# 3f. Full formatted summary
print("\n" + "=" * 60)
print(get_dashboard_summary_text(spark, ASHA_ID))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ All Scenarios Complete
# MAGIC
# MAGIC | Scenario | Feature | Status |
# MAGIC |----------|---------|--------|
# MAGIC | 1 | Patient Registration | ✅ |
# MAGIC | 1 | ANC Schedule Generation | ✅ |
# MAGIC | 1 | Hindi Language Chat (RAG) | ✅ |
# MAGIC | 2 | Manual EHR Entry | ✅ |
# MAGIC | 2 | Automated Risk Assessment | ✅ |
# MAGIC | 2 | Condition-specific Ration Plan | ✅ |
# MAGIC | 3 | Village Dashboard | ✅ |
# MAGIC | 3 | Alerts & Upcoming Deliveries | ✅ |
# MAGIC | 3 | Village Ration Distribution | ✅ |
