# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Demo Scenarios: End-to-End Walk-through
# MAGIC
# MAGIC Three demo scenarios that exercise the full ASHA-Sahayak pipeline:
# MAGIC 1. New patient registration + Hindi voice chat
# MAGIC 2. EHR upload + automated risk detection
# MAGIC 3. Village dashboard + ration distribution

# COMMAND ----------

# DBTITLE 1,Cell 2
# MAGIC %pip install 'numpy<2' faiss-cpu httpx pdfplumber PyPDF2 --disable-pip-version-check

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 3
import os, sys, json
from datetime import date, timedelta

# Project root
project_root = "/Workspace/Users/hemasrisail@iisc.ac.in/Asha_Sahayak"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use Unity Catalog instead of DBFS
os.environ["ASHA_CATALOG"] = "workspace"
os.environ["ASHA_SCHEMA"] = "default"
os.environ["ASHA_FAISS_PATH"] = "/Volumes/workspace/default/asha_sahayak/faiss"

# Force reload delta_utils to get Unity Catalog version
if 'src.utils.delta_utils' in sys.modules:
    del sys.modules['src.utils.delta_utils']

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

# DBTITLE 1,Load API Keys from Secrets
# Load API keys from Databricks secrets
try:
    # Try to load Databricks token for Foundation Model API
    databricks_token = dbutils.secrets.get(scope="asha-sahayak", key="databricks-token")
    os.environ["DATABRICKS_TOKEN"] = databricks_token
    print("✅ Loaded DATABRICKS_TOKEN from secrets")
except Exception as e:
    print(f"⚠️ Could not load DATABRICKS_TOKEN: {e}")
    # Fallback to personal access token if available
    try:
        import requests
        # Use notebook context token
        context_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        os.environ["DATABRICKS_TOKEN"] = context_token
        print("✅ Using notebook context token as DATABRICKS_TOKEN")
    except:
        print("❌ No DATABRICKS_TOKEN available")

try:
    # Try to load Sarvam AI API key for Hindi translation
    sarvam_key = dbutils.secrets.get(scope="asha-sahayak", key="sarvam_api_key")
    os.environ["SARVAM_API_KEY"] = sarvam_key
    print("✅ Loaded SARVAM_API_KEY from secrets")
except Exception as e:
    print(f"⚠️ Could not load SARVAM_API_KEY: {e}")

# Set DATABRICKS_HOST from workspace URL
try:
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    os.environ["DATABRICKS_HOST"] = f"https://{workspace_url}"
    print(f"✅ Set DATABRICKS_HOST to: https://{workspace_url}")
except Exception as e:
    print(f"⚠️ Could not set DATABRICKS_HOST: {e}")

# Force reimport of delta_utils with new schemas
import importlib
import sys

# Remove cached modules
if 'src.utils.delta_utils' in sys.modules:
    del sys.modules['src.utils.delta_utils']
if 'src.services.chat_service' in sys.modules:
    del sys.modules['src.services.chat_service']
if 'src.pipeline.risk_engine' in sys.modules:
    del sys.modules['src.pipeline.risk_engine']

# Add missing risk_assessments schema directly to the source file
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, TimestampType, BooleanType
)

risk_assessments_schema = StructType([
    StructField("assessment_id", StringType(), False),
    StructField("patient_id", StringType(), True),
    StructField("assessment_date", TimestampType(), True),
    StructField("risk_level", StringType(), True),
    StructField("risk_factors", StringType(), True),
    StructField("recommended_action", StringType(), True),
    StructField("emergency_flag", BooleanType(), True),
    StructField("auto_appointment_created", BooleanType(), True),
])

# Read the delta_utils file
with open("/Workspace/Users/hemasrisail@iisc.ac.in/Asha_Sahayak/src/utils/delta_utils.py", "r") as f:
    content = f.read()

# Check if risk_assessments schema already exists
if '"risk_assessments"' not in content:
    # Add the risk_assessments schema to SCHEMAS dict
    schema_def = '''
    "risk_assessments": StructType([
        StructField("assessment_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("assessment_date", TimestampType(), True),
        StructField("risk_level", StringType(), True),
        StructField("risk_factors", StringType(), True),
        StructField("recommended_action", StringType(), True),
        StructField("emergency_flag", BooleanType(), True),
        StructField("auto_appointment_created", BooleanType(), True),
    ]),'''
    
    # Insert before the closing brace of SCHEMAS
    insert_pos = content.rfind('}', 0, content.find('# -----', content.find('SCHEMAS')))
    new_content = content[:insert_pos] + schema_def + '\n' + content[insert_pos:]
    
    # Write back
    with open("/Workspace/Users/hemasrisail@iisc.ac.in/Asha_Sahayak/src/utils/delta_utils.py", "w") as f:
        f.write(new_content)
    
    print("✅ Added risk_assessments schema to delta_utils.py")
else:
    print("✓ risk_assessments schema already exists")

# Now reimport with the new schema
from src.utils.delta_utils import get_spark, create_table

# Create missing tables
print("\nCreating missing tables...")
create_table(spark, "conversations")
create_table(spark, "risk_assessments")
print("\n✅ All required tables created")

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

# DBTITLE 1,Cell 12
# 1f. View conversation history
history = get_chat_history(spark, demo_patient_id)
print(f"\n💬 Conversation ({len(history)} messages):")
for h in history:
    print(f"  👩 User: {h['user_message'][:100]}...")
    print(f"  🤖 AI: {h['ai_response'][:100]}...")
    print()

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

# DBTITLE 1,Cell 16
# 2c. View all EHRs
ehrs_df = spark.table("workspace.default.ehr_records").filter(f"patient_id = '{demo_patient_id}'").orderBy("visit_date")
ehrs = ehrs_df.collect()

print(f"\n📋 EHR Records ({len(ehrs)}):")
for e in ehrs:
    bp = f"{e['bp_systolic']}/{e['bp_diastolic']}"
    print(f"  {e['visit_date']}: Hb={e['hemoglobin']}, BP={bp}, Weight={e['weight_kg']}kg, Urine Albumin={e['urine_albumin']}")

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
