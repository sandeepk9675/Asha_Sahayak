# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Seed Sample Data
# MAGIC Loads realistic sample patient data into Delta Lake tables.

# COMMAND ----------

# DBTITLE 1,Cell 2
import sys, os
# Use Unity Catalog instead of DBFS
os.environ["ASHA_CATALOG"] = "workspace"
os.environ["ASHA_SCHEMA"] = "default"

notebook_dir = os.getcwd()
for candidate in [notebook_dir, os.path.dirname(notebook_dir), "/Workspace/Repos/asha-sahayak"]:
    if os.path.isdir(os.path.join(candidate, "src")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

# COMMAND ----------

# DBTITLE 1,Cell 3
from datetime import datetime, date, timedelta
import uuid
import json

# Force reload delta_utils to get latest changes
if 'src.utils.delta_utils' in sys.modules:
    del sys.modules['src.utils.delta_utils']

from src.utils.delta_utils import get_spark, append_rows, read_table

spark = get_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed Patient Profiles

# COMMAND ----------

def uid():
    return str(uuid.uuid4())

now = datetime.now()

patients = [
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Meena Devi",
        "age": 24, "village": "Rampur", "contact": "9876543210",
        "lmp_date": date(2026, 1, 15), "edd": date(2026, 10, 22),
        "blood_group": "B+", "height_cm": 155.0, "pre_pregnancy_weight_kg": 52.0,
        "risk_status": "GREEN", "language_preference": "hi",
        "registration_date": now, "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Priya Devi",
        "age": 28, "village": "Rampur", "contact": "9876543211",
        "lmp_date": date(2025, 10, 1), "edd": date(2026, 7, 8),
        "blood_group": "O+", "height_cm": 158.0, "pre_pregnancy_weight_kg": 48.0,
        "risk_status": "RED", "language_preference": "hi",
        "registration_date": now - timedelta(days=60), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Sita Kumari",
        "age": 22, "village": "Rampur", "contact": "9876543212",
        "lmp_date": date(2025, 12, 1), "edd": date(2026, 9, 7),
        "blood_group": "A+", "height_cm": 160.0, "pre_pregnancy_weight_kg": 55.0,
        "risk_status": "RED", "language_preference": "hi",
        "registration_date": now - timedelta(days=30), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Lakshmi Bai",
        "age": 30, "village": "Sundarpur", "contact": "9876543213",
        "lmp_date": date(2025, 11, 15), "edd": date(2026, 8, 22),
        "blood_group": "AB+", "height_cm": 152.0, "pre_pregnancy_weight_kg": 60.0,
        "risk_status": "GREEN", "language_preference": "hi",
        "registration_date": now - timedelta(days=45), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Kavitha R",
        "age": 26, "village": "Sundarpur", "contact": "9876543214",
        "lmp_date": date(2026, 2, 10), "edd": date(2026, 11, 17),
        "blood_group": "B-", "height_cm": 162.0, "pre_pregnancy_weight_kg": 58.0,
        "risk_status": "GREEN", "language_preference": "kn",
        "registration_date": now - timedelta(days=10), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Radha Devi",
        "age": 35, "village": "Rampur", "contact": "9876543215",
        "lmp_date": date(2025, 9, 20), "edd": date(2026, 6, 27),
        "blood_group": "O-", "height_cm": 150.0, "pre_pregnancy_weight_kg": 65.0,
        "risk_status": "RED", "language_preference": "hi",
        "registration_date": now - timedelta(days=90), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA002", "name": "Anita Sharma",
        "age": 17, "village": "Gopalpur", "contact": "9876543216",
        "lmp_date": date(2025, 12, 20), "edd": date(2026, 9, 26),
        "blood_group": "A-", "height_cm": 148.0, "pre_pregnancy_weight_kg": 45.0,
        "risk_status": "RED", "language_preference": "hi",
        "registration_date": now - timedelta(days=25), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA002", "name": "Durga Patel",
        "age": 29, "village": "Gopalpur", "contact": "9876543217",
        "lmp_date": date(2026, 1, 5), "edd": date(2026, 10, 12),
        "blood_group": "B+", "height_cm": 157.0, "pre_pregnancy_weight_kg": 54.0,
        "risk_status": "GREEN", "language_preference": "hi",
        "registration_date": now - timedelta(days=15), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA002", "name": "Saraswati Kumari",
        "age": 23, "village": "Gopalpur", "contact": "9876543218",
        "lmp_date": date(2025, 10, 15), "edd": date(2026, 7, 22),
        "blood_group": "O+", "height_cm": 165.0, "pre_pregnancy_weight_kg": 50.0,
        "risk_status": "YELLOW", "language_preference": "hi",
        "registration_date": now - timedelta(days=55), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA002", "name": "Geeta Verma",
        "age": 27, "village": "Gopalpur", "contact": "9876543219",
        "lmp_date": date(2025, 11, 1), "edd": date(2026, 8, 8),
        "blood_group": "A+", "height_cm": 155.0, "pre_pregnancy_weight_kg": 56.0,
        "risk_status": "GREEN", "language_preference": "hi",
        "registration_date": now - timedelta(days=40), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Parvathi S",
        "age": 31, "village": "Sundarpur", "contact": "9876543220",
        "lmp_date": date(2026, 3, 1), "edd": date(2026, 12, 6),
        "blood_group": "B+", "height_cm": 158.0, "pre_pregnancy_weight_kg": 53.0,
        "risk_status": "GREEN", "language_preference": "ta",
        "registration_date": now - timedelta(days=5), "last_updated": now,
    },
    {
        "patient_id": uid(), "asha_id": "ASHA001", "name": "Deepa Yadav",
        "age": 25, "village": "Rampur", "contact": "9876543221",
        "lmp_date": date(2025, 8, 15), "edd": date(2026, 5, 22),
        "blood_group": "AB-", "height_cm": 156.0, "pre_pregnancy_weight_kg": 51.0,
        "risk_status": "YELLOW", "language_preference": "hi",
        "registration_date": now - timedelta(days=120), "last_updated": now,
    },
]

append_rows(spark, "patients_profiles", patients)
print(f"✅ Seeded {len(patients)} patient profiles")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed EHR Records

# COMMAND ----------

# Get patient IDs for seeding EHR
patients_df = read_table(spark, "patients_profiles").toPandas()
patient_ids = dict(zip(patients_df["name"], patients_df["patient_id"]))

ehr_records = [
    {
        "record_id": uid(), "patient_id": patient_ids.get("Priya Devi", uid()),
        "visit_date": date(2026, 3, 15), "trimester": 2, "gestational_weeks": 24,
        "hemoglobin": 6.5, "bp_systolic": 120, "bp_diastolic": 80,
        "weight_kg": 52.0, "urine_albumin": "Normal", "urine_sugar": "Normal",
        "blood_sugar_fasting": 85.0, "blood_sugar_pp": 120.0,
        "hiv_status": "Non-reactive", "vdrl_status": "Non-reactive",
        "malaria_status": "Negative", "usg_findings": "Normal fetal growth",
        "prescribed_medicines": "IFA tablets, Calcium", "raw_document_path": None,
        "extracted_text": None, "created_at": now,
    },
    {
        "record_id": uid(), "patient_id": patient_ids.get("Sita Kumari", uid()),
        "visit_date": date(2026, 3, 20), "trimester": 2, "gestational_weeks": 18,
        "hemoglobin": 10.5, "bp_systolic": 155, "bp_diastolic": 100,
        "weight_kg": 58.0, "urine_albumin": "+", "urine_sugar": "Normal",
        "blood_sugar_fasting": 90.0, "blood_sugar_pp": 130.0,
        "hiv_status": "Non-reactive", "vdrl_status": "Non-reactive",
        "malaria_status": "Negative", "usg_findings": "Normal",
        "prescribed_medicines": "Labetalol, IFA, Calcium", "raw_document_path": None,
        "extracted_text": None, "created_at": now,
    },
    {
        "record_id": uid(), "patient_id": patient_ids.get("Meena Devi", uid()),
        "visit_date": date(2026, 3, 25), "trimester": 1, "gestational_weeks": 10,
        "hemoglobin": 12.0, "bp_systolic": 110, "bp_diastolic": 70,
        "weight_kg": 53.0, "urine_albumin": "Normal", "urine_sugar": "Normal",
        "blood_sugar_fasting": 80.0, "blood_sugar_pp": 110.0,
        "hiv_status": "Non-reactive", "vdrl_status": "Non-reactive",
        "malaria_status": "Negative", "usg_findings": "Single live intrauterine pregnancy",
        "prescribed_medicines": "Folic acid, IFA", "raw_document_path": None,
        "extracted_text": None, "created_at": now,
    },
    {
        "record_id": uid(), "patient_id": patient_ids.get("Radha Devi", uid()),
        "visit_date": date(2026, 3, 10), "trimester": 3, "gestational_weeks": 28,
        "hemoglobin": 9.0, "bp_systolic": 130, "bp_diastolic": 85,
        "weight_kg": 70.0, "urine_albumin": "Normal", "urine_sugar": "+",
        "blood_sugar_fasting": 130.0, "blood_sugar_pp": 180.0,
        "hiv_status": "Non-reactive", "vdrl_status": "Non-reactive",
        "malaria_status": "Negative", "usg_findings": "Adequate amniotic fluid",
        "prescribed_medicines": "Insulin, IFA, Calcium", "raw_document_path": None,
        "extracted_text": None, "created_at": now,
    },
    {
        "record_id": uid(), "patient_id": patient_ids.get("Saraswati Kumari", uid()),
        "visit_date": date(2026, 3, 18), "trimester": 2, "gestational_weeks": 24,
        "hemoglobin": 9.5, "bp_systolic": 118, "bp_diastolic": 76,
        "weight_kg": 54.0, "urine_albumin": "Normal", "urine_sugar": "Normal",
        "blood_sugar_fasting": 88.0, "blood_sugar_pp": 125.0,
        "hiv_status": "Non-reactive", "vdrl_status": "Non-reactive",
        "malaria_status": "Negative", "usg_findings": "Normal",
        "prescribed_medicines": "IFA (double dose), Calcium", "raw_document_path": None,
        "extracted_text": None, "created_at": now,
    },
]

append_rows(spark, "ehr_records", ehr_records)
print(f"✅ Seeded {len(ehr_records)} EHR records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Seeded Data

# COMMAND ----------

for table_name in ["patients_profiles", "ehr_records"]:
    df = read_table(spark, table_name)
    print(f"\n{table_name}: {df.count()} rows")
    df.show(5, truncate=False)

print("\n✅ Seed data loaded successfully!")
