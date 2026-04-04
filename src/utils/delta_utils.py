"""
Delta Lake utility functions for ASHA-Sahayak.
Provides read/write helpers for all Delta tables.
Works with Unity Catalog managed tables in serverless compute.
"""

import os
import json
from datetime import datetime, date
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType,
    DateType, TimestampType, BooleanType
)


# ---------------------------------------------------------------------------
# Spark session (reuses existing on Databricks, creates local otherwise)
# ---------------------------------------------------------------------------

def get_spark() -> SparkSession:
    """Get or create SparkSession with Delta Lake support."""
    try:
        # On Databricks, spark is pre-configured
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is not None:
            return spark
    except Exception:
        pass

    # Local / fallback
    spark = (
        SparkSession.builder
        .appName("ASHA-Sahayak")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark


# ---------------------------------------------------------------------------
# Unity Catalog table helpers
# ---------------------------------------------------------------------------

def _catalog_name() -> str:
    """Return catalog name from environment or default."""
    return os.environ.get("ASHA_CATALOG", "workspace")


def _schema_name() -> str:
    """Return schema name from environment or default."""
    return os.environ.get("ASHA_SCHEMA", "default")


def table_name(table: str) -> str:
    """Return fully qualified Unity Catalog table name."""
    return f"{_catalog_name()}.{_schema_name()}.{table}"

def table_path(table: str) -> str:
    """Get the data path for a Delta table.
    For Unity Catalog tables, this returns the table's data location.
    """
    spark = get_spark()
    full_name = table_name(table)
    
    # For Unity Catalog, get the table location from metadata
    try:
        table_details = spark.sql(f"DESCRIBE DETAIL {full_name}").collect()
        if table_details:
            return table_details[0]["location"]
    except Exception:
        pass
    
    # Fallback: construct path (though this shouldn't be used for UC tables)
    catalog = _catalog_name()
    schema = _schema_name()
    return f"/Volumes/{catalog}/{schema}/asha_sahayak/tables/{table}"



# ---------------------------------------------------------------------------
# Schema definitions for all tables
# ---------------------------------------------------------------------------

SCHEMAS = {
    "patients_profiles": StructType([
        StructField("patient_id", StringType(), False),
        StructField("asha_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("village", StringType(), True),
        StructField("contact", StringType(), True),
        StructField("lmp_date", DateType(), True),
        StructField("edd", DateType(), True),
        StructField("blood_group", StringType(), True),
        StructField("height_cm", FloatType(), True),
        StructField("pre_pregnancy_weight_kg", FloatType(), True),
        StructField("risk_status", StringType(), True),
        StructField("language_preference", StringType(), True),
        StructField("registration_date", TimestampType(), True),
        StructField("last_updated", TimestampType(), True),
    ]),
    "ehr_records": StructType([
        StructField("record_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("visit_date", DateType(), True),
        StructField("trimester", IntegerType(), True),
        StructField("gestational_weeks", IntegerType(), True),
        StructField("hemoglobin", FloatType(), True),
        StructField("bp_systolic", IntegerType(), True),
        StructField("bp_diastolic", IntegerType(), True),
        StructField("weight_kg", FloatType(), True),
        StructField("urine_albumin", StringType(), True),
        StructField("urine_sugar", StringType(), True),
        StructField("blood_sugar_fasting", FloatType(), True),
        StructField("blood_sugar_pp", FloatType(), True),
        StructField("hiv_status", StringType(), True),
        StructField("vdrl_status", StringType(), True),
        StructField("malaria_status", StringType(), True),
        StructField("anemia_status", StringType(), True),
        StructField("complications", StringType(), True),
        StructField("notes", StringType(), True),
        StructField("created_at", TimestampType(), True),
    ]),
    "ifa_logs": StructType([
        StructField("log_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("dispensed_date", DateType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("reported_adherence", FloatType(), True),
        StructField("side_effects", StringType(), True),
        StructField("created_at", TimestampType(), True),
    ]),
    "anc_visits": StructType([
        StructField("visit_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("visit_number", IntegerType(), True),
        StructField("visit_date", DateType(), True),
        StructField("gestational_weeks", IntegerType(), True),
        StructField("weight_kg", FloatType(), True),
        StructField("bp_systolic", IntegerType(), True),
        StructField("bp_diastolic", IntegerType(), True),
        StructField("fundal_height_cm", FloatType(), True),
        StructField("fetal_heart_rate", IntegerType(), True),
        StructField("fetal_presentation", StringType(), True),
        StructField("edema", BooleanType(), True),
        StructField("danger_signs", StringType(), True),
        StructField("tt_dose_given", IntegerType(), True),
        StructField("next_visit_date", DateType(), True),
        StructField("created_at", TimestampType(), True),
    ]),
    "checkup_schedules": StructType([
        StructField("schedule_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("visit_number", IntegerType(), True),
        StructField("scheduled_date", DateType(), True),
        StructField("actual_date", DateType(), True),
        StructField("visit_type", StringType(), True),
        StructField("tests_due", StringType(), True),
        StructField("status", StringType(), True),
    ]),
    "conversations": StructType([
        StructField("conversation_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("asha_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("input_type", StringType(), True),
        StructField("original_input", StringType(), True),
        StructField("translated_input", StringType(), True),
        StructField("ai_response", StringType(), True),
        StructField("translated_response", StringType(), True),
        StructField("extracted_health_updates", StringType(), True),
    ]),
    "nutrition_logs": StructType([
        StructField("log_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("log_date", DateType(), True),
        StructField("meal_type", StringType(), True),
        StructField("food_items", StringType(), True),
        StructField("calories_est", IntegerType(), True),
        StructField("protein_est_g", FloatType(), True),
        StructField("iron_est_mg", FloatType(), True),
        StructField("calcium_est_mg", FloatType(), True),
        StructField("created_at", TimestampType(), True),
    ]),
    "risk_scores": StructType([
        StructField("score_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("score_date", DateType(), True),
        StructField("model_version", StringType(), True),
        StructField("risk_score", FloatType(), True),
        StructField("risk_category", StringType(), True),
        StructField("anemia_risk", FloatType(), True),
        StructField("preeclampsia_risk", FloatType(), True),
        StructField("gdm_risk", FloatType(), True),
        StructField("preterm_risk", FloatType(), True),
        StructField("top_factors", StringType(), True),
        StructField("recommendations", StringType(), True),
        StructField("created_at", TimestampType(), True),
    ]),
    "notifications": StructType([
        StructField("notification_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("asha_id", StringType(), True),
        StructField("notification_type", StringType(), True),
        StructField("message", StringType(), True),
        StructField("priority", StringType(), True),
        StructField("status", StringType(), True),
        StructField("scheduled_at", TimestampType(), True),
        StructField("sent_at", TimestampType(), True),
        StructField("created_at", TimestampType(), True),
    ]),

    "risk_assessments": StructType([
        StructField("assessment_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("assessment_date", TimestampType(), True),
        StructField("risk_level", StringType(), True),
        StructField("risk_factors", StringType(), True),
        StructField("recommended_action", StringType(), True),
        StructField("emergency_flag", BooleanType(), True),
        StructField("auto_appointment_created", BooleanType(), True),
    ]),
    "ration_plans": StructType([
        StructField("plan_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("week_start_date", DateType(), True),
        StructField("week_end_date", DateType(), True),
        StructField("trimester", IntegerType(), True),
        StructField("daily_calorie_target", IntegerType(), True),
        StructField("protein_target_g", IntegerType(), True),
        StructField("ration_items", StringType(), True),  # JSON string
        StructField("supplements", StringType(), True),  # JSON string
        StructField("special_notes", StringType(), True),
        StructField("generated_by_model", StringType(), True),
    ]),

    "appointments": StructType([
        StructField("appointment_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("facility_name", StringType(), True),
        StructField("facility_type", StringType(), True),
        StructField("scheduled_datetime", TimestampType(), True),
        StructField("appointment_type", StringType(), True),
        StructField("status", StringType(), True),
        StructField("notes", StringType(), True),
    ]),
}


# ---------------------------------------------------------------------------
# Table creation and management
# ---------------------------------------------------------------------------

def create_table(spark: SparkSession, table: str) -> None:
    """Create an empty Unity Catalog managed table if it doesn't exist."""
    schema = SCHEMAS[table]
    full_name = table_name(table)
    
    try:
        # Check if table exists
        spark.read.table(full_name)
        print(f"✓ Table already exists: {full_name}")
    except Exception:
        # Create empty DataFrame and save as managed table
        df = spark.createDataFrame([], schema)
        df.write.format("delta").mode("overwrite").saveAsTable(full_name)
        print(f"✅ Created table: {full_name}")


def create_all_tables(spark: SparkSession) -> None:
    """Create all Delta tables in Unity Catalog."""
    print(f"Creating tables in {_catalog_name()}.{_schema_name()}...")
    for table in SCHEMAS:
        create_table(spark, table)
    print("\n✅ All Delta tables created successfully.")


def read_table(spark: SparkSession, table: str) -> DataFrame:
    """Read a Unity Catalog table as a Spark DataFrame."""
    return spark.read.table(table_name(table))


def append_rows(spark: SparkSession, table: str, rows: list[dict]) -> None:
    """Append rows (list of dicts) to a Unity Catalog table."""
    schema = SCHEMAS[table]
    df = spark.createDataFrame(rows, schema)
    df.write.format("delta").mode("append").saveAsTable(table_name(table))


def upsert_row(spark: SparkSession, table: str, row: dict, key_col: str) -> None:
    """Upsert a single row into a Unity Catalog table using merge."""
    from delta.tables import DeltaTable
    schema = SCHEMAS[table]
    full_name = table_name(table)
    new_df = spark.createDataFrame([row], schema)
    
    try:
        delta_table = DeltaTable.forName(spark, full_name)
        (
            delta_table.alias("target")
            .merge(new_df.alias("source"), f"target.{key_col} = source.{key_col}")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    except Exception:
        # Table might not exist yet; write directly
        new_df.write.format("delta").mode("append").saveAsTable(full_name)


def delete_row(spark: SparkSession, table: str, key_col: str, key_val: str) -> None:
    """Delete a row from a Unity Catalog table."""
    from delta.tables import DeltaTable
    full_name = table_name(table)
    delta_table = DeltaTable.forName(spark, full_name)
    delta_table.delete(f"{key_col} = '{key_val}'")


def query(spark: SparkSession, table: str, filter_expr: str = None) -> DataFrame:
    """Query a Unity Catalog table with optional filter."""
    df = read_table(spark, table)
    if filter_expr:
        df = df.filter(filter_expr)
    return df


def read_table_pandas(spark: SparkSession, table: str):
    """Read a Unity Catalog table as a pandas DataFrame."""
    return read_table(spark, table).toPandas()