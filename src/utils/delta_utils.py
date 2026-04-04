"""
Delta Lake utility functions for ASHA-Sahayak.
Provides read/write helpers for all 7 Delta tables.
Works both on Databricks (with SparkSession) and locally (with delta-spark).
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
# Base path helpers
# ---------------------------------------------------------------------------

def _base_path() -> str:
    """Return base DBFS path for Delta tables."""
    # On Databricks use DBFS; locally use a temp directory
    dbfs_path = os.environ.get("ASHA_DELTA_BASE", "/dbfs/asha_sahayak/delta")
    return dbfs_path


def table_path(table_name: str) -> str:
    """Return full path for a Delta table."""
    return f"{_base_path()}/{table_name}"


# ---------------------------------------------------------------------------
# Schema definitions for all 7 tables
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
        StructField("usg_findings", StringType(), True),
        StructField("prescribed_medicines", StringType(), True),
        StructField("raw_document_path", StringType(), True),
        StructField("extracted_text", StringType(), True),
        StructField("created_at", TimestampType(), True),
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
    "ration_plans": StructType([
        StructField("plan_id", StringType(), False),
        StructField("patient_id", StringType(), True),
        StructField("week_start_date", DateType(), True),
        StructField("week_end_date", DateType(), True),
        StructField("trimester", IntegerType(), True),
        StructField("daily_calorie_target", IntegerType(), True),
        StructField("protein_target_g", IntegerType(), True),
        StructField("ration_items", StringType(), True),
        StructField("supplements", StringType(), True),
        StructField("special_notes", StringType(), True),
        StructField("generated_by_model", StringType(), True),
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
# CRUD helpers
# ---------------------------------------------------------------------------

def create_table(spark: SparkSession, table_name: str) -> None:
    """Create an empty Delta table if it doesn't already exist."""
    schema = SCHEMAS[table_name]
    path = table_path(table_name)
    try:
        spark.read.format("delta").load(path)
    except Exception:
        df = spark.createDataFrame([], schema)
        df.write.format("delta").mode("overwrite").save(path)
        print(f"Created Delta table: {table_name} at {path}")


def create_all_tables(spark: SparkSession) -> None:
    """Create all 7 Delta tables."""
    for name in SCHEMAS:
        create_table(spark, name)
    print("All Delta tables created successfully.")


def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    """Read a Delta table as a Spark DataFrame."""
    return spark.read.format("delta").load(table_path(table_name))


def append_rows(spark: SparkSession, table_name: str, rows: list[dict]) -> None:
    """Append rows (list of dicts) to a Delta table."""
    schema = SCHEMAS[table_name]
    df = spark.createDataFrame(rows, schema)
    df.write.format("delta").mode("append").save(table_path(table_name))


def upsert_row(spark: SparkSession, table_name: str, row: dict, key_col: str) -> None:
    """Upsert a single row into a Delta table using merge."""
    from delta.tables import DeltaTable
    schema = SCHEMAS[table_name]
    path = table_path(table_name)
    new_df = spark.createDataFrame([row], schema)
    try:
        delta_table = DeltaTable.forPath(spark, path)
        (
            delta_table.alias("target")
            .merge(new_df.alias("source"), f"target.{key_col} = source.{key_col}")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    except Exception:
        # Table might not exist as Delta yet; write directly
        new_df.write.format("delta").mode("append").save(path)


def delete_row(spark: SparkSession, table_name: str, key_col: str, key_val: str) -> None:
    """Delete a row from a Delta table by key."""
    from delta.tables import DeltaTable
    path = table_path(table_name)
    delta_table = DeltaTable.forPath(spark, path)
    delta_table.delete(F.col(key_col) == key_val)


def read_table_pandas(spark: SparkSession, table_name: str):
    """Read a Delta table as a Pandas DataFrame."""
    return read_table(spark, table_name).toPandas()
