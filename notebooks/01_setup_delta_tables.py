# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Setup Delta Lake Tables
# MAGIC Creates all 7 Delta Lake tables for ASHA-Sahayak.

# COMMAND ----------

import sys, os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(".")))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# On Databricks, the notebooks folder is the CWD; adjust to find src/
notebook_dir = os.getcwd()
# Try to find the project root that contains src/
for candidate in [notebook_dir, os.path.dirname(notebook_dir), "/Workspace/Repos/asha-sahayak", "/Workspace/Users"]:
    if os.path.isdir(os.path.join(candidate, "src")):
        project_root = candidate
        break

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# COMMAND ----------

# Set base path for Delta tables on DBFS
import os
os.environ["ASHA_DELTA_BASE"] = "/dbfs/asha_sahayak/delta"

# Create the base directory on DBFS
dbutils.fs.mkdirs("dbfs:/asha_sahayak/delta")
print("Base directory created: dbfs:/asha_sahayak/delta")

# COMMAND ----------

from src.utils.delta_utils import get_spark, create_all_tables, SCHEMAS

spark = get_spark()

# COMMAND ----------

# Create all 7 Delta tables
create_all_tables(spark)

# COMMAND ----------

# Verify tables were created
for table_name in SCHEMAS.keys():
    path = f"/dbfs/asha_sahayak/delta/{table_name}"
    try:
        df = spark.read.format("delta").load(f"dbfs:/asha_sahayak/delta/{table_name}")
        print(f"✅ {table_name}: {df.count()} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ {table_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Schemas Summary

# COMMAND ----------

for name, schema in SCHEMAS.items():
    print(f"\n{'='*60}")
    print(f"Table: {name}")
    print(f"{'='*60}")
    for field in schema.fields:
        print(f"  {field.name:30s} {str(field.dataType):20s} nullable={field.nullable}")

print("\n✅ All 7 Delta Lake tables are ready!")
