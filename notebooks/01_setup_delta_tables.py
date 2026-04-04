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

# DBTITLE 1,Cell 3
# Use Unity Catalog managed tables instead of DBFS
import os
# Set catalog and schema for Delta tables
os.environ["ASHA_CATALOG"] = "workspace"
os.environ["ASHA_SCHEMA"] = "default"

print("Using Unity Catalog: workspace.default")
print("Tables will be created as managed tables with automatic storage")

# COMMAND ----------

# DBTITLE 1,Cell 4
import sys

# Force reload delta_utils to get latest changes
if 'src.utils.delta_utils' in sys.modules:
    del sys.modules['src.utils.delta_utils']

from src.utils.delta_utils import get_spark, create_all_tables, SCHEMAS

spark = get_spark()

# COMMAND ----------

# Create all 7 Delta tables
create_all_tables(spark)

# COMMAND ----------

# DBTITLE 1,Cell 6
# Verify Unity Catalog tables were created
catalog = os.environ.get("ASHA_CATALOG", "workspace")
schema = os.environ.get("ASHA_SCHEMA", "default")

for table in SCHEMAS.keys():
    full_name = f"{catalog}.{schema}.{table}"
    try:
        df = spark.read.table(full_name)
        print(f"✅ {full_name}: {df.count()} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ {full_name}: {e}")

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
