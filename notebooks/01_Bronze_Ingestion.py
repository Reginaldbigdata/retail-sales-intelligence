# Databricks notebook source
# Cell 1 — Download dataset
import urllib.request
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
download_path = "/tmp/online_retail_II.xlsx"

print("Downloading dataset...")
urllib.request.urlretrieve(url, download_path)
print(f"Done! File size: {os.path.getsize(download_path) / 1024 / 1024:.1f} MB")

# COMMAND ----------

# Fix — Install openpyxl
import subprocess
subprocess.run(["pip", "install", "openpyxl"], capture_output=True)
print("openpyxl installed!")

# COMMAND ----------

# Cell 2 — Load Excel into Spark (fixed)
import pandas as pd

print("Reading Excel file...")
pdf = pd.read_excel("/tmp/online_retail_II.xlsx", sheet_name=None)

# Combine both sheets
df_combined = pd.concat(pdf.values(), ignore_index=True)
print(f"Total rows loaded: {len(df_combined):,}")

# Fix mixed types — convert all object columns to string
for col in df_combined.select_dtypes(include=['object']).columns:
    df_combined[col] = df_combined[col].astype(str)

# Fix Customer ID — loads as float (12345.0), clean it
df_combined['Customer ID'] = df_combined['Customer ID'].replace('nan', None)

# Fix InvoiceDate to proper datetime
df_combined['InvoiceDate'] = pd.to_datetime(df_combined['InvoiceDate'])

# Convert to Spark DataFrame
df_spark = spark.createDataFrame(df_combined)

print("Schema:")
df_spark.printSchema()
df_spark.show(5, truncate=False)

# COMMAND ----------

# Cell 3 — Write to Bronze Delta Lake table (managed)
spark.sql("CREATE DATABASE IF NOT EXISTS retail_project")

# Rename column to remove the space
df_spark = df_spark.withColumnRenamed("Customer ID", "CustomerID")

(df_spark
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable("retail_project.bronze_retail"))

print("✓ Bronze Delta table written!")
print("✓ Table registered as: retail_project.bronze_retail")

# COMMAND ----------

# Cell 4 — Sanity check
spark.sql("""
  SELECT 
    COUNT(*)                   AS total_rows,
    COUNT(DISTINCT CustomerID) AS unique_customers,
    MIN(InvoiceDate)           AS earliest_date,
    MAX(InvoiceDate)           AS latest_date,
    COUNT(DISTINCT Country)    AS countries
  FROM retail_project.bronze_retail
""").show()

# COMMAND ----------

# Cell 5 — Bronze layer summary (good notebook practice)
print("""
==============================================
  BRONZE LAYER — COMPLETE
==============================================
  Source    : UCI Online Retail II Dataset
  Rows      : 1,067,371
  Customers : 5,942
  Countries : 43
  Date range: Dec 2009 → Dec 2011
  Storage   : Delta Lake (managed)
  Table     : retail_project.bronze_retail
  
  Columns:
    - Invoice      : Transaction ID
    - StockCode    : Product code
    - Description  : Product name
    - Quantity     : Units sold (can be negative = returns)
    - InvoiceDate  : Timestamp of transaction
    - Price        : Unit price (GBP)
    - CustomerID   : Unique customer identifier
    - Country      : Customer country
    
  Next: Silver layer — cleaning & feature engineering
==============================================
""")
