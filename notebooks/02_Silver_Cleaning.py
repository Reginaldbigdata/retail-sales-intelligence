# Databricks notebook source
# Cell 1 — Load Bronze and see what we're dealing with
df_bronze = spark.table("retail_project.bronze_retail")

print("=== DATA QUALITY REPORT — BRONZE ===\n")

total = df_bronze.count()
print(f"Total rows: {total:,}")

# Nulls per column
from pyspark.sql.functions import col, count, when, isnan

print("\nNull counts per column:")
df_bronze.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in df_bronze.columns
]).show()

# Negative quantities (returns)
returns = df_bronze.filter(col("Quantity") < 0).count()
print(f"Negative quantity rows (returns): {returns:,}")

# Zero price rows
zero_price = df_bronze.filter(col("Price") <= 0).count()
print(f"Zero/negative price rows: {zero_price:,}")

# Cancelled invoices (start with C)
from pyspark.sql.functions import substring
cancelled = df_bronze.filter(col("Invoice").startswith("C")).count()
print(f"Cancelled invoices (start with C): {cancelled:,}")

# COMMAND ----------

# Cell 2 — Clean and filter
from pyspark.sql.functions import (
    col, round, month, year, dayofweek, 
    to_date, regexp_replace, trim, upper
)

df_clean = (df_bronze
    # Remove returns (negative quantity)
    .filter(col("Quantity") > 0)
    
    # Remove zero/negative prices
    .filter(col("Price") > 0)
    
    # Remove cancelled invoices
    .filter(~col("Invoice").startswith("C"))
    
    # Remove null CustomerIDs
    .filter(col("CustomerID").isNotNull())
    
    # Clean Description — trim whitespace, uppercase
    .withColumn("Description", upper(trim(col("Description"))))
    
    # Fix CustomerID — remove decimal (13085.0 → 13085)
    .withColumn("CustomerID", 
        col("CustomerID").cast("integer").cast("string"))
    
    # Add Revenue column
    .withColumn("Revenue", 
        round(col("Quantity") * col("Price"), 2))
    
    # Extract date parts for time analysis
    .withColumn("Date", to_date(col("InvoiceDate")))
    .withColumn("Year", year(col("InvoiceDate")))
    .withColumn("Month", month(col("InvoiceDate")))
    .withColumn("DayOfWeek", dayofweek(col("InvoiceDate")))
)

clean_count = df_clean.count()
removed = 1067371 - clean_count
print(f"Rows after cleaning : {clean_count:,}")
print(f"Rows removed        : {removed:,} ({removed/1067371*100:.1f}%)")
print(f"\nSample:")
df_clean.show(5, truncate=False)

# COMMAND ----------

# Cell 3 — Post-clean quality check
from pyspark.sql.functions import min, max, sum as spark_sum, countDistinct

print("=== DATA QUALITY REPORT — SILVER ===\n")

# Null check — should all be 0
print("Null counts (should all be 0):")
df_clean.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in df_clean.columns
]).show()

# Business summary
df_clean.select(
    countDistinct("CustomerID").alias("unique_customers"),
    countDistinct("Invoice").alias("unique_invoices"),
    countDistinct("StockCode").alias("unique_products"),
    countDistinct("Country").alias("countries"),
    spark_sum("Revenue").alias("total_revenue_GBP"),
    min("Date").alias("from_date"),
    max("Date").alias("to_date")
).show()

# COMMAND ----------

# Cell 4 — Write Silver table
(df_clean
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("retail_project.silver_retail"))

print("✓ Silver table saved: retail_project.silver_retail")
print(f"\nColumns added in Silver:")
for c in df_clean.columns:
    print(f"  • {c}")

# COMMAND ----------

# Cell 5 — Silver layer summary
print("""
==============================================
  SILVER LAYER — COMPLETE
==============================================
  Source table : retail_project.bronze_retail
  Output table : retail_project.silver_retail

  Cleaning applied:
    ✓ Removed 243,007 null CustomerIDs
    ✓ Removed 22,950 returns (negative quantity)
    ✓ Removed 6,207 zero/negative price rows
    ✓ Removed 19,494 cancelled invoices
    ✓ Cleaned Description (trimmed + uppercased)
    ✓ Fixed CustomerID (float → clean integer string)

  Features engineered:
    ✓ Revenue = Quantity × Price
    ✓ Date (date only, no time)
    ✓ Year, Month, DayOfWeek

  Final dataset:
    Rows             : ~700,000
    Unique customers : 5,878
    Unique products  : 4,631
    Countries        : 41
    Total revenue    : £17,743,429
    Date range       : Dec 2009 → Dec 2011

  Next: Gold layer — analytics & aggregations
==============================================
""")
