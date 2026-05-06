# Databricks notebook source
# Cell 1 — Load Silver table
df_silver = spark.table("retail_project.silver_retail")

print(f"✓ Silver table loaded: {df_silver.count():,} rows")
print(f"✓ Columns: {df_silver.columns}")

# COMMAND ----------

# Cell 2 — Monthly Revenue Trend
from pyspark.sql.functions import (
    col, sum as spark_sum, count, 
    countDistinct, round, concat, lit, lpad
)

df_monthly_revenue = (df_silver
    .groupBy("Year", "Month")
    .agg(
        round(spark_sum("Revenue"), 2).alias("total_revenue"),
        countDistinct("Invoice").alias("total_orders"),
        countDistinct("CustomerID").alias("unique_customers"),
        round(spark_sum("Revenue") / countDistinct("Invoice"), 2).alias("avg_order_value")
    )
    .withColumn("YearMonth", 
        concat(col("Year"), lit("-"), lpad(col("Month").cast("string"), 2, "0")))
    .orderBy("Year", "Month")
)

print("=== MONTHLY REVENUE TREND ===")
df_monthly_revenue.show(25, truncate=False)

# COMMAND ----------

# Cell 3 — Revenue by Country
df_country = (df_silver
    .groupBy("Country")
    .agg(
        round(spark_sum("Revenue"), 2).alias("total_revenue"),
        countDistinct("CustomerID").alias("unique_customers"),
        countDistinct("Invoice").alias("total_orders"),
        round(spark_sum("Revenue") / countDistinct("Invoice"), 2).alias("avg_order_value")
    )
    .orderBy(col("total_revenue").desc())
)

print("=== REVENUE BY COUNTRY (Top 15) ===")
df_country.show(15, truncate=False)

# COMMAND ----------

# Cell 4 — Top 20 Products
df_top_products = (df_silver
    .groupBy("StockCode", "Description")
    .agg(
        round(spark_sum("Revenue"), 2).alias("total_revenue"),
        spark_sum("Quantity").alias("total_units_sold"),
        countDistinct("Invoice").alias("times_ordered"),
        round(spark_sum("Revenue") / spark_sum("Quantity"), 2).alias("avg_unit_price")
    )
    .orderBy(col("total_revenue").desc())
)

print("=== TOP 20 PRODUCTS BY REVENUE ===")
df_top_products.show(20, truncate=False)

# COMMAND ----------

# Cell 5 — RFM Segmentation (fixed scoring)
from pyspark.sql.functions import (
    max as spark_max, datediff, lit,
    percent_rank, when, count
)
from pyspark.sql.window import Window

reference_date = "2011-12-10"

# Step 1 — Raw RFM metrics
df_rfm_raw = (df_silver
    .groupBy("CustomerID", "Country")
    .agg(
        datediff(lit(reference_date),
            spark_max("Date")).alias("Recency"),
        countDistinct("Invoice").alias("Frequency"),
        round(spark_sum("Revenue"), 2).alias("Monetary")
    )
)

# Step 2 — Score using percent_rank (more stable than ntile)
w_r = Window.orderBy(col("Recency"))        # low recency = bought recently = good
w_f = Window.orderBy(col("Frequency"))      # high frequency = good
w_m = Window.orderBy(col("Monetary"))       # high monetary = good

df_rfm_scored = (df_rfm_raw
    .withColumn("R_Score",
        (5 - (percent_rank().over(w_r) * 4)).cast("integer"))
    .withColumn("F_Score",
        (1 + (percent_rank().over(w_f) * 4)).cast("integer"))
    .withColumn("M_Score",
        (1 + (percent_rank().over(w_m) * 4)).cast("integer"))
)

# Step 3 — RFM total score + segments
df_rfm = (df_rfm_scored
    .withColumn("RFM_Score",
        col("R_Score") + col("F_Score") + col("M_Score"))
    .withColumn("Segment",
        when(col("RFM_Score") >= 11, "Champion")
        .when(col("RFM_Score") >= 9,  "Loyal Customer")
        .when(col("RFM_Score") >= 7,  "Potential Loyalist")
        .when(col("RFM_Score") >= 5,  "At Risk")
        .otherwise("Lost"))
)

print("=== RFM CUSTOMER SEGMENTS ===")
df_rfm.groupBy("Segment").agg(
    count("CustomerID").alias("customer_count"),
    round(spark_sum("Monetary"), 2).alias("segment_revenue"),
    round(spark_sum("Monetary") / count("CustomerID"), 2).alias("avg_revenue_per_customer")
).orderBy(col("segment_revenue").desc()).show()

print("\n=== SAMPLE — TOP 10 CHAMPIONS ===")
df_rfm.filter(col("Segment") == "Champion") \
    .orderBy(col("Monetary").desc()) \
    .select("CustomerID", "Country", "Recency", "Frequency", "Monetary", "RFM_Score") \
    .show(10)

# COMMAND ----------

# Cell 6 — Save all Gold tables
tables = {
    "retail_project.gold_monthly_revenue" : df_monthly_revenue,
    "retail_project.gold_country_revenue" : df_country,
    "retail_project.gold_top_products"    : df_top_products,
    "retail_project.gold_rfm_segments"    : df_rfm,
}

for table_name, df in tables.items():
    (df.write
       .format("delta")
       .mode("overwrite")
       .saveAsTable(table_name))
    print(f"✓ Saved: {table_name}")

print("\n✓ All Gold tables saved!")

# COMMAND ----------

# Cell 7 — Gold layer summary
print("""
==============================================
  GOLD LAYER — COMPLETE
==============================================
  Tables created:
    ✓ gold_monthly_revenue  — 25 months of revenue trends
    ✓ gold_country_revenue  — 41 countries ranked by revenue
    ✓ gold_top_products     — 4,631 products ranked by revenue
    ✓ gold_rfm_segments     — 5,891 customers segmented

  Key business insights:
    💰 Total revenue        : £17,743,429
    🏆 Top country          : United Kingdom (83% of revenue)
    🛍️  Top product          : REGENCY CAKESTAND 3 TIER (£286,486)
    👑 Champions (19%)      : Generate 69% of all revenue
    ⚠️  At Risk + Lost (42%) : £1,005,177 recoverable revenue
    📈 Peak months          : Oct, Nov every year (seasonal spike)
    🌍 Top intl market      : EIRE with £1,096 avg order value

  Next: Phase 4 — Sales forecasting with Prophet + XGBoost
==============================================
""")
