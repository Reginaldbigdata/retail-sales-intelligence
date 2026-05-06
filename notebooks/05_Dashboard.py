# Databricks notebook source
# Cell 1 — Load all Gold tables
df_monthly  = spark.table("retail_project.gold_monthly_revenue").toPandas()
df_country  = spark.table("retail_project.gold_country_revenue").toPandas()
df_products = spark.table("retail_project.gold_top_products").toPandas()
df_rfm      = spark.table("retail_project.gold_rfm_segments")
df_forecast = spark.table("retail_project.gold_forecast").toPandas()

# RFM summary
df_rfm_summary = (df_rfm
    .groupBy("Segment")
    .agg(
        {"CustomerID": "count", "Monetary": "sum"}
    )
    .toPandas()
    .rename(columns={"count(CustomerID)": "customers", "sum(Monetary)": "revenue"})
    .sort_values("revenue", ascending=False)
)

print("✓ All Gold tables loaded!")
print(f"  Monthly  : {len(df_monthly)} rows")
print(f"  Country  : {len(df_country)} rows")
print(f"  Products : {len(df_products)} rows")
print(f"  RFM      : {len(df_rfm_summary)} segments")
print(f"  Forecast : {len(df_forecast)} rows")

# COMMAND ----------

# Cell 2 — Dashboard Page 1: Revenue & Geography
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Prep data
df_monthly["YearMonth"] = pd.to_datetime(df_monthly["YearMonth"])
df_monthly = df_monthly.sort_values("YearMonth")
top10_countries = df_country[df_country.Country != "United Kingdom"].head(10)
df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
future_forecast = df_forecast[df_forecast.Date > pd.Timestamp("2011-12-09")]

# Colours
BLUE   = "#1F4E79"
ORANGE = "#E07B39"
GREEN  = "#27500A"
GOLD   = "#C08B2A"
GRAY   = "#666666"

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- KPI Banner ---
kpi_ax = fig.add_axes([0.02, 0.91, 0.96, 0.07])
kpi_ax.set_facecolor(BLUE)
kpi_ax.axis("off")
kpis = [
    ("Total Revenue", "£17,743,429"),
    ("Total Orders",  "36,969"),
    ("Customers",     "5,878"),
    ("Countries",     "41"),
    ("Top Product",   "Regency Cakestand"),
    ("Forecast (30d)","£231,008"),
]
for i, (label, value) in enumerate(kpis):
    x = 0.08 + i * 0.155
    kpi_ax.text(x, 0.72, value, transform=kpi_ax.transAxes,
                fontsize=13, fontweight="bold", color="white",
                ha="center", va="center")
    kpi_ax.text(x, 0.22, label, transform=kpi_ax.transAxes,
                fontsize=8, color="#AACCEE", ha="center", va="center")

# Title
fig.text(0.5, 0.99, "Retail Sales Intelligence Dashboard",
         ha="center", va="top", fontsize=18,
         fontweight="bold", color=BLUE)
fig.text(0.5, 0.965, "Online Retail II Dataset  |  Dec 2009 – Dec 2011  |  Built with Databricks + Delta Lake",
         ha="center", va="top", fontsize=9, color=GRAY)

# --- Chart 1: Monthly Revenue Trend ---
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("white")
bars = ax1.bar(df_monthly.YearMonth, df_monthly.total_revenue,
               color=BLUE, alpha=0.7, width=20, label="Monthly Revenue")
# Highlight peak months
peak_mask = df_monthly.total_revenue > 1_000_000
ax1.bar(df_monthly.YearMonth[peak_mask],
        df_monthly.total_revenue[peak_mask],
        color=ORANGE, alpha=0.9, width=20, label="Peak months (>£1M)")
# Trend line
z = np.polyfit(range(len(df_monthly)), df_monthly.total_revenue, 1)
p = np.poly1d(z)
ax1.plot(df_monthly.YearMonth,
         p(range(len(df_monthly))),
         color=GREEN, linewidth=2, linestyle="--", label="Trend")
ax1.set_title("Monthly Revenue Trend with Growth Trajectory",
              fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Revenue (£)")
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)

# --- Chart 2: Top 10 International Markets ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("white")
colors_bar = [ORANGE if i == 0 else BLUE for i in range(len(top10_countries))]
bars2 = ax2.barh(top10_countries.Country[::-1],
                 top10_countries.total_revenue[::-1],
                 color=colors_bar[::-1])
ax2.set_title("Top 10 International Markets\n(excl. United Kingdom)",
              fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Revenue (£)")
ax2.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="x", alpha=0.3)

# --- Chart 3: 30-Day Forecast ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("white")
ax3.plot(future_forecast.Date, future_forecast.Forecast,
         color=GREEN, linewidth=2, label="Forecast")
ax3.fill_between(future_forecast.Date,
                 future_forecast.Forecast_Lower,
                 future_forecast.Forecast_Upper,
                 alpha=0.2, color=GREEN, label="Confidence interval")
ax3.set_title("30-Day Revenue Forecast\n(Prophet Model · MAPE 33.2%)",
              fontsize=11, fontweight="bold", pad=10)
ax3.set_ylabel("Revenue (£)")
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

plt.savefig("/tmp/dashboard_page1.png", dpi=150,
            bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("✓ Dashboard Page 1 saved!")

# COMMAND ----------

# Cell 3 — Dashboard Page 2: Products & RFM (self-contained)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pyspark.sql.functions import sum as spark_sum

# Colours (redefine since cluster restarted)
BLUE   = "#1F4E79"
ORANGE = "#E07B39"
GREEN  = "#27500A"
GOLD   = "#C08B2A"
GRAY   = "#666666"

# Reload RFM summary and products in case they were lost too
df_rfm = spark.table("retail_project.gold_rfm_segments")
df_rfm_summary = (df_rfm
    .groupBy("Segment")
    .agg({"CustomerID": "count", "Monetary": "sum"})
    .toPandas()
    .rename(columns={"count(CustomerID)": "customers", "sum(Monetary)": "revenue"})
    .sort_values("revenue", ascending=False)
)
df_products = spark.table("retail_project.gold_top_products").toPandas()

# --- Now the actual chart code ---
fig2 = plt.figure(figsize=(18, 11))
fig2.patch.set_facecolor("#F8F9FA")
gs2  = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.4)

fig2.text(0.5, 0.97, "Customer Intelligence & Product Performance",
          ha="center", fontsize=16, fontweight="bold", color=BLUE)
fig2.text(0.5, 0.935, "RFM Segmentation + Top Products  |  Retail Sales Intelligence Project",
          ha="center", fontsize=9, color=GRAY)

# --- Chart 4: RFM Donut ---
ax4 = fig2.add_subplot(gs2[0, 0])
ax4.set_facecolor("white")

segment_colors = {
    "Champion"          : "#1F4E79",
    "Loyal Customer"    : "#2E75B6",
    "Potential Loyalist": "#C08B2A",
    "At Risk"           : "#E07B39",
    "Lost"              : "#C00000",
}
seg_order    = ["Champion","Loyal Customer","Potential Loyalist","At Risk","Lost"]
df_rfm_plot  = df_rfm_summary.set_index("Segment").reindex(seg_order).reset_index()
colors_rfm   = [segment_colors[s] for s in df_rfm_plot.Segment]

wedges, texts, autotexts = ax4.pie(
    df_rfm_plot.revenue,
    labels=None,
    colors=colors_rfm,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(width=0.55)
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("white")
    at.set_fontweight("bold")

legend_labels = [
    f"{row.Segment}  ({int(row.customers):,} customers · £{row.revenue:,.0f})"
    for _, row in df_rfm_plot.iterrows()
]
ax4.legend(wedges, legend_labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=False)
ax4.set_title("Customer Segments by Revenue\n(RFM Model)",
              fontsize=12, fontweight="bold", pad=15)
ax4.text(0, 0, "£17.7M\nTotal", ha="center", va="center",
         fontsize=11, fontweight="bold", color=BLUE)

# --- Chart 5: Top 15 Products ---
ax5 = fig2.add_subplot(gs2[0, 1])
ax5.set_facecolor("white")
top15 = df_products.head(15).copy()
top15["short_name"] = top15.Description.str[:28]
bar_colors = [ORANGE if i == 0 else BLUE for i in range(len(top15))]
ax5.barh(top15.short_name[::-1],
         top15.total_revenue[::-1],
         color=bar_colors[::-1])
ax5.set_title("Top 15 Products by Revenue",
              fontsize=12, fontweight="bold", pad=10)
ax5.set_xlabel("Revenue (£)")
ax5.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))
ax5.spines[["top","right"]].set_visible(False)
ax5.grid(axis="x", alpha=0.3)
ax5.tick_params(axis="y", labelsize=8)

plt.savefig("/tmp/dashboard_page2.png", dpi=150,
            bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("✓ Dashboard Page 2 saved!")

# COMMAND ----------

# Cell 4 — Project complete!
print("""
╔══════════════════════════════════════════════════════╗
║     RETAIL SALES INTELLIGENCE — PROJECT COMPLETE     ║
╠══════════════════════════════════════════════════════╣
║  ARCHITECTURE                                        ║
║  ✓ Bronze  → Raw Delta table  (1,067,371 rows)       ║
║  ✓ Silver  → Clean Delta table  (805,549 rows)       ║
║  ✓ Gold    → 5 analytics tables                      ║
║                                                      ║
║  ANALYTICS                                           ║
║  ✓ Monthly revenue trend  (25 months)                ║
║  ✓ Country breakdown  (41 countries)                 ║
║  ✓ Top products  (4,631 SKUs ranked)                 ║
║  ✓ RFM segmentation  (5,878 customers)               ║
║                                                      ║
║  FORECASTING                                         ║
║  ✓ Prophet  — MAPE 33.2%  (winner)                   ║
║  ✓ XGBoost  — MAPE 36.4%                             ║
║  ✓ 30-day forecast  → £231,008                       ║
║                                                      ║
║  TECH STACK                                          ║
║  ✓ Databricks Community Edition                      ║
║  ✓ Delta Lake  (Bronze/Silver/Gold)                  ║
║  ✓ Apache Spark + Spark SQL                          ║
║  ✓ Python · Prophet · XGBoost · Matplotlib           ║
║  ✓ Medallion Architecture                            ║
╚══════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# Full self-contained dashboard generator + download
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from IPython.display import Image, display
import os

# Colours
BLUE   = "#1F4E79"
ORANGE = "#E07B39"
GREEN  = "#27500A"
GOLD   = "#C08B2A"
GRAY   = "#666666"

# --- Reload all Gold tables ---
df_monthly  = spark.table("retail_project.gold_monthly_revenue").toPandas()
df_country  = spark.table("retail_project.gold_country_revenue").toPandas()
df_products = spark.table("retail_project.gold_top_products").toPandas()
df_forecast = spark.table("retail_project.gold_forecast").toPandas()
df_rfm      = spark.table("retail_project.gold_rfm_segments")
df_rfm_summary = (df_rfm
    .groupBy("Segment")
    .agg({"CustomerID": "count", "Monetary": "sum"})
    .toPandas()
    .rename(columns={"count(CustomerID)": "customers", "sum(Monetary)": "revenue"})
    .sort_values("revenue", ascending=False))

# Prep
df_monthly["YearMonth"] = pd.to_datetime(df_monthly["YearMonth"])
df_monthly = df_monthly.sort_values("YearMonth")
top10_countries = df_country[df_country.Country != "United Kingdom"].head(10)
df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
future_forecast = df_forecast[df_forecast.Date > pd.Timestamp("2011-12-09")]

print("✓ All tables loaded!")

# ================================================
# PAGE 1 — Revenue & Geography
# ================================================
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# KPI Banner
kpi_ax = fig.add_axes([0.02, 0.91, 0.96, 0.07])
kpi_ax.set_facecolor(BLUE)
kpi_ax.axis("off")
kpis = [
    ("Total Revenue",  "£17,743,429"),
    ("Total Orders",   "36,969"),
    ("Customers",      "5,878"),
    ("Countries",      "41"),
    ("Top Product",    "Regency Cakestand"),
    ("Forecast (30d)", "£231,008"),
]
for i, (label, value) in enumerate(kpis):
    x = 0.08 + i * 0.155
    kpi_ax.text(x, 0.72, value, transform=kpi_ax.transAxes,
                fontsize=13, fontweight="bold", color="white",
                ha="center", va="center")
    kpi_ax.text(x, 0.22, label, transform=kpi_ax.transAxes,
                fontsize=8, color="#AACCEE", ha="center", va="center")

fig.text(0.5, 0.99, "Retail Sales Intelligence Dashboard",
         ha="center", va="top", fontsize=18, fontweight="bold", color=BLUE)
fig.text(0.5, 0.965,
         "Online Retail II Dataset  |  Dec 2009 – Dec 2011  |  Built with Databricks + Delta Lake",
         ha="center", va="top", fontsize=9, color=GRAY)

# Monthly Revenue
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("white")
ax1.bar(df_monthly.YearMonth, df_monthly.total_revenue,
        color=BLUE, alpha=0.7, width=20, label="Monthly Revenue")
peak_mask = df_monthly.total_revenue > 1_000_000
ax1.bar(df_monthly.YearMonth[peak_mask],
        df_monthly.total_revenue[peak_mask],
        color=ORANGE, alpha=0.9, width=20, label="Peak months (>£1M)")
z = np.polyfit(range(len(df_monthly)), df_monthly.total_revenue, 1)
p = np.poly1d(z)
ax1.plot(df_monthly.YearMonth, p(range(len(df_monthly))),
         color=GREEN, linewidth=2, linestyle="--", label="Trend")
ax1.set_title("Monthly Revenue Trend with Growth Trajectory",
              fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Revenue (£)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)

# Top 10 International Markets
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("white")
colors_bar = [ORANGE if i == 0 else BLUE for i in range(len(top10_countries))]
ax2.barh(top10_countries.Country[::-1],
         top10_countries.total_revenue[::-1],
         color=colors_bar[::-1])
ax2.set_title("Top 10 International Markets\n(excl. United Kingdom)",
              fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Revenue (£)")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="x", alpha=0.3)

# 30-Day Forecast
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("white")
ax3.plot(future_forecast.Date, future_forecast.Forecast,
         color=GREEN, linewidth=2, label="Forecast")
ax3.fill_between(future_forecast.Date,
                 future_forecast.Forecast_Lower,
                 future_forecast.Forecast_Upper,
                 alpha=0.2, color=GREEN, label="Confidence interval")
ax3.set_title("30-Day Revenue Forecast\n(Prophet Model · MAPE 33.2%)",
              fontsize=11, fontweight="bold", pad=10)
ax3.set_ylabel("Revenue (£)")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

plt.savefig("/tmp/dashboard_page1.png", dpi=150,
            bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("✓ Page 1 rendered!")

# ================================================
# PAGE 2 — Products & RFM
# ================================================
fig2 = plt.figure(figsize=(18, 11))
fig2.patch.set_facecolor("#F8F9FA")
gs2  = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.4)

fig2.text(0.5, 0.97, "Customer Intelligence & Product Performance",
          ha="center", fontsize=16, fontweight="bold", color=BLUE)
fig2.text(0.5, 0.935,
          "RFM Segmentation + Top Products  |  Retail Sales Intelligence Project",
          ha="center", fontsize=9, color=GRAY)

# RFM Donut
ax4 = fig2.add_subplot(gs2[0, 0])
ax4.set_facecolor("white")
segment_colors = {
    "Champion"          : "#1F4E79",
    "Loyal Customer"    : "#2E75B6",
    "Potential Loyalist": "#C08B2A",
    "At Risk"           : "#E07B39",
    "Lost"              : "#C00000",
}
seg_order   = ["Champion","Loyal Customer","Potential Loyalist","At Risk","Lost"]
df_rfm_plot = df_rfm_summary.set_index("Segment").reindex(seg_order).reset_index()
colors_rfm  = [segment_colors[s] for s in df_rfm_plot.Segment]
wedges, texts, autotexts = ax4.pie(
    df_rfm_plot.revenue, labels=None, colors=colors_rfm,
    autopct="%1.1f%%", startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.55))
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("white")
    at.set_fontweight("bold")
legend_labels = [
    f"{row.Segment}  ({int(row.customers):,} customers · £{row.revenue:,.0f})"
    for _, row in df_rfm_plot.iterrows()
]
ax4.legend(wedges, legend_labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=False)
ax4.set_title("Customer Segments by Revenue\n(RFM Model)",
              fontsize=12, fontweight="bold", pad=15)
ax4.text(0, 0, "£17.7M\nTotal", ha="center", va="center",
         fontsize=11, fontweight="bold", color=BLUE)

# Top 15 Products
ax5 = fig2.add_subplot(gs2[0, 1])
ax5.set_facecolor("white")
top15 = df_products.head(15).copy()
top15["short_name"] = top15.Description.str[:28]
bar_colors = [ORANGE if i == 0 else BLUE for i in range(len(top15))]
ax5.barh(top15.short_name[::-1], top15.total_revenue[::-1],
         color=bar_colors[::-1])
ax5.set_title("Top 15 Products by Revenue",
              fontsize=12, fontweight="bold", pad=10)
ax5.set_xlabel("Revenue (£)")
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))
ax5.spines[["top","right"]].set_visible(False)
ax5.grid(axis="x", alpha=0.3)
ax5.tick_params(axis="y", labelsize=8)

plt.savefig("/tmp/dashboard_page2.png", dpi=150,
            bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("✓ Page 2 rendered!")

# ================================================
# DISPLAY FOR DOWNLOAD
# ================================================
print("\n📥 Click images below to right-click → Save:")
print("=== DASHBOARD PAGE 1 ===")
display(Image(filename="/tmp/dashboard_page1.png", width=1200))
print("\n=== DASHBOARD PAGE 2 ===")
display(Image(filename="/tmp/dashboard_page2.png", width=1200))

print("\n✓ Both dashboards ready — right-click each image → Save image as")
