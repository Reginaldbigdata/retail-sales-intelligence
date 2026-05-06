# Databricks notebook source
# Cell 1 — Install Prophet and load data
import subprocess
subprocess.run(["pip", "install", "prophet", "xgboost"], capture_output=True)
print("✓ Libraries installed!")

# Load Silver table
from pyspark.sql.functions import col, sum as spark_sum, round
import pandas as pd

df_silver = spark.table("retail_project.silver_retail")

# Aggregate to daily revenue — Prophet needs daily time series
df_daily = (df_silver
    .groupBy("Date")
    .agg(round(spark_sum("Revenue"), 2).alias("Revenue"))
    .orderBy("Date")
)

# Convert to Pandas — Prophet and XGBoost work in Pandas
pdf_daily = df_daily.toPandas()
pdf_daily["Date"] = pd.to_datetime(pdf_daily["Date"])
pdf_daily = pdf_daily.sort_values("Date").reset_index(drop=True)

print(f"✓ Daily data loaded: {len(pdf_daily)} days")
print(f"  Date range : {pdf_daily.Date.min()} → {pdf_daily.Date.max()}")
print(f"  Avg daily revenue : £{pdf_daily.Revenue.mean():,.2f}")
print(f"  Max daily revenue : £{pdf_daily.Revenue.max():,.2f}")
pdf_daily.head()

# COMMAND ----------

# Cell 2 — Split data and visualize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 80/20 train/test split
split_idx = int(len(pdf_daily) * 0.8)
train = pdf_daily[:split_idx].copy()
test  = pdf_daily[split_idx:].copy()

print(f"Training set : {len(train)} days ({train.Date.min().date()} → {train.Date.max().date()})")
print(f"Test set     : {len(test)} days ({test.Date.min().date()} → {test.Date.max().date()})")

# Visualize the full time series
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train.Date, train.Revenue, color="#1F4E79", linewidth=1.2, label="Training data")
ax.plot(test.Date,  test.Revenue,  color="#E07B39", linewidth=1.2, label="Test data")
ax.axvline(x=train.Date.max(), color="gray", linestyle="--", linewidth=1, label="Train/Test split")
ax.set_title("Daily Revenue — Full Time Series", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Revenue (£)")
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/01_time_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Chart saved!")

# COMMAND ----------

# Cell 3 — Prophet forecasting model
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Prophet needs columns named ds (date) and y (value)
train_prophet = train.rename(columns={"Date": "ds", "Revenue": "y"})
test_prophet  = test.rename(columns={"Date": "ds", "Revenue": "y"})

# Build and train model
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",  # revenue scales seasonally
    changepoint_prior_scale=0.05        # controls trend flexibility
)
prophet_model.fit(train_prophet)
print("✓ Prophet model trained!")

# Forecast on test period + 30 days ahead
future = prophet_model.make_future_dataframe(
    periods=len(test) + 30, freq="D"
)
forecast = prophet_model.predict(future)

# Extract test period predictions
test_forecast = forecast[forecast.ds.isin(test_prophet.ds)][["ds", "yhat", "yhat_lower", "yhat_upper"]]
test_forecast = test_forecast.merge(test_prophet, on="ds")

# Metrics
mae  = mean_absolute_error(test_forecast.y, test_forecast.yhat)
rmse = np.sqrt(mean_squared_error(test_forecast.y, test_forecast.yhat))
mape = (abs((test_forecast.y - test_forecast.yhat) / test_forecast.y).mean()) * 100

print(f"\n=== PROPHET MODEL PERFORMANCE ===")
print(f"  MAE  : £{mae:,.2f}")
print(f"  RMSE : £{rmse:,.2f}")
print(f"  MAPE : {mape:.1f}%")

# COMMAND ----------

# Cell 4 — XGBoost forecasting model
import xgboost as xgb

def make_features(df):
    """Engineer time-based features for XGBoost"""
    d = df.copy()
    d["dayofweek"]  = d["Date"].dt.dayofweek
    d["month"]      = d["Date"].dt.month
    d["year"]       = d["Date"].dt.year
    d["dayofyear"]  = d["Date"].dt.dayofyear
    d["weekofyear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["quarter"]    = d["Date"].dt.quarter
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    # Lag features — previous days revenue
    d["lag_7"]  = d["Revenue"].shift(7)
    d["lag_14"] = d["Revenue"].shift(14)
    d["lag_30"] = d["Revenue"].shift(30)
    # Rolling averages
    d["rolling_7"]  = d["Revenue"].shift(1).rolling(7).mean()
    d["rolling_30"] = d["Revenue"].shift(1).rolling(30).mean()
    return d

FEATURES = ["dayofweek", "month", "year", "dayofyear",
            "weekofyear", "quarter", "is_weekend",
            "lag_7", "lag_14", "lag_30",
            "rolling_7", "rolling_30"]

# Build features on full dataset so lags are available
full_featured = make_features(pdf_daily)

# Split again using index
train_xgb = full_featured[:split_idx].dropna()
test_xgb  = full_featured[split_idx:].dropna()

X_train = train_xgb[FEATURES]
y_train = train_xgb["Revenue"]
X_test  = test_xgb[FEATURES]
y_test  = test_xgb["Revenue"]

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("✓ XGBoost model trained!")

# Predictions
xgb_preds = xgb_model.predict(X_test)
mae_xgb  = mean_absolute_error(y_test, xgb_preds)
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_preds))
mape_xgb = (abs((y_test - xgb_preds) / y_test).mean()) * 100

print(f"\n=== XGBOOST MODEL PERFORMANCE ===")
print(f"  MAE  : £{mae_xgb:,.2f}")
print(f"  RMSE : £{rmse_xgb:,.2f}")
print(f"  MAPE : {mape_xgb:.1f}%")

# COMMAND ----------

# Cell 5 — Model comparison
print("=" * 45)
print(f"{'MODEL COMPARISON':^45}")
print("=" * 45)
print(f"{'Metric':<12} {'Prophet':>15} {'XGBoost':>15}")
print("-" * 45)
print(f"{'MAE':<12} £{mae:>13,.2f} £{mae_xgb:>13,.2f}")
print(f"{'RMSE':<12} £{rmse:>13,.2f} £{rmse_xgb:>13,.2f}")
print(f"{'MAPE':<12} {mape:>14.1f}% {mape_xgb:>14.1f}%")
print("-" * 45)

winner = "Prophet" if rmse < rmse_xgb else "XGBoost"
print(f"\n🏆 Better model (lower RMSE): {winner}")
print(f"\nInterpretation:")
print(f"  Prophet MAPE {mape:.1f}% means predictions are")
print(f"  off by {mape:.1f}% on average")
print(f"  XGBoost MAPE {mape_xgb:.1f}% means predictions are")
print(f"  off by {mape_xgb:.1f}% on average")

# COMMAND ----------

# Fix — reinstall prophet at top of cell
import subprocess
subprocess.run(["pip", "install", "prophet"], capture_output=True, check=True)
print("✓ Prophet installed!")

# Then continue with the rest — paste everything below this line
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Reload data ---
df_silver = spark.table("retail_project.silver_retail")
from pyspark.sql.functions import sum as spark_sum, round as spark_round
pdf_daily = (df_silver
    .groupBy("Date")
    .agg(spark_round(spark_sum("Revenue"), 2).alias("Revenue"))
    .orderBy("Date")
    .toPandas())
pdf_daily["Date"] = pd.to_datetime(pdf_daily["Date"])
pdf_daily = pdf_daily.sort_values("Date").reset_index(drop=True)

# --- Rebuild train/test split ---
split_idx     = int(len(pdf_daily) * 0.8)
train         = pdf_daily[:split_idx].copy()
test          = pdf_daily[split_idx:].copy()
train_prophet = train.rename(columns={"Date": "ds", "Revenue": "y"})
test_prophet  = test.rename(columns={"Date": "ds", "Revenue": "y"})

# --- Rebuild Prophet ---
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05
)
prophet_model.fit(train_prophet)
print("✓ Prophet trained!")

# --- Forecast ---
future      = prophet_model.make_future_dataframe(periods=len(test) + 30, freq="D")
forecast    = prophet_model.predict(future)
future_only = forecast[forecast.ds > pdf_daily.Date.max()].copy()
print("✓ Forecast done!")

# --- Test metrics ---
test_forecast = forecast[forecast.ds.isin(test_prophet.ds)][["ds","yhat","yhat_lower","yhat_upper"]]
test_forecast = test_forecast.merge(test_prophet, on="ds")
mae  = mean_absolute_error(test_forecast.y, test_forecast.yhat)
rmse = np.sqrt(mean_squared_error(test_forecast.y, test_forecast.yhat))
mape = (abs((test_forecast.y - test_forecast.yhat) / test_forecast.y).mean()) * 100

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.plot(test.Date, test.Revenue,
         color="#1F4E79", linewidth=1.5, label="Actual Revenue")
ax1.plot(test_forecast.ds, test_forecast.yhat,
         color="#E07B39", linewidth=1.5, linestyle="--", label="Prophet Forecast")
ax1.fill_between(test_forecast.ds,
                 test_forecast.yhat_lower,
                 test_forecast.yhat_upper,
                 alpha=0.2, color="#E07B39", label="Confidence interval")
ax1.set_title("Prophet — Actual vs Predicted (Test Period)",
              fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Revenue (£)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

ax2 = axes[1]
last_60 = pdf_daily.tail(60)
ax2.plot(last_60.Date, last_60.Revenue,
         color="#1F4E79", linewidth=1.5, label="Actual Revenue (last 60 days)")
ax2.plot(future_only.ds, future_only.yhat,
         color="#27500A", linewidth=2, linestyle="--", label="30-Day Forecast")
ax2.fill_between(future_only.ds,
                 future_only.yhat_lower,
                 future_only.yhat_upper,
                 alpha=0.2, color="#27500A", label="Confidence interval")
ax2.axvline(x=pdf_daily.Date.max(), color="gray",
            linestyle=":", linewidth=1.5, label="Forecast start")
ax2.set_title("Prophet — 30-Day Revenue Forecast",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("Revenue (£)")
ax2.set_xlabel("Date")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout(pad=3)
plt.savefig("/tmp/02_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n✓ MAPE: {mape:.1f}%  |  RMSE: £{rmse:,.2f}  |  MAE: £{mae:,.2f}")
print(f"\n30-Day Forecast Summary:")
print(f"  Total forecast revenue : £{future_only.yhat.sum():,.2f}")
print(f"  Average daily forecast : £{future_only.yhat.mean():,.2f}")
print(f"  Peak forecast day      : "
      f"{future_only.loc[future_only.yhat.idxmax(), 'ds'].date()} "
      f"(£{future_only.yhat.max():,.2f})")

# COMMAND ----------

# Cell 6 (self-contained) — Rebuild forecast then visualise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Reload data ---
df_silver = spark.table("retail_project.silver_retail")
from pyspark.sql.functions import sum as spark_sum, round as spark_round
pdf_daily = (df_silver
    .groupBy("Date")
    .agg(spark_round(spark_sum("Revenue"), 2).alias("Revenue"))
    .orderBy("Date")
    .toPandas())
pdf_daily["Date"] = pd.to_datetime(pdf_daily["Date"])
pdf_daily = pdf_daily.sort_values("Date").reset_index(drop=True)

# --- Rebuild train/test split ---
split_idx  = int(len(pdf_daily) * 0.8)
train      = pdf_daily[:split_idx].copy()
test       = pdf_daily[split_idx:].copy()
train_prophet = train.rename(columns={"Date": "ds", "Revenue": "y"})
test_prophet  = test.rename(columns={"Date": "ds", "Revenue": "y"})

# --- Rebuild Prophet ---
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05
)
prophet_model.fit(train_prophet)
print("✓ Prophet rebuilt!")

# --- Rebuild forecast ---
future   = prophet_model.make_future_dataframe(periods=len(test) + 30, freq="D")
forecast = prophet_model.predict(future)
print("✓ Forecast rebuilt!")

# --- Test period predictions ---
test_forecast = forecast[forecast.ds.isin(test_prophet.ds)][["ds","yhat","yhat_lower","yhat_upper"]]
test_forecast = test_forecast.merge(test_prophet, on="ds")

# --- Metrics ---
mae  = mean_absolute_error(test_forecast.y, test_forecast.yhat)
rmse = np.sqrt(mean_squared_error(test_forecast.y, test_forecast.yhat))
mape = (abs((test_forecast.y - test_forecast.yhat) / test_forecast.y).mean()) * 100

# --- Future 30 days ---
future_only = forecast[forecast.ds > pdf_daily.Date.max()].copy()

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.plot(test.Date, test.Revenue,
         color="#1F4E79", linewidth=1.5, label="Actual Revenue")
ax1.plot(test_forecast.ds, test_forecast.yhat,
         color="#E07B39", linewidth=1.5, linestyle="--", label="Prophet Forecast")
ax1.fill_between(test_forecast.ds,
                 test_forecast.yhat_lower,
                 test_forecast.yhat_upper,
                 alpha=0.2, color="#E07B39", label="Confidence interval")
ax1.set_title("Prophet — Actual vs Predicted (Test Period)",
              fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Revenue (£)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

ax2 = axes[1]
last_60 = pdf_daily.tail(60)
ax2.plot(last_60.Date, last_60.Revenue,
         color="#1F4E79", linewidth=1.5, label="Actual Revenue (last 60 days)")
ax2.plot(future_only.ds, future_only.yhat,
         color="#27500A", linewidth=2, linestyle="--", label="30-Day Forecast")
ax2.fill_between(future_only.ds,
                 future_only.yhat_lower,
                 future_only.yhat_upper,
                 alpha=0.2, color="#27500A", label="Confidence interval")
ax2.axvline(x=pdf_daily.Date.max(), color="gray",
            linestyle=":", linewidth=1.5, label="Forecast start")
ax2.set_title("Prophet — 30-Day Revenue Forecast",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("Revenue (£)")
ax2.set_xlabel("Date")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout(pad=3)
plt.savefig("/tmp/02_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n✓ MAPE: {mape:.1f}%  |  RMSE: £{rmse:,.2f}  |  MAE: £{mae:,.2f}")
print(f"\n30-Day Forecast Summary:")
print(f"  Total forecast revenue : £{future_only.yhat.sum():,.2f}")
print(f"  Average daily forecast : £{future_only.yhat.mean():,.2f}")
print(f"  Peak forecast day      : {future_only.loc[future_only.yhat.idxmax(), 'ds'].date()} "
      f"(£{future_only.yhat.max():,.2f})")

# COMMAND ----------

# Cell 7 — Save forecast results to Delta
forecast_to_save = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_to_save.columns = ["Date", "Forecast", "Forecast_Lower", "Forecast_Upper"]
forecast_to_save["Model"] = "Prophet"
forecast_to_save["Date"] = pd.to_datetime(forecast_to_save["Date"])

df_forecast_spark = spark.createDataFrame(forecast_to_save)

(df_forecast_spark
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("retail_project.gold_forecast"))

print("✓ Forecast saved to: retail_project.gold_forecast")
print(f"\n=== PHASE 4 COMPLETE ===")
print(f"  ✓ Prophet model trained & evaluated")
print(f"  ✓ XGBoost model trained & evaluated")
print(f"  ✓ Prophet won with MAPE: 33.2%")
print(f"  ✓ 30-day forecast generated")
print(f"  ✓ Results saved to Delta Lake")
print(f"\n  Next: Phase 5 — Dashboard & portfolio delivery")
