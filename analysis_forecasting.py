"""
=========================================================
ANALYSIS_FORECASTING_V3.PY

PHASES 2–5
Data Intelligence Layer

1. Dataset Loading
2. Validation
3. Climate Intelligence
4. AQI Intelligence
5. Forest Intelligence
=========================================================
"""

from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import true
import json

# =========================================================
# FOLDERS
# =========================================================

os.makedirs("output/analytics", exist_ok=True)
os.makedirs("output/charts", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)
os.makedirs("output/dashboard", exist_ok=True)

print("=" * 60)
print("PHASES 2–5 DATA INTELLIGENCE LAYER")
print("=" * 60)

# =========================================================
# LOAD DATASETS
# =========================================================

temp_df = pd.read_csv(
    "data/Temperature_Data.csv",
    skiprows=1
)
print("\n Temperature Columns:")
print(temp_df.columns.tolist())

print("\n Temperature Data sample:")
print(temp_df.head())


aqi_df = pd.read_csv(
    "data/Air_Quality_Data.csv"
)

forest_df = pd.read_csv(
    "output/forest_interpolated.csv"
)

print("\nDatasets Loaded")

print(
    "Temperature Shape :",
    temp_df.shape
)

print(
    "AQI Shape :",
    aqi_df.shape
)

print(
    "Forest Shape :",
    forest_df.shape
)

# =========================================================
# TEMPERATURE DATA PREPARATION
# =========================================================

print("\nProcessing Temperature Dataset")

temp_df.columns = (
    temp_df.columns.str.strip()
)
climate_df = temp_df[
    ["Year", "J-D"]
].copy()

climate_df.columns = [
    "year",
    "temperature_anomaly"
]

climate_df["year"] = pd.to_numeric(
    climate_df["year"],
    errors="coerce"
)

climate_df["temperature_anomaly"] = pd.to_numeric(
    climate_df["temperature_anomaly"],
    errors="coerce"
)

climate_df = climate_df.dropna()

climate_df["year"] = climate_df["year"].astype(int)

print("\nClimate Dataset:")

print(climate_df.head())

print(climate_df.shape)

# =========================================================
# CLIMATE FEATURES
# =========================================================

climate_df["temp_ma_3"] = (
    climate_df["temperature_anomaly"]
    .rolling(
        3,
        min_periods=1
    )
    .mean()
)

climate_df["temp_ma_5"] = (
    climate_df["temperature_anomaly"]
    .rolling(
        5,
        min_periods=1
    )
    .mean()
)

climate_df["temp_ma_10"] = (
    climate_df["temperature_anomaly"]
    .rolling(
        10,
        min_periods=1
    )
    .mean()
)

climate_df["temp_growth_rate"] = (
    climate_df["temperature_anomaly"]
    .pct_change()
    * 100
)

climate_df["warming_acceleration"] = (
    climate_df["temp_growth_rate"]
    .diff()
)

# climate severity index

climate_df["climate_severity_index"] = (

    (
        climate_df["temperature_anomaly"]
        -
        climate_df["temperature_anomaly"].min()
    )

    /

    (

        climate_df["temperature_anomaly"].max()
        -
        climate_df["temperature_anomaly"].min()

    )

) * 100

# =========================================================
# SAVE CLIMATE FEATURES
# =========================================================

climate_df.to_csv(
    "output/analytics/climate_features.csv",
    index=False
)

print(
    "Climate Features Created"
)

# =========================================================
# AQI DATA PREPARATION
# =========================================================

print("\nProcessing AQI Dataset")

aqi_df.columns = (
    aqi_df.columns
    .str.strip()
    .str.lower()
)

aqi_df["date"] = pd.to_datetime(
    aqi_df["date"],
    errors="coerce"
)

aqi_df["year"] = (
    aqi_df["date"]
    .dt.year
)

aqi_df["month"] = (
    aqi_df["date"]
    .dt.month
)

# =========================================================
# YEARLY AQI AGGREGATION
# =========================================================

pollution_df = (

    aqi_df

    .groupby("year")

    .agg({

        "aqi": "mean",

        "pm2.5": "mean",

        "pm10": "mean",

        "no2": "mean",

        "so2": "mean",

        "o3": "mean"

    })

    .reset_index()

)

# =========================================================
# POLLUTION EXPOSURE INDEX
# =========================================================

pollution_df["pollution_exposure_index"] = (

      0.50 * pollution_df["pm2.5"]

    + 0.30 * pollution_df["pm10"]

    + 0.20 * pollution_df["aqi"]

)

# =========================================================
# RISK CATEGORIES
# =========================================================

def pollution_category(value):

    if value < 50:
        return "Low"

    elif value < 100:
        return "Moderate"

    elif value < 150:
        return "High"

    return "Critical"

pollution_df["pollution_risk_category"] = (

    pollution_df["pollution_exposure_index"]

    .apply(
        pollution_category
    )

)

# =========================================================
# SAVE POLLUTION FEATURES
# =========================================================

pollution_df.to_csv(
    "output/analytics/pollution_features.csv",
    index=False
)

print(
    "Pollution Features Created"
)

# =========================================================
# FOREST FEATURE ENGINEERING
# =========================================================

print("\nProcessing Forest Dataset")

forest_df["forest_growth_rate"] = (

    forest_df["forest_cover"]

    .pct_change()

    * 100

)

forest_df["forest_ma_3yr"] = (

    forest_df["forest_cover"]

    .rolling(
        3,
        min_periods=1
    )

    .mean()

)

forest_df["forest_ma_5yr"] = (

    forest_df["forest_cover"]

    .rolling(
        5,
        min_periods=1
    )

    .mean()

)

# forest sustainability index

forest_df["forest_sustainability_index"] = (

    forest_df["forest_cover"]

    /

    forest_df["forest_cover"].max()

) * 100

# forest loss indicator

forest_df["forest_loss_indicator"] = (

    forest_df["forest_cover"]

    .diff()

)

# =========================================================
# SAVE FOREST FEATURES
# =========================================================

forest_df.to_csv(
    "output/analytics/forest_features.csv",
    index=False
)

print(
    "Forest Features Created"
)

# =========================================================
# VALIDATION REPORT
# =========================================================

validation = {

    "temperature_rows":
        int(len(climate_df)),

    "aqi_rows":
        int(len(pollution_df)),

    "forest_rows":
        int(len(forest_df)),

    "temperature_missing":
        int(
            climate_df.isna().sum().sum()
        ),

    "aqi_missing":
        int(
            pollution_df.isna().sum().sum()
        ),

    "forest_missing":
        int(
            forest_df.isna().sum().sum()
        )

}

pd.Series(
    validation
).to_json(
    "output/reports/validation_report.json"
)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PHASES 2–5 COMPLETE")
print("=" * 60)

print(
    "\nGenerated Files:"
)

print(
    "output/analytics/climate_features.csv"
)

print(
    "output/analytics/pollution_features.csv"
)

print(
    "output/analytics/forest_features.csv"
)

print(
    "output/reports/validation_report.json"
)

print("=" * 60)
# =========================================================
# PHASE 6
# ENVIRONMENTAL HEALTH INDEX
# =========================================================

print("\n" + "=" * 60)
print("PHASE 6 - ENVIRONMENTAL HEALTH INDEX")
print("=" * 60)

forest_score = (
    forest_df["forest_sustainability_index"]
    .mean()
)

climate_score = (

    100

    -

    climate_df[
        "climate_severity_index"
    ].mean()

)

pollution_score = (

    100

    -

    min(

        pollution_df[
            "pollution_exposure_index"
        ].mean(),

        100

    )

)

environmental_health_index = (

      0.40 * forest_score

    + 0.30 * climate_score

    + 0.30 * pollution_score

)

ehi_df = pd.DataFrame({

    "environmental_health_index": [
        environmental_health_index
    ],

    "forest_score": [
        forest_score
    ],

    "climate_score": [
        climate_score
    ],

    "pollution_score": [
        pollution_score
    ]

})

ehi_df.to_csv(
    "output/analytics/environmental_health_index.csv",
    index=False
)

print(
    f"EHI Created: {environmental_health_index:.2f}"
)

# =========================================================
# PHASE 7
# RISK ENGINE
# =========================================================

print("\n" + "=" * 60)
print("PHASE 7 - RISK ENGINE")
print("=" * 60)

risk_score = (

    100

    -

    environmental_health_index

)

if risk_score < 30:

    risk_category = "Low"

elif risk_score < 60:

    risk_category = "Moderate"

elif risk_score < 80:

    risk_category = "High"

else:

    risk_category = "Critical"

risk_drivers = {

    "Climate":

        climate_df[
            "climate_severity_index"
        ].mean(),

    "Pollution":

        pollution_df[
            "pollution_exposure_index"
        ].mean(),

    "Forest":

        (
            100
            -
            forest_score
        )

}

risk_driver = max(
    risk_drivers,
    key=risk_drivers.get
)

risk_df = pd.DataFrame({

    "environmental_risk_score": [
        risk_score
    ],

    "risk_category": [
        risk_category
    ],

    "risk_driver": [
        risk_driver
    ]

})

risk_df.to_csv(

    "output/analytics/risk_dataset.csv",

    index=False

)

print(
    f"Risk Category: {risk_category}"
)

# =========================================================
# PHASE 8
# SUSTAINABILITY ENGINE
# =========================================================

print("\n" + "=" * 60)
print("PHASE 8 - SUSTAINABILITY ENGINE")
print("=" * 60)

sustainability_score = (
    environmental_health_index
)

if sustainability_score < 40:

    sustainability_category = "Poor"

elif sustainability_score < 60:

    sustainability_category = "Fair"

elif sustainability_score < 80:

    sustainability_category = "Good"

else:

    sustainability_category = "Excellent"

sustainability_df = pd.DataFrame({

    "sustainability_score": [
        sustainability_score
    ],

    "sustainability_category": [
        sustainability_category
    ]

})

sustainability_df.to_csv(

    "output/analytics/sustainability_dataset.csv",

    index=False

)

print(
    f"Sustainability Category: {sustainability_category}"
)

# =========================================================
# PHASE 9
# MACHINE LEARNING
# =========================================================

print("\n" + "=" * 60)
print("PHASE 9 - MACHINE LEARNING")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

X = climate_df[
    ["year"]
]

y = climate_df[
    "temperature_anomaly"
]

# -------- Linear Regression --------

lr_model = LinearRegression()

lr_model.fit(
    X,
    y
)

lr_predictions = (
    lr_model.predict(X)
)

# -------- Random Forest --------

rf_model = RandomForestRegressor(

    n_estimators=300,

    random_state=42,

    max_depth=12

)

rf_model.fit(
    X,
    y
)

rf_predictions = (
    rf_model.predict(X)
)

print("Models Trained Successfully")

# =========================================================
# PHASE 10
# CROSS VALIDATION
# =========================================================

print("\n" + "=" * 60)
print("PHASE 10 - CROSS VALIDATION")
print("=" * 60)

from sklearn.model_selection import cross_val_score

lr_cv_scores = cross_val_score(

    lr_model,

    X,

    y,

    cv=5,

    scoring="r2"

)

rf_cv_scores = cross_val_score(

    rf_model,

    X,

    y,

    cv=5,

    scoring="r2"

)

cv_df = pd.DataFrame({

    "model": [

        "Linear Regression",

        "Random Forest"

    ],

    "mean_cv_score": [

        lr_cv_scores.mean(),

        rf_cv_scores.mean()

    ],

    "std_cv_score": [

        lr_cv_scores.std(),

        rf_cv_scores.std()

    ]

})

cv_df.to_csv(

    "output/analytics/cross_validation_report.csv",

    index=False

)

print("Cross Validation Complete")

# =========================================================
# PHASE 11
# MODEL EVALUATION
# =========================================================

print("\n" + "=" * 60)
print("PHASE 11 - MODEL EVALUATION")
print("=" * 60)

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)

model_results = []

models = {

    "Linear Regression":
        lr_predictions,

    "Random Forest":
        rf_predictions

}

for model_name, predictions in models.items():

    mae = mean_absolute_error(

        y,

        predictions

    )

    rmse = np.sqrt(

        mean_squared_error(

            y,

            predictions

        )

    )

    r2 = r2_score(

        y,

        predictions

    )

    model_results.append({

        "model":
            model_name,

        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2

    })

model_comparison = pd.DataFrame(
    model_results
)

model_comparison.to_csv(

    "output/analytics/model_comparison.csv",

    index=False

)

best_model = (

    model_comparison

    .sort_values(

        "r2",

        ascending=False

    )

    .head(1)

)

best_model.to_csv(

    "output/analytics/best_model.csv",

    index=False

)

best_row = best_model.iloc[0]

model_metrics = {

    "best_model":
        best_row["model"],

    "r2":
        float(best_row["r2"]),

    "mae":
        float(best_row["mae"]),

    "rmse":
        float(best_row["rmse"])

}

with open(

    "output/reports/model_metrics.json",

    "w"

) as file:

    json.dump(
        model_metrics,
        file,
        indent=4
    )

print("model_metrics.json created")
print("\nBest Model")

print(best_model)

# =========================================================
# PHASES 6-11 SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PHASES 6–11 COMPLETE")
print("=" * 60)

print(
    "environmental_health_index.csv"
)

print(
    "risk_dataset.csv"
)

print(
    "sustainability_dataset.csv"
)

print(
    "cross_validation_report.csv"
)

print(
    "model_comparison.csv"
)

print(
    "best_model.csv"
)

print("=" * 60)
# =========================================================
# PHASE 12
# FEATURE IMPORTANCE
# =========================================================

print("\n" + "=" * 60)
print("PHASE 12 - FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    "output/analytics/feature_importance.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "output/charts/feature_importance.png"
)

plt.close()

print("Feature Importance Generated")

# =========================================================
# PHASE 13
# RESIDUAL DIAGNOSTICS
# =========================================================

print("\n" + "=" * 60)
print("PHASE 13 - RESIDUAL DIAGNOSTICS")
print("=" * 60)

best_predictions = rf_predictions

residuals = y - best_predictions

# Actual vs Predicted

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    best_predictions,
    alpha=0.6
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "output/charts/actual_vs_predicted.png"
)

plt.close()

# Residual Distribution

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=20
)

plt.title(
    "Residual Distribution"
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "output/charts/residual_distribution.png"
)

plt.close()

# Residuals vs Predicted

plt.figure(figsize=(8, 6))

plt.scatter(
    best_predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.title(
    "Residuals vs Predicted"
)

plt.tight_layout()

plt.savefig(
    "output/charts/residuals_vs_predicted.png"
)

plt.close()

print("Residual Diagnostics Generated")

# =========================================================
# PHASE 14
# FORECASTING
# =========================================================

print("\n" + "=" * 60)
print("PHASE 14 - FORECASTING")
print("=" * 60)

last_year = int(
    climate_df["year"].max()
)

future_years = pd.DataFrame({

    "year": range(
        last_year + 1,
        last_year + 6
    )

})

#use linear regression model for future forecasting
future_predictions = lr_model.predict(
    future_years
)

prediction_std = residuals.std()

forecast_df = future_years.copy()

forecast_df["predicted_temperature"] = (
    future_predictions
)

forecast_df["lower_bound"] = (
    future_predictions
    -
    (1.96 * prediction_std)
)

forecast_df["upper_bound"] = (
    future_predictions
    +
    (1.96 * prediction_std)
)

forecast_df.to_csv(
    "output/analytics/forecast_results.csv",
    index=False
)

# Forecast Chart

plt.figure(figsize=(12, 6))

plt.plot(
    climate_df["year"],
    climate_df["temperature_anomaly"],
    label="Historical"
)

plt.plot(
    forecast_df["year"],
    forecast_df["predicted_temperature"],
    label="Forecast",
    linewidth=3
)

plt.legend()

plt.title(
    "Temperature Forecast"
)

plt.xlabel("Year")

plt.ylabel(
    "Temperature Anomaly"
)

plt.tight_layout()

plt.savefig(
    "output/charts/forecast.png"
)

plt.close()

# Confidence Band

plt.figure(figsize=(12, 6))

plt.plot(
    forecast_df["year"],
    forecast_df["predicted_temperature"],
    color="blue"
)

plt.fill_between(
    forecast_df["year"],
    forecast_df["lower_bound"],
    forecast_df["upper_bound"],
    alpha=0.3
)

plt.title(
    "Forecast Uncertainty"
)

plt.xlabel("Year")

plt.ylabel(
    "Temperature Anomaly"
)

plt.tight_layout()

plt.savefig(
    "output/charts/forecast_uncertainty.png"
)

plt.close()

print("Forecasting Completed")

# =========================================================
# PHASE 15
# SCENARIO FORECASTING
# =========================================================

print("\n" + "=" * 60)
print("PHASE 15 - SCENARIO FORECASTING")
print("=" * 60)

baseline_forecast = (
    forecast_df[
        "predicted_temperature"
    ].mean()
)

scenario_df = pd.DataFrame({

    "scenario": [

        "Baseline",

        "Green Policy",

        "Aggressive Green",

        "Pollution Growth"

    ],

    "adjustment_factor": [

        1.00,

        0.98,

        0.95,

        1.03

    ]

})

scenario_df["projected_temperature"] = (

    baseline_forecast
    *
    scenario_df[
        "adjustment_factor"
    ]

)

scenario_df.to_csv(
    "output/analytics/scenario_forecast.csv",
    index=False
)

plt.figure(figsize=(8, 6))

plt.bar(
    scenario_df["scenario"],
    scenario_df[
        "projected_temperature"
    ]
)

plt.title(
    "Scenario Forecast Comparison"
)

plt.ylabel(
    "Projected Temperature"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "output/charts/scenario_comparison.png"
)

plt.close()

print("Scenario Forecasting Completed")

# =========================================================
# PHASES 12–15 COMPLETE
# =========================================================

print("\n" + "=" * 60)
print("PHASES 12-15 COMPLETE")
print("=" * 60)

print("feature_importance.csv")
print("forecast_results.csv")
print("scenario_forecast.csv")

print("feature_importance.png")
print("actual_vs_predicted.png")
print("residual_distribution.png")
print("residuals_vs_predicted.png")
print("forecast.png")
print("forecast_uncertainty.png")
print("scenario_comparison.png")

print("=" * 60)
# =========================================================
# PHASE 16
# KMEANS CLUSTERING
# =========================================================

print("\n" + "=" * 60)
print("PHASE 16 - KMEANS CLUSTERING")
print("=" * 60)

from sklearn.cluster import KMeans

cluster_df = pd.DataFrame({

    "climate_score": [
        climate_score
    ],

    "pollution_score": [
        pollution_score
    ],

    "forest_score": [
        forest_score
    ]

})

# Create synthetic neighborhood around environment profile
# to make clustering meaningful

cluster_data = []

for i in range(100):

    cluster_data.append({

        "climate_score":
            climate_score +
            np.random.normal(0, 5),

        "pollution_score":
            pollution_score +
            np.random.normal(0, 5),

        "forest_score":
            forest_score +
            np.random.normal(0, 5)

    })

cluster_data = pd.DataFrame(
    cluster_data
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

cluster_data["cluster"] = (
    kmeans.fit_predict(
        cluster_data[
            [
                "climate_score",
                "pollution_score",
                "forest_score"
            ]
        ]
    )
)

cluster_data.to_csv(
    "output/analytics/clustering_results.csv",
    index=False
)

# cluster plot

plt.figure(figsize=(8, 6))

plt.scatter(

    cluster_data["climate_score"],

    cluster_data["forest_score"],

    c=cluster_data["cluster"]

)

plt.xlabel("Climate Score")

plt.ylabel("Forest Score")

plt.title(
    "Environmental Clusters"
)

plt.tight_layout()

plt.savefig(
    "output/charts/clusters.png"
)

plt.close()

print("Clustering Complete")

# =========================================================
# PHASE 17
# BUSINESS RECOMMENDATION ENGINE
# =========================================================

print("\n" + "=" * 60)
print("PHASE 17 - BUSINESS RECOMMENDATIONS")
print("=" * 60)

recommendations = []

if risk_score > 60:

    recommendations.append(
        "Environmental risk is high. Increase environmental monitoring."
    )

if pollution_score < 60:

    recommendations.append(
        "Implement pollution-control programs to reduce AQI and PM2.5 levels."
    )

if forest_score < 95:

    recommendations.append(
        "Increase afforestation and conservation initiatives."
    )

if climate_score < 60:

    recommendations.append(
        "Develop long-term climate adaptation strategies."
    )

if len(recommendations) == 0:

    recommendations.append(
        "Environmental indicators are stable. Continue monitoring."
    )
recommendations.append(
    "Increase afforestation initiatives to improve long-term environmental sustainability."
)

recommendations.append(
    "Strengthen climate adaptation and resilience strategies."
)

recommendations.append(
    "Improve air-quality monitoring in urban areas."
)
with open(

    "output/reports/business_recommendations.txt",

    "w"

) as file:

    for rec in recommendations:

        file.write(rec + "\n")

print("Business Recommendations Generated")

# =========================================================
# PHASE 18
# EXECUTIVE SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PHASE 18 - EXECUTIVE SUMMARY")
print("=" * 60)

best_model_name = best_model.iloc[0]["model"]

forecast_trend = "Increasing"

if forecast_df["predicted_temperature"].mean() < \
   climate_df["temperature_anomaly"].iloc[-1]:

    forecast_trend = "Decreasing"

summary = f"""
=====================================================
ENVIRONMENTAL INTELLIGENCE REPORT
=====================================================

Environmental Health Index : {environmental_health_index:.2f}

Environmental Risk Score   : {risk_score:.2f}

Risk Category              : {risk_category}

Sustainability Score       : {sustainability_score:.2f}

Sustainability Category    : {sustainability_category}

Best Model                 : {best_model_name}

Forecast Trend             : {forecast_trend}

Primary Risk Driver        : {risk_driver}

Average AQI               : {pollution_df['aqi'].mean():.2f}

Average PM2.5             : {pollution_df['pm2.5'].mean():.2f}

Forecast Horizon          : 5 Years

=====================================================
"""

with open(

    "output/reports/executive_summary.txt",

    "w"

) as file:

    file.write(summary)

print("Executive Summary Generated")

# =========================================================
# PHASE 19
# POWER BI STAR SCHEMA
# =========================================================

print("\n" + "=" * 60)
print("PHASE 19 - POWER BI STAR SCHEMA")
print("=" * 60)

# -----------------------------
# DIM TIME
# -----------------------------

dim_time = pd.DataFrame({

    "year":
        climate_df["year"]

})

dim_time["time_key"] = (
    range(1, len(dim_time) + 1)
)

dim_time.to_csv(

    "output/dashboard/dim_time.csv",

    index=False

)

# -----------------------------
# DIM RISK
# -----------------------------

dim_risk = pd.DataFrame({

    "risk_category":
        [risk_category],

    "risk_driver":
        [risk_driver]

})

dim_risk["risk_key"] = [1]

dim_risk.to_csv(

    "output/dashboard/dim_risk.csv",

    index=False

)

# -----------------------------
# DIM SCENARIO
# -----------------------------

dim_scenario = scenario_df.copy()

dim_scenario["scenario_key"] = (

    range(
        1,
        len(dim_scenario) + 1
    )

)

dim_scenario.to_csv(

    "output/dashboard/dim_scenario.csv",

    index=False

)

# -----------------------------
# FACT TABLE
# -----------------------------

fact_environment = climate_df[[
    "year",
    "temperature_anomaly"
]].copy()

fact_environment["risk_score"] = (
    risk_score
)

fact_environment["sustainability_score"] = (
    sustainability_score
)

fact_environment["environmental_health_index"] = (
    environmental_health_index
)

fact_environment.to_csv(

    "output/dashboard/fact_environment.csv",

    index=False

)

print("Star Schema Created")

# =========================================================
# PHASE 20
# DASHBOARD MASTER DATASET
# =========================================================

print("\n" + "=" * 60)
print("PHASE 20 - DASHBOARD MASTER DATASET")
print("=" * 60)

dashboard_master = pd.DataFrame({

    "generated_date":[
        datetime.now().strftime("%Y-%m-%d")
    ],

    "environmental_health_index":[
        environmental_health_index
    ],
    "environmental_risk_score": [
        risk_score
    ],

    "sustainability_score": [
        sustainability_score
    ],

    "risk_category": [
        risk_category
    ],

    "risk_driver": [
        risk_driver
    ],

    "forest_score": [
        forest_score
    ],

    "climate_score": [
        climate_score
    ],

    "pollution_score": [
        pollution_score
    ],

    "best_model": [
        best_model_name
    ]

})

dashboard_master.to_csv(

    "output/dashboard/dashboard_master_dataset.csv",

    index=False

)

print("Dashboard Dataset Created")

# =========================================================
# FINAL PROJECT SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("ALL PHASES COMPLETE")
print("=" * 60)

print("\nAnalytics Files")

print("climate_features.csv")
print("pollution_features.csv")
print("forest_features.csv")
print("environmental_health_index.csv")
print("risk_dataset.csv")
print("sustainability_dataset.csv")
print("model_comparison.csv")
print("forecast_results.csv")
print("scenario_forecast.csv")
print("clustering_results.csv")

print("\nReports")

print("executive_summary.txt")
print("business_recommendations.txt")

print("\nPower BI Files")

print("fact_environment.csv")
print("dim_time.csv")
print("dim_risk.csv")
print("dim_scenario.csv")
print("dashboard_master_dataset.csv")

print("\nProject completed successfully.")
print("=" * 60)