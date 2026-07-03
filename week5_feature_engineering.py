"""
================================================
WEEK 5 FEATURE ENGINEERING
Environmental Intelligence Dataset Builder
================================================
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ==================================================
# OUTPUT FOLDERS
# ==================================================

os.makedirs("output", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

# ==================================================
# LOAD FILES
# ==================================================

temp_df = pd.read_csv("data/Temperature_Data.csv")
aqi_df = pd.read_csv("data/Air_Quality_Data.csv")
geo_df = pd.read_csv("data/Geospatial_Data.csv")

# ==================================================
# CLEAN COLUMN NAMES
# ==================================================

temp_df.columns = (
    temp_df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

aqi_df.columns = (
    aqi_df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

geo_df.columns = (
    geo_df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

# ==================================================
# COLUMN INFORMATION
# ==================================================

print("\nTemperature Columns:")
print(temp_df.columns.tolist())

print("\nAQI Columns:")
print(aqi_df.columns.tolist())

print("\nGeospatial Columns:")
print(geo_df.columns.tolist())

# ==================================================
# FILTER GLOBAL FOREST COVER
# ==================================================

forest_df = geo_df[
    (
        geo_df["land"]
        == "Total, all countries or areas"
    )
    &
    (
        geo_df["series"]
        == "Forest cover (thousand hectares)"
    )
].copy()

forest_df = forest_df[
    ["year", "value"]
]

forest_df.rename(
    columns={
        "value": "forest_cover"
    },
    inplace=True
)

forest_df["year"] = forest_df["year"].astype(int)

print("\nForest Records:")
print(forest_df)

# ==================================================
# FOREST INTERPOLATION
# ==================================================

forest_interp = forest_df.set_index("year")

forest_interp = forest_interp.reindex(
    range(
        forest_interp.index.min(),
        forest_interp.index.max() + 1
    )
)

forest_interp["forest_cover"] = (
    forest_interp["forest_cover"]
    .interpolate(method="linear")
)

# ==================================================
# FOREST ENGINEERED FEATURES
# ==================================================

forest_interp["forest_growth_rate"] = (
    forest_interp["forest_cover"]
    .pct_change()
    * 100
)

forest_interp["forest_ma_3yr"] = (
    forest_interp["forest_cover"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
)

forest_interp["forest_ma_5yr"] = (
    forest_interp["forest_cover"]
    .rolling(
        window=5,
        min_periods=1
    )
    .mean()
)

forest_interp.reset_index(inplace=True)

forest_interp.rename(
    columns={"index": "year"},
    inplace=True
)

# ==================================================
# SAVE INTERPOLATED FOREST DATASET
# ==================================================

forest_interp.to_csv(
    "output/forest_interpolated.csv",
    index=False
)

# ==================================================
# ENVIRONMENTAL KPI CALCULATIONS
# ==================================================

forest_mean = forest_interp[
    "forest_cover"
].mean()

# Placeholder KPI values
# Advanced calculations will be done in analysis_forecasting_v3.py

environmental_risk_score = 50.0

sustainability_score = 75.0

# ==================================================
# RISK CATEGORY
# ==================================================

def get_risk_category(score):

    if score < 30:
        return "Low"

    elif score < 60:
        return "Moderate"

    elif score < 80:
        return "High"

    return "Critical"


risk_category = get_risk_category(
    environmental_risk_score
)

# ==================================================
# ENVIRONMENTAL KPI DATASET
# ==================================================

risk_df = pd.DataFrame({

    "environmental_risk_score":
        [environmental_risk_score],

    "sustainability_score":
        [sustainability_score],

    "risk_category":
        [risk_category]

})

risk_df.to_csv(
    "output/environmental_risk_score.csv",
    index=False
)

# ==================================================
# DASHBOARD MASTER DATASET
# ==================================================

dashboard_df = pd.DataFrame({

    "metric": [
        "forest_cover"
    ],

    "average_value": [
        forest_mean
    ],

    "environmental_risk_score": [
        environmental_risk_score
    ],

    "sustainability_score": [
        sustainability_score
    ],

    "risk_category": [
        risk_category
    ],

    "generated_date": [
        datetime.now().strftime(
            "%Y-%m-%d"
        )
    ]

})

dashboard_df.to_csv(
    "output/dashboard_master_dataset.csv",
    index=False
)

# ==================================================
# DATA QUALITY REPORT
# ==================================================

quality_report = {

    "temperature_rows":
        int(len(temp_df)),

    "aqi_rows":
        int(len(aqi_df)),

    "forest_rows_original":
        int(len(forest_df)),

    "forest_rows_interpolated":
        int(len(forest_interp)),

    "generated_on":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

}

with open(
    "output/reports/data_quality_report.json",
    "w"
) as f:

    json.dump(
        quality_report,
        f,
        indent=4
    )

# ==================================================
# METADATA REPORT
# ==================================================

metadata = {

    "temperature_rows":
        int(len(temp_df)),

    "aqi_rows":
        int(len(aqi_df)),

    "forest_rows_original":
        int(len(forest_df)),

    "forest_rows_interpolated":
        int(len(forest_interp)),

    "environmental_risk_score":
        environmental_risk_score,

    "sustainability_score":
        sustainability_score,

    "risk_category":
        risk_category

}

with open(
    "output/reports/metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("WEEK 5 FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print("\nGenerated Files:")

print("output/forest_interpolated.csv")
print("output/environmental_risk_score.csv")
print("output/dashboard_master_dataset.csv")
print("output/reports/metadata.json")
print("output/reports/data_quality_report.json")

print("\nOriginal Forest Records :", len(forest_df))
print("Interpolated Records    :", len(forest_interp))

print("\nEnvironmental Risk Score :", environmental_risk_score)
print("Sustainability Score     :", sustainability_score)
print("Risk Category            :", risk_category)

print("=" * 60)