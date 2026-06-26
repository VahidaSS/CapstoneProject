"""
===========================================================
Environmental Data Analytics Capstone Project (Team 14)
Week 4 – Feature Engineering Pipeline

Author : Team 14
Description:
Production-ready Feature Engineering Pipeline

Features:
✓ Configuration Driven
✓ Logging
✓ Validation
✓ PostgreSQL Integration
✓ Exception Handling
✓ Modular Design
===========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import json
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================================
# START TIMER
# ==========================================================

PIPELINE_START_TIME = time.time()

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

os.makedirs("output", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/metadata", exist_ok=True)

# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = f"output/logs/pipeline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("ENVIRONMENTAL FEATURE ENGINEERING PIPELINE STARTED")
logger.info("=" * 70)

# ==========================================================
# LOAD CONFIGURATION
# ==========================================================

def load_config():

    """
    Load configuration from config.json
    """

    try:

        with open("config.json", "r") as file:
            config = json.load(file)

        logger.info("Configuration loaded successfully")

        return config

    except FileNotFoundError:

        logger.exception("config.json not found")
        raise

    except Exception as e:

        logger.exception(e)
        raise


config = load_config()

DB = config["db"]

OUTPUT_PATH = config.get("output_path", "output/")

# ==========================================================
# DATABASE CONNECTION
# ==========================================================

def create_database_connection():

    """
    Create PostgreSQL connection
    """

    try:

        engine = create_engine(

            f"postgresql://{DB['user']}:{DB['password']}@"
            f"{DB['host']}:{DB['port']}/{DB['database']}"

        )

        connection = engine.connect()

        connection.close()

        logger.info("PostgreSQL connection established")

        return engine

    except SQLAlchemyError as e:

        logger.exception(e)
        raise

    except Exception as e:

        logger.exception(e)
        raise


engine = create_database_connection()

# ==========================================================
# LOAD DATA
# ==========================================================

def load_dataset():

    """
    Read merged dataset from PostgreSQL
    """

    logger.info("Loading merged dataset...")

    try:

        df = pd.read_sql(
            "SELECT * FROM environment_merged",
            engine
        )

        logger.info(f"Dataset Loaded Successfully")

        logger.info(f"Rows    : {df.shape[0]}")
        logger.info(f"Columns : {df.shape[1]}")

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
        )

        logger.info("Column names standardized")

        return df

    except Exception as e:

        logger.exception(e)
        raise

# ==========================================================
# DATA VALIDATION
# ==========================================================

REQUIRED_COLUMNS = [

    "year",

    "temperature_anomaly",

    "aqi",

    "pm2.5"

]

def validate_dataset(df):

    """
    Validate dataset before feature engineering
    """

    logger.info("Running data validation")

    # Empty Dataset
    if df.empty:

        raise ValueError("Dataset is empty")

    logger.info("Dataset is not empty")

    # Required Columns
    missing_columns = [

        col

        for col in REQUIRED_COLUMNS

        if col not in df.columns

    ]

    if missing_columns:

        raise ValueError(

            f"Missing Columns : {missing_columns}"

        )

    logger.info("Required columns present")

    # Duplicate Rows

    duplicate_count = df.duplicated().sum()

    logger.info(

        f"Duplicate Rows : {duplicate_count}"

    )

    # Missing Values

    logger.info("Missing Values")

    logger.info(df.isnull().sum())

    # Memory Usage

    memory = (

        df.memory_usage(deep=True).sum()

        / 1024

    )

    logger.info(

        f"Memory Usage : {memory:.2f} KB"

    )

    # Data Types

    logger.info("Data Types")

    logger.info(df.dtypes)

    logger.info("Validation Completed")

# ==========================================================
# PIPELINE SUMMARY
# ==========================================================

def pipeline_summary(df):

    """
    Display pipeline summary
    """

    print("\n")

    print("=" * 60)

    print("PIPELINE SUMMARY")

    print("=" * 60)

    print(f"Rows            : {df.shape[0]}")

    print(f"Columns         : {df.shape[1]}")

    print(f"Duplicates      : {df.duplicated().sum()}")

    print(f"Missing Values  : {df.isnull().sum().sum()}")

    print("=" * 60)


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def engineer_features(df):
    """
    Create engineered features from the integrated dataset.
    """

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING STARTED")
    logger.info("=" * 60)

    try:

        # --------------------------------------------------
        # Sort by Year
        # --------------------------------------------------

        df = df.sort_values("year").reset_index(drop=True)

        logger.info("Dataset sorted by year")

        # --------------------------------------------------
        # Rolling Average Features
        # --------------------------------------------------

        logger.info("Generating rolling average features")

        df["temp_5yr_avg"] = (
            df["temperature_anomaly"]
            .rolling(window=5, min_periods=1)
            .mean()
        )

        df["aqi_3yr_avg"] = (
            df["aqi"]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        if "forest_cover" in df.columns:

            df["forest_3yr_avg"] = (
                df["forest_cover"]
                .rolling(window=3, min_periods=1)
                .mean()
            )

        # --------------------------------------------------
        # Trend Features
        # --------------------------------------------------

        logger.info("Generating trend features")

        df["temp_trend"] = (
            df["temperature_anomaly"]
            .diff()
            .fillna(0)
        )

        df["aqi_trend"] = (
            df["aqi"]
            .diff()
            .fillna(0)
        )

        if "forest_cover" in df.columns:

            df["forest_trend"] = (
                df["forest_cover"]
                .diff()
                .fillna(0)
            )

        # --------------------------------------------------
        # Percentage Growth
        # --------------------------------------------------

        logger.info("Generating growth features")

        df["temp_growth_pct"] = (
            df["temperature_anomaly"]
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        df["aqi_growth_pct"] = (
            df["aqi"]
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        # --------------------------------------------------
        # Lag Features
        # --------------------------------------------------

        logger.info("Generating lag features")

        df["temp_previous_year"] = (
            df["temperature_anomaly"]
            .shift(1)
            .fillna(df["temperature_anomaly"])
        )

        df["aqi_previous_year"] = (
            df["aqi"]
            .shift(1)
            .fillna(df["aqi"])
        )

        # --------------------------------------------------
        # Cumulative Metrics
        # --------------------------------------------------

        logger.info("Generating cumulative metrics")

        df["cumulative_temperature"] = (
            df["temperature_anomaly"]
            .cumsum()
        )

        df["cumulative_pm25"] = (
            df["pm2.5"]
            .cumsum()
        )

        df["cumulative_aqi"] = (
            df["aqi"]
            .cumsum()
        )

        if "forest_cover" in df.columns:

            df["cumulative_forest"] = (
                df["forest_cover"]
                .cumsum()
            )

        # --------------------------------------------------
        # Running Maximum
        # --------------------------------------------------

        df["max_temperature"] = (
            df["temperature_anomaly"]
            .cummax()
        )

        df["max_aqi"] = (
            df["aqi"]
            .cummax()
        )

        # --------------------------------------------------
        # Running Minimum
        # --------------------------------------------------

        df["min_temperature"] = (
            df["temperature_anomaly"]
            .cummin()
        )

        df["min_aqi"] = (
            df["aqi"]
            .cummin()
        )

        # --------------------------------------------------
        # Environmental Ratios
        # --------------------------------------------------

        logger.info("Generating environmental ratios")

        df["aqi_to_temp_ratio"] = (
            df["aqi"]
            /
            (df["temperature_anomaly"] + 1)
        )

        df["pm25_to_aqi_ratio"] = (
            df["pm2.5"]
            /
            (df["aqi"] + 1)
        )

        # --------------------------------------------------
        # Composite Environmental Score
        # --------------------------------------------------

        logger.info("Generating environmental score")

        df["env_score"] = (
            (
                df["aqi"]
                *
                df["temperature_anomaly"]
            )
            +
            df["pm2.5"]
        )

        # --------------------------------------------------
        # Environmental Index
        # --------------------------------------------------

        logger.info("Generating environmental index")

        df["environmental_index"] = (

            df[
                [
                    "temperature_anomaly",
                    "aqi",
                    "pm2.5"
                ]

            ].mean(axis=1)

        )

        # --------------------------------------------------
        # Risk Category
        # --------------------------------------------------

        logger.info("Assigning AQI risk category")

        df["risk_category"] = pd.cut(

            df["aqi"],

            bins=[
                -1,
                50,
                100,
                200,
                300,
                np.inf
            ],

            labels=[
                "Good",
                "Moderate",
                "Poor",
                "Very Poor",
                "Severe"
            ]

        )

        # --------------------------------------------------
        # Missing Values
        # --------------------------------------------------

        df = df.fillna(0)

        logger.info("Missing values handled")

        logger.info(
            f"Total Features Created : {len(df.columns)}"
        )

        logger.info("Feature Engineering Completed Successfully")

        return df

    except Exception as e:

        logger.exception(e)

        raise
    # ==========================================================
# NORMALIZATION & SCALING
# ==========================================================

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def normalize_features(df):
    """
    Normalize numerical features using
    Z-score Standardization and Min-Max Scaling.
    """

    logger.info("=" * 60)
    logger.info("NORMALIZATION STARTED")
    logger.info("=" * 60)

    try:

        numerical_columns = [
            "temperature_anomaly",
            "aqi",
            "pm2.5"
        ]

        if "forest_cover" in df.columns:
            numerical_columns.append("forest_cover")

        # ----------------------------------------------
        # Standard Scaling (Z-score)
        # ----------------------------------------------

        logger.info("Applying StandardScaler")

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            df[numerical_columns]
        )

        scaled_df = pd.DataFrame(

            scaled,

            columns=[
                col + "_zscore"
                for col in numerical_columns
            ]

        )

        df = pd.concat(
            [df, scaled_df],
            axis=1
        )

        # ----------------------------------------------
        # Min-Max Scaling
        # ----------------------------------------------

        logger.info("Applying MinMaxScaler")

        minmax = MinMaxScaler()

        minmax_scaled = minmax.fit_transform(
            df[numerical_columns]
        )

        minmax_df = pd.DataFrame(

            minmax_scaled,

            columns=[
                col + "_minmax"
                for col in numerical_columns
            ]

        )

        df = pd.concat(
            [df, minmax_df],
            axis=1
        )

        logger.info(
            "Normalization completed successfully"
        )

        return df

    except Exception as e:

        logger.exception(e)

        raise   
    # ==========================================================
# FEATURE METADATA GENERATION
# ==========================================================

def generate_feature_metadata(df):
    """
    Generate metadata for all engineered features.
    """

    logger.info("=" * 60)
    logger.info("FEATURE METADATA GENERATION")
    logger.info("=" * 60)

    try:

        metadata = []

        descriptions = {

            "temp_5yr_avg":
                "5-Year Rolling Average of Temperature",

            "aqi_3yr_avg":
                "3-Year Rolling Average of AQI",

            "forest_3yr_avg":
                "3-Year Rolling Average of Forest Cover",

            "temp_trend":
                "Year-to-Year Temperature Change",

            "aqi_trend":
                "Year-to-Year AQI Change",

            "forest_trend":
                "Year-to-Year Forest Cover Change",

            "temp_growth_pct":
                "Temperature Growth Percentage",

            "aqi_growth_pct":
                "AQI Growth Percentage",

            "temp_previous_year":
                "Previous Year Temperature",

            "aqi_previous_year":
                "Previous Year AQI",

            "cumulative_temperature":
                "Running Total Temperature",

            "cumulative_pm25":
                "Running Total PM2.5",

            "cumulative_aqi":
                "Running Total AQI",

            "cumulative_forest":
                "Running Total Forest Cover",

            "aqi_to_temp_ratio":
                "AQI divided by Temperature",

            "pm25_to_aqi_ratio":
                "PM2.5 divided by AQI",

            "env_score":
                "Composite Environmental Score",

            "environmental_index":
                "Average Environmental Indicator",

            "risk_category":
                "AQI Risk Classification"

        }

        for column in df.columns:

            dtype = str(df[column].dtype)

            metadata.append({

                "Feature": column,

                "Data_Type": dtype,

                "Description":
                    descriptions.get(
                        column,
                        "Original Dataset Feature"
                    )

            })

        metadata_df = pd.DataFrame(metadata)

        metadata_file = (
            OUTPUT_PATH +
            f"/metadata/feature_metadata_{timestamp}.csv"
        )

        metadata_df.to_csv(
            metadata_file,
            index=False
        )

        logger.info(
            "Feature metadata generated successfully"
        )

        logger.info(
            f"Metadata saved at {metadata_file}"
        )

        return metadata_df

    except Exception as e:

        logger.exception(e)

        raise
    # ==========================================================
# DATA QUALITY REPORT
# ==========================================================

def generate_data_quality_report(df):
    """
    Generate a comprehensive data quality report.
    """

    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    try:

        report = []

        report.append("=" * 70)
        report.append("ENVIRONMENTAL DATA QUALITY REPORT")
        report.append("=" * 70)
        report.append(f"Generated : {datetime.now()}")
        report.append("")

        # ------------------------------------------
        # Dataset Information
        # ------------------------------------------

        report.append("DATASET SUMMARY")
        report.append("-" * 40)
        report.append(f"Rows                : {df.shape[0]}")
        report.append(f"Columns             : {df.shape[1]}")
        report.append("")

        # ------------------------------------------
        # Missing Values
        # ------------------------------------------

        report.append("MISSING VALUES")
        report.append("-" * 40)

        missing = df.isnull().sum()

        for col, value in missing.items():
            report.append(f"{col:35} {value}")

        report.append("")

        # ------------------------------------------
        # Duplicate Records
        # ------------------------------------------

        duplicates = df.duplicated().sum()

        report.append("DUPLICATE RECORDS")
        report.append("-" * 40)
        report.append(f"Duplicate Rows      : {duplicates}")
        report.append("")

        # ------------------------------------------
        # Data Types
        # ------------------------------------------

        report.append("COLUMN DATA TYPES")
        report.append("-" * 40)

        for col in df.columns:
            report.append(f"{col:35} {df[col].dtype}")

        report.append("")

        # ------------------------------------------
        # Memory Usage
        # ------------------------------------------

        memory = (
            df.memory_usage(deep=True).sum() / 1024
        )

        report.append("MEMORY USAGE")
        report.append("-" * 40)
        report.append(f"{memory:.2f} KB")
        report.append("")

        # ------------------------------------------
        # Numeric Summary
        # ------------------------------------------

        report.append("NUMERIC SUMMARY")
        report.append("-" * 40)

        numeric = df.select_dtypes(include="number")

        for col in numeric.columns:

            report.append(f"\n{col}")

            report.append(
                f"   Min  : {numeric[col].min():.2f}"
            )

            report.append(
                f"   Max  : {numeric[col].max():.2f}"
            )

            report.append(
                f"   Mean : {numeric[col].mean():.2f}"
            )

            report.append(
                f"   Std  : {numeric[col].std():.2f}"
            )

        report.append("")

        # ------------------------------------------
        # Validation Status
        # ------------------------------------------

        report.append("VALIDATION STATUS")
        report.append("-" * 40)

        if duplicates == 0:
            report.append("✓ No Duplicate Records")
        else:
            report.append("⚠ Duplicate Records Found")

        if missing.sum() == 0:
            report.append("✓ No Missing Values")
        else:
            report.append("⚠ Missing Values Present")

        report.append("")

        report.append("=" * 70)
        report.append("REPORT COMPLETED")
        report.append("=" * 70)

        report_path = (
            OUTPUT_PATH +
            f"/reports/data_quality_report_{timestamp}.txt"
        )

        with open(report_path, "w" ,encoding="utf-8") as file:

            file.write("\n".join(report))

        logger.info("Data Quality Report Generated")

        logger.info(report_path)

        return report_path

    except Exception as e:

        logger.exception(e)

        raise
    # ==========================================================
# SAFE PCA & CORRELATION ANALYSIS
# ==========================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_advanced_analysis(df):
    """
    Perform correlation analysis and PCA only when
    sufficient observations are available.
    """

    logger.info("=" * 60)
    logger.info("ADVANCED ANALYSIS")
    logger.info("=" * 60)

    try:

        numeric_df = df.select_dtypes(include="number")

        # ------------------------------------------
        # Correlation Analysis
        # ------------------------------------------

        if len(df) >= 5:

            logger.info("Generating Correlation Matrix")

            corr = numeric_df.corr()

            corr_file = (
                OUTPUT_PATH +
                f"/reports/correlation_matrix_{timestamp}.csv"
            )

            corr.to_csv(corr_file)

            plt.figure(figsize=(10,8))

            plt.imshow(corr, cmap="coolwarm")

            plt.xticks(
                range(len(corr.columns)),
                corr.columns,
                rotation=90
            )

            plt.yticks(
                range(len(corr.columns)),
                corr.columns
            )

            plt.colorbar()

            plt.tight_layout()

            heatmap_file = (
                OUTPUT_PATH +
                f"/plots/correlation_heatmap_{timestamp}.png"
            )

            plt.savefig(heatmap_file)

            plt.close()

            logger.info("Correlation analysis completed")

        else:

            logger.warning(
                "Correlation skipped - insufficient records"
            )

        # ------------------------------------------
        # PCA
        # ------------------------------------------

        pca_columns = []

        for column in [

            "temperature_anomaly",

            "aqi",

            "pm2.5"

        ]:

            if column in df.columns:

                pca_columns.append(column)

        if len(df) >= 3 and len(pca_columns) >= 2:

            logger.info("Running PCA")

            scaler = StandardScaler()

            scaled = scaler.fit_transform(
                df[pca_columns]
            )

            pca = PCA(n_components=2)

            components = pca.fit_transform(scaled)

            df["pc1"] = components[:,0]
            df["pc2"] = components[:,1]

            variance = pd.DataFrame({

                "Component":[
                    "PC1",
                    "PC2"
                ],

                "Explained Variance":

                    pca.explained_variance_ratio_

            })

            variance_file = (

                OUTPUT_PATH +

                f"/reports/pca_variance_{timestamp}.csv"

            )

            variance.to_csv(

                variance_file,

                index=False

            )

            plt.figure(figsize=(8,6))

            plt.scatter(

                df["pc1"],

                df["pc2"],

                s=80

            )

            plt.xlabel("Principal Component 1")

            plt.ylabel("Principal Component 2")

            plt.title("Principal Component Analysis")

            plt.grid(True)

            pca_plot = (

                OUTPUT_PATH +

                f"/plots/pca_plot_{timestamp}.png"

            )

            plt.savefig(pca_plot)

            plt.close()

            logger.info("PCA completed successfully")

        else:

            logger.warning(

                "PCA skipped - insufficient observations"

            )

            df["pc1"] = 0

            df["pc2"] = 0

        return df

    except Exception as e:

        logger.exception(e)

        raise
    # ==========================================================
# SAVE OUTPUTS & DATABASE VALIDATION
# ==========================================================

def save_outputs(df, metadata_df):
    """
    Save engineered datasets, metadata and feature summary.
    """

    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    try:

        feature_dataset = (
            OUTPUT_PATH +
            f"/feature_dataset_{timestamp}.csv"
        )

        df.to_csv(
            feature_dataset,
            index=False
        )

        logger.info("Feature dataset saved")

        metadata_file = (
            OUTPUT_PATH +
            f"/metadata/feature_metadata_{timestamp}.csv"
        )

        metadata_df.to_csv(
            metadata_file,
            index=False
        )

        logger.info("Metadata saved")

        summary = df.describe(include="all").transpose()

        summary_file = (
            OUTPUT_PATH +
            f"/reports/feature_summary_{timestamp}.csv"
        )

        summary.to_csv(summary_file)

        logger.info("Feature summary saved")

        return feature_dataset

    except Exception as e:

        logger.exception(e)
        raise


# ==========================================================
# SAVE TO DATABASE
# ==========================================================

def save_to_database(df):

    logger.info("=" * 60)
    logger.info("DATABASE UPDATE")
    logger.info("=" * 60)

    try:

        df.to_sql(
            "feature_engineered",
            engine,
            if_exists="replace",
            index=False
        )

        logger.info(
            "Table feature_engineered updated"
        )

    except Exception as e:

        logger.exception(e)
        raise


# ==========================================================
# DATABASE VALIDATION
# ==========================================================

def validate_database(df):

    logger.info("=" * 60)
    logger.info("DATABASE VALIDATION")
    logger.info("=" * 60)

    try:

        validation = pd.read_sql(

            """
            SELECT COUNT(*) AS total_rows
            FROM feature_engineered
            """,

            engine

        )

        db_rows = int(validation.iloc[0]["total_rows"])

        csv_rows = len(df)

        logger.info(f"CSV Rows      : {csv_rows}")
        logger.info(f"Database Rows : {db_rows}")

        if db_rows == csv_rows:

            logger.info(
                "DATABASE VALIDATION PASSED"
            )

            return True

        logger.warning(
            "DATABASE VALIDATION FAILED"
        )

        return False

    except Exception as e:

        logger.exception(e)

        return False


# ==========================================================
# EXECUTION REPORT
# ==========================================================

def generate_execution_report(df):

    logger.info("=" * 60)
    logger.info("GENERATING EXECUTION REPORT")
    logger.info("=" * 60)

    runtime = time.time() - PIPELINE_START_TIME

    report = [

        "=" * 70,
        "FEATURE ENGINEERING EXECUTION REPORT",
        "=" * 70,
        f"Execution Time : {runtime:.2f} Seconds",
        "",
        f"Rows Processed : {len(df)}",
        f"Columns        : {len(df.columns)}",
        "",
        f"Features Generated : {len(df.columns)}",
        "",
        f"Duplicates : {df.duplicated().sum()}",
        f"Missing Values : {df.isnull().sum().sum()}",
        "",
        f"Pipeline Status : SUCCESS",
        "=" * 70

    ]

    report_file = (

        OUTPUT_PATH +

        f"/reports/execution_report_{timestamp}.txt"

    )

    with open(report_file, "w") as file:

        file.write("\n".join(report))

    logger.info("Execution report created")

    return report_file


# ==========================================================
# FINAL PIPELINE SUMMARY
# ==========================================================

def print_pipeline_summary(df):

    runtime = time.time() - PIPELINE_START_TIME

    print("\n")
    print("=" * 70)
    print("ENVIRONMENTAL FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    print(f"Rows Processed      : {len(df)}")
    print(f"Columns Generated   : {len(df.columns)}")
    print(f"Execution Time      : {runtime:.2f} sec")

    print(f"Duplicate Rows      : {df.duplicated().sum()}")
    print(f"Missing Values      : {df.isnull().sum().sum()}")

    print("\nPipeline Status : SUCCESS")

    print("=" * 70)

    logger.info("Pipeline Completed Successfully")

#==========================================================
# MAIN
# ==========================================================

def main():

    logger.info("Pipeline Started")

    df = load_dataset()

    validate_dataset(df)

    pipeline_summary(df)

    df = engineer_features(df)

    df = normalize_features(df)

    metadata_df = generate_feature_metadata(df)

    quality_report = generate_data_quality_report(df)

    df = perform_advanced_analysis(df)

    save_outputs(df, metadata_df)

    save_to_database(df)

    validate_database(df)

    generate_execution_report(df)

    print_pipeline_summary(df)

    logger.info("Pipeline Finished Successfully")

    return df

if __name__ == "__main__":

    dataframe = main()