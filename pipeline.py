import pandas as pd
import os
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine, text

# ==========================================
# CONFIGURATION
# ==========================================
with open("config.json") as f:
    config = json.load(f)

BASE_PATH = config["data_path"]
OUTPUT_PATH = config["output_path"]
DB = config["db"]

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("Pipeline started...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================================
# DATABASE CONNECTION
# ==========================================
engine = create_engine(
    f"postgresql://{DB['user']}:{DB['password']}@{DB['host']}:{DB['port']}/{DB['database']}"
)

# ==========================================
# CREATE FOLDERS
# ==========================================
def create_folders():
    for folder in ["temperature", "aqi", "geo", "merged"]:
        os.makedirs(os.path.join(OUTPUT_PATH, folder), exist_ok=True)
    logging.info(" Output folders ready")

# ==========================================
# LOAD DATA
# ==========================================
def load_data():
    try:
        temp = pd.read_csv(BASE_PATH + "Temperature_Data.csv", skiprows=1)
        aq = pd.read_csv(BASE_PATH + "Air_Quality_Data.csv")
        geo = pd.read_csv(BASE_PATH + "Geospatial_Data.csv")

        logging.info(" Data loaded successfully")
        return temp, aq, geo

    except Exception as e:
        logging.error(f" Data load error: {e}")
        raise

# ==========================================
# DATA VALIDATION 
# ==========================================
def validate_data(df, name):
    if df.empty:
        logging.warning(f" {name} dataset is EMPTY")

    if df.isnull().sum().sum() > 0:
        logging.warning(f" {name} contains missing values")

    logging.info(f" {name} validation completed")

# ==========================================
# PROCESS TEMPERATURE
# ==========================================
def process_temperature(df):
    df.columns = df.columns.str.strip()

    df = df[['Year', 'J-D']]
    df.rename(columns={'J-D': 'Temperature_Anomaly'}, inplace=True)

    df['Temperature_Anomaly'] = pd.to_numeric(df['Temperature_Anomaly'], errors='coerce')
    df = df.dropna()

    df = df.sort_values(by='Year')
    df['Temp_MA'] = df['Temperature_Anomaly'].rolling(3).mean()

    df = df[df['Year'] >= 2000]

    df.to_csv(f"{OUTPUT_PATH}temperature/temp_{timestamp}.csv", index=False)

    logging.info(f" Temperature processed | Rows: {len(df)}")
    return df

# ==========================================
# PROCESS AQI
# ==========================================
def process_aqi(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    df = df.ffill()

    df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

    df = df.dropna(subset=['PM2.5', 'AQI'])

    monthly = df.groupby(['City', 'Year', 'Month'], as_index=False)[['PM2.5', 'AQI']].mean()

    #  Partition by year
    for year in monthly['Year'].unique():
        monthly[monthly['Year'] == year].to_csv(
            f"{OUTPUT_PATH}aqi/aqi_{year}_{timestamp}.csv", index=False
        )

    yearly = monthly.groupby('Year')[['PM2.5', 'AQI']].mean().reset_index()

    logging.info(f" AQI processed | Years: {len(yearly)}")
    return yearly

# ==========================================
# PROCESS GEO
# ==========================================
def process_geo(df):
    df = df[df['Series'].str.contains("Forest cover", case=False)]

    df = df[['Year', 'Value']].dropna()
    df.rename(columns={'Value': 'Forest_Cover'}, inplace=True)

    df = df.groupby('Year')[['Forest_Cover']].mean().reset_index()
    df['Forest_MA'] = df['Forest_Cover'].rolling(2).mean()

    for year in df['Year'].unique():
        df[df['Year'] == year].to_csv(
            f"{OUTPUT_PATH}geo/geo_{year}_{timestamp}.csv", index=False
        )

    logging.info(f" Geospatial processed | Rows: {len(df)}")
    return df

# ==========================================
# MERGE DATA
# ==========================================
def merge_data(temp, aq, geo):
    common = set(temp['Year']) & set(aq['Year']) & set(geo['Year'])

    temp = temp[temp['Year'].isin(common)]
    aq = aq[aq['Year'].isin(common)]
    geo = geo[geo['Year'].isin(common)]

    merged = temp.merge(aq, on='Year').merge(geo, on='Year')

    merged.to_csv(f"{OUTPUT_PATH}merged/merged_{timestamp}.csv", index=False)

    logging.info(f" Data merged | Rows: {len(merged)}")
    return merged

# ==========================================
# METADATA LOGGING 
# ==========================================
def log_metadata(temp, aq, geo, merged):
    logging.info(f"Temp rows: {len(temp)}")
    logging.info(f"AQI rows: {len(aq)}")
    logging.info(f"Geo rows: {len(geo)}")
    logging.info(f"Merged rows: {len(merged)}")

# ==========================================
# SAVE TO POSTGRES 
# ==========================================
def save_to_postgres(temp, aq, geo, merged):
    try:
        temp.to_sql("temperature_data", engine, if_exists="replace", index=False)
        aq.to_sql("air_quality", engine, if_exists="replace", index=False)
        geo.to_sql("geospatial", engine, if_exists="replace", index=False)
        merged.to_sql("environment_merged", engine, if_exists="replace", index=False)

        #  Validate DB insert
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM environment_merged"))
            count = result.scalar()

        logging.info(f" Data stored in PostgreSQL | Rows: {count}")

    except Exception as e:
        logging.error(f"DB error: {e}")
        raise

# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline():
    start_time = datetime.now()

    create_folders()
    temp, aq, geo = load_data()

    temp_clean = process_temperature(temp)
    aq_clean = process_aqi(aq)
    geo_clean = process_geo(geo)

    #  VALIDATION
    validate_data(temp_clean, "Temperature")
    validate_data(aq_clean, "AQI")
    validate_data(geo_clean, "Geospatial")

    merged = merge_data(temp_clean, aq_clean, geo_clean)

    # METADATA
    log_metadata(temp_clean, aq_clean, geo_clean, merged)

    # DATABASE SAVE
    save_to_postgres(temp_clean, aq_clean, geo_clean, merged)

    print("Pipeline executed successfully!")

    logging.info(f"Pipeline completed in {datetime.now() - start_time}")


if __name__ == "__main__":
    run_pipeline()
    