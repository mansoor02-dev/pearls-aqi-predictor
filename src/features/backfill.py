import sys
from pathlib import Path

# 1. Resolve project root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config"

# 2. Add PROJECT_ROOT to Python path BEFORE importing local modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import hopsworks

from config.settings import settings
from src.utils.logger import setup_logger
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.features.feature_engineering import AQIFeatureEngineer
from src.features.feature_store import HopsworksFeatureStore


logger = setup_logger("BackfillScript")


def backfill_historical_data(start_date: str, end_date: str):
    """
    Fetch, validate, and engineer features for the full historical window
    in one pass, then insert into the feature store.
    """
    logger.info(f"Starting backfill from {start_date} to {end_date}")

    client = APIClientFactory.get_primary_client()
    validator = DataValidator()

    aqi_records = client.fetch_historical(
        city=settings.CITY, start_date=start_date, end_date=end_date,
    )
    weather_records = client.fetch_historical_weather(
        city=settings.CITY, start_date=start_date, end_date=end_date,
    )
    
    aqi_df = pd.DataFrame(aqi_records)
    weather_df = pd.DataFrame(weather_records)
    
    merged_df = pd.merge(weather_df, aqi_df, on=['date', 'city', 'lat', 'lon'])
    logger.info(f"Validating {len(merged_df)} raw rows")
    validated_df = validator.validate_raw_data(merged_df)

    logger.info("Engineering features")
    feature_engineer = AQIFeatureEngineer(forecast_horizon=settings.FORECAST_HORIZON)
    engineered_df = feature_engineer.fit_transform(validated_df)

    logger.info(f"Validating {len(engineered_df)} engineered rows")
    engineered_df = validator.validate_features(engineered_df)

    project = hopsworks.login(
        api_key_value=settings.HOPSWORKS_API_KEY,
        project=settings.HOPSWORKS_PROJECT_NAME,
        host=settings.HOPSWORKS_HOST,
    )
    fs = HopsworksFeatureStore(project)
    
    fg = fs.create_or_get_feature_group(
        name=settings.FEATURE_GROUP_NAME,
        version=settings.FEATURE_GROUP_VERSION,
        primary_key=["city", "date"],
        event_time="date",
    )

    feature_cols = [c for c in engineered_df.columns if not c.startswith("aqi_next_")]
    engineered_df = engineered_df.dropna(subset=feature_cols)
    logger.info(f"Dropped rows with incomplete features — {len(engineered_df)} rows remain")

    fs.insert_features(fg, engineered_df)

    training_data = fs.get_training_data(
        settings.FEATURE_VIEW_NAME, start_date=start_date, end_date=end_date
    )
    return training_data