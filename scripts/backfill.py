from datetime import datetime
import pandas as pd

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

    # Fetch the ENTIRE range in a single call each — not looped per day.
    # Lag/rolling features need continuous history to compute correctly.
    aqi_records = client.fetch_historical(
        city="Lahore", lat=31.558, lon=74.351,
        start_date=start_date, end_date=end_date,
    )
    weather_records = client.fetch_historical_weather(
        city="Lahore", lat=31.558, lon=74.351,
        start_date=start_date, end_date=end_date,
    )

    aqi_df = pd.DataFrame(aqi_records)
    weather_df = pd.DataFrame(weather_records)
    raw_df = pd.merge(aqi_df, weather_df, on=["date", "city", "lat", "lon"])

    logger.info(f"Validating {len(raw_df)} raw rows")
    validated_df = validator.validate_raw_data(raw_df)

    logger.info("Engineering features")
    feature_engineer = AQIFeatureEngineer(forecast_horizon=settings.FORECAST_HORIZON)
    engineered_df = feature_engineer.fit_transform(validated_df)

    logger.info(f"Validating {len(engineered_df)} engineered rows")
    engineered_df = validator.validate_features(engineered_df)

    fs = HopsworksFeatureStore(
        api_key=settings.HOPSWORKS_API_KEY,
        project_name=settings.HOPSWORKS_PROJECT_NAME,
        host=settings.HOPSWORKS_HOST,
    )
    fg = fs.create_or_get_feature_group(
        name=settings.FEATURE_GROUP_NAME,
        version=settings.FEATURE_GROUP_VERSION,
        primary_key=["city", "date"],
        event_time="date",
    )
    fs.insert_features(fg, engineered_df)

    training_data = fs.get_training_data(
        settings.FEATURE_VIEW_NAME, start_date=start_date, end_date=end_date
    )
    return training_data