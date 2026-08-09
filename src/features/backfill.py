"""
src/features/backfill.py

Core data ingestion and feature generation pipeline.
Used by backfill scripts and scheduled feature pipeline updates.
"""
import pandas as pd

from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.hopsworks_utils import login_hopsworks
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.features.feature_engineering import AQIFeatureEngineer
from src.features.feature_store import HopsworksFeatureStore

logger = setup_logger("BackfillScript")


def backfill_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch, validate, engineer features, and upload to Hopsworks Feature Store.
    """
    logger.info(f"Starting feature ingestion from {start_date} to {end_date}")

    client = APIClientFactory.get_primary_client()
    validator = DataValidator()

    merged_df = client.fetch_merged_historical(
        city=settings.CITY, start_date=start_date, end_date=end_date
    )
    logger.info(f"Validating {len(merged_df)} raw rows")
    validated_df = validator.validate_raw_data(merged_df)

    logger.info("Engineering features")
    feature_engineer = AQIFeatureEngineer(forecast_horizon=settings.FORECAST_HORIZON)
    engineered_df = feature_engineer.fit_transform(validated_df)

    logger.info(f"Validating {len(engineered_df)} engineered rows")
    engineered_df = validator.validate_features(engineered_df)

    feature_cols = [c for c in engineered_df.columns if not c.startswith("aqi_next_")]
    engineered_df = engineered_df.dropna(subset=feature_cols)
    logger.info(f"Filtered incomplete feature rows — {len(engineered_df)} valid rows remain")

    project = login_hopsworks()
    fs = HopsworksFeatureStore(project)

    fg = fs.create_or_get_feature_group(
        name=settings.FEATURE_GROUP_NAME,
        version=settings.FEATURE_GROUP_VERSION,
        primary_key=["city", "date"],
        event_time="date",
    )

    fs.insert_features(fg, engineered_df, wait_for_job=True)
    logger.info(f"Successfully completed ingestion! {len(engineered_df)} feature rows stored in Hopsworks.")
    return engineered_df