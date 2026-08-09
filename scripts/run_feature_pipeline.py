from datetime import datetime, timedelta

import pandas as pd
import hopsworks

from src.features.feature_engineering import AQIFeatureEngineer
from src.features.feature_store import HopsworksFeatureStore
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("FeaturePipeline")

LOOKBACK_DAYS = 5


def update_features(lookback_days: int = LOOKBACK_DAYS):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    logger.info(f"Starting feature pipeline update from {start_date} to {end_date}")

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

    feature_cols = [c for c in engineered_df.columns if not c.startswith("aqi_next_")]
    engineered_df = engineered_df.dropna(subset=feature_cols)
    logger.info(f"{len(engineered_df)} rows with complete features to upsert")

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
    fs.insert_features(fg, engineered_df)


if __name__ == "__main__":
    update_features()