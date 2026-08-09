import pandas as pd
import hopsworks

from src.features.feature_engineering import AQIFeatureEngineer
from src.features.feature_store import HopsworksFeatureStore
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.utils.logger import setup_logger
from config.settings import settings


logger = setup_logger("FeaturePipeline")

def current_features(start_date:str, end_date: str):
    logger.info(f"Starting backfill from {start_date} to {end_date}")

    client = APIClientFactory.get_primary_client()
    validator = DataValidator()

    aqi_records = client.fetch_current(
        city=settings.CITY, start_date=start_date, end_date=end_date,
    )
    weather_records = client.fetch_current_weather(
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
    fs.append_features(fg, engineered_df)

if __name__ == "__main__":
    current_features(start_date="2022-01-01", end_date="2026-07-31")