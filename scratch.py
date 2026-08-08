# scratch.py, run with: uv run python scratch.py
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.features.feature_engineering import AQIFeatureEngineer
import pandas as pd

client = APIClientFactory.get_primary_client()
aqi = client.fetch_historical("Lahore", "2026-07-01", "2026-07-07")
weather = client.fetch_historical_weather("Lahore", "2026-07-01", "2026-07-07")
merged = pd.merge(pd.DataFrame(weather), pd.DataFrame(aqi), on=["date","city","lat","lon"])

validator = DataValidator()
validated = validator.validate_raw_data(merged)

fe = AQIFeatureEngineer(forecast_horizon=3)
features = fe.fit_transform(validated)
print(features.shape, features.columns.tolist())