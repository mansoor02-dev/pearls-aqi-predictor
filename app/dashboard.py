# src/inference.py — shared by app/api.py and app/dashboard.py
from datetime import datetime, timedelta

import pandas as pd
import hopsworks

from config.settings import settings
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.features.feature_engineering import AQIFeatureEngineer
from src.models.model_registry import HopsworksModelRegistry
from src.models.sklearn_model import SklearnAQIModel
from src.models.deeplearning_model import FeedForwardAQIModel, LSTMAQIModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CANDIDATE_MODELS = {
    "linear": ("linear", SklearnAQIModel, "joblib"),
    "random_forest": ("random_forest", SklearnAQIModel, "joblib"),
    "xgboost": ("xgboost", SklearnAQIModel, "joblib"),
    "ffn": (None, FeedForwardAQIModel, "pt"),
    "lstm": (None, LSTMAQIModel, "pt"),
}

LOOKBACK_DAYS = 7


def get_hopsworks_registry() -> HopsworksModelRegistry:
    project = hopsworks.login(
        api_key_value=settings.HOPSWORKS_API_KEY,
        project=settings.HOPSWORKS_PROJECT_NAME,
        host=settings.HOPSWORKS_HOST,
    )
    return HopsworksModelRegistry(project)


def load_production_model(mr: HopsworksModelRegistry, horizon: int):
    for name, (model_type, cls, ext) in CANDIDATE_MODELS.items():
        registry_name = f"aqi_{name}_h{horizon}"
        hw_model = mr.get_production_model(registry_name)
        if hw_model is None:
            continue

        model_dir = hw_model.download()
        instance = cls(registry_name, model_type, forecast_horizon=horizon) if model_type \
            else cls(registry_name, forecast_horizon=horizon)
        instance.load(f"{model_dir}/{name}_h{horizon}.{ext}")

        rmse = (hw_model.training_metrics or {}).get("rmse") if hasattr(hw_model, "training_metrics") else None
        return instance, name, hw_model.version, rmse

    return None, None, None, None


def get_latest_features(city: str) -> pd.DataFrame:
    client = APIClientFactory.get_primary_client()
    validator = DataValidator()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    aqi_df = pd.DataFrame(client.fetch_historical(city, start_date, end_date))
    weather_df = pd.DataFrame(client.fetch_historical_weather(city, start_date, end_date))
    merged_df = pd.merge(weather_df, aqi_df, on=["date", "city", "lat", "lon"])

    validated_df = validator.validate_raw_data(merged_df)
    return AQIFeatureEngineer(forecast_horizon=settings.FORECAST_HORIZON).fit_transform(validated_df)


def predict_horizon(model, engineered_df: pd.DataFrame, current_aqi: float):
    feature_cols = model.feature_names_
    latest_rows = engineered_df[feature_cols]

    if isinstance(model, LSTMAQIModel):
        seq_len = model.sequence_length
        if len(latest_rows) < seq_len:
            return None
        X_input = latest_rows.iloc[-seq_len:].to_numpy().reshape(1, seq_len, -1)
    else:
        X_input = latest_rows.iloc[[-1]]

    return float(model.predict(X_input, current_aqi)[0])