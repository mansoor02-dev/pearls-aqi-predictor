# src/inference.py — shared by app/api.py and app/dashboard.py
from datetime import datetime, timedelta

import pandas as pd
import hopsworks

from config.settings import settings
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.features.feature_engineering import AQIFeatureEngineer
from src.models.model_registry import HopsworksModelRegistry
from src.models.sklearn_models import SklearnAQIModel
from src.models.deep_learning import FeedForwardAQIModel, LSTMAQIModel
from src.utils.logger import setup_logger
from src.utils.hopsworks_utils import login_hopsworks


logger = setup_logger(__name__)

CANDIDATE_MODELS = {
    "linear": ("linear", SklearnAQIModel, "joblib"),
    "random_forest": ("random_forest", SklearnAQIModel, "joblib"),
    "xgboost": ("xgboost", SklearnAQIModel, "joblib"),
    "ffn": (None, FeedForwardAQIModel, "pt"),
    "lstm": (None, LSTMAQIModel, "pt"),
}

LOOKBACK_DAYS = 16


def get_hopsworks_registry() -> HopsworksModelRegistry:
    project = login_hopsworks()
    return HopsworksModelRegistry(project)


def load_production_model(mr: HopsworksModelRegistry, horizon: int):
    best_candidate = None
    best_rmse = float("inf")

    for name, (model_type, cls, ext) in CANDIDATE_MODELS.items():
        registry_name = f"aqi_{name}_h{horizon}"
        try:
            hw_model = mr.get_production_model(registry_name)
            if hw_model is None:
                continue

            metrics = getattr(hw_model, "training_metrics", {}) or {}
            rmse = metrics.get("rmse")
            current_rmse = float(rmse) if (rmse is not None and isinstance(rmse, (int, float))) else 999999.0

            if current_rmse < best_rmse:
                model_dir = hw_model.download()
                instance = cls(registry_name, model_type, forecast_horizon=horizon) if model_type \
                    else cls(registry_name, forecast_horizon=horizon)
                instance.load(f"{model_dir}/{name}_h{horizon}.{ext}")
                best_candidate = (instance, name, hw_model.version, rmse, model_dir)
                best_rmse = current_rmse
        except Exception as e:
            logger.warning(f"Could not load candidate model '{registry_name}': {e}")
            continue

    if best_candidate is not None:
        return best_candidate

    return None, None, None, None, None


def get_latest_features(city: str) -> pd.DataFrame:
    client = APIClientFactory.get_primary_client()
    validator = DataValidator()

    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS + 1)).strftime("%Y-%m-%d")


    merged_df = client.fetch_merged_historical(city, start_date, end_date)
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