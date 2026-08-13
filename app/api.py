# app/api.py
import os
from datetime import datetime, timedelta
from typing import Optional

import hopsworks
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from config.settings import settings
from src.data.api_client import APIClientFactory
from src.data.data_validator import DataValidator
from src.features.feature_engineering import AQIFeatureEngineer
from src.models.model_registry import HopsworksModelRegistry
from src.models.sklearn_models import SklearnAQIModel
from src.models.deep_learning import FeedForwardAQIModel, LSTMAQIModel
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

class AQIPredictionRequest(BaseModel):
    city: str = Field(...)
    days: int = Field(3, ge=1, le=3)


class DayPrediction(BaseModel):
    day: int
    predicted_aqi: float
    confidence_interval: Optional[list] = None
    model_used: str
    model_version: int


class AQIPredictionResponse(BaseModel):
    city: str
    current_aqi: float
    predictions: list[DayPrediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    project = hopsworks.login(
        api_key_value=settings.HOPSWORKS_API_KEY,
        project=settings.HOPSWORKS_PROJECT_NAME,
        host=settings.HOPSWORKS_HOST,
    )
    app.state.mr = HopsworksModelRegistry(project)
    app.state.client = APIClientFactory.get_primary_client()
    app.state.validator = DataValidator()
    yield 

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def _load_production_model(mr: HopsworksModelRegistry, horizon: int):
    from src.inference import load_production_model
    instance, name, version, rmse, _ = load_production_model(mr, horizon)
    return instance, version, rmse


def _latest_features(client, validator, city: str) -> pd.DataFrame:
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS + 1)).strftime("%Y-%m-%d")

    merged_df = client.fetch_merged_historical(city, start_date, end_date)
    validated_df = validator.validate_raw_data(merged_df)

    engineered_df = AQIFeatureEngineer(forecast_horizon=settings.FORECAST_HORIZON).fit_transform(validated_df)

    return engineered_df


@app.post("/predict", response_model=AQIPredictionResponse)
def predict_aqi(request: AQIPredictionRequest):
    if request.city.lower() != settings.CITY.lower():
        raise HTTPException(
            status_code=400,
            detail=f"Only '{settings.CITY}' is currently supported — models are trained on a single city's data.",
        )

    engineered_df = _latest_features(app.state.client, app.state.validator, request.city)
    current_aqi = float(engineered_df["european_aqi"].iloc[-1])

    predictions = []
    for day in range(1, request.days + 1):
        model, version, rmse = _load_production_model(app.state.mr, horizon=day)
        if model is None:
            logger.warning(f"No production model for horizon {day}d yet — skipping")
            continue

        feature_cols = model.feature_names_
        latest_rows = engineered_df[feature_cols]

        if isinstance(model, LSTMAQIModel):
            seq_len = model.sequence_length
            if len(latest_rows) < seq_len:
                logger.warning(f"Not enough history for LSTM (h={day}) — need {seq_len} rows, have {len(latest_rows)}")
                continue
            X_input = latest_rows.iloc[-seq_len:].to_numpy().reshape(1, seq_len, -1)
        else:
            X_input = latest_rows.iloc[[-1]]

        pred_raw = float(model.predict(X_input, current_aqi)[0])

        confidence_interval = [pred_raw - 1.96 * rmse, pred_raw + 1.96 * rmse] if rmse else None

        predictions.append(DayPrediction(
            day=day, predicted_aqi=pred_raw, confidence_interval=confidence_interval,
            model_used=model.model_name, model_version=version,
        ))

    if not predictions:
        raise HTTPException(status_code=503, detail="No production models available yet — run the training pipeline first.")

    return AQIPredictionResponse(city=request.city, current_aqi=current_aqi, predictions=predictions)