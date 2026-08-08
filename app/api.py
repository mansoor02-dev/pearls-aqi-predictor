from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.features.feature_store import HopsworksFeatureStore

class AQIPredictionRequest:
    city: str = Field(...)
    days: int = Field(3, ge=1, le=3)

class AQIPredictionResponse:
    city: str
    current_aqi: float
    predictions: list
    confidence_intervals: list
    model_version: str

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "1.2.3"}

@app.post("/predict", response_class=AQIPredictionResponse)
def predict_aqi(request: AQIPredictionResponse):
    fg = HopsworksFeatureStore.create_or_get_feature_group()
    
    pass