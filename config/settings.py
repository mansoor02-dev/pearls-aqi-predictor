from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # API Keys (loaded from .env, never hardcoded)
    OPENWEATHER_API_KEY: str  # API For Backup 
    
    # Hopsworks
    HOPSWORKS_API_KEY: str
    HOPSWORKS_PROJECT_NAME: str = "pearl_aqi_feature_store"
    HOPSWORKS_HOST: str = "eu-west.cloud.hopsworks.ai"
    
    # Feature Store Config
    FEATURE_GROUP_NAME: str = "aqi_features"
    FEATURE_GROUP_VERSION: int = 1
    FEATURE_VIEW_NAME: str = "aqi_training_view"
    FEATURE_VIEW_VERSION: int = 1
    
    # Model Registry
    MODEL_NAME: str = "aqi_forecaster"
    
    # Data
    CITY: str = "Lahore" 
    RAW_DATA_PATH: Path = Path("data/raw")
    PROCESSED_DATA_PATH: Path = Path("data/processed")
    
    # Training
    FORECAST_HORIZON: int = 3  # Days ahead
    RANDOM_STATE: int = 42

settings: Settings = Settings()