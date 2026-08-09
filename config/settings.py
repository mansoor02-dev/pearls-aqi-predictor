from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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

    # Alerts — all optional; system logs to file even when none are configured
    SLACK_WEBHOOK_URL: Optional[str] = None   # set to enable Slack alerts
    ALERT_EMAIL_TO: Optional[str] = None      # recipient, e.g. "ops@example.com"
    ALERT_EMAIL_FROM: Optional[str] = None    # sender
    SMTP_HOST: Optional[str] = None           # e.g. "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None

    # Monitoring
    DRIFT_THRESHOLD: float = 0.3   # share of drifted features that triggers an alert
    AQI_ALERT_THRESHOLD: int = 150  # AQI level considered Unhealthy


settings: Settings = Settings()