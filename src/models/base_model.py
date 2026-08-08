# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger


class BaseAQIModel(ABC):
    """
    Abstract base class for all AQI forecasting models.
    Each instance predicts ONE forecast horizon's delta target
    (aqi_next_{horizon}d - european_aqi); train a separate instance per horizon.
    """

    def __init__(self, model_name: str, forecast_horizon: int = 1):
        self.model_name = model_name
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
        self.logger = setup_logger(f"{self.__class__.__name__}-{model_name}")

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, horizon: int = None, test_frac: float = 0.2) -> Dict[str, Any]:
        """Builds X/y/current_aqi splits for this model's forecast_horizon.
        Returns a dict with (at minimum): X_train, X_test, y_train_delta,
        y_test_delta, y_test_raw, current_aqi_test."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train_delta: np.ndarray,
              X_val: np.ndarray = None, y_val_delta: np.ndarray = None) -> Dict[str, float]:
        """Train on the DELTA target. Return training metrics (e.g. final_loss)."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, current_aqi: np.ndarray) -> np.ndarray:
        """Predicts AQI change, reconstructs to raw AQI (current_aqi + predicted
        delta). Returns an array of shape (n_samples,) — raw AQI for THIS
        model's horizon, not one column per day."""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test_raw: np.ndarray,
                 current_aqi_test: np.ndarray, y_naive: np.ndarray = None) -> Dict[str, float]:
        """Scores on the RAW AQI scale. Returns r2, mae, mse, rmse, and
        skill_vs_naive (1 - rmse/naive_rmse) when y_naive is supplied."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance (if available). Override in subclasses."""
        return {}