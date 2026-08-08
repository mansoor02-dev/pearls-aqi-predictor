from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

class BaseAQIModel(ABC):
    """
    Abstract base class for all AQI forecasting models.
    Enforces consistent API across sklearn and deep learning models.
    """
    
    def __init__(self, model_name: str, forecast_horizon: int = 1):
        self.model_name = model_name
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.logger = setup_logger(f"{self.__class__.__name__}-{model_name}")
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame to model-ready arrays.
        Returns: (X, y) where y has shape (n_samples, forecast_horizon)
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train the model. Return training metrics.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict AQI for next N days.
        Returns: array of shape (n_samples, forecast_horizon)
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate RMSE, MAE, R² for each forecast day.
        Returns: {
            'rmse_day1': ..., 'rmse_day2': ..., 'rmse_day3': ...,
            'mae_day1': ..., 'mae_avg': ..., 'r2_avg': ...
        }
        """
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
        """
        Return feature importance (if available).
        Override in subclasses.
        """
        return {}