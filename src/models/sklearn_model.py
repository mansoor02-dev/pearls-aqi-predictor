import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import joblib # from making process parallel

from src.utils.logger import setup_logger
from src.models.base_model import BaseAQIModel

class SklearnAQIModel(BaseAQIModel):
    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name)
        
        if model_type == "random_forest":
            base_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                min_samples_split=5,
                n_jobs=-1
            )
        elif model_type == "linear":
            base_model = Ridge(alpha=0.1, random_state=42, solver='auto')
        else:
            raise ValueError(f"Unknown model type {model_type}")
        
        self.model = MultiOutputRegressor(base_model)
        self.model_type = model_type
        
    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num_cols = df.select_dtypes(include="number")
        cat_cols = df.select_dtypes(include="object")