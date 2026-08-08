import joblib
import pandas as pd
import numpy as np
from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

from src.utils.logger import setup_logger
from src.models.base_model import BaseAQIModel

logger = setup_logger(__name__)


class SklearnAQIModel(BaseAQIModel):
    def __init__(self, model_name: str, model_type: str, forecast_horizon: int = 1):
        super().__init__(model_name, forecast_horizon=forecast_horizon)

        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42,
                min_samples_split=5, n_jobs=-1,
            )
        elif model_type == "linear":
            self.model = Ridge(alpha=0.1, random_state=42, solver="auto")
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                n_estimators=1000, learning_rate=0.03, max_depth=4,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=5.0,
                early_stopping_rounds=50, random_state=42, n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type {model_type}")

        self.model_type = model_type
        self.scaler = RobustScaler()

    def preprocess(self, df: pd.DataFrame, horizon: int = None, test_frac: float = 0.2) -> dict:
        horizon = horizon or self.forecast_horizon
        BASE_DROP_COLS = ["date", "city", "lat", "lon"]
        target_col = f"aqi_next_{horizon}d"
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' not found — was AQIFeatureEngineer built "
                              f"with forecast_horizon >= {horizon}?")

        d = df.dropna(subset=[target_col]).copy()
        d = d.dropna()
        d["target_delta"] = d[target_col] - d["european_aqi"]

        other_horizon_cols = [c for c in d.columns if c.startswith("aqi_next_") and c != target_col]
        drop_cols = BASE_DROP_COLS + [target_col, "target_delta"] + other_horizon_cols

        X = d.drop(columns=drop_cols)
        y_delta = d["target_delta"]
        y_raw = d[target_col]
        current_aqi = d["european_aqi"]

        split_idx = int(len(X) * (1 - test_frac))
        self.feature_names_ = X.columns.tolist()   # needed by get_feature_importance() below
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        if self.model_type == 'linear':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train_delta": y_delta.iloc[:split_idx], "y_test_delta": y_delta.iloc[split_idx:],
            "y_train_raw": y_raw.iloc[:split_idx], "y_test_raw": y_raw.iloc[split_idx:],
            "current_aqi_train": current_aqi.iloc[:split_idx],
            "current_aqi_test": current_aqi.iloc[split_idx:],
        }

    def train(self, X_train, y_train_delta, X_val=None, y_val_delta=None) -> Dict[str, float]:
        if self.model_type == "xgboost" and X_val is not None:
            self.model.fit(X_train, y_train_delta, eval_set=[(X_val, y_val_delta)], verbose=False)
        else:
            self.model.fit(X_train, y_train_delta)
        self.is_trained = True
        return {"status": "trained"}

    def predict(self, X, current_aqi) -> np.ndarray:
        """Returns RAW AQI predictions, not delta — current_aqi + predicted_delta."""
        pred_delta = self.model.predict(X)
        return current_aqi.values + pred_delta if hasattr(current_aqi, "values") else current_aqi + pred_delta

    def evaluate(self, X_test, y_test_raw, current_aqi_test, y_naive=None) -> Dict[str, float]:
        pred_raw = self.predict(X_test, current_aqi_test)

        r2 = r2_score(y_test_raw, pred_raw)
        mae = mean_absolute_error(y_test_raw, pred_raw)
        mse = mean_squared_error(y_test_raw, pred_raw)
        rmse = root_mean_squared_error(y_test_raw, pred_raw)

        skill = None
        if y_naive is not None:
            naive_rmse = root_mean_squared_error(y_test_raw, y_naive)
            skill = 1 - (rmse / naive_rmse) if naive_rmse > 0 else np.nan

        self.metrics = {
            "model": self.model_name, "horizon": f"{self.forecast_horizon}d",
            "r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "skill_vs_naive": skill,
        }
        skill_str = f"   Skill: {skill:+.3f}" if skill is not None else ""
        logger.info(f"{self.model_name:32s}  R2: {r2:.4f}   MAE: {mae:.3f}   RMSE: {rmse:.3f}{skill_str}")
        return self.metrics

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or self.feature_names_ is None:
            return {}
        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            values = np.abs(self.model.coef_)
        else:
            return {}
        return dict(zip(self.feature_names_, values))

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "model_type": self.model_type,
                     "forecast_horizon": self.forecast_horizon,
                     "feature_names_": getattr(self, "feature_names_", None)}, path)
        logger.info(f"Saved {self.model_name} to {path}")

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self.model = state["model"]
        self.model_type = state["model_type"]
        self.forecast_horizon = state["forecast_horizon"]
        self.feature_names_ = state.get("feature_names_")
        self.is_trained = True
        logger.info(f"Loaded {self.model_name} from {path}")