# Design Pattern: Scikit-learn Transformer (fit/transform)
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

LAG_HOURS = [1, 3]
ROLLING_WINDOWS = [6, 12]


class AQIFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for AQI feature engineering.
    Follows sklearn API for pipeline compatibility.
    """

    def __init__(self, forecast_horizon: int = 3):
        self.forecast_horizon = forecast_horizon
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.sort_values(["city", "date"]).reset_index(drop=True)

        # 1. Temporal features
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # 2. Cyclical encoding (CRITICAL — don't use raw hour 0-23)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Group once, reused for all per-city time-series ops below
        g = df.groupby("city")["european_aqi"]

        # 3. Lag features
        for lag in LAG_HOURS:
            df[f"aqi_lag_{lag}h"] = g.shift(lag)

        # 4. Rolling statistics (shift(1) first so window never includes current row)
        for window in ROLLING_WINDOWS:
            df[f"aqi_roll_mean_{window}h"] = (
                df.groupby("city")["european_aqi"]
                .transform(lambda x: x.shift(1).rolling(window).mean())
            )
            df[f"aqi_roll_std_{window}h"] = (
                df.groupby("city")["european_aqi"]
                .transform(lambda x: x.shift(1).rolling(window).std())
            )

        # 5. Rate of change
        df["aqi_change_1h"] = g.shift(1).diff(1)
        df["aqi_change_24h"] = g.shift(1).diff(24)

        # 7. Target creation (shift AQI forward, per city)
        for day in range(1, self.forecast_horizon + 1):
            df[f"aqi_next_{day}d"] = df.groupby("city")["european_aqi"].shift(-day * 24)
            
        self.feature_names_ = [c for c in df.columns if c not in ["date", "city"]]
        return df

    def get_feature_names_out(self, input_features=None) -> list:
        return self.feature_names_