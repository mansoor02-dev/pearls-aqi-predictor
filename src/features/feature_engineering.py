# Design Pattern: Scikit-learn Transformer (fit/transform)
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

LAG_HOURS = [1, 3, 6, 24]
ROLLING_WINDOWS = [4, 6, 12, 24]


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

        # 2. Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df['wind_u'] = -df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))
        df['wind_v'] = -df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))

        # 3. Pollutant interaction features
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)          # pm10 never near 0 — safe as-is
        df["no2_o3_ratio"] = df["nitrogen_dioxide"] / df["ozone"].clip(lower=5.0)

        # Continuous transition signals
        df['temp_trend_7d'] = df['temperature_2m'].rolling(24*7).mean().diff(24*7)  # is it warming/cooling fast?
        df['rain_accum_7d'] = df['rain'].rolling(24*7).sum()   # monsoon "arriving" shows up here

        g = df.groupby("city")["european_aqi"]

        # 4. Lag features
        for lag in LAG_HOURS:
            df[f"aqi_lag_{lag}h"] = g.shift(lag)

        # 5. Rolling statistics — shift(1) first, see the note above on why
        shifted = g.shift(1)
        for window in ROLLING_WINDOWS:
            df[f"aqi_roll_mean_{window}h"] = shifted.groupby(df["city"]).rolling(window).mean().reset_index(level=0, drop=True)
            df[f"aqi_roll_std_{window}h"] = shifted.groupby(df["city"]).rolling(window).std().reset_index(level=0, drop=True)

        # 6. Exponential rolling statistic — weights recent hours more than a flat window average
        df["aqi_ewm_6h"] = df["european_aqi"].ewm(span=6).mean()

        # 7. Rate of change
        df["aqi_change_1h"] = g.shift(1).diff(1)
        df["aqi_change_24h"] = g.shift(1).diff(24)

        # 8. Target creation — one column per day out to forecast_horizon
        for day in range(1, self.forecast_horizon + 1):
            df[f"aqi_next_{day}d"] = df.groupby("city")["european_aqi"].shift(-day * 24)

        self.feature_names_ = [c for c in df.columns if c not in ["date", "city"]]
        return df

    def get_feature_names_out(self, input_features=None) -> list:
        return self.feature_names_