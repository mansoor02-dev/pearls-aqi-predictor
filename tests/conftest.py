# tests/conftest.py
"""
Shared pytest fixtures for the Pearls AQI Predictor test suite.

All fixtures that need Hopsworks or the live Open-Meteo API are
stubbed with pytest-mock / simple fakes so tests run fully offline
in CI (no HOPSWORKS_API_KEY required).
"""
import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 48, city: str = "Lahore") -> pd.DataFrame:
    """
    Synthetic raw DataFrame that looks like a merged AQI + weather fetch.
    48 rows = 2 days of hourly data — enough for all lag/rolling features.
    """
    dates = pd.date_range("2026-07-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date":                 dates,
        "city":                 city,
        "lat":                  31.558,
        "lon":                  74.351,
        # weather
        "temperature_2m":       rng.uniform(25, 42, n),
        "wind_direction_10m":   rng.uniform(0, 360, n),
        "wind_speed_10m":       rng.uniform(0, 30, n),
        "rain":                 rng.uniform(0, 5, n),
        "weather_code":         rng.integers(0, 3, n).astype(float),
        "wind_gusts_10m":       rng.uniform(0, 50, n),
        "cloud_cover":          rng.uniform(0, 100, n),
        "relative_humidity_2m": rng.uniform(30, 95, n),
        # AQI pollutants
        "pm10":                 rng.uniform(10, 200, n),
        "pm2_5":                rng.uniform(5, 120, n),
        "carbon_monoxide":      rng.uniform(200, 900, n),
        "nitrogen_dioxide":     rng.uniform(5, 60, n),
        "sulphur_dioxide":      rng.uniform(3, 20, n),
        "ozone":                rng.uniform(20, 160, n),
        "uv_index":             rng.uniform(0, 8, n),
        "aerosol_optical_depth": rng.uniform(0.3, 2.0, n),
        "european_aqi":         rng.uniform(40, 200, n),
    })


def _make_engineered_df(n: int = 200, horizon: int = 3) -> pd.DataFrame:
    """
    Run AQIFeatureEngineer on synthetic raw data.
    `n` must be large enough that lag/rolling features are non-NaN after dropna.
    """
    from src.features.feature_engineering import AQIFeatureEngineer
    raw = _make_raw_df(n=n)
    fe = AQIFeatureEngineer(forecast_horizon=horizon)
    return fe.fit_transform(raw)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df():
    """Single-city 48-row raw DataFrame."""
    return _make_raw_df(n=48)


@pytest.fixture(scope="session")
def large_raw_df():
    """200-row raw DataFrame — needed for rolling windows + targets."""
    return _make_raw_df(n=200)


@pytest.fixture(scope="session")
def engineered_df(large_raw_df):
    """Engineered features from the 200-row raw DataFrame (horizon=3)."""
    from src.features.feature_engineering import AQIFeatureEngineer
    fe = AQIFeatureEngineer(forecast_horizon=3)
    return fe.fit_transform(large_raw_df)


@pytest.fixture
def sklearn_model():
    """A freshly initialised (untrained) Random Forest SklearnAQIModel."""
    from src.models.sklearn_models import SklearnAQIModel
    return SklearnAQIModel(name="aqi_random_forest_h1",
                           model_type="random_forest", forecast_horizon=1)


@pytest.fixture
def trained_sklearn_model(engineered_df):
    """A trained Random Forest model, ready for predict/evaluate calls."""
    from src.models.sklearn_models import SklearnAQIModel
    model = SklearnAQIModel(name="aqi_random_forest_h1",
                            model_type="random_forest", forecast_horizon=1)
    X, y = model.preprocess(engineered_df, target_day=1)
    model.train(X, y)
    return model
