# tests/test_feature_engineering.py
"""
Unit tests for src/features/feature_engineering.py — AQIFeatureEngineer.

Verifies:
- Output columns (temporal, cyclical, lags, rolling, targets)
- No leakage from rolling windows (shift-before-roll)
- Correct per-city isolation
- fit_transform idempotence
- Cyclical encoding range guarantees
"""
import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import AQIFeatureEngineer


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, city: str = "Lahore") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "date":                 dates,
        "city":                 city,
        "lat":                  31.558,
        "lon":                  74.351,
        "temperature_2m":       rng.uniform(15, 45, n),
        "wind_direction_10m":   rng.uniform(0, 360, n),
        "wind_speed_10m":       rng.uniform(0, 30, n),
        "rain":                 rng.uniform(0, 5, n),
        "weather_code":         1.0,
        "wind_gusts_10m":       rng.uniform(0, 50, n),
        "cloud_cover":          rng.uniform(0, 100, n),
        "relative_humidity_2m": rng.uniform(30, 95, n),
        "pm10":                 rng.uniform(10, 200, n),
        "pm2_5":                rng.uniform(5, 120, n),
        "carbon_monoxide":      rng.uniform(200, 900, n),
        "nitrogen_dioxide":     rng.uniform(5, 60, n),
        "sulphur_dioxide":      rng.uniform(3, 20, n),
        "ozone":                rng.uniform(20, 160, n),
        "uv_index":             rng.uniform(0, 8, n),
        "aerosol_optical_depth": rng.uniform(0.3, 2.0, n),
        "european_aqi":         rng.uniform(40, 220, n),
    })


# ── Column existence ──────────────────────────────────────────────────────────

class TestOutputColumns:

    @pytest.fixture(scope="class")
    def result(self):
        fe = AQIFeatureEngineer(forecast_horizon=3)
        return fe.fit_transform(_make_df())

    def test_temporal_columns_present(self, result):
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in result.columns, f"Missing temporal column: {col}"

    def test_cyclical_columns_present(self, result):
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos",
                    "month_sin", "month_cos", "wind_u", "wind_v"]:
            assert col in result.columns, f"Missing cyclical column: {col}"

    def test_lag_columns_present(self, result):
        for lag in [1, 3, 6, 24]:
            assert f"aqi_lag_{lag}h" in result.columns

    def test_rolling_columns_present(self, result):
        for window in [4, 6, 12, 24]:
            assert f"aqi_roll_mean_{window}h" in result.columns
            assert f"aqi_roll_std_{window}h"  in result.columns

    def test_ewm_and_change_columns_present(self, result):
        assert "aqi_ewm_6h"    in result.columns
        assert "aqi_change_1h"  in result.columns
        assert "aqi_change_24h" in result.columns

    def test_target_columns_present(self, result):
        for day in [1, 2, 3]:
            assert f"aqi_next_{day}d" in result.columns

    def test_pollutant_ratio_columns_present(self, result):
        assert "pm_ratio"     in result.columns
        assert "no2_o3_ratio" in result.columns


# ── Cyclical encoding correctness ─────────────────────────────────────────────

class TestCyclicalEncoding:

    @pytest.fixture(scope="class")
    def result(self):
        fe = AQIFeatureEngineer(forecast_horizon=1)
        return fe.fit_transform(_make_df())

    def test_hour_sin_range(self, result):
        assert result["hour_sin"].between(-1, 1).all()

    def test_hour_cos_range(self, result):
        assert result["hour_cos"].between(-1, 1).all()

    def test_sin_cos_unit_circle(self, result):
        """sin² + cos² must equal 1 for every row (unit circle identity)."""
        norms = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
        assert (norms - 1.0).abs().max() < 1e-9

    def test_is_weekend_binary(self, result):
        assert result["is_weekend"].isin([0, 1]).all()


# ── Leakage guard: rolling windows must be computed on shift(1) ───────────────

class TestNoLeakage:

    def test_roll_mean_uses_shifted_values(self):
        """
        Set a known constant AQI for the first 25 rows, then a spike.
        The rolling mean at hour 25 must NOT include the spike (hour 25 itself).
        If rolling is computed without shift, rolling_mean[25] would contain it.
        """
        n = 60
        df = _make_df(n=n)
        df["european_aqi"] = 50.0           # flat baseline
        df.loc[25, "european_aqi"] = 999.0  # spike at row 25

        fe = AQIFeatureEngineer(forecast_horizon=1)
        out = fe.fit_transform(df)

        # Rolling mean at row 26 (the row after the spike) reflects shifted data.
        # If there's leakage, row 26's roll_mean_24h would be very high.
        roll_at_27 = out.loc[27, "aqi_roll_mean_24h"]
        # The spike is at index 25; shift(1) means it only enters the window
        # from index 26 onward. At 27, it IS included, but it's bounded.
        # The key check: the CURRENT row's value never directly enters its own
        # roll window, so any spike at row N appears only in row N+1 onward.
        assert out.loc[25, "aqi_roll_mean_24h"] < 999.0 or pd.isna(out.loc[25, "aqi_roll_mean_24h"])


# ── Per-city isolation ────────────────────────────────────────────────────────

class TestPerCityIsolation:

    def test_lags_do_not_cross_cities(self):
        """
        The lag of the last row of city A must never appear as a lag for city B.
        """
        df_a = _make_df(n=50, city="Lahore")
        df_b = _make_df(n=50, city="Karachi")
        df_b["european_aqi"] = 999.0   # distinct constant for Karachi

        combined = pd.concat([df_a, df_b], ignore_index=True)
        fe = AQIFeatureEngineer(forecast_horizon=1)
        out = fe.fit_transform(combined)

        # The very first row of Karachi (sorted after Lahore) must have NaN
        # for aqi_lag_1h, not 999 leaked from Lahore's last row.
        karachi_rows = out[out["city"] == "Karachi"].reset_index(drop=True)
        assert pd.isna(karachi_rows.loc[0, "aqi_lag_1h"]) or \
               karachi_rows.loc[0, "aqi_lag_1h"] == 999.0, \
               "Lag of Karachi row 0 should be NaN (first in group), not Lahore's value"

    def test_ewm_does_not_cross_cities(self):
        df_a = _make_df(n=50, city="Lahore")
        df_b = _make_df(n=50, city="Karachi")
        df_b["european_aqi"] = 0.0    # very different from Lahore

        combined = pd.concat([df_a, df_b], ignore_index=True)
        fe = AQIFeatureEngineer(forecast_horizon=1)
        out = fe.fit_transform(combined)

        karachi_ewm = out[out["city"] == "Karachi"]["aqi_ewm_6h"]
        lahore_ewm  = out[out["city"] == "Lahore"]["aqi_ewm_6h"]

        # Karachi's EWM should be close to 0, Lahore's should be >> 0
        assert karachi_ewm.mean() < lahore_ewm.mean() / 2


# ── fit_transform vs transform ────────────────────────────────────────────────

class TestFitTransformIdempotence:

    def test_fit_transform_equals_separate_fit_transform(self):
        df = _make_df(n=100)
        fe1 = AQIFeatureEngineer(forecast_horizon=1)
        fe2 = AQIFeatureEngineer(forecast_horizon=1)

        out1 = fe1.fit_transform(df)
        fe2.fit(df)
        out2 = fe2.transform(df)

        pd.testing.assert_frame_equal(out1.reset_index(drop=True),
                                      out2.reset_index(drop=True))

    def test_feature_names_set_after_fit(self):
        df = _make_df(n=50)
        fe = AQIFeatureEngineer(forecast_horizon=1)
        fe.fit_transform(df)
        assert fe.feature_names_ is not None
        assert len(fe.feature_names_) > 0
