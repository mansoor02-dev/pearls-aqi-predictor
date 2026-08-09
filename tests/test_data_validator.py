# tests/test_data_validator.py
"""
Unit tests for src/data/data_validator.py — DataValidator.

All tests are pure Python; no network or Hopsworks access needed.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.data_validator import DataValidator
from src.utils.exceptions import DataValidationError


# ── Helpers ────────────────────────────────────────────────────────────────────

def _good_df(n: int = 10) -> pd.DataFrame:
    """Minimal valid raw DataFrame."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-07-01", periods=n, freq="h")
    return pd.DataFrame({
        "date":                 dates,
        "city":                 "Lahore",
        "lat":                  31.558,
        "lon":                  74.351,
        "temperature_2m":       rng.uniform(25, 40, n),
        "wind_direction_10m":   rng.uniform(0, 360, n),
        "wind_speed_10m":       rng.uniform(0, 20, n),
        "rain":                 0.0,
        "weather_code":         1.0,
        "wind_gusts_10m":       rng.uniform(0, 30, n),
        "cloud_cover":          rng.uniform(0, 100, n),
        "relative_humidity_2m": rng.uniform(30, 90, n),
        "pm10":                 rng.uniform(10, 100, n),
        "pm2_5":                rng.uniform(5, 80, n),
        "carbon_monoxide":      rng.uniform(200, 700, n),
        "nitrogen_dioxide":     rng.uniform(5, 50, n),
        "sulphur_dioxide":      rng.uniform(3, 18, n),
        "ozone":                rng.uniform(30, 150, n),
        "uv_index":             rng.uniform(0, 6, n),
        "aerosol_optical_depth": rng.uniform(0.3, 1.5, n),
        "european_aqi":         rng.uniform(40, 180, n),
    })


# ── validate_raw_data ──────────────────────────────────────────────────────────

class TestValidateRawData:

    def test_passes_valid_dataframe(self):
        df = _good_df()
        validator = DataValidator()
        result = validator.validate_raw_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_raises_on_negative_pm2_5(self):
        df = _good_df()
        df.loc[0, "pm2_5"] = -5.0          # invalid: ge=0 violated
        validator = DataValidator()
        with pytest.raises(Exception):      # pydantic ValidationError or DataValidationError
            validator.validate_raw_data(df)

    def test_raises_on_aqi_above_500(self):
        df = _good_df()
        df.loc[0, "european_aqi"] = 501.0
        validator = DataValidator()
        with pytest.raises(Exception):
            validator.validate_raw_data(df)

    def test_raises_when_null_rate_exceeds_5_percent(self):
        df = _good_df(n=100)
        # Introduce NaN in 10% of rows for one column
        df.loc[:10, "pm10"] = np.nan
        validator = DataValidator()
        with pytest.raises((ValueError, DataValidationError)):
            validator.validate_raw_data(df)

    def test_accepts_occasional_null_below_threshold(self):
        """A single NaN in 100 rows is 1% — below the 5% threshold."""
        df = _good_df(n=100)
        df.loc[0, "pm10"] = np.nan
        validator = DataValidator()
        # Should not raise — returns cleaned DataFrame
        result = validator.validate_raw_data(df)
        assert result is not None


# ── build_reference_stats ─────────────────────────────────────────────────────

class TestBuildReferenceStats:

    def test_returns_dict_with_mean_and_std(self):
        df = _good_df(n=50)
        validator = DataValidator()
        stats = validator.build_reference_stats(df)
        assert isinstance(stats, dict)
        # Every numeric column should be in the stats
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            assert col in stats
            assert "mean" in stats[col]
            assert "std" in stats[col]

    def test_mean_matches_dataframe(self):
        df = _good_df(n=50)
        validator = DataValidator()
        stats = validator.build_reference_stats(df)
        assert abs(stats["european_aqi"]["mean"] - df["european_aqi"].mean()) < 1e-9


# ── validate_features ─────────────────────────────────────────────────────────

class TestValidateFeatures:

    def test_passes_without_reference_stats(self):
        df = _good_df()
        validator = DataValidator()
        # No reference_stats → should return the df unchanged (just logs)
        result = validator.validate_features(df)
        assert len(result) == len(df)

    def test_returns_dataframe_when_no_drift(self):
        df = _good_df(n=50)
        validator = DataValidator()
        ref_stats = validator.build_reference_stats(df)
        result = validator.validate_features(df, reference_stats=ref_stats)
        assert len(result) == len(df)

    def test_does_not_raise_on_drift(self, capsys):
        """
        Drift detection should warn (print/log), NOT raise an exception —
        a real smog event is expected to drift pollutant columns.
        """
        ref_df = _good_df(n=50)
        validator = DataValidator()
        ref_stats = validator.build_reference_stats(ref_df)

        # Massively drift european_aqi
        drifted_df = _good_df(n=50)
        drifted_df["european_aqi"] = 490.0

        # Should NOT raise
        result = validator.validate_features(drifted_df, reference_stats=ref_stats)
        assert result is not None
