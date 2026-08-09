# tests/test_sklearn_models.py
"""
Unit tests for src/models/sklearn_models.py — SklearnAQIModel.

Tests cover:
- preprocess: shapes, delta-target calculation, feature list
- train: model fits, feature_names_ set, no crash
- predict: output shape, returns float, delta is added back to current AQI
- evaluate: returns dict with expected metric keys
- save / load round-trip (uses tmp_path)
- unsupported model_type raises clearly
"""
import numpy as np
import pandas as pd
import pytest

from src.models.sklearn_models import SklearnAQIModel


# ── Helpers ───────────────────────────────────────────────────────────────────

MODEL_TYPES = ["linear", "random_forest", "xgboost"]


def _engineered_df(n: int = 200) -> pd.DataFrame:
    """Small engineered DataFrame via conftest helper."""
    from tests.conftest import _make_engineered_df
    return _make_engineered_df(n=n, horizon=3)


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:

    @pytest.fixture(scope="class")
    def data(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        return model, model.preprocess(df, target_day=1)

    def test_returns_x_and_y(self, data):
        _, (X, y) = data
        assert X is not None and y is not None

    def test_x_is_dataframe(self, data):
        _, (X, y) = data
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series_or_array(self, data):
        _, (X, y) = data
        assert hasattr(y, "__len__")

    def test_no_target_column_in_x(self, data):
        _, (X, y) = data
        for col in X.columns:
            assert not col.startswith("aqi_next_"), \
                f"Target column '{col}' leaked into features"

    def test_x_y_same_length(self, data):
        _, (X, y) = data
        assert len(X) == len(y)

    def test_y_is_delta_not_raw_aqi(self, data):
        """y must be the change (delta), not raw future AQI — roughly centred near 0."""
        _, (X, y) = data
        y_arr = np.array(y)
        # Delta values should have much smaller magnitude than raw AQI (40-220)
        assert np.abs(y_arr).mean() < 100, \
            "y looks like raw AQI, not a delta — preprocess may be returning wrong target"


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrain:

    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_train_does_not_crash(self, model_type):
        df = _engineered_df()
        model = SklearnAQIModel(f"aqi_{model_type}_h1", model_type, forecast_horizon=1)
        X, y = model.preprocess(df, target_day=1)
        model.train(X, y)   # must not raise
        assert model.model is not None

    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_feature_names_set_after_train(self, model_type):
        df = _engineered_df()
        model = SklearnAQIModel(f"aqi_{model_type}_h1", model_type, forecast_horizon=1)
        X, y = model.preprocess(df, target_day=1)
        model.train(X, y)
        assert model.feature_names_ is not None
        assert len(model.feature_names_) > 0

    def test_raises_on_unsupported_model_type(self):
        with pytest.raises((ValueError, KeyError)):
            SklearnAQIModel("bad", "neural_net_v99", forecast_horizon=1)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:

    @pytest.fixture(scope="class")
    def trained(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        X, y = model.preprocess(df, target_day=1)
        model.train(X, y)
        return model, X

    def test_predict_returns_array(self, trained):
        model, X = trained
        current_aqi = 80.0
        preds = model.predict(X, current_aqi)
        assert len(preds) == len(X)

    def test_predictions_are_floats(self, trained):
        model, X = trained
        preds = model.predict(X, current_aqi=80.0)
        assert all(isinstance(float(p), float) for p in preds)

    def test_predict_adds_current_aqi_back(self, trained):
        """
        The model predicts delta. predict() must add current_aqi back.
        So the result should NOT be a near-zero delta — it should be
        roughly in the range of AQI values (20-300).
        """
        model, X = trained
        # Use a single row for clarity
        preds = model.predict(X.iloc[[0]], current_aqi=100.0)
        pred_val = float(preds[0])
        assert 0 < pred_val < 600, \
            f"Prediction {pred_val} outside plausible AQI range — delta may not have been added back"


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:

    @pytest.fixture(scope="class")
    def metrics(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        X, y = model.preprocess(df, target_day=1)
        model.train(X, y)
        return model.evaluate(X, y, current_aqi=80.0)

    def test_returns_dict(self, metrics):
        assert isinstance(metrics, dict)

    def test_has_rmse(self, metrics):
        assert "rmse" in metrics

    def test_has_mae(self, metrics):
        assert "mae" in metrics

    def test_has_r2(self, metrics):
        assert "r2" in metrics

    def test_rmse_is_non_negative(self, metrics):
        assert metrics["rmse"] >= 0

    def test_r2_is_at_most_one(self, metrics):
        assert metrics["r2"] <= 1.0


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_and_load_random_forest(self, tmp_path):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        X, y = model.preprocess(df, target_day=1)
        model.train(X, y)

        path = str(tmp_path / "rf_h1.joblib")
        model.save(path)

        # Load into a new instance
        loaded = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        loaded.load(path)

        # Predictions from original and loaded must match
        preds_orig   = model.predict(X.iloc[:5], current_aqi=80.0)
        preds_loaded = loaded.predict(X.iloc[:5], current_aqi=80.0)
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-6)

    def test_feature_names_survive_round_trip(self, tmp_path):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        X, y = model.preprocess(df, target_day=1)
        model.train(X, y)
        original_names = list(model.feature_names_)

        path = str(tmp_path / "rf_h1.joblib")
        model.save(path)

        loaded = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        loaded.load(path)
        assert list(loaded.feature_names_) == original_names
