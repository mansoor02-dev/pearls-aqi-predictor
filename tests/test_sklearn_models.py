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


MODEL_TYPES = ["linear", "random_forest", "xgboost"]


def _engineered_df(n: int = 200) -> pd.DataFrame:
    from tests.conftest import _make_engineered_df
    return _make_engineered_df(n=n, horizon=3)


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:

    @pytest.fixture(scope="class")
    def data(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        return model, model.preprocess(df, horizon=1)

    def test_returns_dict_with_expected_keys(self, data):
        _, d = data
        for key in ["X_train", "X_test", "y_train_delta", "y_test_delta",
                    "y_train_raw", "y_test_raw", "current_aqi_train", "current_aqi_test"]:
            assert key in d, f"Missing key '{key}' in preprocess() output"

    def test_x_train_is_dataframe(self, data):
        _, d = data
        assert isinstance(d["X_train"], pd.DataFrame)

    def test_y_train_delta_has_length(self, data):
        _, d = data
        assert hasattr(d["y_train_delta"], "__len__")

    def test_no_target_column_in_x(self, data):
        _, d = data
        for col in d["X_train"].columns:
            assert not col.startswith("aqi_next_"), \
                f"Target column '{col}' leaked into features"

    def test_x_y_same_length(self, data):
        _, d = data
        assert len(d["X_train"]) == len(d["y_train_delta"])
        assert len(d["X_test"]) == len(d["y_test_delta"])

    def test_y_delta_is_delta_not_raw_aqi(self, data):
        """y_train_delta must be the change (delta), not raw future AQI —
        roughly centred near 0, much smaller magnitude than raw AQI (40-220)."""
        _, d = data
        y_arr = np.array(d["y_train_delta"])
        assert np.abs(y_arr).mean() < 100, \
            "y_train_delta looks like raw AQI, not a delta"

    def test_feature_names_set_after_preprocess(self, data):
        model, _ = data
        assert model.feature_names_ is not None
        assert len(model.feature_names_) > 0


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrain:

    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_train_does_not_crash(self, model_type):
        df = _engineered_df()
        model = SklearnAQIModel(f"aqi_{model_type}_h1", model_type, forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])
        assert model.model is not None
        assert model.is_trained

    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_feature_names_set_after_train(self, model_type):
        df = _engineered_df()
        model = SklearnAQIModel(f"aqi_{model_type}_h1", model_type, forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])
        assert model.feature_names_ is not None
        assert len(model.feature_names_) > 0

    def test_raises_on_unsupported_model_type(self):
        with pytest.raises(ValueError):
            SklearnAQIModel("bad", "neural_net_v99", forecast_horizon=1)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:

    @pytest.fixture(scope="class")
    def trained(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])
        return model, data

    def test_predict_returns_array(self, trained):
        model, data = trained
        preds = model.predict(data["X_test"], data["current_aqi_test"])
        assert len(preds) == len(data["X_test"])

    def test_predictions_are_floats(self, trained):
        model, data = trained
        preds = model.predict(data["X_test"], data["current_aqi_test"])
        assert all(isinstance(float(p), float) for p in preds)

    def test_predict_adds_current_aqi_back(self, trained):
        """The model predicts delta. predict() must add current_aqi back, so
        the result should be in the plausible AQI range, not a near-zero delta."""
        model, data = trained
        preds = model.predict(data["X_test"].iloc[[0]], pd.Series([100.0]))
        pred_val = float(preds[0])
        assert 0 < pred_val < 600, \
            f"Prediction {pred_val} outside plausible AQI range — delta may not have been added back"


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:

    @pytest.fixture(scope="class")
    def metrics(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])
        return model.evaluate(data["X_test"], data["y_test_raw"], data["current_aqi_test"])

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

    def test_skill_vs_naive_present_when_y_naive_given(self):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])
        metrics = model.evaluate(
            data["X_test"], data["y_test_raw"], data["current_aqi_test"],
            y_naive=data["current_aqi_test"],
        )
        assert metrics["skill_vs_naive"] is not None


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_and_load_random_forest(self, tmp_path):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])

        path = str(tmp_path / "rf_h1.joblib")
        model.save(path)

        loaded = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        loaded.load(path)

        preds_orig = model.predict(data["X_test"].iloc[:5], data["current_aqi_test"].iloc[:5])
        preds_loaded = loaded.predict(data["X_test"].iloc[:5], data["current_aqi_test"].iloc[:5])
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-6)

    def test_feature_names_survive_round_trip(self, tmp_path):
        df = _engineered_df()
        model = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        data = model.preprocess(df, horizon=1)
        model.train(data["X_train"], data["y_train_delta"])
        original_names = list(model.feature_names_)

        path = str(tmp_path / "rf_h1.joblib")
        model.save(path)

        loaded = SklearnAQIModel("aqi_rf_h1", "random_forest", forecast_horizon=1)
        loaded.load(path)
        assert list(loaded.feature_names_) == original_names