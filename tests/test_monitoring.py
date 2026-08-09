# tests/test_monitoring.py
"""
Unit tests for src/monitoring/alerts.py and src/monitoring/drift_detector.py.

All tests are offline — no Slack, no SMTP, no Evidently HTTP calls.
"""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.alerts import AQIAlertSystem


# ════════════════════════════════════════════════════════════════════════════
# AQIAlertSystem
# ════════════════════════════════════════════════════════════════════════════

class TestAQIAlertSystem:

    @pytest.fixture
    def system(self):
        """Alert system with no webhook / SMTP configured — logger-only mode."""
        return AQIAlertSystem()

    # ── check_and_alert ──────────────────────────────────────────────────────

    def test_no_alert_when_aqi_is_good(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.check_and_alert(current_aqi=45.0, predictions=[50.0, 48.0, 52.0])
        # No WARNING log should be emitted for good air quality
        assert len(caplog.records) == 0

    def test_current_aqi_alert_fires_above_threshold(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.check_and_alert(current_aqi=180.0, predictions=[50.0])
        assert any("180" in r.message for r in caplog.records)

    def test_forecast_alert_fires_when_prediction_exceeds_threshold(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.check_and_alert(current_aqi=40.0, predictions=[200.0, 50.0])
        # Alert should mention day 1 prediction
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_no_forecast_alert_when_all_predictions_safe(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.check_and_alert(current_aqi=40.0, predictions=[80.0, 90.0, 100.0])
        assert len(caplog.records) == 0

    def test_none_predictions_do_not_crash(self, system):
        """Predictions list may contain None when a model is unavailable."""
        # Should not raise
        system.check_and_alert(current_aqi=40.0, predictions=[None, 80.0, None])

    # ── send_drift_alert ─────────────────────────────────────────────────────

    def test_drift_alert_logs_warning(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.send_drift_alert(
                drifted_features=["pm2_5", "ozone"],
                share_drifted=0.42,
            )
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_drift_alert_message_contains_feature_names(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.send_drift_alert(
                drifted_features=["pm2_5", "wind_speed_10m"],
                share_drifted=0.5,
            )
        full_log = " ".join(r.message for r in caplog.records)
        assert "pm2_5" in full_log or "wind_speed_10m" in full_log

    # ── send_model_alert ─────────────────────────────────────────────────────

    def test_model_alert_fires_on_degradation(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.send_model_alert(horizon=1, rmse=25.0, baseline_rmse=10.0)  # +150%
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_model_alert_silent_when_within_tolerance(self, system, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            system.send_model_alert(horizon=1, rmse=11.0, baseline_rmse=10.0)  # +10%
        assert len(caplog.records) == 0

    # ── Slack / Email channels — failures don't propagate ────────────────────

    def test_slack_failure_does_not_raise(self, monkeypatch):
        """If the Slack webhook is unreachable, the alert system must not crash."""
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("timeout")))

        system = AQIAlertSystem()
        system.slack_url = "https://hooks.slack.com/fake-url"
        # Must not raise — error is caught and logged
        system.check_and_alert(current_aqi=200.0, predictions=[])


# ════════════════════════════════════════════════════════════════════════════
# DriftDetector
# ════════════════════════════════════════════════════════════════════════════

def _make_numeric_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "european_aqi":   rng.uniform(40, 180, n),
        "pm2_5":          rng.uniform(5, 120, n),
        "pm10":           rng.uniform(10, 200, n),
        "temperature_2m": rng.uniform(15, 45, n),
        "wind_speed_10m": rng.uniform(0, 30, n),
        "ozone":          rng.uniform(20, 160, n),
    })


class TestDriftDetector:

    @pytest.fixture(scope="class")
    def reference_df(self):
        return _make_numeric_df(n=200, seed=0)

    # ── is_drift_detected — no-drift scenario ────────────────────────────────

    def test_no_drift_when_data_is_identical(self, reference_df):
        """Running the same data against itself should report no drift."""
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df, drift_share_threshold=0.3)
        results = detector.detect_drift(reference_df.copy())
        assert not detector.is_drift_detected(results), \
            "Identical reference and current should not trigger drift"

    # ── is_drift_detected — clear-drift scenario ─────────────────────────────

    def test_drift_detected_when_distribution_shifts_massively(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        # Current data shifted far from the reference distribution
        current = reference_df.copy()
        current["european_aqi"]   = 490.0    # pinned near max
        current["pm2_5"]          = 0.001
        current["temperature_2m"] = -50.0

        detector = DriftDetector(reference_df, drift_share_threshold=0.3)
        results = detector.detect_drift(current)
        assert detector.is_drift_detected(results), \
            "Massively shifted data should trigger drift detection"

    # ── get_drifted_features ──────────────────────────────────────────────────

    def test_get_drifted_features_returns_list(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df)
        detector.detect_drift(reference_df.copy())
        drifted = detector.get_drifted_features()
        assert isinstance(drifted, list)

    def test_get_drifted_features_names_are_strings(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df)
        detector.detect_drift(reference_df.copy())
        for name in detector.get_drifted_features():
            assert isinstance(name, str)

    # ── get_drift_summary ─────────────────────────────────────────────────────

    def test_drift_summary_keys(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df)
        detector.detect_drift(reference_df.copy())
        summary = detector.get_drift_summary()

        for key in ["total_columns", "drifted_count", "share_drifted",
                    "dataset_drift", "drifted_features"]:
            assert key in summary, f"Missing key in drift summary: {key}"

    def test_drift_summary_share_is_between_0_and_1(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df)
        detector.detect_drift(reference_df.copy())
        summary = detector.get_drift_summary()
        assert 0.0 <= summary["share_drifted"] <= 1.0

    # ── raises without detect_drift called first ──────────────────────────────

    def test_is_drift_detected_raises_without_prior_run(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df)
        with pytest.raises(RuntimeError, match="detect_drift"):
            detector.is_drift_detected()

    # ── column alignment: extra / missing columns handled gracefully ──────────

    def test_extra_columns_in_current_are_ignored(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        current_extra = reference_df.copy()
        current_extra["brand_new_column"] = 999.0

        detector = DriftDetector(reference_df)
        results = detector.detect_drift(current_extra)   # must not raise
        assert results is not None

    def test_missing_columns_in_current_are_handled(self, reference_df):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        current_missing = reference_df.drop(columns=["ozone"])
        detector = DriftDetector(reference_df)
        results = detector.detect_drift(current_missing)   # must not raise
        assert results is not None

    # ── save_html_report ──────────────────────────────────────────────────────

    def test_save_html_report_creates_file(self, reference_df, tmp_path):
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            pytest.skip("evidently not installed")

        detector = DriftDetector(reference_df)
        detector.detect_drift(reference_df.copy())

        report_path = str(tmp_path / "drift_report.html")
        saved = detector.save_html_report(report_path)

        import os
        assert os.path.exists(saved)
        assert saved.endswith(".html")
