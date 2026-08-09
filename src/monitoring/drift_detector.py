"""
src/monitoring/drift_detector.py

Data drift detection using Evidently AI.

Design
------
- DriftDetector is constructed once with a reference DataFrame (your training
  set, or a representative slice of it).
- Call detect_drift(current_df) each time you want to check for drift —
  typically after the feature pipeline runs.
- is_drift_detected() parses the Evidently result dict and returns a bool.
- get_drifted_features() returns the list of columns whose drift p-value
  falls below the configured significance level.
- save_html_report() persists the full interactive Evidently HTML report for
  human review.

Evidently API note (v0.4+)
--------------------------
The result dict from Report.as_dict() has this shape:
{
  "metrics": [
    {
      "metric": "DatasetDriftMetric",
      "result": {
        "number_of_columns": <int>,
        "number_of_drifted_columns": <int>,
        "share_of_drifted_columns": <float>,   # 0.0 – 1.0
        "dataset_drift": <bool>,               # True if share > dataset_drift_share threshold
        "drift_by_columns": {
          "<col_name>": {
            "stattest_name": <str>,
            "drift_score": <float>,            # p-value or distance, depending on test
            "drift_detected": <bool>,
            ...
          },
          ...
        }
      }
    },
    ...
  ]
}
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Columns that are always metadata — never feed them into the drift tests.
_META_COLS = {"date", "city", "lat", "lon"}


def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata and non-numeric columns so Evidently only sees features."""
    cols = [c for c in df.select_dtypes(include="number").columns if c not in _META_COLS]
    return df[cols]


class DriftDetector:
    """
    Detects data drift between reference (training) and current (live) data.

    Parameters
    ----------
    reference_data : pd.DataFrame
        The baseline dataset — typically the training set or a representative
        recent window. Only numeric, non-metadata columns are used.
    drift_share_threshold : float
        Fraction of features that must drift before dataset-level drift is
        declared. Default 0.3 (i.e. >30% of features drifted → alert).
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_share_threshold: float = 0.3,
    ):
        try:
            from config.settings import settings
            drift_share_threshold = settings.DRIFT_THRESHOLD
        except Exception:
            pass

        self.reference = _numeric_only(reference_data)
        self.drift_share_threshold = drift_share_threshold
        self._last_result: Optional[dict] = None

        # Build report lazily so import failures are surfaced at detect() time.
        self._report = None

    # ── Core API ──────────────────────────────────────────────────────────────

    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Run the Evidently drift report and return the raw result dict.

        Only columns present in BOTH reference and current are tested —
        schema mismatches are logged as warnings rather than raised.

        Parameters
        ----------
        current_data : pd.DataFrame  Live / recent data to compare against reference.

        Returns
        -------
        dict  Raw Evidently result (see module docstring for shape).
        """
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        current = _numeric_only(current_data)

        # Align columns — only test features present in both DataFrames.
        shared_cols = [c for c in self.reference.columns if c in current.columns]
        missing_ref = [c for c in self.reference.columns if c not in current.columns]
        missing_cur = [c for c in current.columns if c not in self.reference.columns]

        if missing_ref:
            logger.warning(f"Columns in reference but not in current data: {missing_ref}")
        if missing_cur:
            logger.info(f"New columns in current data (not in reference): {missing_cur}")

        ref_aligned = self.reference[shared_cols]
        cur_aligned  = current[shared_cols]

        report = Report(metrics=[
            DataDriftPreset(drift_share_threshold=self.drift_share_threshold)
        ])
        report.run(
            reference_data=ref_aligned,
            current_data=cur_aligned,
            column_mapping=ColumnMapping(),
        )
        self._report = report
        self._last_result = report.as_dict()
        return self._last_result

    def is_drift_detected(
        self,
        drift_results: Optional[dict] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Return True if the share of drifted features exceeds `threshold`.

        Parameters
        ----------
        drift_results : dict, optional
            Output of detect_drift(). If None, uses the most recent run.
        threshold : float, optional
            Override the instance-level drift_share_threshold.

        Returns
        -------
        bool
        """
        result = drift_results or self._last_result
        if result is None:
            raise RuntimeError("No drift results available — call detect_drift() first.")

        threshold = threshold if threshold is not None else self.drift_share_threshold

        try:
            dataset_result = self._extract_dataset_result(result)
            share = dataset_result.get("share_of_drifted_columns", 0.0)
            detected = share > threshold
            logger.info(
                f"Drift check: {share * 100:.1f}% of features drifted "
                f"(threshold={threshold * 100:.0f}%) → drift={'YES' if detected else 'NO'}"
            )
            return detected
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(f"Failed to parse drift result: {exc}. Raw keys: {list(result.keys())}")
            return False

    def get_drifted_features(self, drift_results: Optional[dict] = None) -> List[str]:
        """
        Return a list of feature column names whose drift was detected.

        Parameters
        ----------
        drift_results : dict, optional
            Output of detect_drift(). If None, uses the most recent run.
        """
        result = drift_results or self._last_result
        if result is None:
            raise RuntimeError("No drift results available — call detect_drift() first.")

        try:
            dataset_result = self._extract_dataset_result(result)
            by_col: Dict[str, dict] = dataset_result.get("drift_by_columns", {})
            drifted = [col for col, info in by_col.items() if info.get("drift_detected", False)]
            logger.info(f"Drifted features ({len(drifted)}): {drifted}")
            return drifted
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(f"Failed to extract drifted features: {exc}")
            return []

    def get_drift_summary(self, drift_results: Optional[dict] = None) -> dict:
        """
        Return a concise summary dict for logging or display.

        Returns
        -------
        dict with keys: total_columns, drifted_count, share_drifted,
                        dataset_drift (bool), drifted_features (list)
        """
        result = drift_results or self._last_result
        if result is None:
            raise RuntimeError("No drift results available — call detect_drift() first.")

        dataset_result = self._extract_dataset_result(result)
        by_col: Dict[str, dict] = dataset_result.get("drift_by_columns", {})
        drifted_features = [c for c, v in by_col.items() if v.get("drift_detected", False)]

        return {
            "total_columns": dataset_result.get("number_of_columns", len(by_col)),
            "drifted_count": dataset_result.get("number_of_drifted_columns", len(drifted_features)),
            "share_drifted": dataset_result.get("share_of_drifted_columns", 0.0),
            "dataset_drift": dataset_result.get("dataset_drift", False),
            "drifted_features": drifted_features,
        }

    def save_html_report(self, path: str = "docs/drift_report.html") -> str:
        """
        Save the full interactive Evidently HTML report to `path`.

        Parameters
        ----------
        path : str  Destination file path.

        Returns
        -------
        str  Absolute path to the saved report.
        """
        if self._report is None:
            raise RuntimeError("No report to save — call detect_drift() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._report.save_html(path)
        logger.info(f"Drift HTML report saved to: {os.path.abspath(path)}")
        return os.path.abspath(path)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_dataset_result(result: dict) -> dict:
        """
        Navigate the Evidently v0.4+ result dict to the DatasetDriftMetric
        result block. Raises KeyError / IndexError on unexpected shapes so
        callers can catch and log cleanly.
        """
        for metric_block in result.get("metrics", []):
            if "DatasetDrift" in metric_block.get("metric", ""):
                return metric_block["result"]
        # Fallback: first metric's result (older Evidently schema)
        return result["metrics"][0]["result"]