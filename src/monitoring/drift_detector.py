import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DriftDetector:
    """
    Detects data drift between reference (training) and current data.
    Uses Evidently AI for statistical tests.
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        self.report = Report(metrics=[DataDriftPreset()])
    
    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        self.report.run(
            reference_data=self.reference,
            current_data=current_data,
            column_mapping=ColumnMapping()
        )
        return self.report.as_dict()
    
    def is_drift_detected(self, drift_results: dict, threshold: float = 0.5) -> bool:
        """Return True if drift score exceeds threshold."""
        # Parse Evidently output
        pass