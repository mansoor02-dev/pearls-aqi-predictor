# Design Pattern: Domain-specific exceptions
class AQIPredictorException(Exception):
    """Base exception for the project."""
    pass


class APIClientError(AQIPredictorException):
    """Raised when external API fails."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

    def __str__(self):
        if self.status_code is not None:
            return f"{self.args[0]} (Status: {self.status_code})"
        return self.args[0]


class FeatureStoreError(AQIPredictorException):
    """Raised when Hopsworks operations fail."""
    pass


class ModelTrainingError(AQIPredictorException):
    """Raised when model training fails validation."""
    pass


class DataValidationError(AQIPredictorException):
    """Raised when data quality checks fail."""
    pass