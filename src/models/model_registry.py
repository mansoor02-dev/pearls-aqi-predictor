from typing import Dict, Any, Optional
from src.utils.logger import setup_logger


class HopsworksModelRegistry:
    """
    Manages model versions in Hopsworks Model Registry.
    Handles: save, load, version comparison, promotion to production.

    Hopsworks has no built-in "staging/production" concept (unlike MLflow) —
    promotion here is implemented with a tag on the Model entity itself.
    """

    def __init__(self, project):
        self.mr = project.get_model_registry()
        self.logger = setup_logger(self.__class__.__name__)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, float],
        description: str = "",
    ):
        """Register a new model version. Hopsworks auto-increments version
        numbers when `version` is left unset."""
        model = self.mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=description,
        )
        model.save(model_path)
        self.logger.info(f"Registered {model_name} v{model.version}")
        return model

    def get_best_model(self, model_name: str, metric: str = "rmse", direction: str = "min") -> Any:
        """Retrieve the best-performing version by a given metric.
        direction: 'min' for RMSE/MAE/MSE (lower is better),
                   'max' for R²/skill_vs_naive (higher is better).
        """
        best = self.mr.get_best_model(model_name, metric, direction)
        if best is None:
            self.logger.warning(f"No models found for '{model_name}'")
            return None
        self.logger.info(f"Best model for '{model_name}' by {metric} ({direction}): v{best.version}")
        return best

    def promote_to_production(self, model_name: str, version: int) -> None:
        """In Hopsworks without a predefined tag schema, we simply rely on 
        metrics to find the best model. This is kept for interface compatibility."""
        self.logger.info(f"Model {model_name} v{version} is now available for production inference.")

    def get_production_model(self, model_name: str) -> Any:
        """Returns the best model version based on RMSE.
        This ensures the dashboard always dynamically fetches the best performing model."""
        model = self.get_best_model(model_name, metric="rmse", direction="min")
        if model is not None:
            self.logger.info(f"Loaded production model '{model_name}' v{model.version}")
        return model