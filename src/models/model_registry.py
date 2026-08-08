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
        """Tag one specific version as 'production', un-tagging any previous
        production version of the same model first so exactly one is live."""
        current_prod = self.get_production_model(model_name)
        if current_prod is not None and current_prod.version != version:
            current_prod.delete_tag("stage")
            self.logger.info(f"Un-tagged {model_name} v{current_prod.version} as production")

        model = self.mr.get_model(name=model_name, version=version)
        model.set_tag("stage", "production")
        self.logger.info(f"Promoted {model_name} v{version} to production")

    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Returns the version currently tagged 'production', or None if
        nothing has been promoted yet. This is what inference code (the API,
        the dashboard) should call — never hardcode a version number there."""
        for model in self.mr.get_models(name=model_name):
            if model.get_tag("stage") == "production":
                return model
        self.logger.warning(f"No production model tagged for '{model_name}'")
        return None