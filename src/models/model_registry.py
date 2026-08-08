from typing import Dict, Any
from src.utils.logger import setup_logger


class HopsworksModelRegistry:
    """
    Manages model versions in Hopsworks Model Registry.
    Handles: save, load, version comparison, promotion to production.
    """
    
    def __init__(self, project):
        self.mr = project.get_model_registry()
        self.logger = setup_logger(self.__class__.__name__)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,  # Local path to saved model
        metrics: Dict[str, float],
        description: str = ""
    ):
        """
        Register a new model version.
        Hopsworks auto-increments version numbers.
        """
        model = self.mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=description
        )
        model.save(model_path)
        self.logger.info(f"Registered {model_name} v{model.version}")
        return model
    
    def get_best_model(self, model_name: str, metric: str = "rmse_avg") -> Any:
        """
        Retrieve the best model by a given metric.
        For RMSE/MAE: lower is better.
        For R²: higher is better.
        """
        models = self.mr.get_models(name=model_name)
        # Sort by metric and return best
        pass
    
    def promote_to_production(self, model_name: str, version: int):
        """Tag a model version as 'production'."""
        model = self.mr.get_model(name=model_name, version=version)
        # Hopsworks uses tags/labels for staging
        pass