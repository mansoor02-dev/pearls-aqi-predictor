import os
from datetime import datetime, timedelta
from typing import Dict, Any

import matplotlib.pyplot as plt
import torch

from src.features.feature_store import HopsworksFeatureStore
from src.models.model_registry import HopsworksModelRegistry
from src.models.sklearn_models import SklearnAQIModel
from src.models.deep_learning import LSTMAQIModel, FeedForwardAQIModel
from src.utils.logger import setup_logger
from src.utils.hopsworks_utils import login_hopsworks
from config.settings import settings


logger = setup_logger(__name__)

MODEL_SPECS = {
    "linear": ("linear", SklearnAQIModel),
    "random_forest": ("random_forest", SklearnAQIModel),
    "xgboost": ("xgboost", SklearnAQIModel),
    "ffn": (None, FeedForwardAQIModel),
    "lstm": (None, LSTMAQIModel),
}


class TrainingPipeline:
    """
    End-to-end training pipeline, run once per forecast horizon:
    1. Load engineered features from the Feature Store
    2. Train every model type in MODEL_SPECS
    3. Evaluate each on the raw AQI scale, skill-scored against naive persistence
    4. Register every model version in Hopsworks; promote the best per horizon
    5. Generate SHAP explanations for the winner
    """

    def __init__(self, horizons=(1, 2, 3)):
        project = login_hopsworks()
        self.fs = HopsworksFeatureStore(project)
        self.mr = HopsworksModelRegistry(project)
        self.horizons = list(horizons)
        self.logger = logger


    def run(self) -> Dict[int, Dict[str, Any]]:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        training_data = self.fs.get_training_data(
            settings.FEATURE_VIEW_NAME, start_date="2022-01-01", end_date=yesterday
        )

        all_results: Dict[int, Dict[str, Any]] = {}
        registered_versions: Dict[str, int] = {}

        for horizon in self.horizons:
            self.logger.info(f"Training horizon: {horizon}d")
            horizon_results = {}
            horizon_data = {}
            trained_models = {}

            for name, (model_type, cls) in MODEL_SPECS.items():
                model_name = f"{name}_h{horizon}"
                try:
                    model = cls(model_name, model_type, forecast_horizon=horizon) if model_type \
                        else cls(model_name, forecast_horizon=horizon)

                    data = model.preprocess(training_data, horizon=horizon)
                    val_kwargs = {}
                    if model_type == "xgboost":
                        val_kwargs = {"X_val": data["X_test"], "y_val_delta": data["y_test_delta"]}

                    model.train(data["X_train"], data["y_train_delta"], **val_kwargs)
                    metrics = model.evaluate(
                        data["X_test"], data["y_test_raw"], data["current_aqi_test"],
                        y_naive=data["current_aqi_test"],   # naive persistence baseline
                    )

                    save_dir = self._save_path(model_name, model)
                    os.makedirs(save_dir, exist_ok=True)
                    ext = "joblib" if isinstance(model, SklearnAQIModel) else "pt"
                    model_file = f"{save_dir}/{model_name}.{ext}"
                    model.save(model_file)
                    
                    self._generate_shap_explanations(model, data["X_test"], save_dir)
                    
                    registered = self.mr.register_model(
                        model_name=f"aqi_{model_name}",
                        model_path=save_dir,
                        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                    )
                    registered_versions[name] = registered.version
                    horizon_results[name] = metrics
                    horizon_data[name] = data
                    trained_models[name] = model
                except Exception as e:
                    self.logger.error(f"Training failed for '{name}' (h={horizon}): {e}", exc_info=True)
                    continue   # one model failing shouldn't lose the rest

            if not horizon_results:
                self.logger.warning(f"No models trained successfully for horizon {horizon}d — skipping")
                continue

            best_name = max(horizon_results, key=lambda k: horizon_results[k]["skill_vs_naive"] or -float("inf"))
            best_metrics = horizon_results[best_name]
            self.logger.info(
                f"Best model for h={horizon}d: {best_name} "
                f"(skill_vs_naive={best_metrics['skill_vs_naive']:.3f}, rmse={best_metrics['rmse']:.2f})"
            )

            self.mr.promote_to_production(f"aqi_{best_name}_h{horizon}", version=registered_versions[best_name])

            all_results[horizon] = horizon_results

        return all_results

    @staticmethod
    def _save_path(model_name: str, model) -> str:
        return f"models/aqi_models/{model_name}"

    def _generate_shap_explanations(self, model, X_test, output_dir: str) -> None:
        """SHAP on the DELTA prediction (model.model, not model.predict) — the
        reconstruction step (+ current_aqi) is a constant shift per row and
        isn't informative to attribute; the delta model is what's actually
        making the interesting decision."""
        try:
            import shap
        except ImportError:
            self.logger.warning("shap not installed — skipping explanation generation")
            return

        sample = X_test[:200] if hasattr(X_test, "__len__") and len(X_test) > 200 else X_test

        if isinstance(model, LSTMAQIModel):
            self.logger.info("Skipping SHAP for LSTM — needs GradientExplainer for sequence input"); 
            return

        if isinstance(model, SklearnAQIModel) and model.model_type in ("random_forest", "xgboost"):
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(sample)
        elif isinstance(model, SklearnAQIModel) and model.model_type == "linear":
            explainer = shap.LinearExplainer(model.model, sample)
            shap_values = explainer.shap_values(sample)
        else:
            def torch_predict(x):
                model.model.eval()
                with torch.no_grad():
                    return model.model(torch.tensor(x, dtype=torch.float32)).cpu().numpy().ravel()
            background = sample[:50]
            explainer = shap.KernelExplainer(torch_predict, background)
            shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=getattr(model, "feature_names_", None), show=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary.png")
        plt.close()
        self.logger.info(f"Saved SHAP summary to {output_dir}")