
from src.features.feature_store import HopsworksFeatureStore
from src.utils.logger import setup_logger


class TrainingPipeline:
    """
    End-to-end training pipeline:
    1. Load features from Feature Store
    2. Train multiple models
    3. Evaluate and compare
    4. Register best model
    5. Generate SHAP explanations
    """
    
    def __init__(self):
        self.fs = HopsworksFeatureStore(...)
        self.mr = HopsworksModelRegistry(...)
        self.logger = setup_logger(self.__class__.__name__)
    
    def run(self):
        # 1. Load training data
        train_df, val_df, test_df = self._load_and_split_data()
        
        # 2. Train models
        models = {
            'ridge': SklearnAQIModel('ridge_baseline', 'ridge'),
            'random_forest': SklearnAQIModel('rf_v1', 'random_forest'),
            'lstm': LSTMAQIModel('lstm_v1')
        }
        
        results = {}
        for name, model in models.items():
            X_train, y_train = model.preprocess(train_df)
            X_val, y_val = model.preprocess(val_df)
            X_test, y_test = model.preprocess(test_df)
            
            model.train(X_train, y_train, X_val, y_val)
            test_metrics = model.evaluate(X_test, y_test)
            results[name] = test_metrics
            
            # Save locally then register
            model.save(f"models/{name}.joblib")
            self.mr.register_model(
                model_name=f"aqi_{name}",
                model_path=f"models/{name}.joblib",
                metrics=test_metrics
            )
        
        # 3. Pick best model (lowest RMSE)
        best_model = min(results, key=lambda k: results[k]['rmse_avg'])
        self.logger.info(f"Best model: {best_model} with RMSE {results[best_model]['rmse_avg']:.2f}")
        
        # 4. Generate SHAP for best model
        self._generate_shap_explanations(models[best_model], X_test)
        
        return results
    
    def _generate_shap_explanations(self, model, X_test):
        """Generate SHAP values and save summary plot."""
        import shap
        explainer = shap.TreeExplainer(model.model)  # For tree models
        shap_values = explainer.shap_values(X_test)
        # Save plot to file for dashboard
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("docs/shap_summary.png")