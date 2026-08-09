# notebooks/README.md

This directory contains the three core Jupyter Notebooks for the Pearls AQI Predictor project:

1. **`01_eda.ipynb`**: Setup, API Client layer, Data Validation layer, historical data ingestion, Exploratory Data Analysis (EDA), and Outlier Analysis.
2. **`02_feature_engineering.ipynb`**: `AQIFeatureEngineer` implementation, temporal & cyclical encodings, lag features, rolling statistics, delta-target construction, and feature validation.
3. **`03_model_experiments.ipynb`**: Diagnostic framework, Ridge/Random Forest baselines, XGBoost, hyperparameter tuning, walk-forward validation, Deep Learning (FFN & LSTM), multi-horizon comparison (1, 2, and 3-day horizons), and Model Leaderboard.
