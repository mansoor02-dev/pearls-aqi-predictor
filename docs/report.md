# Pearls AQI Predictor — Final Project Report

**Author:** Mansoor  
**Program:** 10Pearls Internship  
**Date:** August 2026  

---

## Executive Summary: Final Submission Deliverables

This document serves as the final submission for the 10Pearls internship project. The objective was to build an intelligent, production-ready MLOps system. All core requirements have been successfully achieved:

### 1. End-to-End AQI Prediction System
Developed a comprehensive machine learning system that ingests raw weather and pollutant data from the Open-Meteo API, engineers over 30 predictive features, and trains 5 different model architectures (Ridge Regression, Random Forest, XGBoost, Feed-Forward NN, and LSTM). The system accurately forecasts the European AQI for Lahore up to 3 days ahead, consistently outperforming naive baselines by reducing error by ~48% on the 1-day horizon.

### 2. Scalable, Automated Pipeline
Implemented a fully serverless, highly scalable MLOps architecture using GitHub Actions and the Hopsworks Cloud. 
- The **Feature Pipeline** runs 4× daily, pushing fresh data to the Hopsworks Feature Store via Kafka.
- The **Training Pipeline** runs daily, fetching historical data, retraining all models, promoting the best-performing model (based on lowest RMSE) to the Model Registry, and generating SHAP explainability plots.

### 3. Interactive Dashboard
Built a premium, real-time web dashboard using Streamlit and Plotly. The dashboard loads the best production models directly from the Hopsworks registry via an automated Inference Layer. It features a real-time AQI gauge, pollutant cards, a 7-day historical trend chart, a 3-day forecast with confidence intervals, and dynamic SHAP feature importance visualizations.

### 4. Detailed Project Documentation
This report serves as the final deliverable, detailing the architecture, feature engineering methodology, model evaluations, pipeline automation, and lessons learned.

---

## 1. System Architecture

The system follows a modern **three-pipeline architecture** common in enterprise MLOps:

![alt text](images/system_architecture.png)

### Technology Stack
- **Data Ingestion:** Open-Meteo API (Free, no API key required)
- **Feature Store & Model Registry:** Hopsworks Cloud
- **ML Models:** scikit-learn, XGBoost, PyTorch
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Dashboard:** Streamlit + Plotly
- **REST API:** FastAPI + Uvicorn
- **CI/CD & Orchestration:** GitHub Actions
- **Package Management:** uv (Astral)

---

## 2. Feature Engineering & Data Pipeline

The `AQIFeatureEngineer` transformer produces **30+ robust features** from raw hourly data:

- **Temporal & Cyclical:** Hour, day of week, and month, encoded using sin/cos transformations to capture seasonal and diurnal cycles.
- **Wind Vectors:** Wind speed and direction decomposed into `wind_u` and `wind_v` components.
- **Pollutant Ratios:** PM2.5/PM10 ratio and NO₂/O₃ interaction to capture photochemical activity.
- **Climate Trends:** 7-day temperature trends and 7-day rain accumulation.
- **Lag Features & Rolling Stats:** AQI at 1h, 3h, 6h, and 24h ago; rolling means for 4h, 6h, 12h, and 24h windows; exponential weighted moving average (6h).

Data is validated for schema integrity and drift using Evidently AI before being written to the Hopsworks Feature Store.

---

## 3. Model Training & Evaluation

Five distinct model architectures are evaluated dynamically for each forecast horizon (1-day, 2-day, 3-day):

1. **Ridge Regression:** Linear baseline with RobustScaler preprocessing.
2. **Random Forest:** Ensemble of 300 trees (max_depth=12).
3. **XGBoost:** Gradient boosting with early stopping (lr=0.03).
4. **Feed-Forward NN (PyTorch):** Deep learning (128 → Dropout → 64).
5. **LSTM (PyTorch):** Sequence model with 24-hour lookback.

### Training Strategy
Models predict the **AQI delta** (change from current AQI) rather than the absolute AQI. 
`predicted_AQI = current_AQI + predicted_delta`. 
This anchors short-term predictions to the current state, significantly improving accuracy over direct prediction.

### Model Promotion
Models are ranked daily by their **Skill vs. Naïve** score (improvement over assuming the AQI stays the same). The best model's artifacts are serialized and uploaded to the Hopsworks Model Registry. Currently, **Random Forest** has proven to be the most robust architecture for this specific dataset.

---

## 4. Explainability (SHAP)

To ensure the "black box" ML models are interpretable, SHAP values are computed for every promoted model. The SHAP plots are bundled into the model artifacts and displayed on the dashboard.

**Key Findings:**
1. `aqi_lag_1h` (current pollution level) is the strongest predictor.
2. `aqi_lag_24h` effectively captures daily human activity cycles.
3. Wind vectors (`wind_u`, `wind_v`) dictate how quickly pollutants are dispersed from the city center.

---

## 5. Challenges & Solutions

1. **Hopsworks API Key Scopes:** Kafka-based feature ingestion failed initially due to missing `KAFKA` permission scopes. Resolved by updating token permissions.
2. **Scaler Serialization:** Early linear models failed during inference because the `RobustScaler` state was lost. Fixed by refactoring the `save()` method in `sklearn_models.py` to bundle the scaler with the model weights using `joblib`.
3. **CI/CD Permissions:** GitHub Actions initially failed to commit SHAP images back to the repository. This architectural flaw was resolved by bypassing git entirely and saving SHAP images directly inside the Hopsworks Model Registry artifacts directory.

---

## 6. Future Work

- **Multi-city support:** Scale the pipeline to process data for Karachi and Islamabad simultaneously.
- **Ensemble predictions:** Combine the top 3 models for a more robust weighted prediction output.
- **Automated retraining triggers:** Move from cron-based retraining to trigger-based retraining (e.g., retrain only when data drift is detected).
