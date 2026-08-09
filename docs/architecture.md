# System Architecture — Pearls AQI Predictor

The system is a 100% serverless, feature-store-first machine learning pipeline for forecasting Air Quality Index (AQI) up to 3 days ahead.

## High-Level Overview

```mermaid
flowchart TD
    subgraph Data Sources
        API[Open-Meteo API\nWeather & Air Quality]
    end

    subgraph Feature Pipeline
        FP[Feature Pipeline\nRuns Hourly via GitHub Actions]
        FE[AQIFeatureEngineer\nLags, Rolling Stats, Sin/Cos]
        FS[(Hopsworks Feature Store\naqi_features)]
    end

    subgraph Training Pipeline
        TP[Training Pipeline\nRuns Daily via GitHub Actions]
        MD[Models Trained\nRidge, RF, XGBoost, Neural Nets]
        MR[(Hopsworks Model Registry\naqi_forecaster)]
    end

    subgraph Serving & User Interface
        API_APP[FastAPI REST API\n/predict]
        DASH[Streamlit Dashboard\nGauge, Charts, SHAP Plots]
        ALERT[Alert & Drift System\nSlack, Email, Logger]
    end

    API --> FP
    FP --> FE
    FE --> FS
    FS --> TP
    TP --> MD
    MD --> MR
    MR --> API_APP
    MR --> DASH
    FS --> ALERT
```

## Key Components

1. **Feature Pipeline**: Runs hourly on GitHub Actions. It fetches live pollutant and weather data from Open-Meteo, validates the schema, computes time/lag/rolling features, and saves them into the **Hopsworks Feature Store**.
2. **Training Pipeline**: Runs daily. It downloads historical features from Hopsworks, trains 5 ML model architectures (Ridge, Random Forest, XGBoost, FFN, LSTM), evaluates performance against a baseline, and registers the best model into the **Hopsworks Model Registry**.
3. **Inference & Serving**: FastAPI and Streamlit fetch the `production`-tagged model from Hopsworks to serve 1-day, 2-day, and 3-day AQI forecasts.
4. **Monitoring & Alerting**: Tracks feature drift with Evidently AI and sends alert notifications via Slack, Email, or Logs when AQI > 150.
