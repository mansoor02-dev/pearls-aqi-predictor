# 🌬️ Pearls AQI Predictor

> **Production-grade Air Quality Index forecasting with MLOps** — built during a 10Pearls internship

[![Feature Pipeline](https://github.com/mansoor02-dev/pearls-aqi-predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/mansoor02-dev/pearls-aqi-predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/mansoor02-dev/pearls-aqi-predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/mansoor02-dev/pearls-aqi-predictor/actions/workflows/training_pipeline.yml)
[![Deploy App](https://github.com/mansoor02-dev/pearls-aqi-predictor/actions/workflows/app_deply.yml/badge.svg)](https://github.com/mansoor02-dev/pearls-aqi-predictor/actions/workflows/app_deply.yml)

---

## 📖 Overview

Pearls AQI Predictor is an end-to-end Machine Learning system that forecasts the **European Air Quality Index (AQI)** for Lahore, Pakistan up to **3 days ahead**. The system follows modern **MLOps best practices** — automated data pipelines, a centralized feature store, a model registry with dynamic promotion, and a real-time interactive dashboard.

### Why This Matters

Lahore consistently ranks among the world's most polluted cities. Accurate AQI forecasts enable residents to plan outdoor activities, help authorities issue timely health advisories, and provide researchers with a reproducible ML pipeline for air quality prediction.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions (CI/CD)                       │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│   │ Feature Pipeline │  │Training Pipeline │  │ Deploy App   │ │
│   │  (4× daily)      │  │  (daily @ 2AM)   │  │  (on push)   │ │
│   └───────┬──────────┘  └───────┬──────────┘  └──────────────┘ │
└───────────┼─────────────────────┼──────────────────────────────┘
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌──────────────────────┐
│   Open-Meteo API  │   │   Hopsworks Cloud    │
│  (Weather + AQI)  │   │  ┌────────────────┐  │
│   No API key!     │   │  │ Feature Store   │  │
└───────┬───────────┘   │  │ (aqi_features)  │  │
        │               │  └────────┬───────┘  │
        ▼               │           │          │
┌───────────────────┐   │  ┌────────▼───────┐  │
│ Feature Engineer  │──▶│  │ Model Registry │  │
│  30+ features     │   │  │ (5 model types │  │
│  lags, rolling,   │   │  │  × 3 horizons) │  │
│  cyclical, ratios │   │  └────────┬───────┘  │
└───────────────────┘   └──────────┼──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   Inference Layer    │
                        │  Best model (RMSE)   │
                        │  + SHAP explanations │
                        └──────┬───────┬──────┘
                               │       │
                    ┌──────────▼┐  ┌───▼──────────┐
                    │ Streamlit  │  │   FastAPI     │
                    │ Dashboard  │  │   REST API    │
                    │ (Port 8501)│  │  (Port 8000)  │
                    └────────────┘  └──────────────┘
```

---

## 🌟 Key Features

### 🔄 Feature Pipeline
- Fetches raw **weather** (temperature, humidity, wind, rain) and **pollutant** data (PM2.5, PM10, NO₂, O₃, CO, SO₂) from the [Open-Meteo API](https://open-meteo.com) — **no API key required**
- Computes **30+ engineered features** including:
  - **Temporal**: hour, day-of-week, month with sin/cos cyclical encoding
  - **Lag features**: AQI at 1h, 3h, 6h, 24h ago
  - **Rolling statistics**: 4h, 6h, 12h, 24h rolling means + exponential weighted means
  - **Derived**: wind vector components (u/v), PM2.5/PM10 ratio, NO₂/O₃ interaction ratio, 7-day temperature trend, 7-day rain accumulation
- Stores processed features in **Hopsworks Feature Store** via Kafka-based ingestion
- Runs **4× daily** (00:00, 06:00, 12:00, 18:00 UTC) via GitHub Actions

### 📊 Historical Data Backfill
- Runs the feature pipeline retroactively for past dates to build a comprehensive training dataset
- Configurable lookback window (default: 16 days)

### 🤖 Training Pipeline
- Fetches historical features from the Hopsworks Feature Store
- Trains and evaluates **5 model architectures**:
  | Model | Type | Key Hyperparameters |
  |-------|------|-------------------|
  | Ridge Regression | Linear | RobustScaler preprocessing |
  | Random Forest | Ensemble | 300 trees, max_depth=12 |
  | XGBoost | Gradient Boosting | 1000 estimators, lr=0.03 |
  | Feed-Forward NN | Deep Learning | 2 hidden layers, dropout=0.3 |
  | LSTM | Sequence Model | 24-step lookback, hidden=64 |
- Evaluates using **RMSE**, **MAE**, **R²**, and **Skill vs. Naïve baseline**
- Automatically promotes the **best model** to the Hopsworks Model Registry
- Generates **SHAP explainability plots** bundled with each model artifact
- Runs **daily at 2:00 AM UTC**

### 📈 Web Dashboard (Streamlit)
- Real-time AQI gauge with color-coded health status
- **8 pollutant cards** showing current readings (PM2.5, PM10, NO₂, O₃, CO, SO₂, UV, Temp)
- 7-day hourly historical AQI chart with AQI zone bands
- **3-day forecast** with 95% confidence intervals
- **SHAP feature importance** visualization loaded directly from the model registry
- Health advisory alerts for hazardous AQI levels

### 🔌 REST API (FastAPI)
- `GET /health` — liveness probe for CI/CD
- `POST /predict` — returns AQI forecasts with confidence intervals and model metadata
- Automatic model loading from Hopsworks on startup

### 🔍 Monitoring & Alerts
- **Data drift detection** using statistical tests (via Evidently AI)
- **Multi-channel alerting** (Slack, Email, Logs) when AQI exceeds 150

---

## 📁 Repository Structure

```text
pearls-aqi-predictor/
├── app/
│   ├── dashboard.py          # Streamlit interactive dashboard
│   └── api.py                # FastAPI REST API
├── config/
│   └── settings.py           # Pydantic-based configuration
├── src/
│   ├── data/
│   │   ├── api_client.py     # Open-Meteo API client (weather + AQI)
│   │   └── data_validator.py # Schema validation & drift checks
│   ├── features/
│   │   ├── feature_engineering.py  # 30+ feature transforms
│   │   ├── feature_store.py        # Hopsworks Feature Store interface
│   │   └── backfill.py             # Historical data backfill orchestrator
│   ├── models/
│   │   ├── sklearn_models.py  # Ridge, Random Forest, XGBoost
│   │   ├── deep_learning.py   # FFN and LSTM (PyTorch)
│   │   ├── model_registry.py  # Hopsworks Model Registry wrapper
│   │   └── evaluate.py        # RMSE, MAE, R², Skill scoring
│   ├── monitoring/
│   │   ├── drift_detector.py  # Statistical drift detection
│   │   └── alerts.py          # Slack/Email/Log alerting
│   ├── pipeline/
│   │   ├── feature_pipeline.py   # Scheduled feature ingestion
│   │   └── training_pipeline.py  # Model training + promotion
│   └── inference.py           # Production model loading & prediction
├── scripts/
│   ├── run_backfill.py        # Manual backfill runner
│   ├── run_feature_pipeline.py
│   └── run_training_pipeline.py
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── .github/workflows/
│   ├── feature_pipeline.yml   # 4× daily data ingestion
│   ├── training_pipeline.yml  # Daily model retraining
│   └── app_deply.yml          # API health check on push
├── tests/                     # Pytest suite
├── pyproject.toml             # Dependencies (managed by uv)
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)
- **Hopsworks account** ([free tier](https://app.hopsworks.ai/))
- Open-Meteo API — **no API key needed**

### 1. Clone & Install

```bash
git clone https://github.com/mansoor02-dev/pearls-aqi-predictor.git
cd pearls-aqi-predictor
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT_NAME=pearl_aqi_feature_store
CITY=Lahore
```

### 3. Run Pipelines

```bash
# Backfill historical data (first-time setup)
uv run python scripts/run_backfill.py

# Run feature pipeline
uv run python scripts/run_feature_pipeline.py

# Train models
uv run python scripts/run_training_pipeline.py
```

### 4. Launch Applications

```bash
# Streamlit Dashboard
uv run streamlit run app/dashboard.py

# FastAPI Server
uv run uvicorn app.api:app --reload --port 8000
```

---

## 🧪 Testing

```bash
uv run pytest
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| ML Models | scikit-learn, XGBoost, PyTorch |
| Feature Store | Hopsworks |
| Model Registry | Hopsworks |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn |
| Data Source | Open-Meteo (Weather + Air Quality) |
| CI/CD | GitHub Actions |
| Package Manager | uv (Astral) |
| Monitoring | Evidently AI |

---

## 📄 License

This project was developed as part of a **10Pearls internship program**.
