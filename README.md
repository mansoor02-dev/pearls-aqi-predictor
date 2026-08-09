# Pearls AQI Predictor

A 100% serverless Machine Learning system that forecasts Air Quality Index (AQI) up to 3 days ahead. Built with Python, Hopsworks Feature Store & Model Registry, XGBoost, Streamlit, and FastAPI.

---

## 🌟 Key Features

- **Serverless Data Ingestion**: Pulls weather & air quality data from Open-Meteo hourly via GitHub Actions.
- **Feature Engineering**: Computes 30+ features including time/cyclical variables ($\sin/\cos$), pollutant ratios, lags, and rolling statistics.
- **Model Suite**: Evaluates Ridge, Random Forest, XGBoost, Neural Nets, and LSTM. Promotes the best model (XGBoost) to Hopsworks Registry.
- **Interactive Web Dashboard**: Built with Streamlit — includes an AQI gauge, historical charts, 3-day forecasts, and SHAP explainability plots.
- **REST API**: Built with FastAPI for live inference (`/predict`).
- **Monitoring & Alerts**: Statistical drift detection (Evidently AI) and multi-channel alerting (Slack, Email, Logs) when AQI > 150.

---

## 📁 Repository Structure

```text
├── app/                  # Streamlit dashboard & FastAPI application
├── config/               # Settings & configuration management
├── docs/                 # Architecture diagram & project report
├── notebooks/            # Jupyter experiments (01_eda, 02_feature_eng, 03_models)
├── scripts/              # Pipeline runner scripts (backfill, feature, training)
├── src/                  # Core source code (data, features, models, monitoring)
└── tests/                # Automated pytest suite
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

Copy `.env.example` to `.env` and fill in your Hopsworks credentials:

```bash
cp .env.example .env
```

Set environment variables in `.env`:
```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT_NAME=pearl_aqi_feature_store
CITY=Lahore
```

### 2. Install Dependencies

Using `uv` (recommended) or standard `pip`:

```bash
uv sync
```

---

## 🚀 Running Pipelines & Services

### Backfill Historical Data
```bash
uv run python -m scripts.run_backfill
# OR
uv run python scripts/run_backfill.py
```

### Run Feature Pipeline (Hourly Ingestion)
```bash
uv run python -m scripts.run_feature_pipeline
# OR
uv run python scripts/run_feature_pipeline.py
```

### Run Model Training Pipeline
```bash
uv run python -m scripts.run_training_pipeline
# OR
uv run python scripts/run_training_pipeline.py
```


### Launch Streamlit Dashboard
```bash
uv run streamlit run app/dashboard.py
```

### Launch FastAPI Server
```bash
uv run uvicorn app.api:app --reload --port 8000
```

---

## 🧪 Running Tests

Run the full offline test suite:

```bash
uv run pytest
```

---

## 📄 Documentation

- [Project Report](docs/report.md)
- [Architecture Overview](docs/architecture.md)
- [Notebooks Guide](notebooks/README.md)
