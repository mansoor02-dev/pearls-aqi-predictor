---
marp: true
theme: default
class:
  - lead
backgroundColor: #f8fafc
color: #0f172a
style: |
  h1, h2, h3 { color: #1e293b; }
  a { color: #2563eb; }
  .box { padding: 20px; background: #ffffff; border: 1px solid #4b535dff; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .center { text-align: center; }
---

# 🌬️ Pearls AQI Predictor

**Production-grade Air Quality Index forecasting with MLOps**

---

## 🎯 Project Intro

A fully automated, serverless ML system that forecasts the **Air Quality Index (AQI)** for Lahore up to 3 days ahead.

- **Data:** Hourly weather & pollutant data (Open-Meteo)
- **Features:** 30+ engineered signals
- **Models:** Evaluates 5 architectures daily
- **Explainability:** SHAP integration
- **Dashboard:** Real-time interactive Streamlit UI

---

## 🏗️ High Level Architecture

![width:900px](../architecture/image.png)

---

## 🚀 Quick Start

**Prerequisites**
- Python 3.12+
- Hopsworks account (free tier)
- OpenMeteo API (No API key needed)

**Installation**
```bash
git clone https://github.com/YOUR_USERNAME/pearls-aqi-predictor.git
cd pearls-aqi-predictor
uv sync
cp .env.example .env  # Fill in your API keys
```

---

## 🔄 Feature Pipeline

**Fully Serverless Data Ingestion**

- **Source:** Open-Meteo API (Free)
- **Schedule:** Runs 4× daily (00:00, 06:00, 12:00, 18:00 UTC) via GitHub Actions
- **Storage:** Data is ingested securely into **Hopsworks Feature Store** using Kafka.
- **Validation:** Built-in schema checks and data drift detection.

---

## 🧪 Feature Engineering

**30+ Engineered Features** calculated on the fly:

<div class="grid">
<div class="box">

- **Temporal & Cyclical:** hour, day, month (with sin/cos encoding)
- **Lags:** AQI from 1h, 3h, 6h, 24h ago
- **Rolling Stats:** 4h, 6h, 12h, 24h means

</div>
<div class="box">

- **Ratios:** PM2.5/PM10, NO₂/O₃
- **Wind Vectors:** u/v components
- **Trends:** 7-day temp shifts, rain accumulation

</div>
</div>

---

## 🤖 Models & Training

**5 Architectures Evaluated Daily:**

1. **Ridge Regression** (Linear baseline)
2. **Random Forest** (Tree ensemble) 🌟 *Current Best*
3. **XGBoost** (Gradient boosting)
4. **Feed-Forward NN** (Deep learning)
5. **LSTM** (Sequence modeling)

The system automatically promotes the model with the lowest **RMSE** to production.

---

## 🔮 Prediction Range

The system is designed to provide short-to-medium term forecasts to help citizens plan their week.

- **Horizon 1:** Next 24 hours (Day 1)
- **Horizon 2:** 24-48 hours (Day 2)
- **Horizon 3:** 48-72 hours (Day 3)

*Each horizon uses a dedicated model optimized for that specific timeframe.*

---

## 📊 Results & Accuracy Metrics

The ML model significantly outperforms a naïve "no change" baseline.

<div class="box">

### Best Model: Random Forest (1-Day Horizon)
- **RMSE:** ~10.3 AQI points
- **Skill vs. Naïve:** ~0.48 

*(A skill score of 0.48 means the ML model reduces prediction errors by 48% compared to assuming the AQI simply stays exactly the same as today).*

</div>

---

## 💻 Dashboard Walkthrough

**Built with Streamlit & Plotly (Light Theme)**

<div class="grid">
<div>

- **Real-time Gauge:** Current AQI & health status
- **Pollutant Cards:** Detailed readings (PM2.5, NO₂, etc.)
- **History:** 7-day rolling chart
- **Forecast:** 3-day prediction with confidence intervals

</div>
<div>

*(Load the dashboard by running)*:
```bash
uv run streamlit run app/dashboard.py
```

</div>
</div>

---

## 🔍 SHAP Explainability

**Why did the model make this prediction?**

The system generates SHAP (SHapley Additive exPlanations) values for every production model, saving the visualizations directly to Hopsworks.

**Top Drivers (Typically):**
1. `aqi_lag_1h` (Current pollution levels)
2. `aqi_lag_24h` (Diurnal patterns)
3. `wind_u` / `wind_v` (Dispersion)

---

## 📈 Future Analysis & Work

- **Multi-City Support:** Expand beyond Lahore to other major hubs (Karachi, Islamabad).
- **Drift Triggers:** Automatically retrain models out-of-schedule if severe data drift is detected.
- **Ensemble:** Combine the top 3 models for a more robust weighted prediction.

---

# Thank You! 🌬️

**Pearls AQI Predictor**
*Production-grade Air Quality Forecasting*
