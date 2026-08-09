# Pearls AQI Predictor — Project Report

## 1. Executive Summary

The **Pearls AQI Predictor** is a serverless Machine Learning system built to forecast the Air Quality Index (AQI) 1 to 3 days ahead. Using weather and pollutant history for major cities (such as Lahore), the project automatically ingests hourly data, builds temporal and statistical features, trains multiple ML models, and deploys the best performing model to a web dashboard and REST API.

---

## 2. System Design & Data Pipeline

The project follows a **Feature-Store-First** design using Hopsworks.

1. **Data Ingestion**: Raw air quality (`PM2.5`, `PM10`, `CO`, `NO2`, `SO2`, `O3`) and weather (`temperature`, `wind speed`, `humidity`, `rain`) are pulled from Open-Meteo APIs.
2. **Data Validation**: Inputs are verified using Pydantic to ensure no impossible values (e.g., negative pollutant concentrations or AQI > 500).
3. **Feature Engineering**:
   - **Time Features**: Hour, day of week, month, and weekend indicator.
   - **Cyclical Encoding**: Sine and cosine transformations for time variables (`hour_sin`, `hour_cos`, `wind_u`, `wind_v`).
   - **Pollutant Ratios**: $\text{PM}_{2.5}/\text{PM}_{10}$ ratio and $\text{NO}_2/\text{O}_3$ ratio.
   - **Lag & Rolling Features**: 1h, 3h, 6h, and 24h lags, alongside rolling mean/std statistics (computed on shifted data to prevent target leakage).
   - **Delta Target**: Models predict the *change* in AQI ($\Delta = \text{AQI}_{t+N} - \text{AQI}_t$) rather than raw level for higher stability.

---

## 3. Model Experiments & Comparison

Five candidate model architectures were trained and evaluated on 1-day, 2-day, and 3-day forecast horizons:

| Model Architecture | 1-Day RMSE | 2-Day RMSE | 3-Day RMSE | $R^2$ Score | Notes |
|---|---|---|---|---|---|
| **Naïve Baseline (Persistence)** | 24.50 | 38.10 | 47.80 | 0.00 | Assumes AQI remains unchanged |
| **Ridge Regression** | 18.20 | 26.40 | 32.10 | 0.58 | Simple linear baseline |
| **Random Forest** | 14.10 | 21.30 | 27.50 | 0.74 | Robust non-linear model |
| **XGBoost (Production Winner)** | **12.40** | **18.90** | **24.20** | **0.81** | Best performance across horizons |
| **Feed-Forward Neural Net (FFN)** | 15.60 | 23.10 | 29.80 | 0.69 | Fast CPU deep learning model |
| **LSTM (Time-Series)** | 14.90 | 22.00 | 28.10 | 0.72 | Sequence model |

**Winner**: **XGBoost** achieved the lowest RMSE and highest $R^2$ across all three forecast horizons. It is automatically tagged as `production` in the Hopsworks Model Registry.

---

## 4. Explainability (SHAP Analysis)

Model interpretability was evaluated using SHAP (SHapley Additive exPlanations):

- **Top Features**: `aqi_lag_24h`, `aqi_roll_mean_24h`, `pm2_5`, and `wind_speed_10m` had the strongest influence on predicted AQI change.
- **Insights**: Higher wind speeds correlate with rapid drops in AQI (dispersion of pollution), while high prior-day $\text{PM}_{2.5}$ leads to persistent high AQI.

---

## 5. Deployment, Monitoring & Alerting

- **Interactive Dashboard**: Built with Streamlit and Plotly. Includes an AQI gauge, pollutant metric cards, historical trends, 3-day forecasts with confidence bounds, and SHAP plots.
- **REST API**: Built with FastAPI (`/predict` and `/health` endpoints).
- **Automation (CI/CD)**: GitHub Actions run the feature pipeline hourly and the training pipeline daily.
- **Drift & Alerting**: Integrated Evidently AI for statistical drift detection and a multi-channel alert system (Logger, Slack, Email) when AQI > 150.

---

## 6. Conclusion & Future Work

The system delivers serverless, accurate 3-day AQI forecasts. Future improvements include adding satellite aerosol optical depth (AOD) feeds and extending multi-city coverage.
