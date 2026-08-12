import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be FIRST Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor · Lahore",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS design system ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- Global dark palette ---------- */
.stApp { background: #0d1117; color: #e6edf3; }

/* ---------- Metric cards ---------- */
.card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.card-title  { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
               text-transform: uppercase; color: #8b949e; margin-bottom: 4px; }
.card-value  { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.card-sub    { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }

/* ---------- AQI level badge ---------- */
.aqi-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ---------- Section header ---------- */
.section-header {
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8b949e;
    border-bottom: 1px solid #21262d; padding-bottom: 6px; margin: 24px 0 14px;
}

/* ---------- Alert boxes ---------- */
.alert-good      { background:#0d2518; border-left:4px solid #3fb950; padding:10px 14px; border-radius:6px; }
.alert-moderate  { background:#2b2000; border-left:4px solid #d29922; padding:10px 14px; border-radius:6px; }
.alert-bad       { background:#2d0f0f; border-left:4px solid #f85149; padding:10px 14px; border-radius:6px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #21262d; }
</style>
""",
    unsafe_allow_html=True,
)

# ── AQI helpers ──────────────────────────────────────────────────────────────
AQI_LEVELS = [
    (50,  "Good",                    "#3fb950", "#0d2518"),
    (100, "Moderate",                "#d29922", "#2b2000"),
    (150, "Unhealthy (Sensitive)",   "#e3812b", "#2d1a00"),
    (200, "Unhealthy",               "#f85149", "#2d0f0f"),
    (300, "Very Unhealthy",          "#bc8cff", "#1e0d33"),
    (500, "Hazardous",               "#ff7b72", "#2d0000"),
]

def hex_to_rgba(hex_str: str, alpha: float = 0.2) -> str:
    """Converts a 6-character HEX string to a Plotly-compatible rgba string."""
    hex_str = hex_str.lstrip("#")
    r, g, b = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

def _aqi_meta(aqi: float):
    """Return (label, color, bg) for a given AQI value."""
    for ceiling, label, color, bg in AQI_LEVELS:
        if aqi <= ceiling:
            return label, color, bg
    return "Hazardous", "#ff7b72", "#2d0000"


def _gauge(value: float, title: str = "Current AQI") -> go.Figure:
    label, color, _ = _aqi_meta(value)
    steps = []
    prev = 0
    palette = [c for _, _, c, _ in AQI_LEVELS]
    ceilings = [c for c, *_ in AQI_LEVELS]
    for ceiling, clr in zip(ceilings, palette):
        steps.append({"range": [prev, ceiling], "color": hex_to_rgba(clr, 0.2)})
        prev = ceiling

    fig = go.Figure(go.Indicator(
        mode="gauge+number",  # Removed "+delta" to clear the green '-' mark
        value=value,
        domain={'x': [0, 1], 'y': [0, 0.75]},  # Pulls the gauge down from the title space
        title={
            "text": f"<b>{title}</b><br><span style='font-size:0.8em5;color:{color}'>{label}</span>",
            "font": {"size": 16, "color": "#e6edf3"}
        },
        number={
            "font": {"size": 46, "color": color}, 
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 300], 
                "tickwidth": 1,
                "tickcolor": "#30363d", 
                "tickfont": {"color": "#8b949e"}
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": "#e6edf3", "width": 2},
                "thickness": 0.75, 
                "value": value,
            },
        },
    ))
    
    fig.update_layout(
        height=260, 
        margin={"t": 40, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="#0d1117", 
        plot_bgcolor="#0d1117", 
        font_color="#e6edf3",
    )
    
    return fig

def _forecast_chart(
    dates: list[str],
    values: list[float],
    ci_low: list[float | None],
    ci_high: list[float | None],
    current_aqi: float,
    current_label: str = "Now",
) -> go.Figure:
    fig = go.Figure()

    # Confidence band
    valid_ci = [(l, h) for l, h in zip(ci_low, ci_high) if l is not None and h is not None]
    if valid_ci:
        ci_dates = [d for d, (l, h) in zip(dates, zip(ci_low, ci_high)) if l is not None]
        fig.add_trace(go.Scatter(
            x=ci_dates + ci_dates[::-1],
            y=[h for _, h in valid_ci] + [l for l, _ in valid_ci],
            fill="toself", fillcolor="rgba(88,166,255,0.10)",
            line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", name="95% CI",
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers+text",
        line={"color": "#58a6ff", "width": 3},
        marker={"size": 10, "color": [_aqi_meta(v)[1] for v in values], "line": {"width": 2, "color": "#0d1117"}},
        text=[f"<b>{v:.0f}</b>" for v in values],
        textposition="top center",
        textfont={"color": "#e6edf3", "size": 12},
        name="Forecast AQI",
    ))

    # Current AQI reference line
    all_x = [current_label] + dates
    fig.add_trace(go.Scatter(
        x=[current_label, dates[0]],
        y=[current_aqi, current_aqi],
        mode="lines",
        line={"color": "#8b949e", "width": 1, "dash": "dot"},
        hoverinfo="skip", showlegend=False,
    ))

    # AQI zone bands
    zone_colors = [
        (0, 50, "#3fb950", "Good"),
        (51, 100, "#d29922", "Moderate"),
        (101, 150, "#e3812b", ""),
        (151, 200, "#f85149", ""),
        (201, 300, "#bc8cff", ""),
    ]
    for lo, hi, clr, _ in zone_colors:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=clr + "18", line_width=0)

    fig.update_layout(
        height=300,
        margin={"t": 10, "b": 30, "l": 10, "r": 10},
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        yaxis={"title": "AQI", "range": [0, max(300, max(values) + 30)],
               "gridcolor": "#21262d", "zerolinecolor": "#21262d"},
        xaxis={"gridcolor": "#21262d"},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#8b949e"}},
        hovermode="x unified",
    )
    return fig


def _history_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["european_aqi"],
        mode="lines",
        line={"color": "#58a6ff", "width": 2},
        fill="tozeroy", 
        fillcolor=hex_to_rgba("#58a6ff", 0.15),
        name="AQI",
        hovertemplate="<b>%{y:.0f}</b><extra>%{x|%b %d %H:%M}</extra>",
    ))
    for lo, hi, clr, lbl in [(0, 50, "#3fb950", "Good"), (51, 100, "#d29922", "Moderate"),
                               (101, 150, "#e3812b", "USG"), (151, 200, "#f85149", "Unhealthy"),
                               (201, 300, "#bc8cff", "Very Unhealthy")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=hex_to_rgba(clr, 0.1), line_width=0)

    fig.update_layout(
        height=250, margin={"t": 10, "b": 30, "l": 10, "r": 10},
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
        yaxis={"title": "AQI", "gridcolor": "#21262d", "zerolinecolor": "#21262d"},
        xaxis={"gridcolor": "#21262d"},
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def _pollutant_card(label: str, value: float, unit: str, color: str) -> str:
    return f"""
<div class="card">
  <div class="card-title">{label}</div>
  <div class="card-value" style="color:{color}">{value:.1f}</div>
  <div class="card-sub">{unit}</div>
</div>"""


# ── Cached data loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Hopsworks …")
def _get_registry():
    """One Hopsworks session for the entire Streamlit process lifetime."""
    from src.inference import get_hopsworks_registry
    return get_hopsworks_registry()


@st.cache_data(ttl=3600, show_spinner="Fetching latest air-quality data …")
def _fetch_data(city: str):
    """Pulls 7 days of hourly AQI + weather, engineers features. TTL = 1 h."""
    from src.inference import get_latest_features
    return get_latest_features(city)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_current(city: str) -> dict:
    from src.data.api_client import APIClientFactory
    client = APIClientFactory.get_primary_client()
    return client.fetch_current(city)


@st.cache_data(ttl=86400, show_spinner=False)
def _load_model_cached(horizon: int):
    """Download production model once per day per horizon."""
    from src.inference import load_production_model
    mr = _get_registry()
    return load_production_model(mr, horizon)


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    city = "Lahore"   # single-city version; driven by settings.CITY

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌬️ Pearls AQI")
        st.caption(f"City: **{city}**")
        st.caption(f"Updated: {datetime.now().strftime('%H:%M, %d %b %Y')}")
        st.divider()

        if st.button("🔄  Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("**Forecast Horizon**")
        horizon_choice = st.radio("Days ahead", [1, 2, 3], index=2, horizontal=True,
                                  label_visibility="collapsed")

        st.divider()
        st.markdown('<div class="section-header">AQI Guide</div>', unsafe_allow_html=True)
        for ceiling, lbl, clr, _ in AQI_LEVELS:
            st.markdown(
                f'<span class="aqi-badge" style="background:{hex_to_rgba(clr, 0.15)}; color:{clr}">'
                f'0–{ceiling}</span> &nbsp; {lbl}',
                unsafe_allow_html=True,
            )

    # ── Fetch current AQI ────────────────────────────────────────────────────
    current_data: dict | None = None
    current_aqi = None
    with st.spinner("Fetching current AQI …"):
        try:
            current_data = _fetch_current(city)
            current_aqi = float(current_data.get("european_aqi", 0))
        except Exception as exc:
            st.error(f"⚠️ Could not fetch current AQI: {exc}")
            current_aqi = 0.0

    label, color, bg = _aqi_meta(current_aqi)

    # ── Page title ───────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='margin-bottom:4px'>🌬️ Pearls AQI Predictor</h1>"
        f"<p style='color:#8b949e;margin-top:0'>Real-time air quality monitoring & ML forecasting · {city}</p>",
        unsafe_allow_html=True,
    )

    # ── Alert banner ─────────────────────────────────────────────────────────
    if current_aqi <= 50:
        st.markdown(f'<div class="alert-good">✅ Air quality is <b>Good</b> — safe for all activities.</div>',
                    unsafe_allow_html=True)
    elif current_aqi <= 100:
        st.markdown(f'<div class="alert-moderate">⚠️ Air quality is <b>Moderate</b> — sensitive individuals should limit prolonged outdoor exertion.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-bad">🚨 Air quality is <b>{label}</b> (AQI {current_aqi:.0f}) — consider reducing outdoor activity.</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Gauge + pollutant cards ───────────────────────────────────────
    col_gauge, col_polls = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(_gauge(current_aqi), width="stretch")

    with col_polls:
        st.markdown('<div class="section-header">Key Pollutants (current hour)</div>',
                    unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        pollutants = [
            ("PM 2.5",  "pm2_5",            "µg/m³", "#f85149"),
            ("PM 10",   "pm10",             "µg/m³", "#e3812b"),
            ("NO₂",     "nitrogen_dioxide", "µg/m³", "#d29922"),
            ("Ozone",   "ozone",            "µg/m³", "#58a6ff"),
        ]
        for col, (lbl, key, unit, clr) in zip([r1, r2, r3, r4], pollutants):
            val = float(current_data.get(key, 0) or 0) if current_data else 0.0
            col.markdown(_pollutant_card(lbl, val, unit, clr), unsafe_allow_html=True)

        # Secondary row
        r5, r6, r7, r8 = st.columns(4)
        pollutants2 = [
            ("CO",          "carbon_monoxide",    "µg/m³", "#bc8cff"),
            ("SO₂",         "sulphur_dioxide",    "µg/m³", "#3fb950"),
            ("UV Index",    "uv_index",           "",      "#ffa657"),
            ("Temperature", "temperature_2m",     "°C",    "#79c0ff"),
        ]
        for col, (lbl, key, unit, clr) in zip([r5, r6, r7, r8], pollutants2):
            val = float(current_data.get(key, 0) or 0) if current_data else 0.0
            col.markdown(_pollutant_card(lbl, val, unit, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Historical + Forecast ─────────────────────────────────────────
    col_hist, col_fc = st.columns(2)

    with col_hist:
        st.markdown('<div class="section-header">📈 7-Day Historical AQI (hourly)</div>',
                    unsafe_allow_html=True)
        try:
            engineered_df = _fetch_data(city)
            raw_cols = ["date", "european_aqi"]
            hist_df = engineered_df[raw_cols].dropna().copy()
            hist_df["date"] = pd.to_datetime(hist_df["date"])
            st.plotly_chart(_history_chart(hist_df), width="stretch")
        except Exception as exc:
            st.warning(f"Historical data unavailable: {exc}")
            engineered_df = None

    with col_fc:
        st.markdown(f'<div class="section-header">🔮 {horizon_choice}-Day Forecast</div>',
                    unsafe_allow_html=True)

        forecast_rows = []
        models_info = []

        if engineered_df is not None:
            for day in range(1, horizon_choice + 1):
                try:
                    model, model_name, version, rmse, model_dir = _load_model_cached(day)
                    if model is None:
                        forecast_rows.append({"day": day, "aqi": None, "ci_low": None, "ci_high": None})
                        continue
                    from src.inference import predict_horizon
                    pred = predict_horizon(model, engineered_df, current_aqi)
                    ci_low  = (pred - 1.96 * rmse) if (pred is not None and rmse) else None
                    ci_high = (pred + 1.96 * rmse) if (pred is not None and rmse) else None
                    forecast_rows.append({"day": day, "aqi": pred, "ci_low": ci_low, "ci_high": ci_high})
                    models_info.append({"Day": f"Day {day}", "Model": model_name or "—",
                                        "Version": version, "RMSE": f"{rmse:.2f}" if rmse else "—"})
                except Exception as exc:
                    st.warning(f"Day-{day} forecast failed: {exc}")
                    forecast_rows.append({"day": day, "aqi": None, "ci_low": None, "ci_high": None})

        # Build date labels
        today = datetime.now()
        day_labels = [(today + timedelta(days=r["day"])).strftime("%a %b %d")
                      for r in forecast_rows]
        fc_values = [r["aqi"] for r in forecast_rows]
        ci_low    = [r["ci_low"] for r in forecast_rows]
        ci_high   = [r["ci_high"] for r in forecast_rows]

        if any(v is not None for v in fc_values):
            # Replace None with interpolated for display
            display_vals = [v if v is not None else current_aqi for v in fc_values]
            st.plotly_chart(
                _forecast_chart(day_labels, display_vals, ci_low, ci_high, current_aqi),
                width="stretch",
            )

            # Alert on bad forecast
            max_fc = max(v for v in fc_values if v is not None)
            lbl_fc, clr_fc, _ = _aqi_meta(max_fc)
            if max_fc > 150:
                st.markdown(
                    f'<div class="alert-bad">🚨 Forecast peak: <b>{max_fc:.0f}</b> ({lbl_fc})</div>',
                    unsafe_allow_html=True,
                )
            elif max_fc > 100:
                st.markdown(
                    f'<div class="alert-moderate">⚠️ Forecast peak: <b>{max_fc:.0f}</b> ({lbl_fc})</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No production models available yet — run the training pipeline first.", icon="⚙️")

    # ── Row 3: Model info + SHAP ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_model, col_shap = st.columns([1, 2])

    with col_model:
        st.markdown('<div class="section-header">🤖 Active Models</div>', unsafe_allow_html=True)
        if models_info:
            st.dataframe(
                pd.DataFrame(models_info),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No models promoted to production yet.")

        # Current weather snapshot
        if current_data:
            st.markdown('<div class="section-header">🌤 Weather Snapshot</div>', unsafe_allow_html=True)
            weather_items = {
                "💧 Humidity":   f"{current_data.get('relative_humidity_2m', '—'):.0f} %" if current_data.get('relative_humidity_2m') else "—",
                "💨 Wind":       f"{current_data.get('wind_speed_10m', '—'):.1f} km/h" if current_data.get('wind_speed_10m') else "—",
                "🌧 Rain":       f"{current_data.get('rain', 0):.1f} mm" if current_data.get('rain') is not None else "—",
                "☁️ Cloud cover": f"{current_data.get('cloud_cover', '—'):.0f} %" if current_data.get('cloud_cover') else "—",
            }
            for k, v in weather_items.items():
                st.markdown(
                    f'<div class="card" style="padding:10px 14px">'
                    f'<span style="color:#8b949e">{k}</span> &nbsp; <b>{v}</b></div>',
                    unsafe_allow_html=True,
                )

    with col_shap:
        st.markdown('<div class="section-header">🔍 Feature Importance (SHAP)</div>',
                    unsafe_allow_html=True)
        shap_found = False
        for day in range(1, 4):
            try:
                _, mname, _, _, model_dir = _load_model_cached(day)
                if model_dir:
                    shap_path = os.path.join(model_dir, "shap_summary.png")
                    if os.path.exists(shap_path):
                        st.caption(f"Model: **{mname}** · Horizon: **{day}d**")
                        st.image(shap_path, width="stretch")
                        shap_found = True
                        break
            except Exception:
                pass

        if not shap_found:
            st.info(
                "SHAP explanations will appear here after the training pipeline runs and uploads them to Hopsworks.",
                icon="ℹ️",
            )
            # Show a placeholder feature list instead
            st.markdown("""
**Top predictors (expected after training):**
1. `aqi_lag_1h` — AQI one hour ago (strongest signal)
2. `aqi_lag_24h` — Yesterday same hour
3. `aqi_roll_mean_24h` — 24-hour rolling average
4. `aqi_ewm_6h` — Exponential weighted mean
5. `wind_u / wind_v` — Wind vector components
6. `temp_trend_7d` — Warming / cooling trend
7. `no2_o3_ratio` — NO₂ × Ozone interaction
8. `rain_accum_7d` — Monsoon accumulation signal
""")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Data: [Open-Meteo](https://open-meteo.com) · "
        "Models: Hopsworks Model Registry · "
        "Built using Streamlit + Plotly"
    )


if __name__ == "__main__":
    main()