import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; }
    .aqi-good { color: #00e400; }
    .aqi-moderate { color: #ffff00; }
    .aqi-unhealthy { color: #ff7e00; }
    .aqi-hazardous { color: #7e0023; }
</style>
""", unsafe_allow_html=True)

def get_aqi_color(aqi: int) -> str:
    """Return color based on AQI level."""
    if aqi <= 50: return "#00e400"  # Good
    elif aqi <= 100: return "#ffff00"  # Moderate
    elif aqi <= 150: return "#ff7e00"  # Unhealthy for Sensitive
    elif aqi <= 200: return "#ff0000"  # Unhealthy
    elif aqi <= 300: return "#8f3f97"  # Very Unhealthy
    else: return "#7e0023"  # Hazardous

def main():
    st.markdown('<p class="main-header">🌬️ Pearls AQI Predictor</p>', unsafe_allow_html=True)
    st.markdown("Predicting Air Quality Index 3 days ahead using Machine Learning")
    
    # Sidebar
    city = st.sidebar.selectbox("Select City", ["Barcelona", "Madrid", "Valencia"])
    model_type = st.sidebar.radio("Model", ["Random Forest", "LSTM", "Ensemble"])
    
    # Load latest features and model from Hopsworks
    # (In real app, you'd cache this with @st.cache_resource)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_aqi = 82  # Fetch from feature store
        color = get_aqi_color(current_aqi)
        st.metric("Current AQI", current_aqi, delta="Moderate")
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_aqi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 500]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#e8f5e9"},
                    {'range': [50, 100], 'color': "#fffde7"},
                    {'range': [100, 150], 'color': "#fff3e0"},
                    {'range': [150, 200], 'color': "#ffebee"},
                    {'range': [200, 300], 'color': "#f3e5f5"},
                    {'range': [300, 500], 'color': "#fce4ec"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("3-Day Forecast")
        # Line chart with predictions
        days = ['Today', 'Tomorrow', 'Day 3']
        predictions = [82, 95, 110]  # From model
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days, y=predictions,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=12)
        ))
        fig.update_layout(yaxis_range=[0, 200])
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("⚠️ Alerts")
        if max(predictions) > 150:
            st.error("🚨 Unhealthy AQI expected in 3 days!")
        elif max(predictions) > 100:
            st.warning("⚠️ Moderate AQI expected")
        else:
            st.success("✅ Good air quality expected")
    
    # SHAP Summary
    st.subheader("🔍 Why this prediction?")
    with st.expander("See feature importance"):
        st.image("docs/shap_summary.png")
        st.markdown("""
        **Top influencing factors:**
        1. AQI 24 hours ago (most important)
        2. Temperature × Humidity interaction
        3. Day of week (weekend vs weekday)
        4. Rolling 24h average AQI
        """)

if __name__ == "__main__":
    main()