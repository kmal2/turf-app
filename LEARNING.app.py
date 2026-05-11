import streamlit as st
import joblib
import numpy as np
import os
import plotly.graph_objects as go

# ==============================
# Load Model Safely
# ==============================
model_path = os.path.join(os.path.dirname(__file__), "turf_model.pkl")
model = joblib.load(model_path)

# ==============================
# Page Config
# ==============================
import streamlit as st

st.set_page_config(
    page_title="LEARNING App",
    page_icon="🌱",
    layout="wide"
)

# ==============================
# Header
# ==============================
st.title("🌱 Smart Turf AI Dashboard")
st.markdown("### Machine Learning System for Turf Health Prediction")

st.divider()

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("📊 Sensor Inputs")

temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 30)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
moisture = st.sidebar.slider("Soil Moisture (%)", 0, 100, 40)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)

# ==============================
# Prediction
# ==============================
input_data = np.array([[temperature, humidity, moisture, ph]])
prediction = model.predict(input_data)[0]

# ==============================
# Result Display
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡 Temperature", f"{temperature} °C")

with col2:
    st.metric("💧 Humidity", f"{humidity} %")

with col3:
    st.metric("🌱 Moisture", f"{moisture} %")

st.divider()

# ==============================
# Prediction Result
# ==============================
st.subheader("🧠 Prediction Result")

if prediction == "Healthy":
    st.success("✅ Turf Condition: HEALTHY")
elif prediction == "Dry":
    st.warning("⚠️ Turf Condition: DRY - Needs Watering")
elif prediction == "Fungus":
    st.error("🚨 Turf Condition: FUNGUS RISK")
else:
    st.info(f"Prediction: {prediction}")

# ==============================
# Recommendation System
# ==============================
st.subheader("📌 Recommendation Engine")

if prediction == "Dry":
    st.write("💡 Increase irrigation immediately")
elif prediction == "Fungus":
    st.write("💡 Apply fungicide + reduce humidity exposure")
else:
    st.write("💡 Maintain current conditions")

# ==============================
# Visualization (Gauge Chart)
# ==============================
st.subheader("📊 Turf Health Indicator")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=moisture,
    title={'text': "Soil Moisture Level"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 30], 'color': "red"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "green"},
        ],
    }
))

st.plotly_chart(fig)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("🚀 Built with Machine Learning + Streamlit")