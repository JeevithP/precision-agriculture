import streamlit as st
import pandas as pd
import numpy as np
from src.data_collection.simulate_sensors import simulate_sensor_data
from src.machine_learning.crop_prediction import train_crop_model, predict_crop
from src.machine_learning.irrigation_prediction import train_irrigation_model, predict_irrigation
from src.machine_learning.fertilizer_prediction import train_fertilizer_model, predict_fertilizer

st.set_page_config(page_title="Crop, Irrigation & Fertilizer", page_icon="🌾", layout="wide")

st.title("🌾 Crop, Irrigation & Fertilizer Prediction")
st.write("Enter soil and weather parameters manually or use simulated random sensor values.")

# ----------------------------- #
# Load ML Models
# ----------------------------- #
@st.cache_resource
def load_models():
    crop_model = train_crop_model()
    irrigation_model = train_irrigation_model()
    fertilizer_model = train_fertilizer_model()
    return crop_model, irrigation_model, fertilizer_model

crop_model, irrigation_model, fertilizer_model = load_models()

st.subheader("📍 Region Information")

district = st.text_input(
    "Enter District (South India only, optional)",
    value="Mandya",
    help="Used to calibrate soil values using South Indian district data"
)

# ----------------------------- #
# Input Section
# ----------------------------- #
use_random = st.checkbox("🌦️ Use Random Sensor Data Instead")

if use_random:
    data = simulate_sensor_data()
else:
    st.subheader("🔢 Enter Sensor Data:")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6, col7 = st.columns(4)

    data = {
        "N": col1.number_input("Nitrogen (N)", min_value=0, max_value=200, value=70),
        "P": col2.number_input("Phosphorus (P)", min_value=0, max_value=200, value=56),
        "K": col3.number_input("Potassium (K)", min_value=0, max_value=200, value=192),
        "temperature": col4.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=34.6),
        "humidity": col5.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=59.8),
        "ph": col6.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.2),
        "rainfall": col7.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=30.3)
    }

# ----------------------------- #
# Prediction Section
# ----------------------------- #
if st.button("🔍 Predict Recommendations"):
    st.subheader("📟 Sensor Data")
    st.json(data)

    crop = predict_crop(crop_model, data, district=district)
    st.success(f"🌾 Recommended Crop: **{crop}**")

    irrigation = predict_irrigation(irrigation_model, data)
    st.info(f"💧 Irrigation Recommendation: **{irrigation}**")

    fertilizer = predict_fertilizer(fertilizer_model, data)
    st.warning(f"🌿 Fertilizer Recommendation: **{fertilizer}**")

