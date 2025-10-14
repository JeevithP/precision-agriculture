import streamlit as st
from src.data_collection.simulate_sensors import simulate_sensor_data
from src.machine_learning.crop_prediction import train_crop_model, predict_crop
from src.machine_learning.irrigation_prediction import train_irrigation_model, predict_irrigation
from src.machine_learning.fertilizer_prediction import train_fertilizer_model, predict_fertilizer

st.title("🌱 Precision Agriculture Dashboard")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model once
MODEL_PATH = "models/plant_disease_model.h5"
if os.path.exists(MODEL_PATH):
    disease_model = load_model(MODEL_PATH)
    # You need class names mapping
    class_names = sorted(os.listdir("data/PlantVillage/train"))
else:
    disease_model = None
    class_names = []

crop_model = train_crop_model()
irrigation_model = train_irrigation_model()
fertilizer_model = train_fertilizer_model()

if st.button("Simulate Sensor Data"):
    data = simulate_sensor_data()
    st.write("Sensor Data:", data)

    # Crop Recommendation
    crop = predict_crop(crop_model, data)
    st.success(f"Recommended Crop: {crop}")

    # Irrigation Recommendation
    irrigation = predict_irrigation(irrigation_model, data)
    st.info(f"Irrigation Recommendation: {irrigation}")

    # Fertilizer Recommendation
    fertilizer = predict_fertilizer(fertilizer_model, data)
    st.warning(f"Fertilizer Recommendation: {fertilizer}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("📊 Data Visualizations")

if st.checkbox("Show Crop Distribution"):
    df = pd.read_csv("data/crop_recommendation.csv")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(x="label", data=df, order=df["label"].value_counts().index, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

if st.checkbox("Show Feature Correlation"):
    df = pd.read_csv("data/crop_recommendation.csv")
    numeric_df = df.drop(columns=["label"])  # remove crop labels
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


from tensorflow.keras.preprocessing import image
import numpy as np

st.subheader("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    if disease_model:
        prediction = disease_model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"Prediction: {pred_class} (Confidence: {confidence:.2f}%)")
    else:
        st.error("🚧 Model not trained yet. Please train plant_disease_model.h5 first.")
