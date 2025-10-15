# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os

# from src.data_collection.simulate_sensors import simulate_sensor_data
# from src.machine_learning.crop_prediction import train_crop_model, predict_crop
# from src.machine_learning.irrigation_prediction import train_irrigation_model, predict_irrigation
# from src.machine_learning.fertilizer_prediction import train_fertilizer_model, predict_fertilizer

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # -----------------------------
# # Page Setup
# # -----------------------------
# st.set_page_config(page_title="Precision Agriculture", page_icon="🌱", layout="centered")
# st.title("🌱 Precision Agriculture Dashboard")
# st.markdown("This system predicts the best **crop**, **fertilizer**, and **irrigation** level based on soil and weather data, and detects **plant diseases** from leaf images.")

# # -----------------------------
# # Load ML Models
# # -----------------------------
# MODEL_PATH = "models/plant_disease_model.h5"
# if os.path.exists(MODEL_PATH):
#     disease_model = load_model(MODEL_PATH)
#     class_names = sorted(os.listdir("data/PlantVillage/train")) if os.path.exists("data/PlantVillage/train") else []
# else:
#     disease_model = None
#     class_names = []

# crop_model = train_crop_model()
# irrigation_model = train_irrigation_model()
# fertilizer_model = train_fertilizer_model()

# # -----------------------------
# # Sensor Input Section
# # -----------------------------
# st.header("🌾 Soil and Weather Data Input")

# use_random = st.checkbox("🌦️ Use Random Sensor Data Instead")

# if not use_random:
#     st.write("You can enter values manually (leave blank to auto-fill random values):")

# col1, col2, col3 = st.columns(3)
# col4, col5, col6, col7 = st.columns(4)

# def get_value(label, min_val, max_val, default=None):
#     """Helper: return user input or random value"""
#     val = st.number_input(f"{label}", min_val, max_val, default if default is not None else min_val)
#     return val

# if use_random:
#     data = simulate_sensor_data()
# else:
#     data = {
#         "N": col1.number_input("Nitrogen (N)", min_value=0, max_value=200, value=70),
#         "P": col2.number_input("Phosphorus (P)", min_value=0, max_value=200, value=56),
#         "K": col3.number_input("Potassium (K)", min_value=0, max_value=200, value=192),
#         "temperature": col4.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=34.6),
#         "humidity": col5.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=59.8),
#         "ph": col6.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.2),
#         "rainfall": col7.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=30.3)
#     }

# # -----------------------------
# # Prediction Button
# # -----------------------------
# if st.button("🔍 Predict Crop, Irrigation, and Fertilizer"):
#     st.subheader("📟 Sensor Data:")
#     st.json(data)

#     # Crop Recommendation
#     crop = predict_crop(crop_model, data)
#     st.success(f"🌾 Recommended Crop: **{crop}**")

#     # Irrigation Recommendation
#     irrigation = predict_irrigation(irrigation_model, data)
#     st.info(f"💧 Irrigation Recommendation: **{irrigation}**")

#     # Fertilizer Recommendation
#     fertilizer = predict_fertilizer(fertilizer_model, data)
#     st.warning(f"🌿 Fertilizer Recommendation: **{fertilizer}**")

# # -----------------------------
# # Data Visualization Section
# # -----------------------------
# st.header("📊 Data Visualizations")

# if st.checkbox("Show Crop Distribution"):
#     df = pd.read_csv("data/crop_recommendation.csv")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.countplot(x="label", data=df, order=df["label"].value_counts().index, ax=ax)
#     plt.xticks(rotation=90)
#     st.pyplot(fig)

# if st.checkbox("Show Feature Correlation Heatmap"):
#     df = pd.read_csv("data/crop_recommendation.csv")
#     numeric_df = df.drop(columns=["label"])
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# # -----------------------------
# # Plant Disease Detection Section
# # -----------------------------
# st.header("🌿 Plant Disease Detection")

# uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     img = image.load_img(uploaded_file, target_size=(128, 128))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

#     if disease_model:
#         prediction = disease_model.predict(img_array)
#         pred_class = class_names[np.argmax(prediction)] if class_names else "Unknown Class"
#         confidence = np.max(prediction) * 100
#         st.success(f"🧠 Prediction: **{pred_class}**  \n🎯 Confidence: **{confidence:.2f}%**")
#     else:
#         st.error("🚧 Model not trained yet. Please train `plant_disease_model.h5` first.")

import streamlit as st

st.set_page_config(page_title="Precision Agriculture", page_icon="🌱", layout="centered")

st.title("🌱 Precision Agriculture Dashboard")

st.markdown("""
### Welcome to the Precision Agriculture System!  

This project integrates **Machine Learning** and **Visual Computing**  
to assist farmers in making smarter decisions for:
- 🌾 **Crop recommendation**
- 💧 **Irrigation and fertilizer management**
- 🌿 **Plant disease detection** using computer vision  

Use the **sidebar** on the left to navigate between modules:
1. 🌾 Crop, Irrigation & Fertilizer Prediction  
2. 📊 Data Visualizations  
3. 🌿 Plant Disease Detection
""")

st.info("Navigate using the sidebar to explore the different modules.")
