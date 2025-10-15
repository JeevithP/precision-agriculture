import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection")

# Load model
MODEL_PATH = "models/plant_disease_model.h5"
if os.path.exists(MODEL_PATH):
    disease_model = load_model(MODEL_PATH)
    class_names = sorted(os.listdir("data/PlantVillage/train")) if os.path.exists("data/PlantVillage/train") else []
else:
    disease_model = None
    class_names = []

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    if disease_model:
        prediction = disease_model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)] if class_names else "Unknown Class"
        confidence = np.max(prediction) * 100
        st.success(f"🧠 Prediction: **{pred_class}**  \n🎯 Confidence: **{confidence:.2f}%**")
    else:
        st.error("🚧 Model not trained yet. Please train `plant_disease_model.h5` first.")
