# crop_yield_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from PIL import Image

# ---------- Background Image Setup ---------- #
@st.cache_resource

def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("farming_background.jpg")  # Replace with your background image

# ---------- Load ML Model ---------- #
@st.cache_resource

def load_model():
    with open("crop_yield_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- Page Title ---------- #
st.markdown("<h1 style='text-align: center; color: white;'>🌾 Crop Yield Prediction Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lightgray;'>Estimate crop yield (tons/hectare) using AI-driven insights</p>", unsafe_allow_html=True)

# ---------- Sidebar Inputs ---------- #
st.sidebar.header("🌿 Enter Farm Details")

crop_list = [
    "Wheat", "Rice", "Maize", "Barley", "Sugarcane", "Cotton", "Jowar", "Bajra", "Ragi", "Groundnut",
    "Soybean", "Mustard", "Gram", "Tur", "Moong", "Urad", "Sesame", "Lentil", "Castor", "Sunflower",
    "Sorghum", "Millet", "Peas", "Potato", "Tomato", "Onion", "Garlic", "Brinjal", "Chili", "Okra",
    "Cabbage", "Cauliflower", "Spinach", "Coriander", "Carrot", "Beetroot", "Pumpkin", "Bitter Gourd", "Bottle Gourd"
]

crop_type = st.sidebar.selectbox("Crop Type", sorted(crop_list))
state = st.sidebar.selectbox("State", ["Maharashtra", "Punjab", "Bihar", "Karnataka", "Tamil Nadu"])
rainfall = st.sidebar.slider("Annual Rainfall (mm)", 100, 3000, 1200)
temperature = st.sidebar.slider("Average Temperature (°C)", 10, 45, 28)
fertilizer_usage = st.sidebar.slider("Fertilizer Usage (kg/ha)", 0, 500, 150)

# ---------- Prediction Logic ---------- #
def preprocess_inputs():
    crop_map = {name: idx for idx, name in enumerate(sorted(crop_list))}
    state_map = {"Maharashtra": 0, "Punjab": 1, "Bihar": 2, "Karnataka": 3, "Tamil Nadu": 4}
    return np.array([[
        crop_map[crop_type],
        state_map[state],
        rainfall,
        temperature,
        fertilizer_usage
    ]])

if st.sidebar.button("🔍 Predict Yield"):
    input_data = preprocess_inputs()
    predicted_yield = model.predict(input_data)[0]
    st.subheader("📊 Predicted Crop Yield")
    st.success(f"🌱 Estimated Yield: {predicted_yield:.2f} tons/hectare")

    st.info("Note: This prediction is based on trained data. Real outcomes may vary depending on multiple factors including soil, pests, and market conditions.")

# ---------- Footer ---------- #
st.markdown("---")
st.markdown("<p style='text-align: center; color: lightgray;'>Built for Indian agriculture using AI and open data 🌍</p>", unsafe_allow_html=True)
