import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("turf_model.pkl")

# Title
st.title("Smart Turf Predictor")

st.write("Predict Turf Health Using Machine Learning")

# Inputs
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
moisture = st.number_input("Moisture")
ph = st.number_input("pH")

# Prediction Button
if st.button("Predict"):

    data = np.array([[temperature, humidity, moisture, ph]])

    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")

    # Recommendations
    if prediction[0] == "Dry":
        st.warning("Recommendation: Increase Watering")

    elif prediction[0] == "Fungus":
        st.error("Recommendation: Apply Fungicide")

    else:
        st.success("Turf Condition is Healthy")
        import os

print("Current Folder:")
print(os.getcwd())

print("\nFiles in folder:")
print(os.listdir())