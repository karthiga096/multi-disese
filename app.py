import streamlit as st
import numpy as np
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩸",
    layout="centered"
)

st.title("🩸 Diabetes Prediction System")
st.write("Predict if a person is diabetic using Machine Learning")

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "diabetes_model.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "diabetes_model.pkl")

# ---------------- DEBUG INFO ----------------
st.write("BASE_DIR:", BASE_DIR)
st.write("MODEL_DIR exists:", os.path.exists(MODEL_DIR))
if os.path.exists(MODEL_DIR):
    st.write("Files in models folder:", os.listdir(MODEL_DIR))
else:
    st.error("❌ models folder not found")
    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    st.write("Loading model from:", path)
    return pickle.load(open(path, "rb"))

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ {MODEL_FILE} not found!")
    st.stop()

diabetes_model = load_model(MODEL_FILE)

# ---------------- INPUT FORM ----------------
st.subheader("Enter patient data:")

pregnancies = st.number_input("Number of Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 300)
blood_pressure = st.number_input("Blood Pressure", 0, 200)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# ---------------- PREDICTION ----------------
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = diabetes_model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Person is Diabetic")
    else:
        st.success("✅ Person is NOT Diabetic")
