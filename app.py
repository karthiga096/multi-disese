import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Multi Disease Prediction System",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model(filename):
    model_path = os.path.join(MODEL_DIR, filename)
    return pickle.load(open(model_path, "rb"))

heart_model = load_model("heart_model.pkl")
diabetes_model = load_model("diabetes_model.pkl")
kidney_model = load_model("kidney_model.pkl")
import streamlit as st
import numpy as np
import pickle

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Multi Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🧠 Multi Disease Prediction System")
st.write("Predict **Heart Disease**, **Diabetes**, and **Kidney Disease** using Machine Learning")

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

heart_model = load_model("models/heart_model.pkl")
diabetes_model = load_model("models/diabetes_model.pkl")
kidney_model = load_model("models/kidney_model.pkl")

# ----------------- SIDEBAR -----------------
st.sidebar.header("Select Disease")
disease = st.sidebar.selectbox(
    "Choose a disease to predict",
    ("Heart Disease", "Diabetes", "Kidney Disease")
)

# ----------------- HEART DISEASE -----------------
if disease == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 50, 250)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG (0–2)", 0, 2)
    thalach = st.number_input("Maximum Heart Rate", 60, 250)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0)

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach, exang, oldpeak]])
        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

# ----------------- DIABETES -----------------
elif disease == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 300)
    blood_pressure = st.number_input("Blood Pressure", 0, 200)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Person is Diabetic")
        else:
            st.success("✅ Person is Not Diabetic")

# ----------------- KIDNEY DISEASE -----------------
else:
    st.subheader("🩺 Kidney Disease Prediction")

    age = st.number_input("Age", 1, 120)
    blood_pressure = st.number_input("Blood Pressure", 50, 200)
    albumin = st.number_input("Albumin", 0, 5)
    sugar = st.number_input("Sugar", 0, 5)
    blood_glucose = st.number_input("Blood Glucose Random", 50, 500)
    blood_urea = st.number_input("Blood Urea", 1, 300)
    serum_creatinine = st.number_input("Serum Creatinine", 0.1, 20.0)
    sodium = st.number_input("Sodium", 100, 200)
    potassium = st.number_input("Potassium", 2.0, 10.0)
    hemoglobin = st.number_input("Hemoglobin", 3.0, 20.0)

    if st.button("Predict Kidney Disease"):
        input_data = np.array([[age, blood_pressure, albumin, sugar,
                                blood_glucose, blood_urea,
                                serum_creatinine, sodium,
                                potassium, hemoglobin]])
        prediction = kidney_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Kidney Disease")
        else:
            st.success("✅ Low Risk of Kidney Disease")
