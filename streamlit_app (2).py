import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Car Price Predictor")

st.title("Car Price Prediction")
st.write("Enter the car details and get an estimated selling price (lakh INR).")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load():
    model = joblib.load(os.path.join(BASE, "model_artifacts", "car_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "model_artifacts", "scaler.pkl"))
    features = joblib.load(os.path.join(BASE, "model_artifacts", "features.pkl"))
    return model, scaler, features

try:
    model, scaler, features = load()
except FileNotFoundError:
    st.error("Couldn't find the model files. Run the notebook first to generate model_artifacts.")
    st.stop()

st.sidebar.header("Car details")
present_price = st.sidebar.number_input("Present price (lakh INR)", 0.1, 50.0, 5.0, 0.1)
kms_driven = st.sidebar.number_input("Kms driven", 0, 500000, 30000, step=1000)
fuel = st.sidebar.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
brand = st.sidebar.text_input("Brand", "Maruti")
age = st.sidebar.slider("Car age (years)", 0, 25, 5)

if st.button("Predict"):
    row = {f: 0 for f in features}
    values = {
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type_enc": {"Petrol": 2, "Diesel": 1, "CNG": 0}[fuel],
        "Seller_Type_enc": {"Dealer": 0, "Individual": 1}[seller],
        "Transmission_enc": {"Manual": 1, "Automatic": 0}[transmission],
        "Owner_enc": {"First Owner": 0, "Second Owner": 2, "Third Owner": 4,
                      "Fourth & Above Owner": 1, "Test Drive Car": 3}[owner],
        "brand_enc": hash(brand) % 1000,
        "age": age,
    }
    for k, v in values.items():
        if k in row:
            row[k] = v

    X = np.array([[row[f] for f in features]])
    pred = model.predict(scaler.transform(X))[0]
    st.success(f"Estimated selling price: {pred:.2f} lakh INR")
