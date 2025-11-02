import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("trained_car_price_model.pkl")
feature_names = model.feature_names_in_

st.title("Used Car Price Predictor")

# User input
brand = st.selectbox("Brand", ["Ford", "Toyota", "Mercedes-Benz", "BMW", "Other"])
model_year = st.number_input("Model Year", min_value=1970, max_value=2025, value=2015)
milage = st.number_input("Milage (km)", min_value=0, max_value=1000000, value=50000)
fuel_type = st.selectbox("Fuel Type", ["Gas", "Diesel", "Electric", "Unknown"])
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
accident = st.selectbox("Accident History", ["Yes", "No", "Unknown"])
clean_title = st.selectbox("Clean Title", ["Yes", "No", "Unknown"])

if st.button("Predict Price"):
    # Create input dataframe with zeros
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # Set numeric columns
    input_df["model_year"] = model_year
    input_df["milage"] = milage

    # Map categorical inputs
    categorical_mapping = {
        "brand": brand,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "accident": accident,
        "clean_title": clean_title
    }

    for col_prefix, cat in categorical_mapping.items():
        col_name = f"{col_prefix}_{cat}"
        if col_name in input_df.columns:
            input_df[col_name] = 1
        # Ignore categories not seen during training

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: {prediction*10:,.0f} NOK")
# prediction er ganget med 10 fordi det kommer i USD