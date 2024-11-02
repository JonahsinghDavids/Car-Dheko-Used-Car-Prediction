import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_filename = "car_price_prediction_model.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)


# Load evaluation metrics
with open("evaluation_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# Access individual metrics
mae = metrics['mae']
mse = metrics['mse']
rmse = metrics['rmse']
r2 = metrics['r2']

# Load extracted features CSV to get unique values for dropdowns
features_df = pd.read_csv("extracted_features.csv")

# Streamlit App
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Session state to manage page navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Navigation Functions
def go_to_home():
    st.session_state['page'] = 'home'

def go_to_predict():
    st.session_state['page'] = 'predict'

def go_to_evaluation():
    st.session_state['page'] = 'evaluation'

# Home Page
if st.session_state['page'] == 'home':
    st.title("Car Dheko - Car Price Prediction")
    st.write("""
        This application helps predict the price of a used car based on various features.
        """)
    st.button("Go to Prediction", on_click=go_to_predict)
    st.button("Go to Model Evaluation", on_click=go_to_evaluation)

# Prediction Page
elif st.session_state['page'] == 'predict':
    st.title("Car Price Prediction")

    # Dropdown for OEM (Manufacturer)
    oem = st.selectbox("OEM (Manufacturer)", features_df['oem'].dropna().unique())

    # Filter models based on selected OEM
    filtered_models = features_df[features_df['oem'] == oem]['model'].dropna().unique()
    selected_model = st.selectbox("Car Model", filtered_models)

    # Filter body types based on selected model
    filtered_body_types = features_df[features_df['model'] == selected_model]['body_type'].dropna().unique()
    body_type = st.selectbox("Body Type", filtered_body_types)

    # Filter variants based on selected model
    filtered_variants = features_df[features_df['model'] == selected_model]['variantName'].dropna().unique()
    variant_name = st.selectbox("Variant Name", filtered_variants)

    # Filter fuel type and transmission based on selected variant
    filtered_fuel_types = features_df[features_df['variantName'] == variant_name]['fuel_type'].dropna().unique()
    fuel_type = st.selectbox("Fuel Type", filtered_fuel_types)

    filtered_transmissions = features_df[features_df['variantName'] == variant_name]['transmission'].dropna().unique()
    transmission = st.selectbox("Transmission", filtered_transmissions)

    # Additional fields
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, max_value=50000, step=1000)
    model_year = st.number_input("Model Year", min_value=int(features_df['model_year'].min()), max_value=int(features_df['model_year'].max()))
    owner_no = st.number_input("Number of Owners", min_value=0, max_value=5, step=1)

    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            'kilometers_driven': [kilometers_driven],
            'model_year': [model_year],
            'ownerNo': [owner_no],
            'oem': [oem],
            'model': [selected_model],
            'fuel_type': [fuel_type],
            'body_type': [body_type],
            'transmission': [transmission],
            'variantName': [variant_name]
        })

        # Predict price using the trained model
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹{predicted_price:.2f}")
    
    st.button("Go to Home", on_click=go_to_home)
    st.button("Go to Model Evaluation", on_click=go_to_evaluation)

# Model Evaluation Page
elif st.session_state['page'] == 'evaluation':
    st.title("Model Evaluation Metrics")
    st.write("Evaluate the model's performance on test data.")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    st.write(f"R-Squared: {r2:.2f}")
    st.button("Go to Home", on_click=go_to_home)
    st.button("Go to Prediction", on_click=go_to_predict)