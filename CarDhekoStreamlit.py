import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats  # Import the stats module
import json
import re

# Load all six datasets
files = [
    r"C:\Users\Lenovo\Downloads\drive-download-20240812T150704Z-001\bangalore_cars.xlsx",
    r'C:\Users\Lenovo\Downloads\drive-download-20240812T150704Z-001\chennai_cars.xlsx',
    r'C:\Users\Lenovo\Downloads\drive-download-20240812T150704Z-001\delhi_cars.xlsx',
    r'C:\Users\Lenovo\Downloads\drive-download-20240812T150704Z-001\hyderabad_cars.xlsx',
    r'C:\Users\Lenovo\Downloads\drive-download-20240812T150704Z-001\jaipur_cars.xlsx',
    r'C:\Users\Lenovo\Downloads\drive-download-20240812T150704Z-001\kolkata_cars.xlsx'
]

dfs = [pd.read_excel(file) for file in files]

# Combine all the datasets into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Function to fix JSON format
def fix_json_format(text):
    if isinstance(text, str):
        text = re.sub(r"'", '"', text)
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        text = text.replace('None', 'null').replace('True', 'true').replace('False', 'false')
        return text
    return text

# Function to safely parse JSON-like columns
def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e} for text: {text[:200]}")
        return None

# Apply the fix and parsing functions to relevant columns
combined_df['new_car_detail'] = combined_df['new_car_detail'].apply(fix_json_format).apply(safe_json_loads)

# Drop rows where 'new_car_detail' could not be parsed
combined_df = combined_df[combined_df['new_car_detail'].notna()]

# Function to extract key information from 'new_car_detail'
def extract_details(detail):
    if isinstance(detail, dict):
        return {
            'price': detail.get('price', None),
            'fuel_type': detail.get('ft', None),
            'body_type': detail.get('bt', None),
            'kilometers_driven': detail.get('km', None),
            'model_year': detail.get('modelYear', None)  # Added model year
        }
    return {'price': None, 'fuel_type': None, 'body_type': None, 'kilometers_driven': None, 'model_year': None}

# Extract the details into separate columns
details_df = combined_df['new_car_detail'].apply(extract_details)
details_df = pd.json_normalize(details_df)

# Merge extracted details back to the combined_df
combined_df = pd.concat([combined_df, details_df], axis=1)

# Dropping original JSON-like columns as we have extracted the necessary details
combined_df.drop(columns=['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs'], inplace=True)

# Updated clean_price function to handle both 'Lakh' and 'Crore'
def clean_price(price_str):
    if isinstance(price_str, str):
        # Remove currency symbols and convert 'Lakh' and 'Crore' to numeric values
        price_str = price_str.replace('₹', '').replace(',', '').strip()
        if 'Lakh' in price_str:
            return float(price_str.replace('Lakh', '').strip()) * 100000
        elif 'Crore' in price_str:
            return float(price_str.replace('Crore', '').strip()) * 10000000
    return np.nan

# Function to clean and convert kilometers driven to numeric
def clean_kilometers(km_str):
    if isinstance(km_str, str):
        return int(km_str.replace(',', '').strip())
    return np.nan

# Clean and convert the 'price' and 'kilometers_driven' columns
combined_df['price'] = combined_df['price'].apply(clean_price)
combined_df['kilometers_driven'] = combined_df['kilometers_driven'].apply(clean_kilometers)

# Remove rows with NaN values in 'price'
combined_df = combined_df.dropna(subset=['price'])

# Re-define X and y
X = combined_df.drop('price', axis=1)
y = combined_df['price']

# Define numeric and categorical features
numeric_features = ['kilometers_driven', 'model_year']
categorical_features = ['fuel_type', 'body_type']

# Ensure no NaN values in X
if X.isna().any().any():
    # Impute missing values in X if necessary
    X = X.fillna(method='ffill')  # or use other imputation methods

# Impute missing values with the median for numeric columns
imputer = SimpleImputer(strategy='median')
X[numeric_features] = imputer.fit_transform(X[numeric_features])

# Remove outliers using Z-score
combined_df = combined_df[(np.abs(stats.zscore(combined_df[numeric_features])) < 3).all(axis=1)]

# Check again if there are any rows left after outlier removal
if combined_df.empty:
    raise ValueError("The dataset is empty after outlier removal.")

# Re-define X and y after outlier removal
X = combined_df.drop('price', axis=1)
y = combined_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Training the model
model.fit(X_train, y_train)

# Streamlit Application
st.title("Car Price Prediction")

# Sidebar inputs
fuel_type = st.selectbox("Fuel Type", combined_df['fuel_type'].unique())
body_type = st.selectbox("Body Type", combined_df['body_type'].unique())
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, max_value=50000, step=1000)
model_year = st.number_input("Model Year", min_value=int(combined_df['model_year'].min()), max_value=int(combined_df['model_year'].max()))

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'fuel_type': [fuel_type],
        'body_type': [body_type],
        'kilometers_driven': [kilometers_driven],
        'model_year': [model_year]
    })
    
    # Predict the price using the trained model
    predicted_price = model.predict(input_data)[0]
    
    st.success(f"Predicted Price: ₹{predicted_price:.2f}")

# Model evaluation metrics
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
st.write(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, model.predict(X_test))):.2f}")
st.write(f"R^2 Score: {r2_score(y_test, model.predict(X_test)):.2f}")
