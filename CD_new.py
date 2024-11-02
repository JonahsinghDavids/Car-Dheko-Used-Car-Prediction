import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import re
import pickle

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
            'model_year': detail.get('modelYear', None),
            'priceActual': detail.get('priceActual', None),
            'oem': detail.get('oem', None),
            'ownerNo': detail.get('ownerNo', None),
            'model': detail.get('model', None),
            'transmission': detail.get('transmission', None),
            'variantName': detail.get('variantName', None)
        }
    return {'price': None, 'fuel_type': None, 'body_type': None, 'kilometers_driven': None, 'model_year': None,
            'priceActual': None, 'oem': None, 'ownerNo': None, 'model': None, 'transmission': None, 'variantName': None}

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
numeric_features = ['kilometers_driven', 'model_year', 'ownerNo']
categorical_features = ['oem', 'model', 'fuel_type', 'body_type', 'transmission', 'variantName']

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

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Save the model to a .pkl file
model_filename = "car_price_prediction_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Save the extracted features into a DataFrame and store it as a CSV (if needed)
X.to_csv("extracted_features.csv", index=False)

# Predicting on the test set
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# Save evaluation metrics to be used in the Streamlit app
with open("evaluation_metrics.pkl", "wb") as f:
    pickle.dump({'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}, f)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")
