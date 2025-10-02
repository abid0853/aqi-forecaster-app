# train_model.py (Version 8 - Final Two-API Solution)

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys

# --- CONFIGURATION ---
# We'll train our model on data from Delhi, India.
LATITUDE = 28.65195
LONGITUDE = 77.23149

# --- 1. GET DATA FROM CORRECT SOURCES ---
print("Fetching data from specialized APIs...")

try:
    # --- API Call 1: Get Air Quality Data ---
    aq_api_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=pm2_5"
        f"&past_days=92"
        f"&timezone=auto"
    )
    print("Fetching PM2.5 data...")
    response_aq = requests.get(aq_api_url)
    response_aq.raise_for_status()
    data_aq = response_aq.json()['hourly']
    df_aq = pd.DataFrame(data_aq)
    df_aq = df_aq.rename(columns={'time': 'date', 'pm2_5': 'pm25'})
    
    # --- API Call 2: Get Weather Data ---
    weather_api_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        f"&past_days=92"
        f"&timezone=auto"
    )
    print("Fetching weather data...")
    response_weather = requests.get(weather_api_url)
    response_weather.raise_for_status()
    data_weather = response_weather.json()['hourly']
    df_weather = pd.DataFrame(data_weather)
    df_weather = df_weather.rename(columns={'time': 'date'})

    # --- Merge DataFrames ---
    df_aq['date'] = pd.to_datetime(df_aq['date'])
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    df = pd.merge(df_aq, df_weather, on='date')
    
    print("✅ Successfully fetched and merged all data.")

except requests.exceptions.RequestException as e:
    print(f"Error fetching data from Open-Meteo: {e}")
    sys.exit(1)
except KeyError:
    print("Error: Could not parse the JSON response. Check API status.")
    sys.exit(1)

# --- 2. PREPARE DATA ---
print("Preparing data for training...")

df.dropna(subset=['pm25'], inplace=True)
if df.empty:
    print("Error: No valid PM2.5 data available for the selected period.")
    sys.exit(1)

df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['pm25_next_hour'] = df['pm25'].shift(-1)
final_df = df.dropna()
print("Data preparation complete.")

# --- 3. TRAIN MODEL ---
print("Training the model...")

features = ['pm25', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'hour', 'day_of_week']
target = 'pm25_next_hour'

X = final_df[features]
y = final_df[target]

if len(X) < 100:
    print(f"Warning: The dataset is very small ({len(X)} samples). The model may not be accurate.")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")

# --- 4. SAVE THE MODEL ---
joblib.dump(model, 'aqi_forecaster.joblib')
print("✅ SUCCESS: Model saved as aqi_forecaster.joblib!")
