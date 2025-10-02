import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from joblib import load
import requests
from datetime import datetime, timedelta
import logging

# --- Imports for fetching real NASA data and loading environment variables ---
import earthaccess
import xarray as xr
from dotenv import load_dotenv

# --- Setup and Configuration ---
load_dotenv() # Load variables from .env file at the start
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Load Predictive Model ---
try:
    model = load('aqi_forecaster.joblib')
    logging.info("✅ Predictive model loaded successfully!")
except FileNotFoundError:
    model = None
    logging.error("❌ Model file 'aqi_forecaster.joblib' not found.")
except Exception as e:
    model = None
    logging.error(f"❌ An error occurred loading the model: {e}")

# --- Helper Functions ---
def get_aqi_level(pm25):
    """Determines the AQI category and color from a PM2.5 value."""
    if pm25 is None or pm25 < 0: return {"level": "Unknown", "color": "#aaaaaa"}
    if pm25 <= 12.0: return {"level": "Good", "color": "#00e400"}
    if pm25 <= 35.4: return {"level": "Moderate", "color": "#ffff00"}
    if pm25 <= 55.4: return {"level": "Unhealthy for Sensitive Groups", "color": "#ff7e00"}
    if pm25 <= 150.4: return {"level": "Unhealthy", "color": "#ff0000"}
    if pm25 <= 250.4: return {"level": "Very Unhealthy", "color": "#8f3f97"}
    return {"level": "Hazardous", "color": "#7e0023"}

# --- REAL-TIME NASA TEMPO DATA FUNCTION ---
def get_tempo_data(lat, lon):
    """
    Fetches real-time TEMPO NO2 data from NASA Earthdata for a given lat/lon.
    """
    logging.info(f"Attempting to fetch real TEMPO data for ({lat}, {lon}).")
    try:
        os.environ['EARTHDATA_USERNAME'] = os.environ.get('EARTHDATA_USER')
        os.environ['EARTHDATA_PASSWORD'] = os.environ.get('EARTHDATA_PASS')

        if not os.environ.get('EARTHDATA_USERNAME') or not os.environ.get('EARTHDATA_PASSWORD'):
            raise ValueError("EARTHDATA_USER and EARTHDATA_PASS not found in .env file.")

        auth = earthaccess.login(strategy="environment")
        if not auth.authenticated:
            logging.warning("NASA Earthdata authentication failed. Check credentials in .env file.")
            return {'status': 'Auth Error', 'value': None}

        search_results = earthaccess.search_data(
            short_name='TEMPO_NRT_L2_NO2',
            bounding_box=(lon-0.1, lat-0.1, lon+0.1, lat+0.1),
            count=1
        )

        if not search_results:
            logging.warning("No TEMPO data found for this location. Expected outside North America or nighttime.")
            return {'status': 'No Data Available', 'value': None}

        filepaths = earthaccess.download(search_results, local_path="temp_data")
        if not filepaths: raise ValueError("Download failed.")

        with xr.open_dataset(filepaths[0], group="product") as ds:
            data_point = ds['nitrogendioxide_tropospheric_column'].sel(
                latitude=lat, longitude=lon, method='nearest'
            )
            no2_value = data_point.item()
        
        try: os.remove(filepaths[0])
        except Exception as e: logging.warning(f"Could not remove temporary file: {e}")
        
        logging.info(f"✅ Successfully fetched real TEMPO NO2 value: {no2_value:.2e}")
        return {'status': 'Real-Time', 'value': no2_value}

    except Exception as e:
        logging.error(f"❌ An error occurred during real TEMPO data fetch: {e}")
        return {'status': 'Simulated (Error)', 'value': 1.5e-5}

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the landing page (index.html)."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Serves the main interactive dashboard (dashboard.html)."""
    return render_template('dashboard.html')

@app.route('/api/geocode')
def geocode():
    city = request.args.get('city')
    if not city: return jsonify({"error": "City parameter is required"}), 400
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            location = data['results'][0]
            return jsonify({"latitude": location['latitude'], "longitude": location['longitude']})
        else:
            return jsonify({"error": "City not found"}), 404
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Geocoding API request failed: {e}")
        return jsonify({"error": "Failed to connect to geocoding service"}), 500

@app.route('/api/data')
def get_sensor_data():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing lat/lon parameters"}), 400

    # 1. Fetch Weather Data
    try:
        weather_url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                       f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m")
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()['current']
    except Exception as e:
        logging.error(f"❌ Could not fetch weather data: {e}")
        return jsonify({"error": "Failed to fetch weather data"}), 500

    # 2. ✨ REAL-TIME Ground Air Quality Data (Using WAQI API) ✨
    pm25_value = 25.0  # Default fallback value
    location_name = "Regional Estimate"
    try:
        waqi_api_key = "863296fb81e839b073505ca569860cdf03f1ce80" 
        city = None
        
        try:
            reverse_geo_url = f"https://geocoding-api.open-meteo.com/v1/reverse?latitude={lat}&longitude={lon}&count=1&language=en&format=json"
            rg_response = requests.get(reverse_geo_url, timeout=5)
            rg_response.raise_for_status()
            rg_data = rg_response.json()
            if rg_data.get('results'):
                city = rg_data['results'][0].get('name')
                logging.info(f"✅ Reverse geocoded coordinates to city: {city}")
        except Exception as rg_e:
            logging.warning(f"⚠️ Could not reverse geocode coordinates: {rg_e}. Will use geo-based search instead.")

        if city:
             waqi_url = f"https://api.waqi.info/feed/{city}/?token={waqi_api_key}"
             logging.info(f"Attempting to fetch WAQI data using city: {city}")
        else:
             waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={waqi_api_key}"
             logging.info(f"Attempting to fetch WAQI data using coordinates: {lat},{lon}")

        waqi_response = requests.get(waqi_url, timeout=10)
        waqi_response.raise_for_status()
        waqi_json = waqi_response.json()

        if waqi_json.get('status') == 'ok' and 'data' in waqi_json:
            waqi_data = waqi_json['data']
            if 'pm25' in waqi_data.get('iaqi', {}):
                pm25_value = waqi_data['iaqi']['pm25']['v']
                location_name = waqi_data.get('city', {}).get('name', 'Unknown Station')
                logging.info(f"✅ Successfully fetched real WAQI data from {location_name}")
            else:
                logging.warning("WAQI API returned data but no PM2.5 value for this location.")
        else:
            logging.warning(f"WAQI API returned status: {waqi_json.get('status')}. Message: {waqi_json.get('data')}")
            
    except Exception as e:
        logging.error(f"❌ Could not fetch WAQI data: {e}. Using default values.")

    # 3. Fetch REAL TEMPO Satellite Data
    tempo_data = get_tempo_data(lat, lon)

    # 4. Generate 6-Hour Forecast
    forecast_results = []
    if model:
        try:
            now = datetime.now()
            feature_data = {
                'hour': now.hour, 'day_of_week': now.weekday(), 'pm25': pm25_value,
                'temperature_2m': weather_data.get('temperature_2m', 18),
                'relative_humidity_2m': weather_data.get('relative_humidity_2m', 60),
                'wind_speed_10m': weather_data.get('wind_speed_10m', 5),
                'wind_direction_10m': weather_data.get('wind_direction_10m', 180)
            }
            feature_order = ['pm25', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'hour', 'day_of_week']
            current_features = pd.DataFrame([feature_data], columns=feature_order)

            for i in range(6):
                next_hour_prediction = model.predict(current_features)[0]
                next_hour_prediction = max(0, next_hour_prediction)
                forecast_time = now + timedelta(hours=i + 1)
                
                forecast_results.append({"hour": forecast_time.strftime("%H:%M"), "aqi": round(next_hour_prediction, 2)})
                
                current_features['pm25'] = next_hour_prediction
                current_features['hour'] = forecast_time.hour
                current_features['day_of_week'] = forecast_time.weekday()
        except Exception as e:
            logging.error(f"❌ Prediction failed: {e}")
    
    if not forecast_results:
        for i in range(6):
             forecast_time = datetime.now() + timedelta(hours=i + 1)
             forecast_results.append({"hour": forecast_time.strftime("%H:%M"), "aqi": round(pm25_value, 2)})
             
    # 5. Assemble final JSON response
    response_data = {
        "ground_data": {"location": location_name, "value": round(pm25_value, 2), "unit": "µg/m³", "level_info": get_aqi_level(pm25_value)},
        "weather_data": weather_data,
        "satellite_data": tempo_data,
        "forecast_data": forecast_results
    }
    return jsonify(response_data)

if __name__ == '__main__':
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    app.run(debug=True, port=5000)

