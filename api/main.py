import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from pathlib import Path

from pearls_aqi.live.openweather import (
    get_live_conditions,
    get_forecast_conditions,
)
from pearls_aqi.models.artifacts import load_model

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL FROM HOPSWORKS
# ============================================================

MODEL_FILENAME = "aqi_next_hour.joblib"

try:
    artifact = load_model(MODEL_FILENAME)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
except Exception as e:
    print(f"Warning: Failed to load model from Hopsworks: {e}")
    model = None
    preprocessor = None


# ============================================================
# REQUIRED FEATURES FOR MANUAL PREDICTION
# ============================================================

REQUIRED_FEATURES = [
    "city", "latitude", "longitude", "pm10", "pm2_5", 
    "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", 
    "ozone", "dust", "temperature", "humidity", "precipitation", 
    "wind_speed", "wind_direction", "pressure", "hour", 
    "day_of_week", "month", "month_name", "year", 
    "is_weekend", "season",
]

CITY_COORDINATES = {
    "Faisalabad": (31.4167, 73.0833),
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Peshawar": (34.0151, 71.5249),
}

# ============================================================
# AQI CATEGORY MAPPING & ALERTS
# ============================================================

def get_aqi_details(aqi_value: float) -> dict:
    """Map numerical AQI to categories and trigger alerts if hazardous."""
    aqi = round(float(aqi_value))
    
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Moderate"
    elif aqi <= 150:
        category = "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        category = "Unhealthy"
    elif aqi <= 300:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"
        
    return {
        "aqi_value": aqi,
        "category": category,
        "is_hazardous": aqi > 300,
        "alert": "⚠️ CRITICAL WARNING: Hazardous air quality detected!" if aqi > 300 else None
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "RandomForestRegressor",
        "service": "Pearls AQI Predictor",
    })


# ============================================================
# MANUAL NEXT-HOUR PREDICTION
# ============================================================

@app.post("/predict")
def predict():
    """Predict the next-hour numerical AQI."""
    if model is None:
        return jsonify({"error": "Model not loaded from Hopsworks."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must contain JSON data."}), 400

    missing = [feature for feature in REQUIRED_FEATURES if feature not in data]
    if missing:
        return jsonify({"error": "Missing required features.", "missing_features": missing}), 400

    try:
        input_data = pd.DataFrame([[data[feature] for feature in REQUIRED_FEATURES]], columns=REQUIRED_FEATURES)
        
        input_data["hour_sin"] = np.sin(2 * np.pi * input_data["hour"] / 24)
        input_data["hour_cos"] = np.cos(2 * np.pi * input_data["hour"] / 24)
        input_data["month_sin"] = np.sin(2 * np.pi * input_data["month"] / 12)
        input_data["month_cos"] = np.cos(2 * np.pi * input_data["month"] / 12)
        
        if "aqi_change_rate" not in input_data.columns:
            input_data["aqi_change_rate"] = 0.0

        transformed = preprocessor.transform(input_data)
        prediction_value = model.predict(transformed)[0]
        
        aqi_details = get_aqi_details(prediction_value)

        response = {
            "prediction": aqi_details["category"], 
            "prediction_value": aqi_details["aqi_value"],
            "model": "RandomForestRegressor",
            "prediction_horizon": "next_hour",
        }

        if aqi_details["is_hazardous"]:
            response["alert"] = aqi_details["alert"]

        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500


# ============================================================
# LIVE NEXT-HOUR PREDICTION
# ============================================================

@app.get("/predict-live")
def predict_live():
    """Fetch OpenWeather data and predict next-hour AQI."""
    if model is None:
        return jsonify({"error": "Model not loaded from Hopsworks."}), 500

    try:
        city = request.args.get("city", "Faisalabad")
        if city not in CITY_COORDINATES:
            return jsonify({"error": "Unsupported city."}), 400

        latitude, longitude = CITY_COORDINATES[city]
        live_data = get_live_conditions(city, latitude, longitude)

        input_data = pd.DataFrame([live_data])

        from datetime import datetime
        now = datetime.now()
        input_data["day_of_week"] = now.strftime("%A")
        input_data["month_name"] = now.strftime("%B")
        
        season_map = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"}
        input_data["season"] = season_map[now.month]

        if "aqi_change_rate" not in input_data.columns:
            input_data["aqi_change_rate"] = 0.0

        input_data = input_data[preprocessor.feature_names_in_]
        transformed = preprocessor.transform(input_data)
        prediction_value = model.predict(transformed)[0]
        
        aqi_details = get_aqi_details(prediction_value)

        response = {
            "city": city,
            "prediction": aqi_details["category"],
            "prediction_value": aqi_details["aqi_value"],
            "model": "RandomForestRegressor",
            "prediction_horizon": "next_hour",
            "live_data": live_data,
        }

        if aqi_details["is_hazardous"]:
            response["alert"] = aqi_details["alert"]

        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": "Live prediction failed.", "details": str(exc)}), 500


# ============================================================
# NEXT 3 DAYS PREDICTION
# ============================================================

@app.get("/predict-3days")
def predict_3days():
    """Fetch 72-hour forecast and generate numerical AQI predictions."""
    if model is None:
        return jsonify({"error": "Model not loaded from Hopsworks."}), 500

    try:
        city = request.args.get("city", "Faisalabad")
        if city not in CITY_COORDINATES:
            return jsonify({"error": "Unsupported city."}), 400

        latitude, longitude = CITY_COORDINATES[city]
        forecast_records = get_forecast_conditions(city, latitude, longitude, hours=72)

        if not forecast_records:
            return jsonify({"error": "No forecast data available."}), 500

        forecast_df = pd.DataFrame(forecast_records)
        from datetime import datetime
        now = datetime.now()
        
        forecast_start = pd.Timestamp(year=now.year, month=now.month, day=now.day)
        forecast_df["forecast_datetime"] = [forecast_start + pd.Timedelta(hours=i) for i in range(len(forecast_df))]
        forecast_df["day_of_week"] = forecast_df["forecast_datetime"].dt.day_name()
        
        month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        forecast_df["month_name"] = forecast_df["month"].map(month_names)
        
        season_map = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"}
        forecast_df["season"] = forecast_df["month"].map(season_map)

        if "aqi_change_rate" not in forecast_df.columns:
            forecast_df["aqi_change_rate"] = 0.0

        model_features = preprocessor.feature_names_in_
        missing_features = [f for f in model_features if f not in forecast_df.columns]

        if missing_features:
            return jsonify({"error": "Forecast data is missing model features."}), 500

        model_input = forecast_df[model_features]
        transformed = preprocessor.transform(model_input)
        predictions = model.predict(transformed)

        predictions_list = []
        has_hazardous = False
        
        for index, prediction_val in enumerate(predictions):
            record = forecast_records[index]
            forecast_datetime = forecast_df.iloc[index]["forecast_datetime"]
            
            aqi_details = get_aqi_details(prediction_val)
            if aqi_details["is_hazardous"]:
                has_hazardous = True

            predictions_list.append({
                "datetime": forecast_datetime.isoformat(),
                "city": city,
                "prediction": aqi_details["category"],
                "prediction_value": aqi_details["aqi_value"],
                "temperature": float(record["temperature"]),
                "humidity": float(record["humidity"]),
                "pm2_5": float(record["pm2_5"]),
                "pm10": float(record["pm10"]),
            })

        response = {
            "city": city,
            "prediction_horizon": "next_72_hours",
            "forecast_hours": len(predictions_list),
            "model": "RandomForestRegressor",
            "predictions": predictions_list,
        }
        
        if has_hazardous:
            response["alert"] = "⚠️ CRITICAL WARNING: Hazardous air quality predicted within the next 72 hours!"

        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": "3-day prediction failed.", "details": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
