import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

from src.pearls_aqi.live.openweather import (
    get_live_conditions,
    get_forecast_conditions,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "local" / "aqi_next_hour.joblib"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pakistan_air_quality_final_clean_v2.csv"


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
preprocessor = artifact["preprocessor"]


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = preprocessor.feature_names_in_.tolist()


# ============================================================
# CITY COORDINATES
# ============================================================

CITY_COORDINATES = {
    "Faisalabad": (31.4504, 73.1350),
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Multan": (30.1575, 71.5249),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Rahim Yar Khan": (28.4212, 70.2989),
    "Rawalpindi": (33.5651, 73.0169),
    "Sialkot": (32.4945, 74.5229),
}


# ============================================================
# AQI CONVERSION
# ============================================================


def pm25_to_aqi(pm25):
    """
    Convert PM2.5 concentration to US EPA-style AQI.
    """

    if pd.isna(pm25):
        return np.nan

    pm25 = float(pm25)

    if pm25 < 0:
        return np.nan

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]

    pm25 = min(pm25, 500.4)

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low

            return round(aqi)

    return np.nan


# ============================================================
# TIME FEATURES
# ============================================================


def add_time_features(dataframe):

    df = dataframe.copy()

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])

        df["day"] = timestamps.dt.day

        df["day_of_week_number"] = timestamps.dt.dayofweek

        df["hour"] = timestamps.dt.hour

        df["month"] = timestamps.dt.month

        df["year"] = timestamps.dt.year

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ============================================================
# VALIDATE MODEL FEATURES
# ============================================================


def validate_model_features(dataframe):

    return [feature for feature in MODEL_FEATURES if feature not in dataframe.columns]


# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_dataframe(dataframe):

    missing = validate_model_features(dataframe)

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    model_input = dataframe[MODEL_FEATURES]

    transformed = preprocessor.transform(model_input)

    predictions = model.predict(transformed)

    return predictions


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
def health():

    return jsonify(
        {
            "status": "healthy",
            "service": "Pearls AQI Predictor",
            "model": type(model).__name__,
            "prediction_type": "numeric_aqi",
            "feature_count": len(MODEL_FEATURES),
        }
    )


# ============================================================
# MODEL INFORMATION
# ============================================================


@app.get("/model-info")
def model_info():

    return jsonify(
        {
            "model": type(model).__name__,
            "prediction_type": "numeric_aqi",
            "prediction_horizon": "next_hour",
            "feature_count": len(MODEL_FEATURES),
            "features": MODEL_FEATURES,
        }
    )


# ============================================================
# MANUAL NEXT-HOUR PREDICTION
# ============================================================


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": ("Request body must contain JSON data.")}), 400

    try:
        input_data = pd.DataFrame([data])

        input_data = add_time_features(input_data)

        missing = validate_model_features(input_data)

        if missing:
            return jsonify(
                {
                    "error": ("Missing required model features."),
                    "missing_features": missing,
                }
            ), 400

        prediction = float(predict_dataframe(input_data)[0])

        prediction = max(
            0.0,
            min(
                prediction,
                500.0,
            ),
        )

        return jsonify(
            {
                "prediction": round(
                    prediction,
                    2,
                ),
                "prediction_type": ("numeric_aqi"),
                "prediction_horizon": ("next_hour"),
                "model": (type(model).__name__),
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": ("Prediction failed."),
                "details": str(exc),
            }
        ), 500


# ============================================================
# BUILD HISTORICAL FEATURES FOR LIVE PREDICTION
# ============================================================


def get_historical_features(city):

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    historical = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    historical = historical[historical["city"] == city].copy()

    if len(historical) < 24:
        raise ValueError("Not enough historical data for this city.")

    historical = historical.sort_values("timestamp").reset_index(drop=True)

    historical["aqi"] = historical["pm2_5"].apply(pm25_to_aqi)

    # Latest 24 records
    history = historical.tail(24).copy()

    aqi_values = history["aqi"].astype(float).tolist()

    latest = history.iloc[-1]

    return {
        "aqi_lag_1": aqi_values[-1],
        "aqi_lag_3": aqi_values[-3],
        "aqi_lag_6": aqi_values[-6],
        "aqi_lag_12": aqi_values[-12],
        "aqi_lag_24": aqi_values[-24],
        "aqi_rolling_mean_3": np.mean(aqi_values[-3:]),
        "aqi_rolling_mean_6": np.mean(aqi_values[-6:]),
        "aqi_rolling_mean_24": np.mean(aqi_values[-24:]),
        "pm2_5_lag_1": float(latest["pm2_5"]),
        "pm10_lag_1": float(latest["pm10"]),
        "carbon_monoxide_lag_1": float(latest["carbon_monoxide"]),
        "nitrogen_dioxide_lag_1": float(latest["nitrogen_dioxide"]),
        "sulphur_dioxide_lag_1": float(latest["sulphur_dioxide"]),
        "ozone_lag_1": float(latest["ozone"]),
        "previous_aqi": aqi_values[-1],
    }


# ============================================================
# LIVE NEXT-HOUR PREDICTION
# ============================================================


@app.get("/predict-live")
def predict_live():

    try:
        city = request.args.get(
            "city",
            "Faisalabad",
        )

        if city not in CITY_COORDINATES:
            return jsonify(
                {
                    "error": "Unsupported city.",
                    "supported_cities": list(CITY_COORDINATES.keys()),
                }
            ), 400

        latitude, longitude = CITY_COORDINATES[city]

        # ----------------------------------------------------
        # 1. Get LIVE OpenWeather data
        # ----------------------------------------------------

        live_data = get_live_conditions(
            city,
            latitude,
            longitude,
        )

        # ----------------------------------------------------
        # 2. Get historical features
        # ----------------------------------------------------

        historical = get_historical_features(city)

        # ----------------------------------------------------
        # 3. Current AQI from LIVE PM2.5
        # ----------------------------------------------------

        current_aqi = pm25_to_aqi(live_data["pm2_5"])

        if pd.isna(current_aqi):
            return jsonify({"error": ("Could not calculate AQI from live PM2.5.")}), 500

        previous_aqi = historical["previous_aqi"]

        # ----------------------------------------------------
        # 4. Current time
        # ----------------------------------------------------

        now = datetime.now()

        # ----------------------------------------------------
        # 5. Create complete model input
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                {
                    # Location
                    "city": city,
                    "latitude": latitude,
                    "longitude": longitude,
                    # LIVE pollution
                    "pm10": live_data["pm10"],
                    "pm2_5": live_data["pm2_5"],
                    "carbon_monoxide": (live_data["carbon_monoxide"]),
                    "nitrogen_dioxide": (live_data["nitrogen_dioxide"]),
                    "sulphur_dioxide": (live_data["sulphur_dioxide"]),
                    "ozone": live_data["ozone"],
                    "dust": live_data["dust"],
                    # LIVE weather
                    "temperature": (live_data["temperature"]),
                    "humidity": (live_data["humidity"]),
                    "precipitation": (live_data["precipitation"]),
                    "wind_speed": (live_data["wind_speed"]),
                    "wind_direction": (live_data["wind_direction"]),
                    "pressure": (live_data["pressure"]),
                    # Time
                    "day": now.day,
                    "day_of_week_number": (now.weekday()),
                    "hour": now.hour,
                    "month": now.month,
                    "year": now.year,
                    "is_weekend": (now.weekday() >= 5),
                    # Cyclical time
                    "hour_sin": np.sin(2 * np.pi * now.hour / 24),
                    "hour_cos": np.cos(2 * np.pi * now.hour / 24),
                    "month_sin": np.sin(2 * np.pi * now.month / 12),
                    "month_cos": np.cos(2 * np.pi * now.month / 12),
                    # Historical AQI
                    "aqi_lag_1": historical["aqi_lag_1"],
                    "aqi_lag_3": historical["aqi_lag_3"],
                    "aqi_lag_6": historical["aqi_lag_6"],
                    "aqi_lag_12": historical["aqi_lag_12"],
                    "aqi_lag_24": historical["aqi_lag_24"],
                    # AQI change
                    "aqi_change": (current_aqi - previous_aqi),
                    "aqi_change_rate": (
                        (current_aqi - previous_aqi) / previous_aqi
                        if previous_aqi != 0
                        else 0.0
                    ),
                    # Rolling AQI
                    "aqi_rolling_mean_3": (historical["aqi_rolling_mean_3"]),
                    "aqi_rolling_mean_6": (historical["aqi_rolling_mean_6"]),
                    "aqi_rolling_mean_24": (historical["aqi_rolling_mean_24"]),
                    # Historical pollution
                    "pm2_5_lag_1": (historical["pm2_5_lag_1"]),
                    "pm10_lag_1": (historical["pm10_lag_1"]),
                    "carbon_monoxide_lag_1": (historical["carbon_monoxide_lag_1"]),
                    "nitrogen_dioxide_lag_1": (historical["nitrogen_dioxide_lag_1"]),
                    "sulphur_dioxide_lag_1": (historical["sulphur_dioxide_lag_1"]),
                    "ozone_lag_1": (historical["ozone_lag_1"]),
                }
            ]
        )

        # ----------------------------------------------------
        # 6. Check features
        # ----------------------------------------------------

        missing = validate_model_features(input_data)

        if missing:
            return jsonify(
                {
                    "error": ("Live prediction is missing model features."),
                    "missing_features": missing,
                }
            ), 500

        # ----------------------------------------------------
        # 7. Predict
        # ----------------------------------------------------

        prediction = float(predict_dataframe(input_data)[0])

        prediction = max(
            0.0,
            min(
                prediction,
                500.0,
            ),
        )

        # ----------------------------------------------------
        # 8. Return result
        # ----------------------------------------------------

        return jsonify(
            {
                "city": city,
                "prediction": round(
                    prediction,
                    2,
                ),
                "prediction_type": ("numeric_aqi"),
                "prediction_horizon": ("next_hour"),
                "model": (type(model).__name__),
                "live_data": live_data,
                "historical_source": ("pakistan_air_quality_final_clean_v2.csv"),
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": ("Live prediction failed."),
                "details": str(exc),
            }
        ), 500


# ============================================================
# 72-HOUR FORECAST
# ============================================================


@app.get("/predict-3days")
def predict_3days():

    try:
        city = request.args.get(
            "city",
            "Faisalabad",
        )

        if city not in CITY_COORDINATES:
            return jsonify(
                {
                    "error": "Unsupported city.",
                    "supported_cities": list(CITY_COORDINATES.keys()),
                }
            ), 400

        latitude, longitude = CITY_COORDINATES[city]

        forecast_records = get_forecast_conditions(
            city,
            latitude,
            longitude,
            hours=72,
        )

        if not forecast_records:
            return jsonify({"error": ("No forecast data available.")}), 500

        return jsonify(
            {
                "city": city,
                "prediction_horizon": ("next_72_hours"),
                "forecast_hours": len(forecast_records),
                "prediction_type": ("weather_pollution_forecast"),
                "model": (type(model).__name__),
                "forecast": forecast_records,
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": ("3-day prediction failed."),
                "details": str(exc),
            }
        ), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
