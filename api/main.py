import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from pathlib import Path

from pearls_aqi.live.openweather import (
    get_live_conditions,
    get_forecast_conditions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "local" / "aqi_next_hour.joblib"


app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
preprocessor = artifact["preprocessor"]


# ============================================================
# REQUIRED FEATURES FOR MANUAL PREDICTION
# ============================================================

REQUIRED_FEATURES = [
    "city",
    "latitude",
    "longitude",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
    "hour",
    "day_of_week",
    "month",
    "month_name",
    "year",
    "is_weekend",
    "season",
]


# ============================================================
# CITY COORDINATES
# ============================================================

CITY_COORDINATES = {
    "Faisalabad": (
        31.4167,
        73.0833,
    ),
    "Lahore": (
        31.5204,
        74.3587,
    ),
    "Islamabad": (
        33.6844,
        73.0479,
    ),
    "Karachi": (
        24.8607,
        67.0011,
    ),
    "Peshawar": (
        34.0151,
        71.5249,
    ),
}


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
def health():
    """Health check endpoint."""

    return jsonify(
        {
            "status": "healthy",
            "model": "RandomForestClassifier",
            "service": "Pearls AQI Predictor",
        }
    )


# ============================================================
# MANUAL NEXT-HOUR PREDICTION
# ============================================================


@app.post("/predict")
def predict():
    """Predict the next-hour AQI category."""

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": ("Request body must contain JSON data.")}), 400

    missing = [feature for feature in REQUIRED_FEATURES if feature not in data]

    if missing:
        return jsonify(
            {
                "error": ("Missing required features."),
                "missing_features": missing,
            }
        ), 400

    try:
        input_data = pd.DataFrame(
            [[data[feature] for feature in REQUIRED_FEATURES]],
            columns=REQUIRED_FEATURES,
        )

        input_data["hour_sin"] = np.sin(2 * np.pi * input_data["hour"] / 24)

        input_data["hour_cos"] = np.cos(2 * np.pi * input_data["hour"] / 24)

        input_data["month_sin"] = np.sin(2 * np.pi * input_data["month"] / 12)

        input_data["month_cos"] = np.cos(2 * np.pi * input_data["month"] / 12)

        transformed = preprocessor.transform(input_data)

        prediction = model.predict(transformed)[0]

        response = {
            "prediction": str(prediction),
            "model": ("RandomForestClassifier"),
            "prediction_horizon": ("next_hour"),
        }

        if hasattr(
            model,
            "predict_proba",
        ):
            probabilities = model.predict_proba(transformed)[0]

            classes = model.classes_

            response["probabilities"] = {
                str(label): round(
                    float(probability),
                    4,
                )
                for label, probability in zip(
                    classes,
                    probabilities,
                )
            }

        return jsonify(response)

    except Exception as exc:
        return jsonify(
            {
                "error": ("Prediction failed."),
                "details": str(exc),
            }
        ), 500


# ============================================================
# LIVE NEXT-HOUR PREDICTION
# ============================================================


@app.get("/predict-live")
def predict_live():
    """
    Fetch current OpenWeather weather and
    pollution data and predict next-hour AQI.
    """

    try:
        city = request.args.get(
            "city",
            "Faisalabad",
        )

        if city not in CITY_COORDINATES:
            return jsonify(
                {
                    "error": ("Unsupported city."),
                    "supported_cities": list(CITY_COORDINATES),
                }
            ), 400

        latitude, longitude = CITY_COORDINATES[city]

        live_data = get_live_conditions(
            city,
            latitude,
            longitude,
        )

        input_data = pd.DataFrame([live_data])

        # Add fields that are expected by
        # the trained model but are not needed
        # from the OpenWeather current API.

        from datetime import datetime

        now = datetime.now()

        input_data["day_of_week"] = now.strftime("%A")

        input_data["month_name"] = now.strftime("%B")

        season_map = {
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
            12: "Winter",
        }

        input_data["season"] = season_map[now.month]

        input_data = input_data[preprocessor.feature_names_in_]

        transformed = preprocessor.transform(input_data)

        prediction = model.predict(transformed)[0]

        response = {
            "city": city,
            "prediction": str(prediction),
            "model": ("RandomForestClassifier"),
            "prediction_horizon": ("next_hour"),
            "live_data": live_data,
        }

        if hasattr(
            model,
            "predict_proba",
        ):
            probabilities = model.predict_proba(transformed)[0]

            classes = model.classes_

            response["probabilities"] = {
                str(label): round(
                    float(probability),
                    4,
                )
                for label, probability in zip(
                    classes,
                    probabilities,
                )
            }

        return jsonify(response)

    except Exception as exc:
        return jsonify(
            {
                "error": ("Live prediction failed."),
                "details": str(exc),
            }
        ), 500


# ============================================================
# NEXT 3 DAYS PREDICTION
# ============================================================


@app.get("/predict-3days")
def predict_3days():
    """
    Fetch 72 hours of OpenWeather weather and
    air-pollution forecast data and generate
    AQI predictions.
    """

    try:
        city = request.args.get(
            "city",
            "Faisalabad",
        )

        if city not in CITY_COORDINATES:
            return jsonify(
                {
                    "error": ("Unsupported city."),
                    "supported_cities": list(CITY_COORDINATES),
                }
            ), 400

        latitude, longitude = CITY_COORDINATES[city]

        # ----------------------------------------------------
        # GET 72 HOURS FROM OPENWEATHER
        # ----------------------------------------------------

        forecast_records = get_forecast_conditions(
            city,
            latitude,
            longitude,
            hours=72,
        )

        if not forecast_records:
            return jsonify({"error": ("No forecast data available.")}), 500

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        forecast_df = pd.DataFrame(forecast_records)

        # ----------------------------------------------------
        # ADD MODEL TIME FEATURES
        # ----------------------------------------------------

        from datetime import datetime

        # The forecast_records already contain
        # hour/month/year. We need a real date
        # for day-of-week calculation.

        current_year = datetime.now().year

        current_month = datetime.now().month

        current_day = datetime.now().day

        forecast_start = pd.Timestamp(
            year=current_year,
            month=current_month,
            day=current_day,
        )

        forecast_df["forecast_datetime"] = [
            forecast_start + pd.Timedelta(hours=index)
            for index in range(len(forecast_df))
        ]

        forecast_df["day_of_week"] = forecast_df["forecast_datetime"].dt.day_name()

        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }

        forecast_df["month_name"] = forecast_df["month"].map(month_names)

        forecast_df["season"] = forecast_df["month"].map(
            {
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Autumn",
                10: "Autumn",
                11: "Autumn",
                12: "Winter",
            }
        )

        # ----------------------------------------------------
        # MODEL FEATURES
        # ----------------------------------------------------

        model_features = preprocessor.feature_names_in_

        missing_features = [
            feature for feature in model_features if feature not in forecast_df.columns
        ]

        if missing_features:
            return jsonify(
                {
                    "error": ("Forecast data is missing model features."),
                    "missing_features": (missing_features),
                }
            ), 500

        model_input = forecast_df[model_features]

        # ----------------------------------------------------
        # TRANSFORM FEATURES
        # ----------------------------------------------------

        transformed = preprocessor.transform(model_input)

        # ----------------------------------------------------
        # PREDICT AQI
        # ----------------------------------------------------

        predictions = model.predict(transformed)

        # ----------------------------------------------------
        # BUILD FORECAST RESPONSE
        # ----------------------------------------------------

        predictions_list = []

        for index, prediction in enumerate(predictions):
            record = forecast_records[index]

            forecast_datetime = forecast_df.iloc[index]["forecast_datetime"]

            predictions_list.append(
                {
                    "datetime": (forecast_datetime.isoformat()),
                    "city": city,
                    "prediction": str(prediction),
                    "temperature": float(record["temperature"]),
                    "humidity": float(record["humidity"]),
                    "pm2_5": float(record["pm2_5"]),
                    "pm10": float(record["pm10"]),
                }
            )

        response = {
            "city": city,
            "prediction_horizon": ("next_72_hours"),
            "forecast_hours": len(predictions_list),
            "model": ("RandomForestClassifier"),
            "predictions": (predictions_list),
        }

        return jsonify(response)

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
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
