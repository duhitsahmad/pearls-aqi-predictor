import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from pathlib import Path

from pearls_aqi.live.openweather import get_live_conditions


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "local" / "aqi_next_hour.joblib"


app = Flask(__name__)


# Load the trained model artifact once when the API starts.
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
preprocessor = artifact["preprocessor"]


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


@app.post("/predict")
def predict():
    """Predict the next-hour AQI category."""

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must contain JSON data."}), 400

    missing = [feature for feature in REQUIRED_FEATURES if feature not in data]

    if missing:
        return jsonify(
            {
                "error": "Missing required features.",
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
            "model": "RandomForestClassifier",
            "prediction_horizon": "next_hour",
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed)[0]
            classes = model.classes_

            response["probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(classes, probabilities)
            }

        return jsonify(response)

    except Exception as exc:
        return jsonify(
            {
                "error": "Prediction failed.",
                "details": str(exc),
            }
        ), 500


@app.get("/predict-live")
def predict_live():
    """Fetch live weather data and predict the next-hour AQI category."""

    try:
        city = request.args.get("city", "Faisalabad")

        city_coordinates = {
            "Faisalabad": (31.4167, 73.0833),
            "Lahore": (31.5204, 74.3587),
            "Islamabad": (33.6844, 73.0479),
            "Karachi": (24.8607, 67.0011),
            "Peshawar": (34.0151, 71.5249),
        }

        if city not in city_coordinates:
            return jsonify(
                {
                    "error": "Unsupported city.",
                    "supported_cities": list(city_coordinates),
                }
            ), 400

        latitude, longitude = city_coordinates[city]

        live_data = get_live_conditions(
            city,
            latitude,
            longitude,
        )

        input_data = pd.DataFrame([live_data])

        # Use exactly the features expected by the trained model.
        input_data = input_data[preprocessor.feature_names_in_]

        transformed = preprocessor.transform(input_data)

        prediction = model.predict(transformed)[0]

        response = {
            "city": city,
            "prediction": str(prediction),
            "model": "RandomForestClassifier",
            "prediction_horizon": "next_hour",
            "live_data": live_data,
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed)[0]
            classes = model.classes_

            response["probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(classes, probabilities)
            }

        return jsonify(response)

    except Exception as exc:
        return jsonify(
            {
                "error": "Live prediction failed.",
                "details": str(exc),
            }
        ), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
