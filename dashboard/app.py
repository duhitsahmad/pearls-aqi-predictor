import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:5000"


CITIES = [
    "Faisalabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rahim Yar Khan",
    "Rawalpindi",
    "Sialkot",
]


# ============================================================
# AQI CATEGORY
# ============================================================


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ============================================================
# API FUNCTIONS
# ============================================================


def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException:
        return None


def get_live_prediction(city):
    try:
        response = requests.get(
            f"{API_URL}/predict-live",
            params={"city": city},
            timeout=30,
        )

        response.raise_for_status()

        return response.json(), None

    except requests.exceptions.HTTPError as exc:
        try:
            error_data = response.json()
        except Exception:
            error_data = {}

        return None, error_data.get(
            "details",
            str(exc),
        )

    except requests.exceptions.RequestException as exc:
        return None, str(exc)

    except Exception as exc:
        return None, str(exc)


# ============================================================
# HEADER
# ============================================================

st.title("🌍 Pearls AQI Predictor")

st.subheader("Live Next-Hour Air Quality Prediction for Pakistan")

st.markdown(
    """
    This dashboard uses **live weather and air-pollution data**
    together with the trained **Random Forest machine-learning
    model** to predict the **next-hour AQI**.
    """
)

st.divider()


# ============================================================
# API STATUS
# ============================================================

api_status = check_api()

if api_status:
    st.success("🟢 Prediction API is online")

else:
    st.error(
        "🔴 Prediction API is offline. Please start Flask with: `python api\\main.py`"
    )

    st.stop()


# ============================================================
# CITY SELECTION
# ============================================================

st.header("📍 Select City")

city = st.selectbox(
    "Choose a Pakistani city",
    CITIES,
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Get Live AQI Prediction",
    type="primary",
    use_container_width=True,
)


# ============================================================
# LIVE PREDICTION
# ============================================================

if predict_button:
    with st.spinner("Fetching live weather and air-quality data..."):
        result, error = get_live_prediction(city)

    if error:
        st.error("Live prediction failed.")

        st.code(str(error))

    else:
        prediction = float(result["prediction"])

        category = get_aqi_category(prediction)

        live_data = result.get(
            "live_data",
            {},
        )

        # ----------------------------------------------------
        # MAIN AQI RESULT
        # ----------------------------------------------------

        st.success("✅ Live prediction generated successfully!")

        st.header(f"🌫️ {city} — Next-Hour AQI")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted AQI",
                f"{prediction:.2f}",
            )

        with col2:
            st.metric(
                "AQI Category",
                category,
            )

        with col3:
            st.metric(
                "Model",
                result.get(
                    "model",
                    "RandomForestRegressor",
                ),
            )

        # ----------------------------------------------------
        # LIVE WEATHER
        # ----------------------------------------------------

        st.divider()

        st.header("🌤️ Live Weather Conditions")

        weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

        with weather_col1:
            st.metric(
                "Temperature",
                f"{live_data.get('temperature', 0):.1f} °C",
            )

        with weather_col2:
            st.metric(
                "Humidity",
                f"{live_data.get('humidity', 0):.0f} %",
            )

        with weather_col3:
            st.metric(
                "Wind Speed",
                f"{live_data.get('wind_speed', 0):.2f} m/s",
            )

        with weather_col4:
            st.metric(
                "Pressure",
                f"{live_data.get('pressure', 0):.0f} hPa",
            )

        # ----------------------------------------------------
        # LIVE AIR QUALITY
        # ----------------------------------------------------

        st.divider()

        st.header("🌫️ Live Air-Quality Measurements")

        air_col1, air_col2, air_col3 = st.columns(3)

        with air_col1:
            st.metric(
                "PM2.5",
                f"{live_data.get('pm2_5', 0):.2f}",
            )

            st.metric(
                "PM10",
                f"{live_data.get('pm10', 0):.2f}",
            )

        with air_col2:
            st.metric(
                "Carbon Monoxide",
                f"{live_data.get('carbon_monoxide', 0):.2f}",
            )

            st.metric(
                "Nitrogen Dioxide",
                f"{live_data.get('nitrogen_dioxide', 0):.2f}",
            )

        with air_col3:
            st.metric(
                "Sulphur Dioxide",
                f"{live_data.get('sulphur_dioxide', 0):.2f}",
            )

            st.metric(
                "Ozone",
                f"{live_data.get('ozone', 0):.2f}",
            )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        st.divider()

        st.header("📍 Location")

        location_col1, location_col2 = st.columns(2)

        with location_col1:
            st.write(f"**City:** {city}")

        with location_col2:
            st.write(
                f"**Coordinates:** "
                f"{live_data.get('latitude', 0):.4f}, "
                f"{live_data.get('longitude', 0):.4f}"
            )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.divider()

        st.header("🤖 Prediction Information")

        st.write(f"**Model:** {result.get('model', 'RandomForestRegressor')}")

        st.write(f"**Prediction type:** {result.get('prediction_type', 'numeric_aqi')}")

        st.write(
            f"**Prediction horizon:** {result.get('prediction_horizon', 'next_hour')}"
        )

        st.write(
            f"**Historical data source:** {result.get('historical_source', 'Dataset')}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pearls AQI Predictor | "
    "Random Forest | "
    "OpenWeather Live Data | "
    "Next-Hour AQI Forecast"
)
