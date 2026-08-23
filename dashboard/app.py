import requests
import streamlit as st

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Pearls AQI Predictor")
st.subheader("Next-Hour Air Quality Prediction for Pakistan")

st.markdown(
    """
    Predict the **next-hour AQI category** using a machine-learning model
    trained on air-quality and weather data from major Pakistani cities.
    """
)

st.divider()

st.header("📍 Prediction Inputs")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "City",
        [
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
        ],
    )

    latitude = st.number_input(
        "Latitude",
        value=31.5204,
        format="%.4f",
    )

    longitude = st.number_input(
        "Longitude",
        value=74.3587,
        format="%.4f",
    )

with col2:
    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=8,
    )

    year = st.number_input(
        "Year",
        min_value=2025,
        max_value=2030,
        value=2026,
    )

st.divider()

st.header("🌫️ Air Quality Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    pm10 = st.number_input("PM10", value=100.0)
    pm2_5 = st.number_input("PM2.5", value=80.0)
    carbon_monoxide = st.number_input(
        "Carbon Monoxide",
        value=1000.0,
    )

with col2:
    nitrogen_dioxide = st.number_input(
        "Nitrogen Dioxide",
        value=50.0,
    )

    sulphur_dioxide = st.number_input(
        "Sulphur Dioxide",
        value=20.0,
    )

    ozone = st.number_input(
        "Ozone",
        value=30.0,
    )

with col3:
    dust = st.number_input(
        "Dust",
        value=40.0,
    )

    temperature = st.number_input(
        "Temperature",
        value=20.0,
    )

    humidity = st.number_input(
        "Humidity",
        value=60.0,
    )

st.divider()

st.header("🌦️ Weather Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    precipitation = st.number_input(
        "Precipitation",
        value=0.0,
    )

with col2:
    wind_speed = st.number_input(
        "Wind Speed",
        value=5.0,
    )

with col3:
    wind_direction = st.number_input(
        "Wind Direction",
        min_value=0.0,
        max_value=360.0,
        value=180.0,
    )

pressure = st.number_input(
    "Pressure",
    value=1010.0,
)

st.divider()

st.header("🕐 Time Information")

day_of_week = st.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)

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

month_name = month_names[month]

is_weekend = 1 if day_of_week in {"Saturday", "Sunday"} else 0

season_map = {
    12: "Winter",
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
}

season = season_map[month]

st.divider()

if st.button(
    "🔮 Predict Next-Hour AQI",
    type="primary",
    use_container_width=True,
):
    api_url = "https://pearls-aqi-api-brgjbtg8g4a7hsej.centralindia-01.azurewebsites.net/predict"

    payload = {
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "pm10": pm10,
        "pm2_5": pm2_5,
        "carbon_monoxide": carbon_monoxide,
        "nitrogen_dioxide": nitrogen_dioxide,
        "sulphur_dioxide": sulphur_dioxide,
        "ozone": ozone,
        "dust": dust,
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "pressure": pressure,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "month_name": month_name,
        "year": year,
        "is_weekend": is_weekend,
        "season": season,
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            probabilities = result.get("probabilities", {})

            descriptions = {
                "Good": "Air quality is considered satisfactory.",
                "Moderate": (
                    "Air quality is acceptable, but some pollutants "
                    "may affect sensitive individuals."
                ),
                "Unhealthy for Sensitive Groups": (
                    "Sensitive groups may experience health effects."
                ),
                "Unhealthy": ("Everyone may begin to experience health effects."),
                "Very Unhealthy": (
                    "Health alert: the risk of health effects is increased."
                ),
                "Hazardous": (
                    "Health emergency conditions. Everyone is likely to be affected."
                ),
            }

            description = descriptions.get(
                prediction,
                "AQI prediction generated by the machine-learning model.",
            )

            st.success("Prediction generated successfully!")

            st.subheader("🎯 Predicted Next-Hour AQI")

            st.markdown(
                f"""
                <div style="
                    padding: 30px;
                    border-radius: 18px;
                    background-color: #f5f5f5;
                    text-align: center;
                    margin: 20px 0;
                    border: 2px solid #dddddd;
                ">
                    <div style="
                        font-size: 18px;
                        font-weight: 600;
                    ">
                        PREDICTED NEXT-HOUR AQI
                    </div>

                    <div style="
                        font-size: 42px;
                        font-weight: 800;
                        margin: 10px 0;
                    ">
                        {prediction}
                    </div>

                    <div style="
                        font-size: 16px;
                    ">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("📊 Prediction Probabilities")

            for category, probability in probabilities.items():
                st.write(f"**{category}** — {probability * 100:.1f}%")
                st.progress(float(probability))

            st.divider()

            st.subheader("🤖 Model Information")

            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:
                st.metric(
                    "Model",
                    result.get(
                        "model",
                        "RandomForestClassifier",
                    ),
                )

            with info_col2:
                st.metric(
                    "Forecast",
                    "Next Hour",
                )

            with info_col3:
                st.metric(
                    "City",
                    city,
                )

        else:
            st.error(f"API request failed with status {response.status_code}")

            try:
                st.json(response.json())
            except ValueError:
                st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the AQI prediction API. "
            "Make sure the Flask API is running on "
            "http://127.0.0.1:5000."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. Please make sure the Flask API is running."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

st.divider()

st.subheader("🌐 Live AQI Prediction")

live_city = st.selectbox(
    "Select City for Live Prediction",
    ["Faisalabad"],
)

if st.button(
    "🌐 Get Live AQI Prediction",
    use_container_width=True,
):
    live_api_url = "https://pearls-aqi-api-brgjbtg8g4a7hsej.centralindia-01.azurewebsites.net/predict-live"

    try:
        live_response = requests.get(
            live_api_url,
            params={"city": live_city},
            timeout=30,
        )

        if live_response.status_code == 200:
            live_result = live_response.json()

            st.success(
                f"Live next-hour AQI prediction for "
                f"{live_result['city']}: {live_result['prediction']}"
            )

            st.metric(
                "Live Prediction",
                live_result["prediction"],
            )

            st.subheader("Live Prediction Probabilities")

            for category, probability in live_result["probabilities"].items():
                st.write(f"**{category}** — {probability * 100:.1f}%")
                st.progress(float(probability))

        else:
            st.error(f"Live API request failed with status {live_response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the AQI prediction API. "
            "Make sure the Flask API is running on "
            "http://127.0.0.1:5000."
        )

    except requests.exceptions.Timeout:
        st.error("The live API request timed out.")

    except requests.exceptions.RequestException as exc:
        st.error(f"Live request failed: {exc}")


st.divider()

st.caption("Pearls AQI Predictor • Machine Learning • Next-Hour AQI Forecasting")
