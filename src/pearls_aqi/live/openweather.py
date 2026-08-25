import numpy as np
import os
from datetime import datetime

import requests


OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"


def get_live_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather data from OpenWeather."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    response = requests.get(
        f"{OPENWEATHER_BASE_URL}/weather",
        params={
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_live_air_pollution(latitude: float, longitude: float) -> dict:
    """Fetch current air-pollution data from OpenWeather."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    response = requests.get(
        f"{OPENWEATHER_BASE_URL}/air_pollution",
        params={
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_forecast_weather(latitude: float, longitude: float) -> dict:
    """Fetch 5-day weather forecast from OpenWeather."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    response = requests.get(
        f"{OPENWEATHER_BASE_URL}/forecast",
        params={
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_forecast_air_pollution(
    latitude: float,
    longitude: float,
) -> dict:
    """Fetch hourly air-pollution forecast from OpenWeather."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    response = requests.get(
        f"{OPENWEATHER_BASE_URL}/air_pollution/forecast",
        params={
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_live_conditions(
    city: str,
    latitude: float,
    longitude: float,
) -> dict:
    """Combine live weather and pollution data into one record."""

    weather = get_live_weather(latitude, longitude)

    pollution = get_live_air_pollution(
        latitude,
        longitude,
    )

    weather_main = weather["main"]

    wind = weather.get(
        "wind",
        {},
    )

    rain = weather.get(
        "rain",
        {},
    )

    components = pollution["list"][0]["components"]

    now = datetime.now()

    hour_sin = np.sin(2 * np.pi * now.hour / 24)

    hour_cos = np.cos(2 * np.pi * now.hour / 24)

    month_sin = np.sin(2 * np.pi * now.month / 12)

    month_cos = np.cos(2 * np.pi * now.month / 12)

    return {
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "pm10": components.get(
            "pm10",
            0.0,
        ),
        "pm2_5": components.get(
            "pm2_5",
            0.0,
        ),
        "carbon_monoxide": components.get(
            "co",
            0.0,
        ),
        "nitrogen_dioxide": components.get(
            "no2",
            0.0,
        ),
        "sulphur_dioxide": components.get(
            "so2",
            0.0,
        ),
        "ozone": components.get(
            "o3",
            0.0,
        ),
        "dust": 0.0,
        "temperature": weather_main.get(
            "temp",
            0.0,
        ),
        "humidity": weather_main.get(
            "humidity",
            0.0,
        ),
        "precipitation": rain.get(
            "1h",
            0.0,
        ),
        "wind_speed": wind.get(
            "speed",
            0.0,
        ),
        "wind_direction": wind.get(
            "deg",
            0.0,
        ),
        "pressure": weather_main.get(
            "pressure",
            0.0,
        ),
        "hour": now.hour,
        "month": now.month,
        "year": now.year,
        "is_weekend": now.weekday() >= 5,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def get_forecast_conditions(
    city: str,
    latitude: float,
    longitude: float,
    hours: int = 72,
) -> list[dict]:
    """Combine OpenWeather weather and pollution forecasts."""

    weather = get_forecast_weather(
        latitude,
        longitude,
    )

    pollution = get_forecast_air_pollution(
        latitude,
        longitude,
    )

    weather_by_time = {
        item["dt"]: item
        for item in weather.get(
            "list",
            [],
        )
    }

    forecast_records = []

    for pollution_item in pollution.get(
        "list",
        [],
    )[:hours]:
        timestamp = pollution_item["dt"]

        if not weather_by_time:
            continue

        closest_timestamp = min(
            weather_by_time,
            key=lambda value: abs(value - timestamp),
        )

        weather_item = weather_by_time[closest_timestamp]

        weather_main = weather_item.get(
            "main",
            {},
        )

        wind = weather_item.get(
            "wind",
            {},
        )

        rain = weather_item.get(
            "rain",
            {},
        )

        components = pollution_item.get(
            "components",
            {},
        )

        forecast_time = datetime.fromtimestamp(timestamp)

        forecast_records.append(
            {
                "city": city,
                "latitude": latitude,
                "longitude": longitude,
                "pm10": components.get(
                    "pm10",
                    0.0,
                ),
                "pm2_5": components.get(
                    "pm2_5",
                    0.0,
                ),
                "carbon_monoxide": components.get(
                    "co",
                    0.0,
                ),
                "nitrogen_dioxide": components.get(
                    "no2",
                    0.0,
                ),
                "sulphur_dioxide": components.get(
                    "so2",
                    0.0,
                ),
                "ozone": components.get(
                    "o3",
                    0.0,
                ),
                "dust": 0.0,
                "temperature": weather_main.get(
                    "temp",
                    0.0,
                ),
                "humidity": weather_main.get(
                    "humidity",
                    0.0,
                ),
                "precipitation": rain.get(
                    "1h",
                    0.0,
                ),
                "wind_speed": wind.get(
                    "speed",
                    0.0,
                ),
                "wind_direction": wind.get(
                    "deg",
                    0.0,
                ),
                "pressure": weather_main.get(
                    "pressure",
                    0.0,
                ),
                "hour": forecast_time.hour,
                "month": forecast_time.month,
                "year": forecast_time.year,
                "is_weekend": (forecast_time.weekday() >= 5),
                "hour_sin": np.sin(2 * np.pi * forecast_time.hour / 24),
                "hour_cos": np.cos(2 * np.pi * forecast_time.hour / 24),
                "month_sin": np.sin(2 * np.pi * forecast_time.month / 12),
                "month_cos": np.cos(2 * np.pi * forecast_time.month / 12),
            }
        )

    return forecast_records
