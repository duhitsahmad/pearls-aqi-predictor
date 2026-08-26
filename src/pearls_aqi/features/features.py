import numpy as np
import pandas as pd


TARGET_COLUMN = "aqi"


# ============================================================
# PM2.5 → AQI
# ============================================================


def pm25_to_aqi(pm25: float) -> float:
    """
    Convert PM2.5 concentration to US EPA-style AQI.

    The breakpoints match the AQI categories already present
    in the project's dataset.
    """

    if pd.isna(pm25):
        return np.nan

    pm25 = float(pm25)

    if pm25 < 0:
        return np.nan

    # PM2.5 AQI breakpoints
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]

    # Clamp extremely high values to the highest supported range.
    pm25 = min(pm25, 500.4)

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low

            return round(aqi)

    return np.nan


def add_numeric_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric AQI calculated from PM2.5.

    Existing aqi_category is preserved.
    """

    data = df.copy()

    data["aqi"] = data["pm2_5"].apply(pm25_to_aqi)

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def create_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create features for next-hour numeric AQI forecasting.

    The model predicts the AQI one hour into the future.
    The API will later use this model recursively to produce
    a 72-hour forecast.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # --------------------------------------------------------
    # Numeric AQI
    # --------------------------------------------------------

    data = add_numeric_aqi(data)

    # --------------------------------------------------------
    # Sort chronologically per city
    # --------------------------------------------------------

    data = data.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Historical AQI features
    # --------------------------------------------------------

    grouped = data.groupby("city")["aqi"]

    data["aqi_lag_1"] = grouped.shift(1)
    data["aqi_lag_3"] = grouped.shift(3)
    data["aqi_lag_6"] = grouped.shift(6)
    data["aqi_lag_12"] = grouped.shift(12)
    data["aqi_lag_24"] = grouped.shift(24)

    # AQI change rate
    previous_aqi = data["aqi_lag_1"]

    data["aqi_change"] = data["aqi"] - previous_aqi

    data["aqi_change_rate"] = data["aqi_change"] / previous_aqi.replace(0, np.nan)

    # --------------------------------------------------------
    # Rolling AQI features
    # --------------------------------------------------------

    data["aqi_rolling_mean_3"] = data.groupby("city")["aqi"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    data["aqi_rolling_mean_6"] = data.groupby("city")["aqi"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )

    data["aqi_rolling_mean_24"] = data.groupby("city")["aqi"].transform(
        lambda x: x.shift(1).rolling(24).mean()
    )

    # --------------------------------------------------------
    # Pollution lag features
    # --------------------------------------------------------

    for column in [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]:
        data[f"{column}_lag_1"] = data.groupby("city")[column].shift(1)

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    data["day"] = data["timestamp"].dt.day

    data["day_of_week_number"] = data["timestamp"].dt.dayofweek

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)

    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)

    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    data["target_next_hour"] = data.groupby("city")["aqi"].shift(-1)

    # --------------------------------------------------------
    # Remove rows without a target
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "target_next_hour",
            "aqi_lag_24",
            "aqi_rolling_mean_24",
        ]
    )

    target = data["target_next_hour"]

    # --------------------------------------------------------
    # Columns that must NOT be model features
    # --------------------------------------------------------

    columns_to_drop = [
        "aqi",
        "aqi_category",
        "target_next_hour",
        "timestamp",
        "date",
        "month_name",
        "day_of_week",
        "season",
    ]

    features = data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    return features, target
