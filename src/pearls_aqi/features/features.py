import numpy as np
import pandas as pd


TARGET_COLUMN = "aqi_category"


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create features for next-hour AQI category prediction."""

    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Sort chronologically within each city.
    data = data.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # The prediction target is the AQI category one hour ahead.
    data["target_next_hour"] = data.groupby("city")[TARGET_COLUMN].shift(-1)

    # Cyclical time features.
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    # Remove rows where there is no next-hour target.
    data = data.dropna(subset=["target_next_hour"])

    target = data["target_next_hour"]

    columns_to_drop = [
        TARGET_COLUMN,
        "target_next_hour",
        "timestamp",
        "date",
        "month_name",
        "day_of_week",
        "season",
    ]

    features = data.drop(columns=columns_to_drop)

    return features, target
