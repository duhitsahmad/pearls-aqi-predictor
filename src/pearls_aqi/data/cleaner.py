import numpy as np
import pandas as pd

# Changed target to numerical AQI
TARGET_COLUMN = "aqi"

def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create features for next-hour AQI prediction."""

    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Sort chronologically within each city.
    data = data.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # ----------------------------------------------------
    # NEW DERIVED FEATURE: AQI Change Rate
    # Calculate percentage change from the previous hour
    # ----------------------------------------------------
    data["aqi_change_rate"] = data.groupby("city")[TARGET_COLUMN].pct_change().fillna(0.0)

    # The prediction target is now the numerical AQI one hour ahead
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
        "aqi_category",     # We don't need the text category anymore
        TARGET_COLUMN,      # Drop current AQI to prevent data leakage
        "target_next_hour",
        "timestamp",
        "date",
        "month_name",
        "day_of_week",
        "season",
    ]
    
    # Safely drop columns only if they exist in the dataframe
    columns_to_drop = [c for c in columns_to_drop if c in data.columns]
    
    features = data.drop(columns=columns_to_drop)

    return features, target
