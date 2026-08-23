import pandas as pd


TARGET_COLUMN = "aqi_category"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the AQI dataset."""

    data = df.copy()

    # Remove completely duplicated records.
    data = data.drop_duplicates()

    # Ensure timestamp is a proper datetime.
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    # Remove rows with invalid timestamps.
    data = data.dropna(subset=["timestamp"])

    # Ensure the target exists and is not missing.
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    data = data.dropna(subset=[TARGET_COLUMN])

    return data.reset_index(drop=True)
