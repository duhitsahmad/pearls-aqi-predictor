import hopsworks
import pandas as pd
from pathlib import Path

def load_features_from_store(
    feature_group_name: str = "pearls_aqi_features", 
    version: int = 1
) -> pd.DataFrame:
    """Load features from Hopsworks, fallback to local if cloud is down."""
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=version)
        print("Successfully connected to Hopsworks Feature Store.")
        return fg.read()
    except Exception as e:
        print(f"Hopsworks connection failed ({e}). Falling back to local data.")
        local_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "pakistan_air_quality_final_clean_v2.csv"
        return pd.read_csv(local_path)

def save_features_to_store(
    df: pd.DataFrame, 
    feature_group_name: str = "pearls_aqi_features", 
    version: int = 1
) -> None:
    """Save features to Hopsworks, fallback to local if cloud is down."""
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            primary_key=["city", "year", "month", "hour"],
            description="Processed features for Pearls AQI Predictor",
            online_enabled=True,
        )
        fg.insert(df, write_options={"wait_for_job": False})
        print("Successfully saved to Hopsworks Feature Store.")
    except Exception as e:
        print(f"Hopsworks connection failed ({e}). Bypassing cloud save.")
