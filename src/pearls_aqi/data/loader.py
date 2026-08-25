import hopsworks
import pandas as pd

def load_features_from_store(
    feature_group_name: str = "pearls_aqi_features", 
    version: int = 1
) -> pd.DataFrame:
    """Load historical features and targets from Hopsworks Feature Store."""
    
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    fg = fs.get_feature_group(
        name=feature_group_name,
        version=version,
    )
    
    # Read the feature group into a Pandas DataFrame
    return fg.read()

def save_features_to_store(
    df: pd.DataFrame, 
    feature_group_name: str = "pearls_aqi_features", 
    version: int = 1
) -> None:
    """Save processed features to Hopsworks Feature Store."""
    
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Create or get the feature group
    fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        primary_key=["city", "year", "month", "hour"],
        description="Processed features for Pearls AQI Predictor",
        online_enabled=True,
    )
    
    # Insert the dataframe into the cloud feature store
    fg.insert(df, write_options={"wait_for_job": False})
