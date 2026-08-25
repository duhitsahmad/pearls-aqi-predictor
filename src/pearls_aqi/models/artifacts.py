from pathlib import Path
import joblib
import hopsworks

def save_model(model, preprocessor, path: str | Path) -> None:
    """Save the model and preprocessor to the Hopsworks Model Registry."""

    # 1. Save locally first to package it
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
    }
    joblib.dump(artifact, path)

    # 2. Push to Hopsworks Model Registry
    project = hopsworks.login()
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(
        name="pearls_aqi_model",
        description="Random Forest Regressor for next-hour numerical AQI prediction",
    )

    # Upload the folder containing the joblib file
    hw_model.save(str(path.parent))


def load_model(path: str | Path) -> dict:
    """Load a saved model artifact from the Hopsworks Model Registry."""

    path = Path(path)

    # 1. Connect to Hopsworks and get the registry
    project = hopsworks.login()
    mr = project.get_model_registry()

    # 2. Retrieve the model metadata (assumes version 1)
    hw_model = mr.get_model("pearls_aqi_model", version=1)

    # 3. Download the model files to the local container/machine
    saved_model_dir = hw_model.download()

    # 4. Load the downloaded artifact using joblib
    downloaded_path = Path(saved_model_dir) / path.name

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Model artifact not found after download: {downloaded_path}")

    return joblib.load(downloaded_path)
