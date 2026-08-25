from pathlib import Path
import joblib
import hopsworks

def save_model(model, preprocessor, path: str | Path) -> None:
    """Save model locally, then attempt to push to Hopsworks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {"model": model, "preprocessor": preprocessor}
    joblib.dump(artifact, path)

    try:
        project = hopsworks.login()
        mr = project.get_model_registry()
        hw_model = mr.python.create_model(
            name="pearls_aqi_model",
            description="Random Forest Regressor for next-hour numerical AQI prediction",
        )
        hw_model.save(str(path.parent))
        print("Successfully saved to Hopsworks Model Registry.")
    except Exception as e:
        print(f"Hopsworks connection failed ({e}). Model saved locally only.")

def load_model(path: str | Path) -> dict:
    """Try loading from Hopsworks, fallback to local artifact."""
    path = Path(path)

    try:
        project = hopsworks.login()
        mr = project.get_model_registry()
        hw_model = mr.get_model("pearls_aqi_model", version=1)
        saved_model_dir = hw_model.download()
        downloaded_path = Path(saved_model_dir) / path.name
        print("Successfully loaded model from Hopsworks.")
        return joblib.load(downloaded_path)
    except Exception as e:
        print(f"Hopsworks connection failed ({e}). Falling back to local model.")
        if not path.exists():
            raise FileNotFoundError(f"Local fallback failed: {path} not found.")
        return joblib.load(path)
