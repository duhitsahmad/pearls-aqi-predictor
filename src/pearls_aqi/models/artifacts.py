from pathlib import Path

import joblib


def save_model(model, preprocessor, path: str | Path) -> None:
    """Save the model and preprocessor together."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
    }

    joblib.dump(artifact, path)


def load_model(path: str | Path) -> dict:
    """Load a saved model artifact."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return joblib.load(path)
