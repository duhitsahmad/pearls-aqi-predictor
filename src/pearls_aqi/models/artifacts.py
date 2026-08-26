from pathlib import Path

import joblib


def save_model(
    model,
    preprocessor,
    path: Path,
) -> None:
    """
    Save the trained model and preprocessor locally.

    Hopsworks/cloud registration is intentionally not performed here.
    This keeps local training independent of the Hopsworks package.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
    }

    joblib.dump(
        artifact,
        path,
        compress=3,
    )

    print(f"Model artifact saved: {path}")
