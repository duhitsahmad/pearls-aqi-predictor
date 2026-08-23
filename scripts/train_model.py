from pathlib import Path

import pandas as pd

from pearls_aqi.data.cleaner import clean_data
from pearls_aqi.features.features import create_features
from pearls_aqi.models.artifacts import save_model
from pearls_aqi.models.classifier import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pakistan_air_quality_final_clean_v2.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "local" / "aqi_next_hour.joblib"


def main() -> None:
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Raw rows: {len(df)}")

    clean = clean_data(df)

    features, target = create_features(clean)

    print(f"Training samples available: {len(features)}")

    model, preprocessor, X_train, X_test, y_train, y_test = train_model(
        features,
        target,
    )

    save_model(
        model,
        preprocessor,
        MODEL_PATH,
    )

    print()
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
