from pathlib import Path

import pandas as pd

from pearls_aqi.data.cleaner import clean_data
from pearls_aqi.features.features import create_features
from pearls_aqi.models.artifacts import load_model
from pearls_aqi.explainability.shap_explainer import explain_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pakistan_air_quality_final_clean_v2.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "local" / "aqi_next_hour.joblib"


def main() -> None:
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    clean = clean_data(df)

    features, target = create_features(clean)

    artifact = load_model(MODEL_PATH)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    explanation, feature_names, X_sample = explain_samples(
        model,
        preprocessor,
        features,
        max_samples=100,
    )

    print()
    print("SHAP explanation created successfully")
    print("Samples explained:", len(X_sample))
    print("Number of processed features:", len(feature_names))
    print("SHAP values shape:", explanation.values.shape)
    print()
    print("First 10 feature names:")
    print(feature_names[:10])


if __name__ == "__main__":
    main()
