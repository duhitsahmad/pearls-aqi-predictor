from pathlib import Path

import numpy as np
import pandas as pd

from pearls_aqi.data.cleaner import clean_data
from pearls_aqi.features.features import create_features
from pearls_aqi.models.artifacts import load_model
from pearls_aqi.explainability.shap_explainer import explain_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pakistan_air_quality_final_clean_v2.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "local"
    / "aqi_next_hour.joblib"
)


def main() -> None:

    print("=" * 60)
    print("PEARLS AQI PREDICTOR - SHAP FEATURE IMPORTANCE")
    print("=" * 60)

    print()
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    clean = clean_data(df)

    features, target = create_features(clean)

    print(f"Samples: {len(features)}")
    print(f"Features: {features.shape[1]}")

    print()
    print("Loading trained model...")

    artifact = load_model(MODEL_PATH)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    print(f"Model: {type(model).__name__}")

    print()
    print("Calculating SHAP importance...")

    explanation, feature_names, _ = explain_samples(
        model,
        preprocessor,
        features,
        max_samples=25,
    )

    shap_values = np.asarray(explanation.values)

    print(f"SHAP values shape: {shap_values.shape}")

    # For a regression model SHAP values are normally:
    # samples x features
    if shap_values.ndim == 2:

        importance = np.mean(
            np.abs(shap_values),
            axis=0,
        )

    elif shap_values.ndim == 3:

        importance = np.mean(
            np.abs(shap_values),
            axis=(0, 2),
        )

    else:

        raise ValueError(
            f"Unexpected SHAP array dimensions: {shap_values.shape}"
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    print()
    print("TOP 15 FEATURES")
    print("-" * 60)

    print(
        importance_df
        .head(15)
        .to_string(index=False)
    )

    output_path = (
        PROJECT_ROOT
        / "models"
        / "local"
        / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"SHAP results saved to: {output_path}")


if __name__ == "__main__":
    main()
