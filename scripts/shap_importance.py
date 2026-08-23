from pathlib import Path

import numpy as np
import pandas as pd

from pearls_aqi.data.cleaner import clean_data
from pearls_aqi.features.features import create_features
from pearls_aqi.models.artifacts import load_model
from pearls_aqi.explainability.shap_explainer import explain_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pakistan_air_quality_final_clean_v2.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "local" / "aqi_next_hour.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    clean = clean_data(df)

    features, target = create_features(clean)

    artifact = load_model(MODEL_PATH)

    explanation, feature_names, _ = explain_samples(
        artifact["model"],
        artifact["preprocessor"],
        features,
        max_samples=25,
    )

    importance = np.mean(
        np.abs(explanation.values),
        axis=(0, 2),
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

    print("Top 15 features by mean absolute SHAP value:")
    print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
