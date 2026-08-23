from pathlib import Path

import pandas as pd

from pearls_aqi.data.cleaner import clean_data
from pearls_aqi.features.features import create_features
from pearls_aqi.models.artifacts import load_model
from pearls_aqi.evaluation.metrics import evaluate_model


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

    # Reproduce the same per-city chronological split used during training.
    combined = features.copy()
    combined["_target"] = target.values

    train_parts = []
    test_parts = []

    for _, city_data in combined.groupby("city", sort=False):
        split_index = int(len(city_data) * 0.8)
        train_parts.append(city_data.iloc[:split_index])
        test_parts.append(city_data.iloc[split_index:])

    test_data = pd.concat(test_parts).reset_index(drop=True)

    X_test = test_data.drop(columns="_target")
    y_test = test_data["_target"]

    results = evaluate_model(
        model,
        preprocessor,
        X_test,
        y_test,
    )

    print()
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print()
    print("Classification report:")
    print(results["classification_report"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()
