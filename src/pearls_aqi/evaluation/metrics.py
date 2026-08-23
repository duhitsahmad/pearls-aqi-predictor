from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(
    model,
    preprocessor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained AQI classification model."""

    X_test_processed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_processed)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_test,
            predictions,
            average="weighted",
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ),
    }
