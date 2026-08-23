import shap
import pandas as pd


def create_explainer(model):
    """Create a SHAP TreeExplainer for the Random Forest model."""
    return shap.TreeExplainer(model)


def explain_samples(
    model,
    preprocessor,
    X: pd.DataFrame,
    max_samples: int = 25,
):
    """Generate SHAP explanations for a small sample."""

    X_sample = X.head(max_samples)

    X_processed = preprocessor.transform(X_sample)

    feature_names = preprocessor.get_feature_names_out()

    explainer = create_explainer(model)

    explanation = explainer(
        X_processed,
        check_additivity=False,
    )

    return explanation, feature_names, X_sample
