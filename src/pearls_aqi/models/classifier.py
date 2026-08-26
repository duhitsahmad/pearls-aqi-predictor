from typing import Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CREATE PREPROCESSOR
# ============================================================


def create_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing pipeline.

    The city column is one-hot encoded.
    Numeric columns are standardized.
    """

    categorical_columns = ["city"]

    numeric_columns = [
        column for column in X_train.columns if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[RandomForestRegressor, ColumnTransformer]:
    """
    Train a memory-efficient Random Forest.

    This configuration is intentionally smaller than the
    previous 300-tree unlimited-depth model so it can run
    on a 512 MB RAM deployment.
    """

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
        max_features="sqrt",
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    return model, preprocessor


# ============================================================
# TRAIN RIDGE
# ============================================================


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Ridge, ColumnTransformer]:
    """
    Train Ridge regression model.
    """

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    model = Ridge(
        alpha=1.0,
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    return model, preprocessor


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
):
    """
    Split the dataset and train the Random Forest model.

    Returns:
        model
        preprocessor
        X_train
        X_test
        y_train
        y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    model, preprocessor = train_random_forest(
        X_train,
        y_train,
    )

    return (
        model,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
    )
