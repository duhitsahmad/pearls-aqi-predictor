import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    train_fraction: float = 0.8,
):
    """
    Perform a chronological train/test split separately for every city.

    This prevents future observations from leaking into the training set.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    combined = features.copy()
    combined["_target"] = target.values

    train_parts = []
    test_parts = []

    for city, city_data in combined.groupby(
        "city",
        sort=False,
    ):
        split_index = int(len(city_data) * train_fraction)

        train_parts.append(city_data.iloc[:split_index])

        test_parts.append(city_data.iloc[split_index:])

    train_data = pd.concat(train_parts).reset_index(drop=True)

    test_data = pd.concat(test_parts).reset_index(drop=True)

    X_train = train_data.drop(columns="_target")

    y_train = train_data["_target"]

    X_test = test_data.drop(columns="_target")

    y_test = test_data["_target"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def create_preprocessor(
    X_train: pd.DataFrame,
):
    """
    Create preprocessing pipeline.

    City is one-hot encoded.
    Numeric features are passed through.
    """

    categorical_columns = ["city"]

    numeric_columns = [
        column for column in X_train.columns if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns,
            ),
        ]
    )

    return preprocessor


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Train Random Forest regression model.
    """

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    return (
        model,
        preprocessor,
    )


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Train Ridge regression model.

    StandardScaler is applied to numeric/encoded features
    so the linear model is trained on comparable scales.
    """

    categorical_columns = ["city"]

    numeric_columns = [
        column for column in X_train.columns if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    model = Ridge(alpha=1.0)

    model.fit(
        X_train_processed,
        y_train,
    )

    return (
        model,
        preprocessor,
    )


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_fraction: float = 0.8,
):
    """
    Backwards-compatible Random Forest training function.

    Existing scripts can continue calling train_model().
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        features,
        target,
        train_fraction,
    )

    (
        model,
        preprocessor,
    ) = train_random_forest(
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
