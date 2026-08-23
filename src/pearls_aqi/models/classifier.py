import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_fraction: float = 0.8,
):
    """Train an AQI classifier using a chronological split."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    # The dataset is already sorted chronologically within each city.
    # Split each city separately so every city contributes to both periods.
    combined = features.copy()
    combined["_target"] = target.values

    train_parts = []
    test_parts = []

    for city, city_data in combined.groupby("city", sort=False):
        split_index = int(len(city_data) * train_fraction)

        train_parts.append(city_data.iloc[:split_index])
        test_parts.append(city_data.iloc[split_index:])

    train_data = pd.concat(train_parts).reset_index(drop=True)
    test_data = pd.concat(test_parts).reset_index(drop=True)

    X_train = train_data.drop(columns="_target")
    y_train = train_data["_target"]

    X_test = test_data.drop(columns="_target")
    y_test = test_data["_target"]

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
            ("numeric", "passthrough", numeric_columns),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    print(classification_report(y_test, predictions))

    return (
        model,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
    )
