import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.features import (
    add_time_features,
    add_lag_features
)


FEATURES = [
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "promotion",
    "price",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]


def build_training_frame(df):

    parts = []

    for _, group in df.groupby(
        ["store_id", "sku"],
        sort=False
    ):

        group = add_time_features(group)
        group = add_lag_features(group)

        parts.append(group)

    frame = pd.concat(
        parts,
        ignore_index=True
    )

    frame = frame.dropna(
        subset=FEATURES + ["demand_simulated"]
    )

    return frame


def train_and_evaluate(df):

    frame = build_training_frame(df)

    frame = frame.sort_values("date")

    unique_dates = np.sort(
        frame["date"].unique()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "Not enough dates for model evaluation."
        )

    split_index = int(len(unique_dates) * 0.8)
    split_index = min(
        max(split_index, 1),
        len(unique_dates) - 1
    )

    cutoff = unique_dates[split_index]

    train = frame[
        frame["date"] < cutoff
    ]

    test = frame[
        frame["date"] >= cutoff
    ]

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[FEATURES],
        train["demand_simulated"]
    )

    predictions = model.predict(
        test[FEATURES]
    )

    predictions = np.maximum(
        0,
        predictions
    )

    mae = mean_absolute_error(
        test["demand_simulated"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["demand_simulated"],
            predictions
        )
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "train_rows": int(len(train)),
        "test_rows": int(len(test))
    }

    return model, metrics, frame