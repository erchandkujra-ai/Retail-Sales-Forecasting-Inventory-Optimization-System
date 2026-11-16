import pandas as pd


def add_time_features(df):

    result = df.copy()

    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["day_of_month"] = result["date"].dt.day
    result["day_of_year"] = result["date"].dt.dayofyear

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    return result


def add_lag_features(group):

    group = group.sort_values("date").copy()

    # Use simulated demand rather than censored sales.
    target = group["demand_simulated"]

    group["lag_1"] = target.shift(1)
    group["lag_7"] = target.shift(7)
    group["lag_14"] = target.shift(14)
    group["lag_28"] = target.shift(28)

    group["rolling_mean_7"] = (
        target.shift(1).rolling(7).mean()
    )

    group["rolling_mean_28"] = (
        target.shift(1).rolling(28).mean()
    )

    return group