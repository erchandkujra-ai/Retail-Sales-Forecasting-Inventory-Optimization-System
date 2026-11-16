import numpy as np


def croston_sba(
    series,
    alpha=0.1,
    horizon=1
):

    values = np.asarray(
        series,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.zeros(horizon)

    if np.all(values == 0):
        return np.zeros(horizon)

    first_positive = np.flatnonzero(
        values > 0
    )[0]

    demand_estimate = values[first_positive]
    interval_estimate = 1.0
    interval = 1

    for value in values[first_positive + 1:]:

        interval += 1

        if value > 0:

            demand_estimate = (
                alpha * value
                + (1 - alpha) * demand_estimate
            )

            interval_estimate = (
                alpha * interval
                + (1 - alpha) * interval_estimate
            )

            interval = 1

    forecast = (
        (1 - alpha / 2)
        * demand_estimate
        / max(interval_estimate, 1e-9)
    )

    forecast = max(0, forecast)

    return np.repeat(
        forecast,
        horizon
    )