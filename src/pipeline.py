import joblib
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    FORECAST_DAYS
)

from src.data_generation import generate_retail_data
from src.preprocessing import load_and_clean_data
from src.forecasting import (
    train_and_evaluate,
    FEATURES
)

from src.inventory import (
    build_inventory_recommendations
)


def run_pipeline(force_generate=False):

    # Step 1: Generate data if required
    if force_generate or not RAW_DATA_PATH.exists():

        print("Generating retail dataset...")

        generate_retail_data()

    # Step 2: Clean data
    print("Cleaning dataset...")

    df = load_and_clean_data()

    # Step 3: Train model
    print("Training forecasting model...")

    model, metrics, training_frame = train_and_evaluate(df)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Step 4: Save trained model
    model_path = (
        MODEL_DIR / "retail_demand_model.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES
        },
        model_path
    )

    print(f"Model saved: {model_path}")

    # Step 5: Generate future forecasts
    forecast_rows = []

    for (store, sku), group in df.groupby(
        ["store_id", "sku"]
    ):

        group = group.sort_values(
            "date"
        )

        demand_history = list(
            group["demand_simulated"].astype(float)
        )

        last_record = group.iloc[-1]

        last_date = group["date"].max()
        price = float(last_record["price"])

        for step in range(
            1,
            FORECAST_DAYS + 1
        ):

            future_date = (
                last_date + pd.Timedelta(days=step)
            )

            recent = demand_history[-28:]

            row = {
                "day_of_week": future_date.dayofweek,
                "month": future_date.month,
                "day_of_month": future_date.day,
                "day_of_year": future_date.dayofyear,
                "is_weekend": int(
                    future_date.dayofweek >= 5
                ),
                "promotion": 0,
                "price": price,

                "lag_1": demand_history[-1],

                "lag_7": (
                    demand_history[-7]
                    if len(demand_history) >= 7
                    else demand_history[-1]
                ),

                "lag_14": (
                    demand_history[-14]
                    if len(demand_history) >= 14
                    else demand_history[-1]
                ),

                "lag_28": (
                    demand_history[-28]
                    if len(demand_history) >= 28
                    else demand_history[-1]
                ),

                "rolling_mean_7": (
                    sum(demand_history[-7:])
                    / min(7, len(demand_history))
                ),

                "rolling_mean_28": (
                    sum(recent)
                    / len(recent)
                )
            }

            input_data = pd.DataFrame(
                [row]
            )[FEATURES]

            prediction = float(
                model.predict(input_data)[0]
            )

            prediction = max(
                0,
                prediction
            )

            demand_history.append(
                prediction
            )

            forecast_rows.append({
                "date": future_date,
                "store_id": store,
                "sku": sku,
                "forecast": round(prediction, 2)
            })

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    # Step 6: Save forecast
    forecast_path = (
        OUTPUT_DIR / "demand_forecast_30_days.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False
    )

    # Step 7: Inventory optimization
    print("Calculating inventory recommendations...")

    inventory_df = build_inventory_recommendations(
        df,
        forecast_df
    )

    inventory_path = (
        OUTPUT_DIR / "inventory_recommendations.csv"
    )

    inventory_df.to_csv(
        inventory_path,
        index=False
    )

    # Step 8: Save model metrics
    metrics_path = (
        OUTPUT_DIR / "model_metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False
    )

    print("Pipeline completed successfully!")

    return (
        df,
        forecast_df,
        inventory_df,
        metrics
    )