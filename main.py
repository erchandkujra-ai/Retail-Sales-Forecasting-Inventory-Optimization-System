import argparse

from src.data_generation import generate_retail_data
from src.pipeline import run_pipeline
from src.config import OUTPUT_DIR

from src.visualization import (
    save_sales_trend,
    save_forecast_plot
)


def main():

    parser = argparse.ArgumentParser(
        description="Retail Sales Forecasting & Inventory Optimization"
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["generate", "run"],
        default="run"
    )

    args = parser.parse_args()

    if args.command == "generate":

        df = generate_retail_data()

        print(
            f"Dataset generated: {len(df):,} rows"
        )

        return

    # Run complete pipeline
    df, forecast, inventory, metrics = run_pipeline()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save charts
    save_sales_trend(
        df,
        OUTPUT_DIR / "sales_trend.png"
    )

    first_record = forecast.iloc[0]

    save_forecast_plot(
        df,
        forecast,
        OUTPUT_DIR / "sample_forecast.png",
        first_record["store_id"],
        first_record["sku"]
    )

    print("\n========== MODEL RESULTS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\n========== INVENTORY RESULTS ==========")

    print(
        inventory.head(10).to_string(index=False)
    )

    print("\n========== OUTPUT FILES ==========")

    print("outputs/demand_forecast_30_days.csv")
    print("outputs/inventory_recommendations.csv")
    print("outputs/model_metrics.csv")
    print("outputs/sales_trend.png")
    print("outputs/sample_forecast.png")


if __name__ == "__main__":
    main()