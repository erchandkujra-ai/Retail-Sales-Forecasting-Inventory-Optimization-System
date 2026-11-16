import matplotlib.pyplot as plt


def save_sales_trend(
    df,
    output_path
):

    daily_sales = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        daily_sales["date"],
        daily_sales["sales"],
        linewidth=1.3
    )

    ax.set_title(
        "Daily Retail Sales Trend"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)


def save_forecast_plot(
    history,
    forecast,
    output_path,
    store,
    sku
):

    historical_data = history[
        (history["store_id"] == store)
        & (history["sku"] == sku)
    ].tail(90)

    future_data = forecast[
        (forecast["store_id"] == store)
        & (forecast["sku"] == sku)
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        historical_data["date"],
        historical_data["demand_simulated"],
        label="Historical Demand"
    )

    ax.plot(
        future_data["date"],
        future_data["forecast"],
        label="Forecast",
        linestyle="--"
    )

    ax.set_title(
        f"Demand Forecast - {store} - {sku}"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Demand Units")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)