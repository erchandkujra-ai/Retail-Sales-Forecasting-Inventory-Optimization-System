import math
import pandas as pd

from src.config import (
    LEAD_TIME_DAYS,
    SERVICE_LEVEL_Z
)


def calculate_inventory_policy(
    history,
    forecast_daily,
    lead_time_days=LEAD_TIME_DAYS,
    service_level_z=SERVICE_LEVEL_Z,
    review_period_days=1,
    holding_cost_rate=0.20,
    order_cost=25.0
):

    demand = pd.to_numeric(
        pd.Series(history),
        errors="coerce"
    ).dropna().clip(lower=0)

    average_demand = (
        float(demand.mean())
        if len(demand) else 0.0
    )

    demand_std = (
        float(demand.std(ddof=1))
        if len(demand) > 1 else 0.0
    )

    lead_time = max(
        1,
        int(lead_time_days)
    )

    # Baseline safety stock
    safety_stock = (
        service_level_z
        * demand_std
        * math.sqrt(lead_time)
    )

    # Reorder point
    reorder_point = (
        average_demand * lead_time
        + safety_stock
    )

    # Annual demand
    annual_demand = average_demand * 365

    # Illustrative unit cost assumption
    unit_cost = 10.0

    annual_holding_cost = max(
        0.01,
        unit_cost * holding_cost_rate
    )

    # Economic Order Quantity
    if annual_demand > 0:

        eoq = math.sqrt(
            (2 * annual_demand * order_cost)
            / annual_holding_cost
        )

    else:
        eoq = 0.0

    # Target stock for lead time + review period
    target_stock = (
        average_demand
        * (lead_time + review_period_days)
        + safety_stock
    )

    return {
        "average_daily_demand": round(average_demand, 2),
        "daily_demand_std": round(demand_std, 2),
        "safety_stock": int(math.ceil(safety_stock)),
        "reorder_point": int(math.ceil(reorder_point)),
        "eoq_reference": int(math.ceil(eoq)),
        "target_stock": int(math.ceil(target_stock)),
        "forecast_daily_demand": round(
            max(0, float(forecast_daily)),
            2
        )
    }


def build_inventory_recommendations(
    df,
    forecast_df,
    lead_time_days=LEAD_TIME_DAYS
):

    recommendations = []

    for (store, sku), forecast_group in forecast_df.groupby(
        ["store_id", "sku"]
    ):

        history = df[
            (df["store_id"] == store)
            & (df["sku"] == sku)
        ].sort_values("date")

        if history.empty:
            continue

        forecast_daily = float(
            forecast_group["forecast"].mean()
        )

        # Illustrative stock estimate for this demo.
        # Replace with actual on-hand inventory in a real system.
        current_stock = int(
            round(
                history.tail(1)["sales"].iloc[0] * 4
            )
        )

        policy = calculate_inventory_policy(
            history["demand_simulated"],
            forecast_daily,
            lead_time_days=lead_time_days
        )

        order_quantity = max(
            0,
            policy["target_stock"] - current_stock
        )

        recommendations.append({
            "store_id": store,
            "sku": sku,
            "current_stock_demo": current_stock,
            **policy,
            "recommended_order_qty": int(order_quantity),
            "reorder_required": bool(
                current_stock <= policy["reorder_point"]
            )
        })

    return pd.DataFrame(
        recommendations
    )