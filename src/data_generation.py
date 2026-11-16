import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    DATA_HISTORY_DAYS,
    NUMBER_OF_STORES,
    NUMBER_OF_PRODUCTS
)


def generate_retail_data(
    days=DATA_HISTORY_DAYS,
    n_stores=NUMBER_OF_STORES,
    n_products=NUMBER_OF_PRODUCTS,
    seed=RANDOM_STATE
):
    """
    Generate synthetic retail sales data.

    Includes:
    - Multiple stores
    - Multiple products
    - Daily sales
    - Promotions
    - Seasonal demand
    - Simulated stockouts
    """

    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    records = []

    weekday_effect = {
        0: 0.88,
        1: 0.92,
        2: 0.98,
        3: 1.00,
        4: 1.08,
        5: 1.25,
        6: 1.18
    }

    for store_number in range(1, n_stores + 1):

        store_id = f"STORE_{store_number:02d}"

        for product_number in range(1, n_products + 1):

            sku = f"SKU_{product_number:03d}"

            base_demand = rng.uniform(8, 45)
            price = round(rng.uniform(5, 80), 2)

            category = (
                "Category_A"
                if product_number <= n_products // 2
                else "Category_B"
            )

            for day_number, date in enumerate(dates):

                # Gradual demand trend
                trend = 1 + (0.00025 * day_number)

                # Annual seasonality
                seasonal_effect = (
                    1 + 0.12 * np.sin(
                        2 * np.pi * day_number / 365.25
                    )
                )

                # Weekly demand pattern
                weekly_effect = weekday_effect[date.dayofweek]

                # Random promotion
                promotion = int(rng.random() < 0.07)

                promotion_effect = (
                    1.65 if promotion == 1 else 1.0
                )

                # Expected daily demand
                expected_demand = (
                    base_demand
                    * trend
                    * seasonal_effect
                    * weekly_effect
                    * promotion_effect
                )

                # Simulated actual customer demand
                demand = int(
                    rng.poisson(max(0, expected_demand))
                )

                # Simulate occasional stockouts
                stockout = int(rng.random() < 0.035)

                if stockout == 1 and demand > 0:
                    sales = int(
                        rng.integers(
                            0,
                            max(1, demand // 2 + 1)
                        )
                    )
                else:
                    sales = demand

                records.append({
                    "date": date,
                    "store_id": store_id,
                    "sku": sku,
                    "category": category,
                    "price": price,
                    "promotion": promotion,
                    "demand_simulated": demand,
                    "sales": sales,
                    "stockout": stockout
                })

    df = pd.DataFrame(records)

    # Create the raw data directory
    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save generated dataset
    df.to_csv(RAW_DATA_PATH, index=False)

    print("Retail dataset generated successfully!")
    print(f"Total records: {len(df):,}")
    print(f"Stores: {df['store_id'].nunique()}")
    print(f"Products: {df['sku'].nunique()}")
    print(f"Dataset saved at: {RAW_DATA_PATH}")

    return df


if __name__ == "__main__":
    generate_retail_data()