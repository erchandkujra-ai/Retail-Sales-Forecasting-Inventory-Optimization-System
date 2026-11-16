import sys
from pathlib import Path

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import streamlit as st

from src.config import (
    OUTPUT_DIR,
    RAW_DATA_PATH
)

from src.pipeline import run_pipeline


st.set_page_config(
    page_title="Retail Forecasting System",
    page_icon="📦",
    layout="wide"
)


st.title(
    "📦 Retail Sales Forecasting & Inventory Optimization"
)

st.caption(
    "Machine learning dashboard for retail demand forecasting "
    "and inventory planning."
)


# Sidebar
st.sidebar.title("Project Navigation")

st.sidebar.info(
    "Retail Analytics | Demand Forecasting | Inventory Optimization"
)


# Run pipeline
if st.button(
    "Run / Refresh Forecast Pipeline",
    type="primary"
):

    with st.spinner(
        "Training model and generating inventory recommendations..."
    ):

        try:

            run_pipeline(
                force_generate=not RAW_DATA_PATH.exists()
            )

            st.success(
                "Forecast pipeline completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Pipeline error: {error}"
            )


# Output files
forecast_path = (
    OUTPUT_DIR / "demand_forecast_30_days.csv"
)

inventory_path = (
    OUTPUT_DIR / "inventory_recommendations.csv"
)

metrics_path = (
    OUTPUT_DIR / "model_metrics.csv"
)


if not forecast_path.exists() or not inventory_path.exists():

    st.warning(
        "No forecast results found. "
        "Click Run / Refresh Forecast Pipeline first."
    )

    st.stop()


forecast = pd.read_csv(
    forecast_path,
    parse_dates=["date"]
)

inventory = pd.read_csv(
    inventory_path
)


if metrics_path.exists():

    metrics = pd.read_csv(
        metrics_path
    )

else:

    metrics = pd.DataFrame()


# KPI cards
st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Store-SKU Combinations",
    len(inventory)
)

col2.metric(
    "Forecast Demand (30 Days)",
    f"{forecast['forecast'].sum():,.0f}"
)

col3.metric(
    "Items Requiring Reorder",
    int(inventory["reorder_required"].sum())
)

col4.metric(
    "Recommended Order Units",
    f"{inventory['recommended_order_qty'].sum():,.0f}"
)


# Model performance
st.subheader("Model Performance")

if not metrics.empty:

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Mean Absolute Error (MAE)",
        f"{metrics.iloc[0]['MAE']:.2f}"
    )

    metric_col2.metric(
        "Root Mean Squared Error (RMSE)",
        f"{metrics.iloc[0]['RMSE']:.2f}"
    )


# Forecast chart
st.subheader("30-Day Demand Forecast")

stores = sorted(
    forecast["store_id"].unique()
)

selected_store = st.selectbox(
    "Select Store",
    stores
)

available_skus = sorted(
    forecast.loc[
        forecast["store_id"] == selected_store,
        "sku"
    ].unique()
)

selected_sku = st.selectbox(
    "Select Product",
    available_skus
)


selected_forecast = forecast[
    (forecast["store_id"] == selected_store)
    & (forecast["sku"] == selected_sku)
]


if not selected_forecast.empty:

    chart_data = selected_forecast.set_index(
        "date"
    )[["forecast"]]

    st.line_chart(
        chart_data
    )


# Inventory table
st.subheader("Inventory Recommendations")

st.dataframe(
    inventory,
    use_container_width=True
)


# Reorder filter
st.subheader("Reorder Alerts")

reorder_items = inventory[
    inventory["reorder_required"] == True
]

if reorder_items.empty:

    st.success(
        "No items currently meet the demo reorder condition."
    )

else:

    st.warning(
        f"{len(reorder_items)} store-SKU combinations "
        "meet the demo reorder condition."
    )

    st.dataframe(
        reorder_items,
        use_container_width=True
    )


# Downloads
st.subheader("Download Reports")

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        label="Download Forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="demand_forecast_30_days.csv",
        mime="text/csv"
    )

with download_col2:

    st.download_button(
        label="Download Inventory CSV",
        data=inventory.to_csv(index=False).encode("utf-8"),
        file_name="inventory_recommendations.csv",
        mime="text/csv"
    )


st.caption(
    "Educational portfolio demonstration using synthetic retail data. "
    "Inventory values are illustrative and should not be used for real purchasing."
)