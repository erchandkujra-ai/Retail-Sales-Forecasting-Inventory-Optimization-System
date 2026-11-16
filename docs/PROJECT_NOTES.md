# Retail Sales Forecasting & Inventory Optimization

## Project Objective

Build a machine learning system that forecasts retail product demand
and generates inventory replenishment recommendations.

## Dataset

The project uses synthetic retail data.

Main fields:

- date
- store_id
- sku
- category
- price
- promotion
- demand_simulated
- sales
- stockout

## Machine Learning

Random Forest Regression is used for daily demand forecasting.

Features include:

- Calendar features
- Lag features
- Rolling averages
- Price
- Promotion indicator

## Inventory Optimization

The system calculates:

- Safety Stock
- Reorder Point
- Economic Order Quantity reference
- Target Stock
- Recommended Order Quantity

## Limitations

This is a learning and portfolio project.

The current stock figure is a demonstration estimate, not actual
inventory data. The safety stock calculation is a baseline using
historical demand variation.

A real deployment would require actual inventory position,
supplier lead times, minimum order quantities, pack sizes,
forecast-error-based uncertainty, and business validation.