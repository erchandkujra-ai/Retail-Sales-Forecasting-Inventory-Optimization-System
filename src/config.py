from pathlib import Path

# Main project directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset paths
RAW_DATA_PATH = RAW_DATA_DIR / "retail_sales.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "clean_retail_sales.csv"

# Model and output directories
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
IMAGE_DIR = ROOT_DIR / "images"

# Project settings
RANDOM_STATE = 42
FORECAST_DAYS = 30

# Synthetic dataset settings
DATA_HISTORY_DAYS = 730
NUMBER_OF_STORES = 3
NUMBER_OF_PRODUCTS = 8

# Inventory settings
LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.65