
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "sku",
    "category",
    "price",
    "promotion",
    "demand_simulated",
    "sales",
    "stockout"
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"


def read_dataset(path):
    """Read CSV or Excel retail dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run python -m src.data_generation first."
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError("Only CSV and Excel files are supported.")


def load_and_clean_data(path=RAW_DATA_PATH):

    print("\n" + "=" * 55)
    print("       RETAIL DATA PREPROCESSING")
    print("=" * 55)

    # 1. Load dataset
    df = read_dataset(path)

    print("\n[1] Dataset loaded successfully")
    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    # 2. Validate columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 3. Save dataset preview
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    preview_path = REPORT_DIR / "dataset_preview.csv"
    df.head(20).to_csv(preview_path, index=False)

    print("\n[2] Dataset preview saved")
    print(f"File: {preview_path}")

    # 4. Missing value analysis
    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    missing_path = REPORT_DIR / "missing_values_report.csv"
    missing_report.to_csv(missing_path, index=False)

    print("\n[3] Missing value analysis")
    print(missing_report.to_string(index=False))

    # 5. Convert data types
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "price",
        "promotion",
        "demand_simulated",
        "sales",
        "stockout"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # 6. Remove invalid and missing records
    before_cleaning = len(df)

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    removed_rows = before_cleaning - len(df)

    # 7. Correct invalid values
    for column in ["sales", "demand_simulated", "price"]:
        df[column] = df[column].clip(lower=0)

    df["promotion"] = df["promotion"].clip(0, 1).astype(int)
    df["stockout"] = df["stockout"].clip(0, 1).astype(int)

    # 8. Remove duplicate records
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["store_id", "sku", "date"],
        keep="last"
    )

    duplicate_rows = before_duplicates - len(df)

    # 9. Sort dataset
    df = df.sort_values(
        ["store_id", "sku", "date"]
    ).reset_index(drop=True)

    # 10. Save processed dataset
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # 11. Final summary
    print("\n" + "=" * 55)
    print("          DATA CLEANING COMPLETED")
    print("=" * 55)

    print(f"Original records     : {before_cleaning:,}")
    print(f"Removed missing rows : {removed_rows:,}")
    print(f"Removed duplicates   : {duplicate_rows:,}")
    print(f"Final records        : {len(df):,}")
    print(f"Stores               : {df['store_id'].nunique()}")
    print(f"Products             : {df['sku'].nunique()}")
    print(f"Categories           : {df['category'].nunique()}")

    print("\nGenerated files:")
    print(f"1. {PROCESSED_DATA_PATH}")
    print(f"2. {preview_path}")
    print(f"3. {missing_path}")

    print("=" * 55)

    return df


if __name__ == "__main__":
    load_and_clean_data()
