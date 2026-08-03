"""
preprocessing.py
----------------
Handles data loading, cleaning, missing value imputation, duplicate
removal, outlier detection, and feature engineering for the retail
sales dataset.
"""

import os

import numpy as np
import pandas as pd

from src.utils import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    get_logger,
    ensure_dirs,
)

logger = get_logger(__name__)


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw sales dataset from CSV."""
    logger.info(f"Loading raw data from {path}")
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error while loading data: {exc}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using sensible strategies per column type:
      - Numeric columns: median imputation
      - Categorical columns: mode imputation
    """
    missing_before = df.isnull().sum().sum()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

    missing_after = df.isnull().sum().sum()
    logger.info(f"Imputed {missing_before - missing_after} missing values")
    return df


def detect_and_cap_outliers(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Detect outliers using the IQR method and cap them (winsorize) rather
    than dropping rows, to preserve sample size.
    """
    if columns is None:
        columns = ["Sales", "Quantity", "Profit", "UnitPrice"]

    outlier_count = 0
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count += mask.sum()
        df[col] = df[col].clip(lower=lower, upper=upper)

    logger.info(f"Capped {outlier_count} outlier values across {columns}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features useful for analysis and modeling."""
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.month_name()
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["ProfitMargin"] = (df["Profit"] / df["Sales"].replace(0, np.nan)) * 100
    df["ProfitMargin"] = df["ProfitMargin"].fillna(0)
    df["AvgOrderValue"] = df["Sales"] / df["Quantity"].replace(0, np.nan)
    df["AvgOrderValue"] = df["AvgOrderValue"].fillna(0)
    df["HighValueOrder"] = (df["Sales"] > df["Sales"].median()).astype(int)

    logger.info("Feature engineering complete: Year, Month, Quarter, ProfitMargin, "
                "AvgOrderValue, HighValueOrder added")
    return df


def run_pipeline(save: bool = True) -> pd.DataFrame:
    """Execute the full preprocessing pipeline end-to-end."""
    ensure_dirs(os.path.dirname(PROCESSED_DATA_PATH))

    df = load_data()
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = detect_and_cap_outliers(df)
    df = engineer_features(df)

    if save:
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved cleaned dataset to {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run_pipeline()
