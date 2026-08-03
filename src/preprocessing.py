"""
src/preprocessing.py
--------------------
Handles loading, cleaning, missing value imputation, duplicate removal,
outlier detection/capping, and feature engineering for the retail sales dataset.

Author: Data Analytics Team
"""

import os
from typing import List, Optional

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
    """
    Load the raw retail sales dataset from a CSV file.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file, by default RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file at `path` does not exist.
    """
    logger.info(f"Loading raw data from {path}")
    try:
        # Check that path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at: {path}")

        df = pd.read_csv(path, parse_dates=["Date"])
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError as exc:
        logger.error(f"File not found at path {path}: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error while loading data: {exc}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.

    Uses the following strategy:
      - Numeric columns: median imputation.
      - Categorical columns: mode imputation (first mode or 'Unknown' if none exists).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to impute (modifies in place or returns a copy).

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame.
    """
    df = df.copy()
    missing_before = int(df.isnull().sum().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            # If median is NaN (e.g. all values in col are NaN), default to 0
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        if df[col].isnull().any():
            modes = df[col].mode()
            mode_val = modes.iloc[0] if not modes.empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

    missing_after = int(df.isnull().sum().sum())
    logger.info(f"Imputed {missing_before - missing_after} missing values")
    return df


def detect_and_cap_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method and cap them.

    Capping (winsorization) is performed rather than row deletion to preserve
    sample sizes while reducing extreme-value skew.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Optional[List[str]], optional
        Columns to cap, by default ["Sales", "Quantity", "Profit", "UnitPrice"].

    Returns
    -------
    pd.DataFrame
        DataFrame with capped outlier values.
    """
    df = df.copy()
    if columns is None:
        columns = ["Sales", "Quantity", "Profit", "UnitPrice"]

    outlier_count = 0
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping outlier capping.")
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count += int(mask.sum())
        df[col] = df[col].clip(lower=lower, upper=upper)

    logger.info(f"Capped {outlier_count} outlier values across columns {columns}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering to create temporal and financial features.

    Features added:
      - Temporal: Year, Month, MonthName, Quarter, DayOfWeek
      - Financial: ProfitMargin, AvgOrderValue
      - Target: HighValueOrder (binary flag for Sales > median Sales)

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned DataFrame containing Date, Sales, Profit, and Quantity columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """
    df = df.copy()

    # Safety check for Date column
    if "Date" not in df.columns:
        raise KeyError("Required 'Date' column is missing from the DataFrame.")

    # Convert to datetime if not already
    df["Date"] = pd.to_datetime(df["Date"])

    # Temporal features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.month_name()
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.day_name()

    # Financial features
    # Prevent division by zero by replacing zero with NaN, then filling with 0
    df["ProfitMargin"] = (df["Profit"] / df["Sales"].replace(0, np.nan)) * 100
    df["ProfitMargin"] = df["ProfitMargin"].fillna(0.0)

    df["AvgOrderValue"] = df["Sales"] / df["Quantity"].replace(0, np.nan)
    df["AvgOrderValue"] = df["AvgOrderValue"].fillna(0.0)

    # Classification Target: order sales strictly greater than the median sales
    median_sales = df["Sales"].median() if not df["Sales"].empty else 0.0
    df["HighValueOrder"] = (df["Sales"] > median_sales).astype(int)

    logger.info(
        "Feature engineering complete: Added temporal and financial metrics. "
        "HighValueOrder target established."
    )
    return df


def run_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Run the entire preprocessing pipeline end-to-end.

    Parameters
    ----------
    save : bool, optional
        Whether to save the resulting DataFrame to PROCESSED_DATA_PATH, by default True.

    Returns
    -------
    pd.DataFrame
        Cleaned, imputed, and engineered DataFrame.
    """
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
