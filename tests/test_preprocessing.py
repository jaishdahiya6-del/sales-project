"""
tests/test_preprocessing.py
---------------------------
Unit tests for the preprocessing, cleaning, and feature engineering modules in src/preprocessing.py.

Author: Data Analytics Team
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    remove_duplicates,
    handle_missing_values,
    detect_and_cap_outliers,
    engineer_features,
)


def test_remove_duplicates():
    """Verify that exact duplicate rows are correctly removed."""
    df = pd.DataFrame({
        "ID": [1, 2, 2, 3],
        "Val": ["A", "B", "B", "C"]
    })
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 3
    assert list(cleaned["ID"]) == [1, 2, 3]


def test_handle_missing_values():
    """Verify that missing values are correctly imputed (median for numeric, mode for categorical)."""
    df = pd.DataFrame({
        "NumericCol": [10.0, 20.0, np.nan, 40.0],
        "CategoricalCol": ["X", "Y", "Y", None]
    })
    imputed = handle_missing_values(df)

    # Median of [10.0, 20.0, 40.0] is 20.0
    assert imputed["NumericCol"].isnull().sum() == 0
    assert imputed["NumericCol"].iloc[2] == 20.0

    # Mode of ['X', 'Y', 'Y'] is 'Y'
    assert imputed["CategoricalCol"].isnull().sum() == 0
    assert imputed["CategoricalCol"].iloc[3] == "Y"


def test_detect_and_cap_outliers():
    """Verify that extreme values outside the IQR bounds are capped (winsorized)."""
    # 10 values, one extreme outlier (1000)
    df = pd.DataFrame({
        "Sales": [10.0, 12.0, 11.0, 14.0, 13.0, 12.0, 11.0, 15.0, 13.0, 1000.0]
    })
    capped = detect_and_cap_outliers(df, columns=["Sales"])

    # Outlier should be capped to upper limit
    q1 = df["Sales"].quantile(0.25)
    q3 = df["Sales"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    assert capped["Sales"].max() <= upper_bound
    assert capped["Sales"].iloc[9] < 1000.0


def test_engineer_features():
    """Verify that time-based and financial features are correctly engineered."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-03-15", "2024-06-20"]),
        "Sales": [200.0, 100.0],
        "Profit": [50.0, -10.0],
        "Quantity": [4, 2]
    })
    engineered = engineer_features(df)

    # Verify temporal features
    assert list(engineered["Year"]) == [2024, 2024]
    assert list(engineered["Month"]) == [3, 6]
    assert list(engineered["Quarter"]) == [1, 2]
    assert list(engineered["MonthName"]) == ["March", "June"]

    # Verify financial features
    # Profit Margin = (Profit / Sales) * 100
    assert list(engineered["ProfitMargin"]) == [25.0, -10.0]

    # Avg Order Value = Sales / Quantity
    assert list(engineered["AvgOrderValue"]) == [50.0, 50.0]

    # High Value Order target
    # Median Sales of [200, 100] is 150. HighValueOrder = (Sales > median)
    assert list(engineered["HighValueOrder"]) == [1, 0]
