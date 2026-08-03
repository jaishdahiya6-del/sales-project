"""
src/analysis.py
---------------
Performs exploratory data analysis (EDA), aggregates statistics, computes
correlation matrices, and maps performance trends for the retail sales dataset.

Author: Data Analytics Team
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import OUTPUTS_DIR, get_logger, ensure_dirs

logger = get_logger(__name__)


def summary_statistics(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Generate robust descriptive statistics for all numeric columns.

    Appends skewness and kurtosis calculations to pandas describe statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    save : bool, optional
        Whether to save the statistics to outputs/summary_statistics.csv, by default True.

    Returns
    -------
    pd.DataFrame
        Descriptive summary DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric columns found in the DataFrame to compute summary statistics.")
        return pd.DataFrame()

    summary = numeric_df.describe().T
    summary["skew"] = numeric_df.skew()
    summary["kurtosis"] = numeric_df.kurtosis()
    summary = summary.round(2)

    if save:
        ensure_dirs(OUTPUTS_DIR)
        out_path = os.path.join(OUTPUTS_DIR, "summary_statistics.csv")
        summary.to_csv(out_path)
        logger.info(f"Saved summary statistics to {out_path}")

    return summary


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Pearson correlation coefficient matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric columns found to calculate correlation matrix.")
        return pd.DataFrame()

    corr = numeric_df.corr(method="pearson", numeric_only=True)
    logger.info("Computed numeric correlation matrix")
    return corr


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate key financial and customer performance metrics by product category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'Category', 'Sales', 'Profit', 'OrderID', and 'SatisfactionScore'.

    Returns
    -------
    pd.DataFrame
        Category performance metrics aggregated.
    """
    required_cols = {"Category", "Sales", "Profit", "OrderID", "SatisfactionScore"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Cannot compute category performance, missing columns: {missing}")
        return pd.DataFrame()

    perf = (
        df.groupby("Category")
        .agg(
            TotalSales=("Sales", "sum"),
            TotalProfit=("Profit", "sum"),
            AvgOrderValue=("Sales", "mean"),
            OrderCount=("OrderID", "count"),
            AvgSatisfaction=("SatisfactionScore", "mean"),
        )
        .round(2)
        .sort_values("TotalSales", ascending=False)
    )
    logger.info("Computed category-level performance summary")
    return perf


def regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales, total profit, and order count by geographical region.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'Region', 'Sales', 'Profit', and 'OrderID'.

    Returns
    -------
    pd.DataFrame
        Regional performance metrics aggregated.
    """
    required_cols = {"Region", "Sales", "Profit", "OrderID"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Cannot compute regional performance, missing columns: {missing}")
        return pd.DataFrame()

    perf = (
        df.groupby("Region")
        .agg(
            TotalSales=("Sales", "sum"),
            TotalProfit=("Profit", "sum"),
            OrderCount=("OrderID", "count"),
        )
        .round(2)
        .sort_values("TotalSales", ascending=False)
    )
    logger.info("Computed regional performance summary")
    return perf


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month-over-month aggregated sales and profit trend.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'Date', 'Sales', and 'Profit'.

    Returns
    -------
    pd.DataFrame
        Month-over-month aggregated sales and profit trend DataFrame.
    """
    required_cols = {"Date", "Sales", "Profit"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Cannot compute monthly trend, missing columns: {missing}")
        return pd.DataFrame()

    # Aggregate monthly
    trend = (
        df.groupby(pd.Grouper(key="Date", freq="MS"))
        .agg(TotalSales=("Sales", "sum"), TotalProfit=("Profit", "sum"))
        .reset_index()
    )
    logger.info("Computed monthly sales/profit trend")
    return trend


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Find the top N products ranked by total sales revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n : int, optional
        Number of top products to return, by default 10.

    Returns
    -------
    pd.DataFrame
        DataFrame of top N products.
    """
    if "Product" not in df.columns or "Sales" not in df.columns:
        logger.error("Missing 'Product' or 'Sales' column for top product analysis.")
        return pd.DataFrame()

    top = (
        df.groupby("Product")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    return top


if __name__ == "__main__":
    from src.preprocessing import run_pipeline

    clean_df = run_pipeline()
    print(summary_statistics(clean_df))
    print(category_performance(clean_df))
