"""
analysis.py
-----------
Exploratory data analysis, statistical summaries, correlation analysis,
and trend analysis for the retail sales dataset.
"""

import os

import numpy as np
import pandas as pd

from src.utils import OUTPUTS_DIR, get_logger, ensure_dirs

logger = get_logger(__name__)


def summary_statistics(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Generate descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
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
    """Compute the Pearson correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    logger.info("Computed correlation matrix")
    return corr


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales, profit, and order counts by product category."""
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
    """Aggregate sales and profit by region."""
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
    """Compute month-over-month sales and profit trend."""
    trend = (
        df.groupby(pd.Grouper(key="Date", freq="MS"))
        .agg(TotalSales=("Sales", "sum"), TotalProfit=("Profit", "sum"))
        .reset_index()
    )
    logger.info("Computed monthly sales/profit trend")
    return trend


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top N products by total sales."""
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
