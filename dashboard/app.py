"""
dashboard/app.py
----------------
Production-Grade Interactive Retail Sales Analytics & Dynamic Business Simulator.

This module provides an enterprise-ready Streamlit dashboard for real-time
retail intelligence, predictive machine learning inference, sales trend forecasting,
seasonality analysis, anomaly detection, Pareto 80/20 performance ranking, and
dual-mode "What-If" order and macro strategy simulations.

Architecture & Dependencies:
  - Source Data: 'data/processed/sales_data_clean.csv' (Fallback to raw data / pipeline)
  - ML Artifacts: 'models/regression_model.joblib', 'models/classification_model.joblib', 'models/encoders.joblib'
  - Pipeline Outputs: 'outputs/model_metrics.json', 'outputs/prediction_results.csv'
  - Modular Helpers: 'src/utils.py', 'src/preprocessing.py'

Run via CLI:
    streamlit run dashboard/app.py

Author: Advanced Retail Analytics Team
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Allow importing from project root src/ directory when executed as `streamlit run dashboard/app.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    get_logger,
)

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Configuration & Schema Constants
# ------------------------------------------------------------------
@dataclass(frozen=True)
class DashboardConfig:
    """Central configuration parameters for schema validation, colors, and layout."""

    PAGE_TITLE: str = "Retail Intelligence & Simulation Hub"
    PAGE_ICON: str = "📊"
    LAYOUT: str = "wide"

    # Color Palette (Tailwind & Plotly inspired)
    PRIMARY_COLOR: str = "#2563eb"     # Royal Blue
    SUCCESS_COLOR: str = "#10b981"     # Emerald Green
    WARNING_COLOR: str = "#f59e0b"     # Amber
    DANGER_COLOR: str = "#ef4444"      # Crimson Red
    INFO_COLOR: str = "#06b6d4"        # Cyan
    DARK_BG: str = "#1e293b"          # Slate 800
    ACCENT_PURPLE: str = "#8b5cf6"    # Violet

    # Schema definition for raw / processed retail dataset validation
    REQUIRED_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "OrderID",
            "Date",
            "Category",
            "Product",
            "Region",
            "UnitPrice",
            "Quantity",
            "DiscountPct",
            "Sales",
            "Cost",
            "Profit",
            "CustomerAge",
            "SatisfactionScore",
        ]
    )

    FEATURE_COLUMNS_NUMERIC: List[str] = field(
        default_factory=lambda: [
            "UnitPrice",
            "Quantity",
            "DiscountPct",
            "CustomerAge",
            "SatisfactionScore",
            "Month",
            "Quarter",
        ]
    )

    FEATURE_COLUMNS_CATEGORICAL: List[str] = field(
        default_factory=lambda: ["Category", "Region"]
    )


CONFIG = DashboardConfig()


# ------------------------------------------------------------------
# Pure Analytical & Simulation Helper Functions (Testable & Decoupled)
# ------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataset schema, data types, and value sanity.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.

    Returns
    -------
    Tuple[bool, List[str]]
        - is_valid: True if required schema is present and readable.
        - issues: List of descriptive warnings or validation errors identified.
    """
    issues: List[str] = []
    if df is None or df.empty:
        return False, ["Dataset is empty or None."]

    missing_cols = [col for col in CONFIG.REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        try:
            pd.to_datetime(df["Date"])
        except Exception as exc:
            issues.append(f"'Date' column cannot be parsed to datetime: {exc}")

    for col in ["Sales", "UnitPrice", "Quantity"]:
        if col in df.columns:
            if (df[col] < 0).any():
                issues.append(f"Column '{col}' contains negative values.")

    missing_val_count = int(df.isnull().sum().sum())
    if missing_val_count > 0:
        issues.append(f"Dataset contains {missing_val_count} missing values across columns.")

    is_valid = len([i for i in issues if "Missing required" in i]) == 0
    return is_valid, issues


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute core financial and operational KPI metrics for a dataset slice.

    Parameters
    ----------
    df : pd.DataFrame
        Input pandas DataFrame.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metric values.
    """
    if df is None or df.empty:
        return {
            "total_sales": 0.0,
            "total_profit": 0.0,
            "total_orders": 0,
            "avg_margin": 0.0,
            "avg_order_value": 0.0,
            "avg_satisfaction": 0.0,
        }

    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(len(df))

    avg_margin = (
        float(df["ProfitMargin"].mean())
        if "ProfitMargin" in df.columns
        else (total_profit / total_sales * 100 if total_sales > 0 else 0.0)
    )

    avg_order_value = (
        float(df["AvgOrderValue"].mean())
        if "AvgOrderValue" in df.columns
        else (total_sales / total_orders if total_orders > 0 else 0.0)
    )

    avg_satisfaction = (
        float(df["SatisfactionScore"].mean())
        if "SatisfactionScore" in df.columns
        else 0.0
    )

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "avg_margin": avg_margin,
        "avg_order_value": avg_order_value,
        "avg_satisfaction": avg_satisfaction,
    }


def compute_sales_forecast(
    df: pd.DataFrame, forecast_months: int = 6, confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Aggregate sales monthly and compute linear-trend forecasting with confidence bands.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing 'Date' and 'Sales'.
    forecast_months : int, optional
        Number of future months to forecast, by default 6.
    confidence_level : float, optional
        Confidence level for prediction intervals, by default 0.95.

    Returns
    -------
    pd.DataFrame
        DataFrame with Historical & Forecast dates, ActualSales, ForecastSales,
        LowerBound, UpperBound, and IsForecast indicator.
    """
    if df is None or df.empty or "Date" not in df.columns or "Sales" not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])

    # Aggregate by Month Start
    monthly = (
        df_copy.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    if len(monthly) < 3:
        return pd.DataFrame()

    monthly["TimeIdx"] = np.arange(len(monthly))

    # Fit ordinary least squares linear trend + exponential trend blend
    x = monthly["TimeIdx"].values
    y = monthly["Sales"].values

    poly_coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = poly_coeffs[0], poly_coeffs[1]

    fitted_y = slope * x + intercept
    residuals = y - fitted_y
    std_err = float(np.std(residuals)) if len(residuals) > 1 else 1000.0

    # Z-score multiplier for confidence interval
    z_score = 1.96 if confidence_level == 0.95 else 1.645

    # Generate historical records
    records = []
    for idx, row in monthly.iterrows():
        records.append({
            "Date": row["Date"],
            "ActualSales": round(float(row["Sales"]), 2),
            "ForecastSales": round(float(slope * row["TimeIdx"] + intercept), 2),
            "LowerBound": round(max(0.0, float(slope * row["TimeIdx"] + intercept - z_score * std_err)), 2),
            "UpperBound": round(float(slope * row["TimeIdx"] + intercept + z_score * std_err), 2),
            "IsForecast": False,
        })

    # Generate future forecast records
    last_date = monthly["Date"].max()
    last_idx = monthly["TimeIdx"].max()

    for i in range(1, forecast_months + 1):
        future_date = last_date + pd.DateOffset(months=i)
        future_idx = last_idx + i
        f_val = slope * future_idx + intercept
        # Expand uncertainty over time horizon
        horizon_std = std_err * (1.0 + 0.05 * i)

        records.append({
            "Date": future_date,
            "ActualSales": np.nan,
            "ForecastSales": round(max(0.0, float(f_val)), 2),
            "LowerBound": round(max(0.0, float(f_val - z_score * horizon_std)), 2),
            "UpperBound": round(float(f_val + z_score * horizon_std), 2),
            "IsForecast": True,
        })

    return pd.DataFrame(records)


def compute_seasonality_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute monthly seasonal indices, month-over-month growth rates, and quarterly trends.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'Date', 'Sales', 'Profit'.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'monthly_index', 'quarterly_summary', and 'mom_growth'.
    """
    if df is None or df.empty or "Date" not in df.columns:
        return {
            "monthly_index": pd.DataFrame(),
            "quarterly_summary": pd.DataFrame(),
            "mom_growth": pd.DataFrame(),
        }

    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    df_copy["Month"] = df_copy["Date"].dt.month
    df_copy["MonthName"] = df_copy["Date"].dt.month_name()
    df_copy["Quarter"] = df_copy["Date"].dt.quarter

    # Monthly Seasonality Index
    month_agg = (
        df_copy.groupby(["Month", "MonthName"])
        .agg(
            TotalSales=("Sales", "sum"),
            AvgSales=("Sales", "mean"),
            TotalProfit=("Profit", "sum"),
            OrderCount=("OrderID", "count"),
        )
        .reset_index()
    )

    overall_mean_sales = month_agg["TotalSales"].mean()
    month_agg["SeasonalIndex"] = (
        (month_agg["TotalSales"] / overall_mean_sales * 100).round(2)
        if overall_mean_sales > 0
        else 100.0
    )

    # Quarterly Breakdown
    quarter_agg = (
        df_copy.groupby("Quarter")
        .agg(
            TotalSales=("Sales", "sum"),
            TotalProfit=("Profit", "sum"),
            OrderCount=("OrderID", "count"),
        )
        .reset_index()
    )
    quarter_agg["SalesShare"] = (
        (quarter_agg["TotalSales"] / quarter_agg["TotalSales"].sum() * 100).round(2)
        if quarter_agg["TotalSales"].sum() > 0
        else 0.0
    )

    # Month-over-Month Growth
    ts_monthly = (
        df_copy.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )
    ts_monthly["MoMGrowthPct"] = ts_monthly["Sales"].pct_change() * 100
    ts_monthly["MoMGrowthPct"] = ts_monthly["MoMGrowthPct"].round(2).fillna(0.0)

    return {
        "monthly_index": month_agg.sort_values("Month"),
        "quarterly_summary": quarter_agg,
        "mom_growth": ts_monthly,
    }


def compute_pareto_analysis(
    df: pd.DataFrame, entity_col: str = "Product"
) -> pd.DataFrame:
    """
    Compute Pareto 80/20 distribution metrics for products or regions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    entity_col : str, optional
        Categorical grouping column (e.g. 'Product', 'Region', 'Category'), by default 'Product'.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by Sales descending, with CumulativeSales, SharePct,
        CumulativeSharePct, and InTop80Percent boolean flag.
    """
    if df is None or df.empty or entity_col not in df.columns or "Sales" not in df.columns:
        return pd.DataFrame()

    agg = (
        df.groupby(entity_col)
        .agg(
            TotalSales=("Sales", "sum"),
            TotalProfit=("Profit", "sum"),
            OrderCount=("OrderID", "count"),
        )
        .reset_index()
        .sort_values("TotalSales", ascending=False)
    )

    grand_total_sales = agg["TotalSales"].sum()
    if grand_total_sales <= 0:
        return pd.DataFrame()

    agg["SharePct"] = (agg["TotalSales"] / grand_total_sales * 100).round(2)
    agg["CumulativeSales"] = agg["TotalSales"].cumsum()
    agg["CumulativeSharePct"] = (
        agg["CumulativeSales"] / grand_total_sales * 100
    ).round(2)
    agg["InTop80Percent"] = agg["CumulativeSharePct"] <= 80.0

    # Ensure at least the top contributor is True
    if not agg.empty:
        agg.iloc[0, agg.columns.get_loc("InTop80Percent")] = True

    return agg


def detect_transaction_anomalies(
    df: pd.DataFrame, contamination: float = 0.03, method: str = "isolation_forest"
) -> pd.DataFrame:
    """
    Detect anomalous transactions using Isolation Forest or IQR statistical bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric features ('Sales', 'UnitPrice', 'Quantity', 'Profit').
    contamination : float, optional
        Expected fraction of outliers/anomalies in the dataset, by default 0.03.
    method : str, optional
        Detection algorithm ('isolation_forest' or 'iqr'), by default 'isolation_forest'.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added 'IsAnomaly' (bool) and 'AnomalyScore' (float) columns.
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()
    feature_cols = [c for c in ["Sales", "UnitPrice", "Quantity", "Profit"] if c in out_df.columns]

    if len(feature_cols) == 0:
        out_df["IsAnomaly"] = False
        out_df["AnomalyScore"] = 0.0
        return out_df

    try:
        if method == "isolation_forest":
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )
            # Impute any missing values temporarily for prediction
            num_data = out_df[feature_cols].fillna(out_df[feature_cols].median())
            preds = clf.fit_predict(num_data)
            scores = clf.decision_function(num_data)

            out_df["IsAnomaly"] = preds == -1
            out_df["AnomalyScore"] = np.round(-scores, 3)
        else:
            # Fallback IQR method on Sales
            q1 = out_df["Sales"].quantile(0.25)
            q3 = out_df["Sales"].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 3.0 * iqr  # Extreme outlier threshold
            lower = max(0.0, q1 - 3.0 * iqr)

            out_df["IsAnomaly"] = (out_df["Sales"] > upper) | (out_df["Sales"] < lower)
            out_df["AnomalyScore"] = (out_df["Sales"] - q3) / (iqr + 1e-6)
    except Exception as exc:
        logger.error(f"Anomaly detection failed: {exc}")
        out_df["IsAnomaly"] = False
        out_df["AnomalyScore"] = 0.0

    return out_df


def simulate_macro_scenario(
    df: pd.DataFrame,
    price_adj_pct: float,
    demand_adj_pct: float,
    cost_adj_pct: float,
    selected_categories: Optional[List[str]] = None,
    selected_regions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Simulate macro business strategic adjustments (Price, Volume, Cost shifts).

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned DataFrame.
    price_adj_pct : float
        Percentage price shift (e.g. +5.0 for +5% price increase).
    demand_adj_pct : float
        Percentage demand/volume shift (e.g. -2.0 for -2% volume drop).
    cost_adj_pct : float
        Percentage cost shift (e.g. +3.0 for +3% supplier cost inflation).
    selected_categories : Optional[List[str]], optional
        Categories affected by strategy, by default None (all categories).
    selected_regions : Optional[List[str]], optional
        Regions affected by strategy, by default None (all regions).

    Returns
    -------
    Dict[str, Any]
        Dictionary of baseline vs simulated financials, net variance, and category-level impacts.
    """
    if df is None or df.empty:
        return {
            "baseline_sales": 0.0,
            "simulated_sales": 0.0,
            "sales_delta": 0.0,
            "sales_delta_pct": 0.0,
            "baseline_profit": 0.0,
            "simulated_profit": 0.0,
            "profit_delta": 0.0,
            "profit_delta_pct": 0.0,
            "category_comparison": pd.DataFrame(),
        }

    sim_df = df.copy()

    # Determine affected mask
    mask = pd.Series(True, index=sim_df.index)
    if selected_categories:
        mask &= sim_df["Category"].isin(selected_categories)
    if selected_regions:
        mask &= sim_df["Region"].isin(selected_regions)

    price_factor = 1.0 + (price_adj_pct / 100.0)
    demand_factor = 1.0 + (demand_adj_pct / 100.0)
    cost_factor = 1.0 + (cost_adj_pct / 100.0)

    # Compute baseline
    sim_df["SimUnitPrice"] = sim_df["UnitPrice"]
    sim_df["SimQuantity"] = sim_df["Quantity"]
    sim_df["SimCost"] = sim_df["Cost"]

    # Apply adjustments to masked rows
    sim_df.loc[mask, "SimUnitPrice"] = sim_df.loc[mask, "UnitPrice"] * price_factor
    sim_df.loc[mask, "SimQuantity"] = (sim_df.loc[mask, "Quantity"] * demand_factor).round()
    sim_df.loc[mask, "SimCost"] = sim_df.loc[mask, "Cost"] * cost_factor

    # Recalculate financial metrics
    discount_mult = 1.0 - (sim_df["DiscountPct"] / 100.0)
    sim_df["SimSales"] = sim_df["SimUnitPrice"] * sim_df["SimQuantity"] * discount_mult
    sim_df["SimTotalCost"] = sim_df["SimCost"] * sim_df["SimQuantity"]
    sim_df["SimProfit"] = sim_df["SimSales"] - sim_df["SimTotalCost"]

    # Aggregate summaries
    base_sales = float(sim_df["Sales"].sum())
    sim_sales = float(sim_df["SimSales"].sum())
    sales_delta = sim_sales - base_sales
    sales_delta_pct = (sales_delta / base_sales * 100) if base_sales > 0 else 0.0

    base_profit = float(sim_df["Profit"].sum())
    sim_profit = float(sim_df["SimProfit"].sum())
    profit_delta = sim_profit - base_profit
    profit_delta_pct = (profit_delta / base_profit * 100) if abs(base_profit) > 0 else 0.0

    # Category level aggregation
    cat_comp = (
        sim_df.groupby("Category")
        .agg(
            BaselineSales=("Sales", "sum"),
            SimulatedSales=("SimSales", "sum"),
            BaselineProfit=("Profit", "sum"),
            SimulatedProfit=("SimProfit", "sum"),
        )
        .reset_index()
    )
    cat_comp["SalesVariance"] = cat_comp["SimulatedSales"] - cat_comp["BaselineSales"]
    cat_comp["ProfitVariance"] = cat_comp["SimulatedProfit"] - cat_comp["BaselineProfit"]

    return {
        "baseline_sales": base_sales,
        "simulated_sales": sim_sales,
        "sales_delta": sales_delta,
        "sales_delta_pct": sales_delta_pct,
        "baseline_profit": base_profit,
        "simulated_profit": sim_profit,
        "profit_delta": profit_delta,
        "profit_delta_pct": profit_delta_pct,
        "category_comparison": cat_comp,
    }


def simulate_order_inference(
    reg_model: Any,
    clf_model: Any,
    encoders: Dict[str, LabelEncoder],
    input_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute real-time ML inference for a custom simulated order profile.

    Parameters
    ----------
    reg_model : Any
        Pre-trained RandomForestRegressor model.
    clf_model : Any
        Pre-trained RandomForestClassifier model.
    encoders : Dict[str, LabelEncoder]
        Dictionary of pre-fitted LabelEncoder instances for Category & Region.
    input_params : Dict[str, Any]
        Dictionary containing transaction inputs (Category, Region, UnitPrice, Quantity,
        DiscountPct, CustomerAge, SatisfactionScore, Month, Quarter).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predicted sales ($), high-value order decision (0/1),
        and high-value probability score.
    """
    if reg_model is None or clf_model is None or not encoders:
        return {
            "predicted_sales": 0.0,
            "is_high_value": 0,
            "high_value_prob": 0.0,
            "error": "Models or Encoders not loaded.",
        }

    try:
        cat_str = str(input_params.get("Category", ""))
        reg_str = str(input_params.get("Region", ""))

        # Encode categorical inputs with unseen label handling fallback
        cat_encoder = encoders.get("Category")
        if cat_encoder and cat_str in cat_encoder.classes_:
            cat_enc = int(cat_encoder.transform([cat_str])[0])
        else:
            cat_enc = 0

        reg_encoder = encoders.get("Region")
        if reg_encoder and reg_str in reg_encoder.classes_:
            reg_enc = int(reg_encoder.transform([reg_str])[0])
        else:
            reg_enc = 0

        # Construct exact feature row matching src/model.py prepare_features
        # FEATURE_COLUMNS = ["UnitPrice", "Quantity", "DiscountPct", "CustomerAge", "SatisfactionScore", "Month", "Quarter"]
        # Categorical columns: Category_enc, Region_enc
        col_headers = [
            "UnitPrice",
            "Quantity",
            "DiscountPct",
            "CustomerAge",
            "SatisfactionScore",
            "Month",
            "Quarter",
            "Category_enc",
            "Region_enc",
        ]

        feature_values = [
            float(input_params.get("UnitPrice", 100.0)),
            int(input_params.get("Quantity", 1)),
            float(input_params.get("DiscountPct", 0.0)),
            float(input_params.get("CustomerAge", 30.0)),
            float(input_params.get("SatisfactionScore", 4.0)),
            int(input_params.get("Month", 1)),
            int(input_params.get("Quarter", 1)),
            cat_enc,
            reg_enc,
        ]

        input_df = pd.DataFrame([feature_values], columns=col_headers)

        pred_sales = float(reg_model.predict(input_df)[0])
        pred_prob = float(clf_model.predict_proba(input_df)[0][1])
        pred_class = int(clf_model.predict(input_df)[0])

        return {
            "predicted_sales": round(pred_sales, 2),
            "is_high_value": pred_class,
            "high_value_prob": round(pred_prob, 4),
            "error": None,
        }
    except Exception as exc:
        logger.error(f"Inference simulation error: {exc}")
        return {
            "predicted_sales": 0.0,
            "is_high_value": 0,
            "high_value_prob": 0.0,
            "error": str(exc),
        }


# ------------------------------------------------------------------
# Cached Asset Loaders
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and validating retail datasets...")
def load_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load, clean, and validate processed and raw datasets for the dashboard.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        - df: Cleaned and engineered dataset.
        - raw_df: Original raw dataset.
        - validation_issues: Validation messages or warnings list.
    """
    df: Optional[pd.DataFrame] = None
    raw_df: Optional[pd.DataFrame] = None

    try:
        if os.path.exists(PROCESSED_DATA_PATH):
            df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
        else:
            # Fallback to run preprocessing pipeline
            from src.preprocessing import run_pipeline
            df = run_pipeline(save=True)
    except Exception as exc:
        logger.error(f"Failed to load processed dataset: {exc}")
        # Secondary fallback directly to raw or synthetic generation
        if os.path.exists(RAW_DATA_PATH):
            df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])

    try:
        if os.path.exists(RAW_DATA_PATH):
            raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
        else:
            raw_df = df.copy() if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.error(f"Failed to load raw dataset: {exc}")
        raw_df = df.copy() if df is not None else pd.DataFrame()

    is_valid, validation_issues = validate_dataset(df)
    logger.info(f"Loaded dataset: {len(df) if df is not None else 0} rows. Valid: {is_valid}")

    return df, raw_df, validation_issues


@st.cache_resource(show_spinner="Loading serialized machine learning models...")
def load_serialized_models() -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, LabelEncoder]]]:
    """
    Load serialized Random Forest models and fitted LabelEncoders.

    Returns
    -------
    Tuple[Optional[Any], Optional[Any], Optional[Dict[str, LabelEncoder]]]
        - regression_model: RandomForestRegressor model instance or None.
        - classification_model: RandomForestClassifier model instance or None.
        - encoders: Dictionary of fitted LabelEncoder mapping or None.
    """
    reg_path = os.path.join(MODELS_DIR, "regression_model.joblib")
    clf_path = os.path.join(MODELS_DIR, "classification_model.joblib")
    enc_path = os.path.join(MODELS_DIR, "encoders.joblib")

    if os.path.exists(reg_path) and os.path.exists(clf_path) and os.path.exists(enc_path):
        try:
            reg_model = joblib.load(reg_path)
            clf_model = joblib.load(clf_path)
            encoders = joblib.load(enc_path)
            logger.info("Successfully loaded all serialized ML models and encoders.")
            return reg_model, clf_model, encoders
        except Exception as exc:
            logger.error(f"Error loading model files: {exc}")
            return None, None, None
    else:
        logger.warning("Serialized model files not found in models/ directory.")
        return None, None, None


# ------------------------------------------------------------------
# Plotly Chart Builder Functions
# ------------------------------------------------------------------
def plot_sales_revenue_velocity(df: pd.DataFrame) -> go.Figure:
    """Plot monthly sales revenue velocity and profit trend line."""
    if df is None or df.empty:
        return go.Figure()

    monthly_agg = (
        df.groupby(pd.Grouper(key="Date", freq="MS"))
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly_agg["Date"],
            y=monthly_agg["Sales"],
            mode="lines+markers",
            name="Revenue Sales ($)",
            line=dict(color=CONFIG.PRIMARY_COLOR, width=3),
            marker=dict(size=6),
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_agg["Date"],
            y=monthly_agg["Profit"],
            mode="lines+markers",
            name="Gross Profit ($)",
            line=dict(color=CONFIG.SUCCESS_COLOR, width=3, dash="dash"),
            marker=dict(size=6),
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Profit:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Monthly Revenue Velocity & Profit Trajectory</b>",
        xaxis_title="Timeline Window",
        yaxis_title="Financial Velocity ($)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Plot historical sales and future trend forecast with confidence intervals."""
    if forecast_df is None or forecast_df.empty:
        return go.Figure()

    fig = go.Figure()

    hist_df = forecast_df[~forecast_df["IsForecast"]]
    future_df = forecast_df[forecast_df["IsForecast"]]

    # Historical Sales
    fig.add_trace(
        go.Scatter(
            x=hist_df["Date"],
            y=hist_df["ActualSales"],
            mode="lines+markers",
            name="Actual Monthly Sales",
            line=dict(color=CONFIG.PRIMARY_COLOR, width=3),
            marker=dict(size=6),
        )
    )

    # Forecast Sales
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["ForecastSales"],
            mode="lines",
            name="Linear Trend Projection",
            line=dict(color=CONFIG.ACCENT_PURPLE, width=2.5, dash="dot"),
        )
    )

    # Confidence Interval Upper & Lower Bounds
    fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["UpperBound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["LowerBound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(139, 92, 246, 0.15)",
            name="95% Confidence Band",
        )
    )

    fig.update_layout(
        title="<b>Sales Trend Forecast & 95% Confidence Projection</b>",
        xaxis_title="Date Horizon",
        yaxis_title="Monthly Revenue ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_pareto_chart(pareto_df: pd.DataFrame, entity_name: str = "Product") -> go.Figure:
    """Plot dual-axis Pareto 80/20 chart combining revenue bars and cumulative percentage line."""
    if pareto_df is None or pareto_df.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    top_df = pareto_df.head(15)

    # Bar chart for revenue
    fig.add_trace(
        go.Bar(
            x=top_df[entity_name],
            y=top_df["TotalSales"],
            name="Total Sales ($)",
            marker_color=CONFIG.PRIMARY_COLOR,
            hovertemplate=f"<b>{entity_name}:</b> %{{x}}<br><b>Sales:</b> $%{{y:,.2f}}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Line chart for cumulative percentage
    fig.add_trace(
        go.Scatter(
            x=top_df[entity_name],
            y=top_df["CumulativeSharePct"],
            name="Cumulative Share (%)",
            mode="lines+markers",
            line=dict(color=CONFIG.WARNING_COLOR, width=3),
            marker=dict(size=7),
            hovertemplate=f"<b>{entity_name}:</b> %{{x}}<br><b>Cum Share:</b> %{{y:.1f}}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # 80% Threshold Line
    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color=CONFIG.DANGER_COLOR,
        annotation_text="80% Cumulative Threshold",
        annotation_position="bottom right",
        secondary_y=True,
    )

    fig.update_layout(
        title=f"<b>Pareto 80/20 Distribution Analysis ({entity_name}s)</b>",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=entity_name, tickangle=-20)
    fig.update_yaxes(title_text="Total Sales Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Revenue Share (%)", range=[0, 105], secondary_y=True)

    return fig


def plot_anomaly_scatter(anomaly_df: pd.DataFrame) -> go.Figure:
    """Plot scatter view highlighting transaction anomalies."""
    if anomaly_df is None or anomaly_df.empty:
        return go.Figure()

    fig = px.scatter(
        anomaly_df,
        x="UnitPrice",
        y="Sales",
        color="IsAnomaly",
        size="Quantity",
        hover_data=["OrderID", "Category", "Region", "Profit", "AnomalyScore"],
        color_discrete_map={False: CONFIG.PRIMARY_COLOR, True: CONFIG.DANGER_COLOR},
        labels={"IsAnomaly": "Flagged Outlier", "UnitPrice": "Unit Price ($)", "Sales": "Order Sales ($)"},
        title="<b>Isolation Forest Transaction Anomaly Map</b>",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_correlation_heatmap_plotly(df: pd.DataFrame) -> go.Figure:
    """Plot correlation heatmap matrix for numeric features."""
    if df is None or df.empty:
        return go.Figure()

    num_df = df.select_dtypes(include=[np.number])
    cols = [c for c in num_df.columns if not c.endswith("_enc")]
    corr_mat = num_df[cols].corr().round(2)

    fig = px.imshow(
        corr_mat,
        x=corr_mat.columns,
        y=corr_mat.columns,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="<b>Multivariate Feature Correlation Matrix (Pearson)</b>",
        template="plotly_white",
    )
    fig.update_layout(height=550, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ------------------------------------------------------------------
# Main UI Execution App
# ------------------------------------------------------------------
def main():
    # Page Setup
    st.set_page_config(
        page_title=CONFIG.PAGE_TITLE,
        page_icon=CONFIG.PAGE_ICON,
        layout=CONFIG.LAYOUT,
        initial_sidebar_state="expanded",
    )

    # Global CSS Theme & Metric Card Styles
    st.markdown(
        """
        <style>
            .main-header {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 24px;
                border-radius: 12px;
                color: white;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .metric-card {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            .metric-title {
                font-size: 0.85rem;
                font-weight: 600;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #0f172a;
                margin: 4px 0;
            }
            .metric-sub {
                font-size: 0.8rem;
                color: #10b981;
                font-weight: 500;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#ffffff;">
                📊 Enterprise Retail Intelligence & Dynamic Simulator
            </h1>
            <p style="margin:8px 0 0 0; font-size:1.05rem; color:#94a3b8;">
                Production-grade retail analytics, sales forecasting, transaction anomaly detection, and predictive What-If simulation engine.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Data & Models
    df, raw_df, validation_issues = load_dashboard_data()
    reg_model, clf_model, encoders = load_serialized_models()

    if df is None or df.empty:
        st.error("❌ **Critical Error:** Unable to load retail datasets. Please verify project data paths.")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar Filtering & Control Panel
    # ------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ Navigation & Control")

    navigation_page = st.sidebar.radio(
        "Select Analytics View",
        [
            "📈 Executive Summary & KPIs",
            "🔮 Trend Forecasting & Seasonality",
            "🏆 Product & Regional Deep-Dive",
            "🚨 Anomaly Detection & Quality",
            "🔗 Multivariate Feature Analytics",
            "🧹 Data Pipeline Audit Log",
            "⚙️ Dynamic 'What-If' Simulator",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🔍 Global Data Filters")

    # Categories
    all_categories = sorted(df["Category"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Product Category",
        options=all_categories,
        default=all_categories,
    )

    # Regions
    all_regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Geographical Region",
        options=all_regions,
        default=all_regions,
    )

    # Date Range
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Analysis Time Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Apply Filtering
    filtered_df = df.copy()

    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if selected_regions:
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_dt = pd.Timestamp(selected_date_range[0])
        end_dt = pd.Timestamp(selected_date_range[1])
        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_dt) & (filtered_df["Date"] <= end_dt)
        ]

    # Global KPI Summary Panel
    kpis = compute_kpis(filtered_df)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Sales Revenue</div>
                <div class="metric-value">${kpis['total_sales']:,.2f}</div>
                <div class="metric-sub">Avg Order: ${kpis['avg_order_value']:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Gross Accumulated Profit</div>
                <div class="metric-value">${kpis['total_profit']:,.2f}</div>
                <div class="metric-sub">Avg Margin: {kpis['avg_margin']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Orders Volume</div>
                <div class="metric-value">{kpis['total_orders']:,}</div>
                <div class="metric-sub">Processed Transactions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Customer Satisfaction</div>
                <div class="metric-value">⭐ {kpis['avg_satisfaction']:.2f} / 5.0</div>
                <div class="metric-sub">Mean Score Across Cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Page Rendering Router
    # ------------------------------------------------------------------

    if navigation_page == "📈 Executive Summary & KPIs":
        st.subheader("📋 Executive Operational Summary & Data Table")

        if filtered_df.empty:
            st.warning("⚠️ No data available matching current global filter criteria.")
        else:
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                st.plotly_chart(
                    plot_sales_revenue_velocity(filtered_df),
                    use_container_width=True,
                )
            with col_t2:
                top_prods = (
                    filtered_df.groupby("Product")["Sales"]
                    .sum()
                    .reset_index()
                    .sort_values("Sales", ascending=True)
                    .tail(8)
                )
                fig_top = px.bar(
                    top_prods,
                    x="Sales",
                    y="Product",
                    orientation="h",
                    title="<b>Top 8 Revenue Generating Products</b>",
                    color="Sales",
                    color_continuous_scale="Viridis",
                    template="plotly_white",
                )
                st.plotly_chart(fig_top, use_container_width=True)

            st.markdown("### 🗂️ Interactive Dataset Explorer")

            col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
            with col_s1:
                sort_col = st.selectbox(
                    "Sort Table By",
                    options=["Date", "Sales", "Profit", "Quantity", "SatisfactionScore", "UnitPrice"],
                    index=0,
                )
            with col_s2:
                sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
            with col_s3:
                row_limit = st.slider("Display Limit", min_value=10, max_value=200, value=50, step=10)

            sorted_view = filtered_df.sort_values(
                by=sort_col, ascending=(sort_order == "Ascending")
            ).head(row_limit)

            st.dataframe(
                sorted_view,
                use_container_width=True,
                column_config={
                    "OrderID": "Order ID",
                    "Date": "Date Purchased",
                    "Sales": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                    "Profit": st.column_config.NumberColumn("Profit ($)", format="$%.2f"),
                    "SatisfactionScore": "Satisfaction",
                },
            )

            st.download_button(
                label="⬇️ Export Filtered View to CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_retail_sales.csv",
                mime="text/csv",
            )

    elif navigation_page == "🔮 Trend Forecasting & Seasonality":
        st.subheader("🔮 Sales Trend Forecasting & Seasonality Analytics")

        if filtered_df.empty:
            st.warning("⚠️ No data available matching current filter selection.")
        else:
            tab_fc, tab_sn = st.tabs(["📈 Trend Forecast", "🗓️ Monthly Seasonality & Growth"])

            with tab_fc:
                col_f1, col_f2 = st.columns([1, 3])
                with col_f1:
                    st.markdown("#### Forecast Settings")
                    horizon = st.slider("Forecast Horizon (Months)", min_value=3, max_value=12, value=6)
                    conf_level = st.selectbox("Confidence Interval", [0.95, 0.90], index=0)

                    forecast_df = compute_sales_forecast(
                        filtered_df, forecast_months=horizon, confidence_level=conf_level
                    )

                    if not forecast_df.empty:
                        future_only = forecast_df[forecast_df["IsForecast"]]
                        proj_total = future_only["ForecastSales"].sum()
                        st.metric("Projected Horizon Sales", f"${proj_total:,.2f}")

                with col_f2:
                    if not forecast_df.empty:
                        fig_fc = plot_forecast_chart(forecast_df)
                        st.plotly_chart(fig_fc, use_container_width=True)
                    else:
                        st.info("Insufficient timeline records to generate linear forecast.")

            with tab_sn:
                season_data = compute_seasonality_analysis(filtered_df)

                col_sn1, col_sn2 = st.columns(2)
                with col_sn1:
                    monthly_idx = season_data["monthly_index"]
                    if not monthly_idx.empty:
                        fig_sn = px.bar(
                            monthly_idx,
                            x="MonthName",
                            y="SeasonalIndex",
                            color="SeasonalIndex",
                            color_continuous_scale="Tealgrn",
                            title="<b>Monthly Seasonal Index (Baseline = 100)</b>",
                            template="plotly_white",
                        )
                        fig_sn.add_hline(y=100.0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_sn, use_container_width=True)

                with col_sn2:
                    mom_df = season_data["mom_growth"]
                    if not mom_df.empty:
                        fig_mom = px.line(
                            mom_df,
                            x="Date",
                            y="MoMGrowthPct",
                            markers=True,
                            title="<b>Month-over-Month Revenue Growth Rate (%)</b>",
                            template="plotly_white",
                        )
                        fig_mom.add_hline(y=0.0, line_color="red", line_dash="dot")
                        st.plotly_chart(fig_mom, use_container_width=True)

    elif navigation_page == "🏆 Product & Regional Deep-Dive":
        st.subheader("🏆 Product & Regional Deep-Dive (Pareto 80/20)")

        if filtered_df.empty:
            st.warning("⚠️ No data available matching selection.")
        else:
            tab_par, tab_reg = st.tabs(["📊 Pareto 80/20 Product Ranking", "🌍 Regional Concentration"])

            with tab_par:
                pareto_df = compute_pareto_analysis(filtered_df, entity_col="Product")
                if not pareto_df.empty:
                    st.plotly_chart(plot_pareto_chart(pareto_df, "Product"), use_container_width=True)

                    top_80_count = pareto_df["InTop80Percent"].sum()
                    total_prod_count = len(pareto_df)
                    st.info(
                        f"💡 **Pareto Rule Insight:** {top_80_count} out of {total_prod_count} products "
                        f"({(top_80_count/total_prod_count*100):.1f}%) generate 80% of total revenue."
                    )
                else:
                    st.info("No product pareto data available.")

            with tab_reg:
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    reg_agg = (
                        filtered_df.groupby("Region")["Sales"]
                        .sum()
                        .reset_index()
                        .sort_values("Sales", ascending=False)
                    )
                    fig_pie = px.pie(
                        reg_agg,
                        names="Region",
                        values="Sales",
                        title="<b>Regional Revenue Market Share</b>",
                        template="plotly_white",
                        hole=0.4,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_r2:
                    cat_reg_agg = (
                        filtered_df.groupby(["Region", "Category"])["Sales"]
                        .sum()
                        .reset_index()
                    )
                    fig_heat = px.density_heatmap(
                        cat_reg_agg,
                        x="Region",
                        y="Category",
                        z="Sales",
                        title="<b>Region vs Category Revenue Matrix</b>",
                        color_continuous_scale="Viridis",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

    elif navigation_page == "🚨 Anomaly Detection & Quality":
        st.subheader("🚨 Transaction Anomaly Detection & Quality Control")

        if filtered_df.empty:
            st.warning("⚠️ No data available matching current filters.")
        else:
            col_a1, col_a2 = st.columns([1, 3])
            with col_a1:
                st.markdown("#### Anomaly Tuning")
                contamination = st.slider("Expected Contamination (%)", 1, 10, 3) / 100.0
                method = st.selectbox("Detection Method", ["isolation_forest", "iqr"])

                anomaly_df = detect_transaction_anomalies(
                    filtered_df, contamination=contamination, method=method
                )
                anom_count = anomaly_df["IsAnomaly"].sum()
                st.metric("Flagged Anomalous Orders", f"{anom_count} / {len(anomaly_df)}")

            with col_a2:
                fig_anom = plot_anomaly_scatter(anomaly_df)
                st.plotly_chart(fig_anom, use_container_width=True)

            if anom_count > 0:
                st.markdown("#### Flagged Anomalous Orders Table")
                st.dataframe(
                    anomaly_df[anomaly_df["IsAnomaly"]].sort_values("AnomalyScore", ascending=False),
                    use_container_width=True,
                )

    elif navigation_page == "🔗 Multivariate Feature Analytics":
        st.subheader("🔗 Multivariate Feature Correlation & Interactions")

        if filtered_df.empty:
            st.warning("⚠️ No data available.")
        else:
            st.plotly_chart(
                plot_correlation_heatmap_plotly(filtered_df),
                use_container_width=True,
            )

            st.markdown("### 📊 Price vs Sales Volume Scatter Matrix")
            fig_scat = px.scatter(
                filtered_df,
                x="UnitPrice",
                y="Sales",
                color="Category",
                size="Quantity",
                hover_data=["Product", "Profit", "SatisfactionScore"],
                title="<b>Unit Price vs Order Sales by Category</b>",
                template="plotly_white",
            )
            st.plotly_chart(fig_scat, use_container_width=True)

    elif navigation_page == "🧹 Data Pipeline Audit Log":
        st.subheader("🧹 Production Pipeline Ingestion & Data Quality Audit")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### Raw Ingested Profile")
            st.write(f"- **Total Rows:** {len(raw_df):,}")
            st.write(f"- **Total Missing Entries:** {int(raw_df.isnull().sum().sum()):,}")
            st.write(f"- **Identified Duplicate Rows:** {int(raw_df.duplicated().sum()):,}")

        with col_p2:
            st.markdown("#### Cleaned Dataset Profile")
            st.write(f"- **Total Clean Rows:** {len(df):,}")
            st.write(f"- **Remaining Missing Entries:** {int(df.isnull().sum().sum()):,}")
            st.write(f"- **Duplicate Rows Remaining:** {int(df.duplicated().sum()):,}")

        if validation_issues:
            st.warning("#### Data Validation Log Messages")
            for issue in validation_issues:
                st.write(f"- ⚠️ {issue}")
        else:
            st.success("✅ Dataset schema and non-negativity checks passed completely.")

    elif navigation_page == "⚙️ Dynamic 'What-If' Simulator":
        st.subheader("⚙️ Dynamic 'What-If' Business & Predictive Simulator")

        sim_tab1, sim_tab2 = st.tabs(
            ["🎯 Single Order Predictive Sandbox", "📊 Macro Strategy Scenario Simulator"]
        )

        with sim_tab1:
            if reg_model is None or clf_model is None or not encoders:
                st.error(
                    "❌ ML models or encoders not loaded. Run `python run_pipeline.py` to serialize model files."
                )
            else:
                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    st.markdown("### ⚙️ Order Simulation Inputs")
                    avail_cats = list(encoders["Category"].classes_)
                    avail_regs = list(encoders["Region"].classes_)

                    sim_cat = st.selectbox("Category", avail_cats, index=0)
                    sim_reg = st.selectbox("Region", avail_regs, index=0)

                    sim_price = st.slider("Unit Price ($)", 5.0, float(df["UnitPrice"].max()), float(df["UnitPrice"].median()), step=0.5)
                    sim_qty = st.slider("Quantity Ordered", 1, 20, 3, step=1)
                    sim_discount = st.selectbox("Discount Percentage (%)", [0, 5, 10, 15, 20, 25, 30], index=0)
                    sim_age = st.slider("Customer Age", 18, 75, 35, step=1)
                    sim_sat = st.slider("Satisfaction Score", 1, 5, 4, step=1)
                    sim_month = st.slider("Transaction Month", 1, 12, 6, step=1)
                    sim_quarter = (sim_month - 1) // 3 + 1

                with col_s2:
                    st.markdown("### 🎯 Predictive Inference Results")

                    input_dict = {
                        "Category": sim_cat,
                        "Region": sim_reg,
                        "UnitPrice": sim_price,
                        "Quantity": sim_qty,
                        "DiscountPct": sim_discount,
                        "CustomerAge": sim_age,
                        "SatisfactionScore": sim_sat,
                        "Month": sim_month,
                        "Quarter": sim_quarter,
                    }

                    res = simulate_order_inference(reg_model, clf_model, encoders, input_dict)

                    if res["error"]:
                        st.error(f"Inference error: {res['error']}")
                    else:
                        st.markdown(
                            f"""
                            <div style="border:2px solid #2563eb; border-radius:10px; padding:20px; background-color:#f0f7ff; margin-bottom:15px;">
                                <h4 style="margin:0; color:#1e40af;">Predicted Order Sales</h4>
                                <div style="font-size:2.4rem; font-weight:800; color:#2563eb;">
                                    ${res['predicted_sales']:,.2f}
                                </div>
                                <span style="font-size:0.85rem; color:#64748b;">Random Forest Regression Prediction</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        prob_pct = res["high_value_prob"] * 100.0
                        if res["is_high_value"] == 1:
                            st.markdown(
                                f"""
                                <div style="border:2px solid #10b981; border-radius:10px; padding:20px; background-color:#f0fdf4;">
                                    <h4 style="margin:0; color:#166534;">High-Value Status: YES</h4>
                                    <div style="font-size:2.4rem; font-weight:800; color:#10b981;">
                                        {prob_pct:.1f}% Probability
                                    </div>
                                    <span style="font-size:0.85rem; color:#64748b;">Order exceeds historical median value threshold</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="border:2px solid #f59e0b; border-radius:10px; padding:20px; background-color:#fffbe0;">
                                    <h4 style="margin:0; color:#b45309;">High-Value Status: NO</h4>
                                    <div style="font-size:2.4rem; font-weight:800; color:#f59e0b;">
                                        {prob_pct:.1f}% Probability
                                    </div>
                                    <span style="font-size:0.85rem; color:#64748b;">Order falls below historical median value threshold</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        with sim_tab2:
            st.markdown("### 📊 Portfolio Macro Strategy Simulator")
            st.caption("Simulate market pricing adjustments, volume demand shifts, or supply inflation across categories.")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                price_adj = st.slider("Price Adjustment (%)", -20.0, 30.0, 0.0, step=1.0)
            with col_m2:
                demand_adj = st.slider("Demand Volume Shift (%)", -30.0, 30.0, 0.0, step=1.0)
            with col_m3:
                cost_adj = st.slider("Cost Inflation Shift (%)", -10.0, 30.0, 0.0, step=1.0)

            macro_res = simulate_macro_scenario(
                filtered_df,
                price_adj_pct=price_adj,
                demand_adj_pct=demand_adj,
                cost_adj_pct=cost_adj,
            )

            col_mc1, col_mc2, col_mc3 = st.columns(3)
            with col_mc1:
                st.metric("Simulated Total Sales", f"${macro_res['simulated_sales']:,.2f}", delta=f"{macro_res['sales_delta_pct']:+.2f}%")
            with col_mc2:
                st.metric("Simulated Total Profit", f"${macro_res['simulated_profit']:,.2f}", delta=f"{macro_res['profit_delta_pct']:+.2f}%")
            with col_mc3:
                st.metric("Net Sales Variance", f"${macro_res['sales_delta']:,.2f}")

            if not macro_res["category_comparison"].empty:
                fig_cat_sim = px.bar(
                    macro_res["category_comparison"],
                    x="Category",
                    y=["BaselineSales", "SimulatedSales"],
                    barmode="group",
                    title="<b>Category Revenue Shift Comparison</b>",
                    template="plotly_white",
                )
                st.plotly_chart(fig_cat_sim, use_container_width=True)


if __name__ == "__main__":
    main()
