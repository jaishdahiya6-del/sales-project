"""
tests/test_dashboard.py
-----------------------
Unit tests for the analytical, forecasting, anomaly detection, Pareto 80/20,
and simulation helper functions in dashboard/app.py.

Author: Advanced Retail Analytics Team
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from dashboard.app import (
    validate_dataset,
    compute_kpis,
    compute_sales_forecast,
    compute_seasonality_analysis,
    compute_pareto_analysis,
    detect_transaction_anomalies,
    simulate_macro_scenario,
    simulate_order_inference,
)


@pytest.fixture
def mock_retail_df():
    """Generate a mock clean retail DataFrame with known properties for testing."""
    dates = pd.date_range(start="2024-01-01", periods=12, freq="MS")
    rows = []
    for i, dt in enumerate(dates):
        rows.append({
            "OrderID": f"ORD_{i+100}",
            "Date": dt,
            "Category": "Electronics" if i % 2 == 0 else "Clothing",
            "Product": f"Product_{i % 3}",
            "Region": "North" if i % 2 == 0 else "South",
            "UnitPrice": 100.0 + i * 10,
            "Quantity": 2 + (i % 3),
            "DiscountPct": 5.0,
            "Sales": (100.0 + i * 10) * (2 + (i % 3)) * 0.95,
            "Cost": 50.0 + i * 5,
            "Profit": ((100.0 + i * 10) * (2 + (i % 3)) * 0.95) - (50.0 + i * 5),
            "CustomerAge": 30 + i,
            "SatisfactionScore": 4.0,
            "ProfitMargin": 25.0,
            "AvgOrderValue": 100.0,
        })
    return pd.DataFrame(rows)


def test_validate_dataset(mock_retail_df):
    """Test schema and non-negativity validation logic."""
    is_valid, issues = validate_dataset(mock_retail_df)
    assert is_valid is True
    assert len(issues) == 0

    # Test invalid dataframe missing columns
    bad_df = pd.DataFrame({"WrongCol": [1, 2, 3]})
    is_valid_bad, bad_issues = validate_dataset(bad_df)
    assert is_valid_bad is False
    assert any("Missing required columns" in issue for issue in bad_issues)


def test_compute_kpis(mock_retail_df):
    """Test KPI calculation accuracy and edge-case empty DataFrame handling."""
    kpis = compute_kpis(mock_retail_df)
    assert kpis["total_orders"] == 12
    assert kpis["total_sales"] > 0
    assert kpis["total_profit"] > 0
    assert kpis["avg_satisfaction"] == 4.0

    # Test empty dataframe
    empty_kpis = compute_kpis(pd.DataFrame())
    assert empty_kpis["total_orders"] == 0
    assert empty_kpis["total_sales"] == 0.0


def test_compute_sales_forecast(mock_retail_df):
    """Test linear-trend forecasting and prediction bounds generation."""
    forecast_df = compute_sales_forecast(mock_retail_df, forecast_months=6, confidence_level=0.95)

    assert not forecast_df.empty
    assert "ForecastSales" in forecast_df.columns
    assert "LowerBound" in forecast_df.columns
    assert "UpperBound" in forecast_df.columns

    future_rows = forecast_df[forecast_df["IsForecast"]]
    assert len(future_rows) == 6


def test_compute_seasonality_analysis(mock_retail_df):
    """Test monthly seasonal indexing and month-over-month growth computation."""
    season_res = compute_seasonality_analysis(mock_retail_df)

    assert "monthly_index" in season_res
    assert "quarterly_summary" in season_res
    assert "mom_growth" in season_res

    m_idx = season_res["monthly_index"]
    assert "SeasonalIndex" in m_idx.columns
    assert len(m_idx) > 0


def test_compute_pareto_analysis(mock_retail_df):
    """Test Pareto 80/20 distribution metrics and cumulative share calculation."""
    pareto_df = compute_pareto_analysis(mock_retail_df, entity_col="Product")

    assert not pareto_df.empty
    assert "CumulativeSharePct" in pareto_df.columns
    assert "InTop80Percent" in pareto_df.columns
    # Check that rows are sorted descending by TotalSales
    assert pareto_df["TotalSales"].is_monotonic_decreasing


def test_detect_transaction_anomalies(mock_retail_df):
    """Test Isolation Forest and IQR anomaly detection flagging."""
    anom_df = detect_transaction_anomalies(mock_retail_df, contamination=0.1, method="isolation_forest")

    assert "IsAnomaly" in anom_df.columns
    assert "AnomalyScore" in anom_df.columns
    assert len(anom_df) == len(mock_retail_df)


def test_simulate_macro_scenario(mock_retail_df):
    """Test portfolio pricing, volume, and cost macro scenario simulations."""
    res = simulate_macro_scenario(
        mock_retail_df,
        price_adj_pct=10.0,
        demand_adj_pct=-5.0,
        cost_adj_pct=0.0,
    )

    assert "simulated_sales" in res
    assert "sales_delta" in res
    assert "category_comparison" in res
    assert isinstance(res["category_comparison"], pd.DataFrame)


def test_simulate_order_inference():
    """Test real-time ML order inference with mock regressor and classifier."""
    X_mock = pd.DataFrame({
        "UnitPrice": [100.0, 150.0],
        "Quantity": [2, 3],
        "DiscountPct": [0.0, 5.0],
        "CustomerAge": [30.0, 40.0],
        "SatisfactionScore": [4.0, 5.0],
        "Month": [1, 6],
        "Quarter": [1, 2],
        "Category_enc": [0, 1],
        "Region_enc": [0, 1],
    })
    y_reg = np.array([200.0, 427.5])
    y_clf = np.array([0, 1])

    reg_model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_mock, y_reg)
    clf_model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_mock, y_clf)

    encoders = {
        "Category": LabelEncoder().fit(["Electronics", "Clothing"]),
        "Region": LabelEncoder().fit(["North", "South"]),
    }

    input_params = {
        "Category": "Electronics",
        "Region": "North",
        "UnitPrice": 100.0,
        "Quantity": 2,
        "DiscountPct": 0.0,
        "CustomerAge": 30.0,
        "SatisfactionScore": 4.0,
        "Month": 1,
        "Quarter": 1,
    }

    res = simulate_order_inference(reg_model, clf_model, encoders, input_params)

    assert res["error"] is None
    assert "predicted_sales" in res
    assert "is_high_value" in res
    assert "high_value_prob" in res
    assert isinstance(res["predicted_sales"], float)
