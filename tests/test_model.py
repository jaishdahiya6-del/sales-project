"""
tests/test_model.py
-------------------
Unit tests for the machine learning training, feature encoding, and model persistence in src/model.py.

Author: Data Analytics Team
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.model import (
    prepare_features,
    train_regression_model,
    train_classification_model,
    save_prediction_results,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
)


@pytest.fixture
def sample_sales_data():
    """Generate a clean, tiny synthetic DataFrame representing prepared retail data."""
    np.random.seed(42)
    rows = []
    categories = ["Electronics", "Clothing"]
    regions = ["North", "South"]

    # 30 samples to satisfy train_test_split minimum sample size
    for i in range(30):
        rows.append({
            "OrderID": f"ORD{1000 + i}",
            "Date": pd.Timestamp("2024-01-15") + pd.to_timedelta(i, unit="D"),
            "Category": np.random.choice(categories),
            "Product": "Widget",
            "Region": np.random.choice(regions),
            "UnitPrice": 100.0,
            "Quantity": 2,
            "DiscountPct": 0.0,
            "Sales": 200.0,
            "Cost": 120.0,
            "Profit": 80.0,
            "CustomerAge": 30.0,
            "SatisfactionScore": 4.0,
            "Year": 2024,
            "Month": 1,
            "MonthName": "January",
            "Quarter": 1,
            "DayOfWeek": "Monday",
            "ProfitMargin": 40.0,
            "AvgOrderValue": 100.0,
            "HighValueOrder": 1 if i % 2 == 0 else 0
        })
    return pd.DataFrame(rows)


def test_prepare_features(sample_sales_data):
    """Verify that features are split and categorical columns are label encoded correctly."""
    X, feature_cols, encoders = prepare_features(sample_sales_data)

    # Check that output features match expectation
    expected_cols = FEATURE_COLUMNS + [c + "_enc" for c in CATEGORICAL_COLUMNS]
    assert list(X.columns) == expected_cols
    assert len(X) == len(sample_sales_data)

    # Check encoders are populated
    assert "Category" in encoders
    assert "Region" in encoders
    assert isinstance(encoders["Category"].classes_, np.ndarray)


def test_train_regression_model(sample_sales_data):
    """Verify that the regressor model trains, generates metrics, and serializes assets."""
    reg_result = train_regression_model(sample_sales_data)

    assert "model" in reg_result
    assert "metrics" in reg_result
    assert "MAE" in reg_result["metrics"]
    assert "R2" in reg_result["metrics"]

    # Check that model file is created
    from src.utils import MODELS_DIR
    assert os.path.exists(os.path.join(MODELS_DIR, "regression_model.joblib"))
    assert os.path.exists(os.path.join(MODELS_DIR, "encoders.joblib"))


def test_train_classification_model(sample_sales_data):
    """Verify that the classifier trains and outputs standard classification metrics."""
    # Obtain encoders first
    _, _, encoders = prepare_features(sample_sales_data)

    clf_result = train_classification_model(sample_sales_data, encoders=encoders)

    assert "model" in clf_result
    assert "metrics" in clf_result
    assert "Accuracy" in clf_result["metrics"]
    assert "ROC_AUC" in clf_result["metrics"]

    # Check that model file is created
    from src.utils import MODELS_DIR
    assert os.path.exists(os.path.join(MODELS_DIR, "classification_model.joblib"))


def test_save_prediction_results(sample_sales_data):
    """Verify that regression and classification test results are combined and saved correctly."""
    reg_result = train_regression_model(sample_sales_data)
    clf_result = train_classification_model(sample_sales_data, encoders=reg_result["encoders"])

    combined = save_prediction_results(sample_sales_data, reg_result, clf_result)

    # Check combined structure
    assert "ActualSales" in combined.columns
    assert "PredictedSales" in combined.columns
    assert "ActualHighValue" in combined.columns
    assert "HighValueProbability" in combined.columns

    from src.utils import OUTPUTS_DIR
    # Check outputs files are written
    assert os.path.exists(os.path.join(OUTPUTS_DIR, "prediction_results.csv"))
    assert os.path.exists(os.path.join(OUTPUTS_DIR, "model_metrics.json"))
