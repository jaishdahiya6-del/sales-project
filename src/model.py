"""
model.py
--------
Trains and evaluates two machine learning models on the retail sales
dataset:

  1. Regression  - predicts order Sales value (RandomForestRegressor)
  2. Classification - predicts whether an order is "High Value"
     (RandomForestClassifier)

Both models are used to generate the visualizations required by the
project (feature importance, prediction vs actual, confusion matrix,
ROC curve, residual plot) and to produce outputs/prediction_results.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from src.utils import OUTPUTS_DIR, get_logger, ensure_dirs

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "UnitPrice", "Quantity", "DiscountPct", "CustomerAge",
    "SatisfactionScore", "Month", "Quarter",
]
CATEGORICAL_COLUMNS = ["Category", "Region"]


def _prepare_features(df: pd.DataFrame):
    """Encode categorical variables and assemble the feature matrix."""
    data = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        data[col + "_enc"] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    feature_cols = FEATURE_COLUMNS + [c + "_enc" for c in CATEGORICAL_COLUMNS]
    X = data[feature_cols]
    return X, feature_cols, encoders


def train_regression_model(df: pd.DataFrame):
    """Train a RandomForestRegressor to predict order Sales."""
    X, feature_cols, _ = _prepare_features(df)
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "R2": round(r2_score(y_test, y_pred), 3),
    }
    logger.info(f"Regression model metrics: {metrics}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def train_classification_model(df: pd.DataFrame):
    """Train a RandomForestClassifier to predict HighValueOrder."""
    X, feature_cols, _ = _prepare_features(df)
    y = df["HighValueOrder"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1": round(f1_score(y_test, y_pred), 3),
        "ROC_AUC": round(roc_auc_score(y_test, y_proba), 3),
    }
    logger.info(f"Classification model metrics: {metrics}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def save_prediction_results(df: pd.DataFrame, reg_result: dict, clf_result: dict):
    """Combine regression + classification test-set predictions and save to CSV."""
    ensure_dirs(OUTPUTS_DIR)

    reg_out = reg_result["X_test"].copy()
    reg_out["ActualSales"] = reg_result["y_test"].values
    reg_out["PredictedSales"] = reg_result["y_pred"]
    reg_out["SalesPredictionError"] = reg_out["ActualSales"] - reg_out["PredictedSales"]

    clf_out = clf_result["y_test"].reset_index(drop=True).to_frame(name="ActualHighValue")
    clf_out["PredictedHighValue"] = clf_result["y_pred"]
    clf_out["HighValueProbability"] = np.round(clf_result["y_proba"], 3)

    reg_out = reg_out.reset_index(drop=True)
    combined = pd.concat([reg_out, clf_out], axis=1)

    out_path = os.path.join(OUTPUTS_DIR, "prediction_results.csv")
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved prediction results to {out_path}")
    return combined


if __name__ == "__main__":
    from src.preprocessing import run_pipeline

    clean_df = run_pipeline()
    reg = train_regression_model(clean_df)
    clf = train_classification_model(clean_df)
    save_prediction_results(clean_df, reg, clf)
    print("Regression metrics:", reg["metrics"])
    print("Classification metrics:", clf["metrics"])
