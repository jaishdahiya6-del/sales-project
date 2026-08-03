"""
src/model.py
------------
Trains, evaluates, and serializes machine learning models on the retail sales dataset.

Two models are built:
  1. Regression: Predicts transaction sales value (RandomForestRegressor).
  2. Classification: Predicts whether an order is 'High Value' (RandomForestClassifier).

Features used:
  - Numeric: UnitPrice, Quantity, DiscountPct, CustomerAge, SatisfactionScore, Month, Quarter
  - Categorical: Category, Region (encoded via LabelEncoder)

Models, encoders, and evaluation metadata are serialized and persisted
for interactive prediction/simulation inside the Streamlit dashboard.

Author: Data Analytics Team
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional

import joblib
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

from src.utils import OUTPUTS_DIR, MODELS_DIR, get_logger, ensure_dirs

logger = get_logger(__name__)

# Feature definitions
FEATURE_COLUMNS: List[str] = [
    "UnitPrice", "Quantity", "DiscountPct", "CustomerAge",
    "SatisfactionScore", "Month", "Quarter",
]
CATEGORICAL_COLUMNS: List[str] = ["Category", "Region"]


def prepare_features(df: pd.DataFrame, encoders: Optional[Dict[str, LabelEncoder]] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]:
    """
    Encode categorical features and assemble the final feature matrix.

    If pre-fitted encoders are provided, they are used to transform the data.
    Otherwise, new LabelEncoder instances are fitted and returned.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    encoders : Optional[Dict[str, LabelEncoder]], optional
        Dictionary mapping column names to pre-fitted LabelEncoder instances, by default None.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]
        - X: Feature matrix (DataFrame) containing numeric and encoded categorical features.
        - feature_cols: List of column names in the feature matrix.
        - encoders: Dictionary of fitted LabelEncoder instances.
    """
    data = df.copy()
    out_encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in data.columns:
            raise KeyError(f"Required categorical column '{col}' is missing from the input data.")

        # Ensure values are cast to string for encoding safety
        col_series = data[col].astype(str)

        if encoders and col in encoders:
            le = encoders[col]
            # Handle unseen categories by mapping them to the first class or handling error safely
            # Since LabelEncoder doesn't support handle_unknown directly, we map unseen values to the mode class
            known_classes = set(le.classes_)
            col_series_safe = col_series.apply(lambda val: val if val in known_classes else le.classes_[0])
            data[col + "_enc"] = le.transform(col_series_safe)
            out_encoders[col] = le
        else:
            le = LabelEncoder()
            data[col + "_enc"] = le.fit_transform(col_series)
            out_encoders[col] = le

    feature_cols = FEATURE_COLUMNS + [c + "_enc" for c in CATEGORICAL_COLUMNS]

    # Validate that numeric features exist
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise KeyError(f"Required numeric feature column '{col}' is missing from the input data.")

    X = data[feature_cols]
    return X, feature_cols, out_encoders


def train_regression_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train a RandomForestRegressor to predict order Sales.

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned and engineered DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing trained model, features list, test split matrices, and evaluation metrics.
    """
    logger.info("Training Sales prediction regression model...")
    X, feature_cols, encoders = prepare_features(df)
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
    }
    logger.info(f"Regression training complete. Metrics: {metrics}")

    # Serialize regression model and encoders
    ensure_dirs(MODELS_DIR)
    reg_model_path = os.path.join(MODELS_DIR, "regression_model.joblib")
    encoders_path = os.path.join(MODELS_DIR, "encoders.joblib")

    joblib.dump(model, reg_model_path)
    joblib.dump(encoders, encoders_path)
    logger.info(f"Serialized regression model -> {reg_model_path}")
    logger.info(f"Serialized encoders -> {encoders_path}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "encoders": encoders,
    }


def train_classification_model(df: pd.DataFrame, encoders: Optional[Dict[str, LabelEncoder]] = None) -> Dict[str, Any]:
    """
    Train a RandomForestClassifier to predict whether an order is 'HighValueOrder'.

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned and engineered DataFrame.
    encoders : Optional[Dict[str, LabelEncoder]], optional
        Pre-fitted LabelEncoder mapping, by default None. If None, fits brand new encoders.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing trained classifier model, features list, test splits, and classification metrics.
    """
    logger.info("Training HighValueOrder prediction classification model...")
    X, feature_cols, out_encoders = prepare_features(df, encoders=encoders)
    y = df["HighValueOrder"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "ROC_AUC": round(roc_auc, 3),
    }
    logger.info(f"Classification training complete. Metrics: {metrics}")

    # Serialize classification model
    ensure_dirs(MODELS_DIR)
    clf_model_path = os.path.join(MODELS_DIR, "classification_model.joblib")
    joblib.dump(model, clf_model_path)
    logger.info(f"Serialized classification model -> {clf_model_path}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def save_prediction_results(df: pd.DataFrame, reg_result: Dict[str, Any], clf_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Combine regression and classification test predictions and output a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    reg_result : Dict[str, Any]
        Regression results.
    clf_result : Dict[str, Any]
        Classification results.

    Returns
    -------
    pd.DataFrame
        Combined test evaluation DataFrame.
    """
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
    logger.info(f"Saved prediction results CSV -> {out_path}")

    # Also save model_metrics.json for monitoring/auditing
    metrics_json_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    combined_metrics = {
        "regression": reg_result["metrics"],
        "classification": clf_result["metrics"],
    }
    try:
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_metrics, f, indent=4)
        logger.info(f"Saved pipeline execution performance metrics JSON -> {metrics_json_path}")
    except Exception as exc:
        logger.error(f"Failed to write model metrics JSON: {exc}")

    return combined


if __name__ == "__main__":
    from src.preprocessing import run_pipeline

    clean_df = run_pipeline()
    reg = train_regression_model(clean_df)
    clf = train_classification_model(clean_df, encoders=reg["encoders"])
    save_prediction_results(clean_df, reg, clf)
    print("Regression metrics:", reg["metrics"])
    print("Classification metrics:", clf["metrics"])
