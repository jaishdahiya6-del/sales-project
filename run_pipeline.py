"""
run_pipeline.py
----------------
Master script that runs the complete data analytics pipeline end to end:

  1. Load & clean raw data
  2. Run exploratory analysis and save summary statistics
  3. Generate all EDA visualizations
  4. Train regression & classification models
  5. Generate model-related visualizations
  6. Save prediction results

Run with:  python run_pipeline.py
"""

from src.preprocessing import run_pipeline as preprocess
from src.analysis import summary_statistics, category_performance, regional_performance
from src.visualization import (
    generate_all_eda_charts,
    plot_feature_importance,
    plot_prediction_vs_actual,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
)
from src.model import (
    train_regression_model,
    train_classification_model,
    save_prediction_results,
)
from src.utils import get_logger

logger = get_logger("run_pipeline")


def main():
    logger.info("=== STEP 1: Preprocessing ===")
    df = preprocess()

    logger.info("=== STEP 2: Exploratory Analysis ===")
    summary_statistics(df)
    print("\nCategory performance:\n", category_performance(df))
    print("\nRegional performance:\n", regional_performance(df))

    logger.info("=== STEP 3: EDA Visualizations ===")
    generate_all_eda_charts(df)

    logger.info("=== STEP 4: Model Training ===")
    reg_result = train_regression_model(df)
    clf_result = train_classification_model(df)

    logger.info("=== STEP 5: Model Visualizations ===")
    plot_feature_importance(reg_result["model"], reg_result["feature_cols"])
    plot_prediction_vs_actual(reg_result["y_test"], reg_result["y_pred"])
    plot_residuals(reg_result["y_test"], reg_result["y_pred"])
    plot_confusion_matrix(clf_result["y_test"], clf_result["y_pred"])
    plot_roc_curve(clf_result["y_test"], clf_result["y_proba"])

    logger.info("=== STEP 6: Save Predictions ===")
    save_prediction_results(df, reg_result, clf_result)

    print("\n=== Regression Metrics ===")
    print(reg_result["metrics"])
    print("\n=== Classification Metrics ===")
    print(clf_result["metrics"])

    logger.info("Pipeline completed successfully.")

    return {
        "df": df,
        "reg_result": reg_result,
        "clf_result": clf_result,
    }


if __name__ == "__main__":
    main()
