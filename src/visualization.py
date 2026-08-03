"""
visualization.py
-----------------
Generates all professional-quality charts required for the analysis
report and dashboard, and saves them into the graphs/ directory.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.utils import GRAPHS_DIR, get_logger, ensure_dirs
from src.analysis import (
    category_performance,
    regional_performance,
    monthly_trend,
    correlation_matrix,
)

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")
PALETTE = "viridis"
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def _save(fig, filename):
    ensure_dirs(GRAPHS_DIR)
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart -> {path}")


def plot_bar_chart(df: pd.DataFrame):
    """Total sales by category - bar chart."""
    perf = category_performance(df)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(x=perf.index, y=perf["TotalSales"], hue=perf.index,
                palette=PALETTE, ax=ax, legend=False)
    ax.set_title("Total Sales by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales ($)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "bar_chart.png")


def plot_line_chart(df: pd.DataFrame):
    """Monthly sales and profit trend - line chart."""
    trend = monthly_trend(df)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(trend["Date"], trend["TotalSales"], marker="o", label="Sales", color="#2563eb")
    ax.plot(trend["Date"], trend["TotalProfit"], marker="o", label="Profit", color="#16a34a")
    ax.set_title("Monthly Sales & Profit Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount ($)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, "line_chart.png")


def plot_pie_chart(df: pd.DataFrame):
    """Sales share by region - pie chart."""
    perf = regional_performance(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette(PALETTE, len(perf))
    ax.pie(
        perf["TotalSales"], labels=perf.index, autopct="%1.1f%%",
        startangle=90, colors=colors, wedgeprops={"edgecolor": "white"}
    )
    ax.set_title("Sales Share by Region")
    _save(fig, "pie_chart.png")


def plot_histogram(df: pd.DataFrame):
    """Distribution of order sales values - histogram."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(df["Sales"], bins=40, kde=False, color="#4338ca", ax=ax)
    ax.set_title("Distribution of Order Sales Values")
    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Frequency")
    _save(fig, "histogram.png")


def plot_scatter(df: pd.DataFrame):
    """Unit price vs quantity, colored by category - scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(
        data=df, x="UnitPrice", y="Sales", hue="Category",
        alpha=0.6, palette=PALETTE, ax=ax
    )
    ax.set_title("Unit Price vs Total Sales by Category")
    ax.set_xlabel("Unit Price ($)")
    ax.set_ylabel("Sales ($)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, "scatter_plot.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix of numeric features - heatmap."""
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5)
    ax.set_title("Correlation Heatmap of Numeric Features")
    _save(fig, "correlation_heatmap.png")


def plot_boxplot(df: pd.DataFrame):
    """Sales distribution by category - box plot."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=df, x="Category", y="Sales", hue="Category",
                palette=PALETTE, ax=ax, legend=False)
    ax.set_title("Sales Distribution by Category")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "boxplot.png")


def plot_distribution(df: pd.DataFrame):
    """KDE distribution of profit margin - distribution plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.kdeplot(df["ProfitMargin"], fill=True, color="#c026d3", ax=ax)
    ax.set_title("Distribution of Profit Margin (%)")
    ax.set_xlabel("Profit Margin (%)")
    _save(fig, "distribution_plot.png")


def plot_feature_importance(model, feature_names):
    """Feature importance from the trained ML model."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        x=importances[order], y=np.array(feature_names)[order],
        hue=np.array(feature_names)[order], palette=PALETTE, ax=ax, legend=False
    )
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    _save(fig, "feature_importance.png")


def plot_prediction_vs_actual(y_true, y_pred):
    """Predicted vs actual values - scatter with reference line."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#0891b2")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="red", label="Perfect Prediction")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Predicted vs Actual Sales")
    ax.legend()
    _save(fig, "prediction_chart.png")


def plot_confusion_matrix(y_true, y_pred, labels=("Low", "High")):
    """Confusion matrix for the classification task."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix - High Value Order Classifier")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save(fig, "confusion_matrix.png")


def plot_roc_curve(y_true, y_proba):
    """ROC curve for the classification task."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - High Value Order Classifier")
    ax.legend(loc="lower right")
    _save(fig, "roc_curve.png")


def plot_residuals(y_true, y_pred):
    """Residual plot for the regression task."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="#ea580c")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot - Sales Prediction Model")
    _save(fig, "residual_plot.png")


def generate_all_eda_charts(df: pd.DataFrame):
    """Generate every EDA-level chart (does not require a trained model)."""
    plot_bar_chart(df)
    plot_line_chart(df)
    plot_pie_chart(df)
    plot_histogram(df)
    plot_scatter(df)
    plot_correlation_heatmap(df)
    plot_boxplot(df)
    plot_distribution(df)
    logger.info("Generated all EDA charts")


if __name__ == "__main__":
    from src.preprocessing import run_pipeline

    clean_df = run_pipeline()
    generate_all_eda_charts(clean_df)
