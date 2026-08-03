# Retail Sales Analytics Report

**Project:** Retail Sales Data Analytics & Prediction
**Prepared by:** Data Analytics Team
**Date:** August 2026

---

## 1. Project Overview

This project delivers an end-to-end data analytics pipeline for a
simulated multi-category retail business. It covers data cleaning,
exploratory data analysis, statistical analysis, correlation analysis,
predictive modeling, and an interactive dashboard, packaged as a
GitHub-ready repository.

## 2. Dataset Description

The dataset (`data/raw/sales_data.csv`) contains **2,015 order-level
records** spanning **January 2022 – December 2024**, across **6 product
categories** (Electronics, Clothing, Home & Kitchen, Sports, Beauty,
Toys) and **5 regions** (North, South, East, West, Central).

| Column | Description |
|---|---|
| OrderID | Unique order identifier |
| Date | Order date |
| Category / Product | Product taxonomy |
| Region | Sales region |
| UnitPrice, Quantity, DiscountPct | Transaction attributes |
| Sales, Cost, Profit | Financial outcomes |
| CustomerAge, SatisfactionScore | Customer attributes |

The raw data intentionally contains missing values, duplicate rows, and
outliers to demonstrate a realistic cleaning workflow.

## 3. Objectives

1. Clean and prepare the raw dataset for analysis.
2. Understand sales, profit, and customer patterns across categories,
   regions, and time.
3. Quantify relationships between key numeric variables.
4. Build predictive models for order value and high-value order
   classification.
5. Present findings through visualizations, a written report, and an
   interactive dashboard.

## 4. Methodology

The pipeline (`run_pipeline.py`) executes six stages:

1. **Preprocessing** (`src/preprocessing.py`) — load, deduplicate,
   impute missing values, cap outliers (IQR method), and engineer
   features (Year, Month, Quarter, ProfitMargin, AvgOrderValue,
   HighValueOrder).
2. **Exploratory Analysis** (`src/analysis.py`) — descriptive
   statistics, category/regional aggregation, monthly trend.
3. **Visualization** (`src/visualization.py`) — 12 chart types saved to
   `graphs/`.
4. **Modeling** (`src/model.py`) — a Random Forest regressor (predicts
   `Sales`) and a Random Forest classifier (predicts `HighValueOrder`).
5. **Prediction Export** — test-set predictions saved to
   `outputs/prediction_results.csv`.
6. **Dashboard** (`dashboard/app.py`) — a Streamlit app for interactive
   exploration of the cleaned data and model outputs.

## 5. Data Cleaning

| Step | Result |
|---|---|
| Duplicate rows removed | 15 exact duplicates dropped |
| Missing values imputed | Numeric columns via median, categorical via mode |
| Outliers capped (IQR method) | 915 extreme values capped across Sales, Quantity, Profit, UnitPrice |
| Features engineered | Year, Month, Quarter, DayOfWeek, ProfitMargin, AvgOrderValue, HighValueOrder |

The cleaned dataset (2,000 rows after deduplication) is saved to
`data/processed/sales_data_clean.csv` and mirrored in
`outputs/cleaned_dataset.csv`.

## 6. Exploratory Data Analysis

- **Total Sales:** $677,423
- **Total Profit:** $218,695
- **Average Order Value:** $338.71
- **Average Profit Margin:** 32.4%

**Sales by category:**

| Category | Total Sales |
|---|---|
| Electronics | $283,316 |
| Home & Kitchen | $121,492 |
| Sports | $92,190 |
| Clothing | $69,312 |
| Toys | $61,749 |
| Beauty | $49,364 |

Electronics contributes over 40% of total revenue despite representing
only one of six categories, driven by a higher average unit price.

**Sales by region** are relatively balanced, ranging from $124K (North)
to $153K (West), suggesting no single region is significantly
underperforming.

## 7. Visualizations

All charts below are saved as high-resolution PNGs in `graphs/`:

- `bar_chart.png` — Total sales by category
- `line_chart.png` — Monthly sales & profit trend
- `pie_chart.png` — Regional sales share
- `histogram.png` — Distribution of order sales values
- `scatter_plot.png` — Unit price vs. sales by category
- `correlation_heatmap.png` — Correlation matrix of numeric features
- `boxplot.png` — Sales distribution by category
- `distribution_plot.png` — Profit margin density
- `feature_importance.png` — Random Forest regression feature importance
- `prediction_chart.png` — Predicted vs. actual sales
- `confusion_matrix.png` — High-value order classifier confusion matrix
- `roc_curve.png` — High-value order classifier ROC curve
- `residual_plot.png` — Regression residuals

## 8. Statistical Findings

- Sales values are right-skewed (skew ≈ 1.09), consistent with a
  long tail of large orders.
- `UnitPrice` and `Sales` show a positive correlation, as expected;
  `Quantity` also contributes meaningfully to total order value.
- `SatisfactionScore` shows negligible correlation with `Sales` or
  `Profit`, suggesting satisfaction is not strongly value-driven in
  this simulated dataset — a signal worth validating against real
  customer data.
- Profit margin is fairly stable across categories, averaging around
  32%, with a moderately tight spread (std ≈ 7 points).

## 9. Machine Learning

### 9.1 Regression — Predicting Order Sales

A `RandomForestRegressor` (200 trees, max depth 8) was trained on
transaction and customer attributes to predict order-level `Sales`.

| Metric | Value |
|---|---|
| MAE | 18.73 |
| RMSE | 44.09 |
| R² | 0.971 |

### 9.2 Classification — High-Value Order Detection

A `RandomForestClassifier` was trained to flag orders above the median
sales value (`HighValueOrder`).

| Metric | Value |
|---|---|
| Accuracy | 0.980 |
| Precision | 0.980 |
| Recall | 0.980 |
| F1 Score | 0.980 |
| ROC AUC | 0.998 |

## 10. Model Evaluation

Both models perform strongly, which is expected given `Sales` is
partly deterministic from `UnitPrice`, `Quantity`, and `DiscountPct` in
this simulated dataset. The residual plot (`residual_plot.png`) shows
residuals centered near zero with no strong pattern, indicating the
regression model is not systematically biased. The ROC curve
(`roc_curve.png`) confirms strong class separability for the
high-value classifier.

**Caveat:** On real-world data, expect noisier relationships and lower
performance; these results should be treated as a template
methodology rather than literal benchmarks.

## 11. Business Insights

1. **Electronics is the primary revenue driver** and merits continued
   inventory and marketing investment.
2. **Regional performance is balanced**, so growth efforts may be
   better spent on cross-region initiatives (e.g., loyalty programs)
   rather than fixing an underperforming region.
3. **Discounting is used sparingly** (mean ≈ 3.3%), and could be tested
   more aggressively in lower-performing categories (Beauty, Toys) to
   assess price elasticity.
4. **Profit margin is consistent across categories**, indicating
   pricing and cost structures are currently well-balanced.

## 12. Recommendations

- Expand Electronics assortment and consider premium SKUs given
  strong average order value.
- Run targeted A/B discount tests in Beauty and Toys to grow their
  smaller revenue share.
- Use the `HighValueOrder` classifier to prioritize high-value
  customers for retention campaigns.
- Monitor `SatisfactionScore` alongside NPS or churn data in
  production, since it showed limited correlation with sales here.

## 13. Future Improvements

- Incorporate real transactional data and seasonality/holiday effects.
- Add time-series specific models (e.g., Prophet, ARIMA) for
  forecasting future sales rather than point-in-time prediction.
- Add customer-level (not just order-level) features such as lifetime
  value and repeat-purchase rate.
- Deploy the dashboard with authentication and a scheduled data
  refresh job for production use.

## 14. Conclusion

This project demonstrates a complete, reproducible retail analytics
workflow — from raw, messy data to cleaned datasets, rich
visualizations, validated predictive models, and an interactive
dashboard. The modular `src/` package and `run_pipeline.py` entry
point make it straightforward to swap in a real dataset and reproduce
the full analysis.
