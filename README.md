# 📊 Advanced Retail Sales Analytics, Machine Learning & Simulation Hub

An end-to-end, production-ready retail intelligence system that transforms raw, transactional sales data into clean datasets, robust analytics dashboards, and predictive models. Includes a fully automated pipeline (`run_pipeline.py`) with model serialization and an interactive Streamlit application featuring real-time "What-If" order simulation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen?style=for-the-badge&logo=pytest)](https://pytest.org/)

---

## 📖 Project Architecture & Purpose

This enterprise-grade repository simulates a high-performance retail operations environment. We ingest multi-category, multi-region sales transactions containing raw data defects (e.g. missing customer attributes, outliers, duplicates) and execute a structured, highly reproducible processing and modeling pipeline.

```
┌─────────────────┐       ┌──────────────────────┐       ┌───────────────────┐
│  Raw Data (CSV) ├──────►│ Preprocessing Module ├──────►│ Clean Data (CSV)  │
└─────────────────┘       └──────────┬───────────┘       └─────────┬─────────┘
                                     │                             │
                                     ▼                             ▼
┌─────────────────┐       ┌──────────────────────┐       ┌───────────────────┐
│ Streamlit App   │◄──────┤ Serialized Models    │◄──────┤ Model Pipeline    │
│  & Simulation   │       │  & Label Encoders    │       │ (Random Forests)  │
└─────────────────┘       └──────────────────────┘       └───────────────────┘
```

The codebase is strictly typed, modularized, fully verified by automated unit tests (`pytest`), and integrates a modern Streamlit user interface mapping real-time predictions.

---

## ✨ Enterprise-Grade Features

1. **Robust Preprocessing Pipeline (`src/preprocessing.py`)**
   - Implements automated median imputation for numeric properties and mode imputation for categorical fields.
   - Drops exact duplicates and handles extreme outliers using Interquartile Range (IQR) winsorization bounds.
   - Engineers temporal factors (Year, Month, Month Name, Quarter, Day of Week) and financial cohorts (Profit Margin, Average Order Value).

2. **Advanced ML Pipeline (`src/model.py`)**
   - **Order Sales Predictor (Regression):** A `RandomForestRegressor` predicting order-level sales value.
   - **High-Value Classifier (Classification):** A `RandomForestClassifier` predicting whether transaction sales will rank above the historical midpoint.
   - **Serialization & Persistence:** Serializes trained models and respective category encoders using `joblib` into a persistent `models/` directory. Saves execution metadata in `outputs/model_metrics.json`.

3. **Production-Ready Dashboard UI (`dashboard/app.py`)**
   - Multi-dimensional filters (Category, Region, and Date windows).
   - Metrics cards detailing sales velocity, aggregated profits, transaction volumes, and average margins.
   - Interactive, hardware-accelerated Plotly visualizations mapping monthly growth, segment dispersion, and correlation matrices.
   - **Interactive 'What-If' Simulation:** A dedicated predictive sandbox allowing real-time forecasting. Users adjust price, quantity, category, region, age, and satisfaction using sliders to view immediate predicted order value and high-value status probability.
   - Supports live CSV table filtering and single-click dataset exports.

4. **Comprehensive Test Suite (`tests/`)**
   - Full automated unit tests verifying data integrity, imputation rules, capping logic, and model training predictions.

---

## 📂 System Directory Layout

```
.
├── data/
│   ├── raw/                # Original synthetic dataset and raw generator
│   └── processed/          # Imputed and cleaned dataset
├── src/
│   ├── preprocessing.py    # Load, impute, deduplicate, and engineer features
│   ├── analysis.py         # Descriptive stats, aggregations, and trends
│   ├── visualization.py    # Headless Matplotlib and Seaborn plotting
│   ├── model.py            # Model training, evaluation, and joblib serialization
│   └── utils.py            # Logger handlers and directory managers
├── dashboard/
│   ├── app.py              # Feature-rich Streamlit application
│   └── requirements.txt    # Streamlit dependencies
├── models/
│   ├── regression_model.joblib      # Serialized RandomForestRegressor
│   ├── classification_model.joblib  # Serialized RandomForestClassifier
│   └── encoders.joblib              # Serialized Category/Region LabelEncoders
├── outputs/
│   ├── cleaned_dataset.csv          # Pipeline clean target mirror
│   ├── prediction_results.csv       # Model test-set actuals vs predictions
│   ├── summary_statistics.csv       # Transposed descriptive stats
│   └── model_metrics.json           # Serialized evaluation metrics (R², F1, MAE)
├── graphs/                 # High-resolution PNG evaluation plots
├── tests/
│   ├── test_preprocessing.py  # Unit tests for preprocessing & imputation
│   └── test_model.py          # Unit tests for modeling pipelines & persistence
├── run_pipeline.py         # Unified automation entrypoint
├── requirements.txt        # Production dependency pins
├── LICENSE                 # MIT License
└── README.md               # Extensive project documentation
```

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.10+** installed.

```bash
# 1. Clone the repository
git clone https://github.com/jaishdahiya6-del/sales-project.git
cd sales-project

# 2. Configure a Virtual Environment (Highly Recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install All Pinned Dependencies
pip install -r requirements.txt
pip install pytest  # For running test suite
```

---

## ▶️ Execution Guide

### 1. Execute the Unified Pipeline
Runs cleaning, feature engineering, statistical profiling, exports 13 high-resolution visualization charts, trains and serializes both Random Forest models, and exports performance metrics.

```bash
python run_pipeline.py
```

### 2. Launch the Streamlit Interactive Dashboard
Launches the server. Connect on `http://localhost:8501`.

```bash
streamlit run dashboard/app.py
```

### 3. Run Automated Tests
Executes the comprehensive test suite verifying the integrity of data structures and prediction modeling.

```bash
python3 -m pytest -v
```

---

## 📈 Model Performance Metadata

Both models are trained on an 80/20 train-test split. Evaluation results are persisted in `outputs/model_metrics.json`:

| Model / Metric | Metric Name | Value | Description |
|---|---|---|---|
| **Sales Predictor (Regression)** | R² Score | **0.971** | Explains 97.1% of test-set sales variance |
| | MAE | **$18.67** | Average point-prediction absolute error |
| | RMSE | **$43.96** | Root Mean Square Error weighting outlier errors |
| **High-Value (Classification)** | Accuracy | **98.0%** | Overall correct class assignments |
| | F1-Score | **0.980** | Balanced Precision & Recall harmonic mean |
| | ROC AUC | **0.998** | High degree of class separability |

---

## 🚀 Key Technological Stack

- **Data Processing:** Python 3, Pandas, NumPy, Statsmodels
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-Learn
- **Serialization:** Joblib, JSON
- **Web UI:** Streamlit
- **Testing:** Pytest

---

## 👤 Author

**Data Analytics Team**
Developed as an end-to-end template demonstrating production-ready retail analytics.
