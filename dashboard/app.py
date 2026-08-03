"""
dashboard/app.py
----------------
Interactive production-ready Streamlit dashboard for the Retail Sales Analytics project.

Integrates real-time predictions using serialized Random Forest models to provide
a "What-If" order simulation. Features robust, cached data handling, multi-dimensional
filtering, sorting, KPI metrics, interactive Plotly visualizations, and CSV exports.

Run with:
    streamlit run dashboard/app.py

Author: Data Analytics Team
"""

import json
import os
import sys
from typing import Dict, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Allow importing from src/ when run as `streamlit run dashboard/app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, MODELS_DIR

# Set wide layout and tab titles
st.set_page_config(
    page_title="Retail Sales Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Cached Data Loading & Model Loading
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading datasets...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned and raw datasets for the dashboard.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - df: Cleaned and processed dataset.
        - raw_df: Original raw dataset.
    """
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        # Fallback to run pipeline if not pre-generated
        from src.preprocessing import run_pipeline
        df = run_pipeline(save=True)

    try:
        raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        raw_df = df.copy()

    return df, raw_df


@st.cache_resource(show_spinner="Loading serialized models...")
def load_serialized_models() -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, LabelEncoder]]]:
    """
    Load serialized Random Forest models and LabelEncoders for on-demand inference.

    Returns
    -------
    Tuple[Optional[Any], Optional[Any], Optional[Dict[str, LabelEncoder]]]
        - regression_model: RandomForestRegressor or None if load fails.
        - classification_model: RandomForestClassifier or None if load fails.
        - encoders: Dict of fitted LabelEncoder mapping or None if load fails.
    """
    reg_path = os.path.join(MODELS_DIR, "regression_model.joblib")
    clf_path = os.path.join(MODELS_DIR, "classification_model.joblib")
    enc_path = os.path.join(MODELS_DIR, "encoders.joblib")

    if (
        os.path.exists(reg_path)
        and os.path.exists(clf_path)
        and os.path.exists(enc_path)
    ):
        try:
            reg_model = joblib.load(reg_path)
            clf_model = joblib.load(clf_path)
            encoders = joblib.load(enc_path)
            return reg_model, clf_model, encoders
        except Exception as exc:
            st.sidebar.error(f"Error loading models: {exc}")
            return None, None, None
    else:
        return None, None, None


# ------------------------------------------------------------------
# Core Dashboard Page Execution
# ------------------------------------------------------------------
def main():
    # Load assets
    df, raw_df = load_data()
    reg_model, clf_model, encoders = load_serialized_models()

    # Title Banner
    st.markdown(
        """
        <div style="background-color:#1e293b;padding:20px;border-radius:10px;margin-bottom:25px;color:white;">
            <h1 style="margin:0;font-size:2.2rem;font-weight:700;">📊 Retail Sales Analytics & Prediction Hub</h1>
            <p style="margin:5px 0 0 0;font-size:1.1rem;color:#94a3b8;font-weight:400;">
                Interactive operational insights, predictive modeling summaries, and dynamic business simulation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar Filters
    # ------------------------------------------------------------------
    st.sidebar.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=300&q=80",
        use_container_width=True,
    )
    st.sidebar.markdown("### Dashboard Control Panel")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Data Processing Audit",
            "Exploratory Trend Analysis",
            "Multivariate Correlation",
            "Model Performance Metadata",
            "Interactive 'What-If' Simulation",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Global Filters**")

    # Categories Filter
    unique_categories = sorted(df["Category"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Product Category",
        options=unique_categories,
        default=unique_categories,
    )

    # Regions Filter
    unique_regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Geographical Region",
        options=unique_regions,
        default=unique_regions,
    )

    # Date Range Filter
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Analysis Date Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Apply global filters
    filtered_df = df.copy()

    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    else:
        filtered_df = filtered_df[filtered_df["Category"] == "None"]  # Empty filter handler

    if selected_regions:
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]
    else:
        filtered_df = filtered_df[filtered_df["Region"] == "None"]

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date = pd.Timestamp(selected_date_range[0])
        end_date = pd.Timestamp(selected_date_range[1])
        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_date) & (filtered_df["Date"] <= end_date)
        ]

    # ------------------------------------------------------------------
    # KPI Display Panel (Always Visible)
    # ------------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_sales = filtered_df["Sales"].sum() if not filtered_df.empty else 0.0
    total_profit = filtered_df["Profit"].sum() if not filtered_df.empty else 0.0
    total_orders = len(filtered_df)
    avg_profit_margin = (
        filtered_df["ProfitMargin"].mean() if not filtered_df.empty else 0.0
    )

    with kpi_col1:
        st.metric(
            label="Total Sales Revenue",
            value=f"${total_sales:,.2f}",
            delta=f"Avg AOV: ${filtered_df['AvgOrderValue'].mean():,.2f}"
            if not filtered_df.empty
            else None,
        )
    with kpi_col2:
        st.metric(
            label="Gross Accumulated Profit",
            value=f"${total_profit:,.2f}",
            delta=f"Avg: ${filtered_df['Profit'].mean():,.2f}/order"
            if not filtered_df.empty
            else None,
        )
    with kpi_col3:
        st.metric(
            label="Order Volume",
            value=f"{total_orders:,}",
        )
    with kpi_col4:
        st.metric(
            label="Mean Profit Margin",
            value=f"{avg_profit_margin:.2f}%",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Rendering Pages
    # ------------------------------------------------------------------

    if page == "Overview Dashboard":
        st.subheader("📋 Operational Data & Summary Overview")

        # Table controls: Sorting
        st.markdown("**Data Table Explorer**")
        col_sort_1, col_sort_2 = st.columns([1, 3])
        with col_sort_1:
            sort_by = st.selectbox(
                "Sort table by",
                options=["Date", "Sales", "Profit", "Quantity", "SatisfactionScore"],
                index=0,
            )
        with col_sort_2:
            sort_order = st.radio(
                "Order",
                options=["Descending", "Ascending"],
                horizontal=True,
                index=0,
            )

        ascending = sort_order == "Ascending"
        sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        # Main Table View
        st.dataframe(
            sorted_df.head(100),
            use_container_width=True,
            column_config={
                "OrderID": "Order ID",
                "Date": "Date Purchased",
                "Sales": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                "Profit": st.column_config.NumberColumn("Profit ($)", format="$%.2f"),
                "SatisfactionScore": "Satisfaction",
            },
        )

        st.caption(f"Showing top 100 rows matching filters (sorted by {sort_by}).")

        # Download option
        st.download_button(
            label="⬇️ Export Filtered View to CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_sales_data.csv",
            mime="text/csv",
        )

        st.divider()

        # Dynamic charts on dashboard
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📈 Monthly Revenue Velocity")
            if not filtered_df.empty:
                monthly_agg = (
                    filtered_df.groupby(pd.Grouper(key="Date", freq="MS"))
                    .agg({"Sales": "sum", "Profit": "sum"})
                    .reset_index()
                )
                fig_trend = px.line(
                    monthly_agg,
                    x="Date",
                    y=["Sales", "Profit"],
                    labels={"value": "Amount ($)", "variable": "Financial Metric"},
                    title="Revenue & Profit Velocity Trend",
                    markers=True,
                    color_discrete_sequence=["#3b82f6", "#10b981"],
                )
                fig_trend.update_layout(hovermode="x unified")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No data available for current selection.")

        with c2:
            st.markdown("### 🏆 Top 10 Best Selling Products")
            if not filtered_df.empty:
                top_prods = (
                    filtered_df.groupby("Product")["Sales"]
                    .sum()
                    .reset_index()
                    .sort_values(by="Sales", ascending=True)
                    .tail(10)
                )
                fig_bar = px.bar(
                    top_prods,
                    x="Sales",
                    y="Product",
                    orientation="h",
                    title="Top Products by Accumulated Sales ($)",
                    color="Sales",
                    color_continuous_scale="blues",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No data available for current selection.")

    elif page == "Data Processing Audit":
        st.subheader("🧹 Production Data Ingestion & Imputation Log")

        col_audit_1, col_audit_2 = st.columns(2)

        with col_audit_1:
            st.markdown("#### Raw Ingested Dataset Profile")
            st.write(f"- **Total Rows:** {len(raw_df):,}")
            st.write(f"- **Total Missing Values Detected:** {int(raw_df.isnull().sum().sum()):,}")
            st.write(f"- **Total Duplicate Records Identified:** {int(raw_df.duplicated().sum()):,}")
            st.write("- **Primary Anomaly Handling:** Complete missing entries. Flag outlier thresholds.")

        with col_audit_2:
            st.markdown("#### Cleaned Dataset Profile")
            st.write(f"- **Total Rows:** {len(df):,}")
            st.write(f"- **Total Missing Values Post-Imputation:** {int(df.isnull().sum().sum()):,}")
            st.write(f"- **Total Duplicate Records Remaining:** {int(df.duplicated().sum()):,}")
            st.write("- **Outlier Treatment:** Capped high-variance observations using IQR bounds.")

        st.info(
            "💡 **Pipeline Rulebook:** Numerical missing values are replaced by respective column medians "
            "to secure a midpoint. Categorical missing values are filled with column mode. Outliers are capped "
            "between [Q1 - 1.5 * IQR] and [Q3 + 1.5 * IQR] to limit skew without introducing variance bias."
        )

        st.markdown("#### Feature Engineering Manifest")
        features_manifest = {
            "Feature Column": [
                "Year", "Month", "MonthName", "Quarter",
                "DayOfWeek", "ProfitMargin", "AvgOrderValue", "HighValueOrder"
            ],
            "Data Type": [
                "int64", "int64", "object", "int64",
                "object", "float64", "float64", "int64"
            ],
            "Logical Purpose": [
                "Temporal cohort extraction",
                "Seasonal indexing",
                "Visualization formatting",
                "Quarterly reporting alignment",
                "Weekday behavioral modeling",
                "Gross financial margin tracking ((Profit/Sales)*100)",
                "Transaction density metrics (Sales/Quantity)",
                "Target flag indicating order sales exceeds historical median"
            ]
        }
        st.table(pd.DataFrame(features_manifest))

    elif page == "Exploratory Trend Analysis":
        st.subheader("📊 Visual Market Segmentation & Performance")

        tab_cat, tab_reg, tab_cust = st.tabs([
            "Category Analysis", "Regional Distribution", "Customer & Price Analytics"
        ])

        with tab_cat:
            st.markdown("### Category-Level Segment Performance")
            if not filtered_df.empty:
                cat_summary = (
                    filtered_df.groupby("Category")
                    .agg(
                        Sales=("Sales", "sum"),
                        Profit=("Profit", "sum"),
                        AvgOrderValue=("Sales", "mean"),
                        AverageSatisfaction=("SatisfactionScore", "mean"),
                    )
                    .reset_index()
                )
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    fig_cat_rev = px.bar(
                        cat_summary,
                        x="Category",
                        y="Sales",
                        color="Profit",
                        title="Sales and Profitability by Category",
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig_cat_rev, use_container_width=True)
                with col_c2:
                    fig_cat_sat = px.bar(
                        cat_summary,
                        x="Category",
                        y="AverageSatisfaction",
                        title="Average Customer Satisfaction by Category",
                        range_y=[1, 5],
                        color="AverageSatisfaction",
                        color_continuous_scale="Tealgrn",
                    )
                    st.plotly_chart(fig_cat_sat, use_container_width=True)
            else:
                st.info("No data matching filters.")

        with tab_reg:
            st.markdown("### Regional Revenue Dispersion")
            if not filtered_df.empty:
                reg_summary = (
                    filtered_df.groupby("Region")["Sales"]
                    .sum()
                    .reset_index()
                )
                fig_reg_pie = px.pie(
                    reg_summary,
                    names="Region",
                    values="Sales",
                    title="Regional Sales Revenue Share",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig_reg_pie, use_container_width=True)
            else:
                st.info("No data matching filters.")

        with tab_cust:
            st.markdown("### Unit Price vs Order Value Density")
            if not filtered_df.empty:
                fig_scatter = px.scatter(
                    filtered_df,
                    x="UnitPrice",
                    y="Sales",
                    color="Category",
                    size="Quantity",
                    hover_data=["Product", "Profit"],
                    title="Sales Correlation with Unit Price & Transaction Volume",
                    color_discrete_sequence=px.colors.qualitative.D3,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No data matching filters.")

    elif page == "Multivariate Correlation":
        st.subheader("🔗 Multivariate Feature Correlation Heatmap")
        st.markdown("Assess the linear relationships among numerical attributes.")

        if not filtered_df.empty:
            num_df = filtered_df.select_dtypes(include=[np.number])
            # Exclude classification target or standard encoding columns if they reduce readability
            cols_to_corr = [c for c in num_df.columns if not c.endswith("_enc")]
            corr_mat = num_df[cols_to_corr].corr()

            fig_heat = px.imshow(
                corr_mat,
                x=corr_mat.columns,
                y=corr_mat.columns,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Pearson Linear Correlation Coefficient Matrix",
            )
            fig_heat.update_layout(height=600)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No data available to calculate correlation.")

    elif page == "Model Performance Metadata":
        st.subheader("📉 Machine Learning Pipeline Evaluation Metrics")

        # Load metadata JSON
        metrics_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "model_metrics.json"
        )
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)

            col_met_1, col_met_2 = st.columns(2)

            with col_met_1:
                st.markdown("#### Regression Model (Order Sales Predictor)")
                reg_m = metrics_dict.get("regression", {})
                st.metric("R² Score (Goodness of Fit)", f"{reg_m.get('R2', 'N/A')}")
                st.metric("Mean Absolute Error (MAE)", f"${reg_m.get('MAE', 'N/A')}")
                st.metric("Root Mean Squared Error (RMSE)", f"${reg_m.get('RMSE', 'N/A')}")

            with col_met_2:
                st.markdown("#### Classification Model (High-Value Order Classifier)")
                clf_m = metrics_dict.get("classification", {})
                st.metric("ROC AUC Area Under Curve", f"{clf_m.get('ROC_AUC', 'N/A')}")
                st.metric("Classification F1-Score", f"{clf_m.get('F1', 'N/A')}")
                st.metric("Accuracy Score", f"{clf_m.get('Accuracy', 'N/A')}")

            st.divider()

            # Plot predictions vs actual from prediction results
            pred_path = os.path.join(
                os.path.dirname(__file__), "..", "outputs", "prediction_results.csv"
            )
            if os.path.exists(pred_path):
                preds_df = pd.read_csv(pred_path)
                st.markdown("#### Predicted vs Actual Sales (Test Cohort)")
                fig_val = px.scatter(
                    preds_df,
                    x="ActualSales",
                    y="PredictedSales",
                    trendline="ols",
                    labels={"ActualSales": "Actual Sales ($)", "PredictedSales": "Predicted Sales ($)"},
                    title="Evaluation Plot: Model Predictions vs actual ground-truth",
                    color_discrete_sequence=["#0891b2"],
                )
                st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.warning(
                "⚠️ **Pipeline Metadata Missing:** Please execute the master analytics pipeline "
                "via `python run_pipeline.py` to generate the serialized metrics files."
            )

    elif page == "Interactive 'What-If' Simulation":
        st.subheader("🔮 Dynamic Predictive Business Simulator ('What-If' Analysis)")
        st.markdown(
            "Use the sidebar controls below to simulate custom transactions and evaluate predicted "
            "sales values and high-value status in real-time."
        )

        if reg_model is None or clf_model is None or encoders is None:
            st.error(
                "❌ **Serialized Models or Encoders Not Found.** "
                "Ensure that the training pipeline has run and created files inside `models/`."
            )
            st.info("Run: `python run_pipeline.py` in your workspace.")
        else:
            sim_col_l, sim_col_r = st.columns([1, 1])

            with sim_col_l:
                st.markdown("### ⚙️ Transaction Simulation Parameters")

                # Map inputs using fitted encoders classes
                available_categories = list(encoders["Category"].classes_)
                available_regions = list(encoders["Region"].classes_)

                sim_cat = st.selectbox("Product Category", options=available_categories, index=0)
                sim_region = st.selectbox("Geographical Region", options=available_regions, index=0)

                sim_unit_price = st.slider(
                    "Unit Price ($)",
                    min_value=5.0,
                    max_value=float(df["UnitPrice"].max() if not df.empty else 1000.0),
                    value=float(df["UnitPrice"].mean() if not df.empty else 100.0),
                    step=0.5,
                )

                sim_quantity = st.slider("Quantity Ordered", min_value=1, max_value=20, value=3, step=1)
                sim_discount = st.selectbox("Discount Percentage (%)", options=[0, 5, 10, 15, 20, 25, 30], index=0)

                sim_age = st.slider("Customer Age", min_value=18, max_value=75, value=35, step=1)
                sim_satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=4, step=1)

                sim_month = st.slider("Month of Transaction", min_value=1, max_value=12, value=6, step=1)
                sim_quarter = (sim_month - 1) // 3 + 1

            with sim_col_r:
                st.markdown("### 🎯 Simulation Inference Outcomes")

                # Assemble features row
                # Encode categorical variables using loaded encoders
                try:
                    cat_enc = int(encoders["Category"].transform([sim_cat])[0])
                    reg_enc = int(encoders["Region"].transform([sim_region])[0])
                except Exception as exc:
                    st.error(f"Encoding mismatch error: {exc}")
                    cat_enc = 0
                    reg_enc = 0

                # Must match EXACTLY the order in model.prepare_features:
                # FEATURE_COLUMNS = ["UnitPrice", "Quantity", "DiscountPct", "CustomerAge", "SatisfactionScore", "Month", "Quarter"]
                # CATEGORICAL_COLUMNS = ["Category", "Region"] (encoded columns have "_enc" appended: Category_enc, Region_enc)
                feature_row = [
                    sim_unit_price,
                    sim_quantity,
                    sim_discount,
                    sim_age,
                    sim_satisfaction,
                    sim_month,
                    sim_quarter,
                    cat_enc,
                    reg_enc,
                ]

                # Convert to dataframe with same headers as model expectation
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
                input_df = pd.DataFrame([feature_row], columns=col_headers)

                # Make Predictions
                pred_sales = float(reg_model.predict(input_df)[0])
                pred_hv_prob = float(clf_model.predict_proba(input_df)[0][1])
                pred_hv = int(clf_model.predict(input_df)[0])

                # Layout Outcomes
                st.markdown(
                    f"""
                    <div style="border: 2px solid #2563eb; border-radius: 8px; padding: 15px; margin-bottom: 20px; background-color: #f0f7ff;">
                        <h4 style="margin-top: 0; color: #1e40af;">Predicted Order Sales</h4>
                        <p style="font-size: 2.2rem; font-weight: 800; color: #2563eb; margin: 5px 0;">
                            ${pred_sales:,.2f}
                        </p>
                        <span style="color: #6b7280; font-size: 0.9rem;">
                            Regression point prediction matching simulated transaction profile.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if pred_hv == 1:
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #16a34a; border-radius: 8px; padding: 15px; background-color: #f0fdf4;">
                            <h4 style="margin-top: 0; color: #166534;">High-Value Status: YES</h4>
                            <p style="font-size: 2.2rem; font-weight: 800; color: #16a34a; margin: 5px 0;">
                                {pred_hv_prob * 100:.1f}% Probability
                            </p>
                            <span style="color: #6b7280; font-size: 0.9rem;">
                                Order sales is predicted to rank above the historical midpoint of order values.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #ea580c; border-radius: 8px; padding: 15px; background-color: #fff7ed;">
                            <h4 style="margin-top: 0; color: #9a3412;">High-Value Status: NO</h4>
                            <p style="font-size: 2.2rem; font-weight: 800; color: #ea580c; margin: 5px 0;">
                                {pred_hv_prob * 100:.1f}% Probability
                            </p>
                            <span style="color: #6b7280; font-size: 0.9rem;">
                                Order sales is predicted to fall below the historical midpoint of order values.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Summary explanation of mathematical determinants
                st.markdown("#### Determinant Diagnostics")
                st.caption(
                    "💡 **How this was calculated:** The Unit Price, Quantity, and Discount are primary determinants "
                    "for the Regression model. The Random Forest classifier maps the multi-feature vector "
                    "against decision thresholds trained on actual historical transactions to identify segment membership."
                )


if __name__ == "__main__":
    main()
