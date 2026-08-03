"""
app.py
------
Interactive Streamlit dashboard for the Retail Sales Analytics project.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from src/ when run as `streamlit run dashboard/app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "sales_data_clean.csv"
)
RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "sales_data.csv"
)

st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_clean_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_PATH)


def kpi_card(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


def main():
    st.title("📊 Retail Sales Analytics Dashboard")
    st.caption("End-to-end interactive view of the cleaned retail sales dataset")

    df = load_clean_data()
    raw_df = load_raw_data()

    # ------------------------------------------------------------------
    # Sidebar Navigation & Filters
    # ------------------------------------------------------------------
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Data Cleaning", "Exploratory Analysis", "Correlation",
         "Model Predictions", "Download Results"],
    )

    st.sidebar.header("Filters")
    categories = st.sidebar.multiselect(
        "Category", options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique()),
    )
    regions = st.sidebar.multiselect(
        "Region", options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique()),
    )
    date_range = st.sidebar.date_input(
        "Date range",
        value=(df["Date"].min().date(), df["Date"].max().date()),
    )

    filtered = df[
        df["Category"].isin(categories) & df["Region"].isin(regions)
    ]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]

    # ------------------------------------------------------------------
    # KPI Cards (shown on every page)
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Sales", f"${filtered['Sales'].sum():,.0f}")
    with col2:
        kpi_card("Total Profit", f"${filtered['Profit'].sum():,.0f}")
    with col3:
        kpi_card("Orders", f"{len(filtered):,}")
    with col4:
        avg_margin = filtered["ProfitMargin"].mean() if len(filtered) else 0
        kpi_card("Avg. Profit Margin", f"{avg_margin:.1f}%")

    st.divider()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    if page == "Overview":
        st.subheader("Dataset Preview")
        st.dataframe(filtered.head(50), use_container_width=True)

        st.subheader("Sales Trend Over Time")
        trend = filtered.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"].sum().reset_index()
        fig = px.line(trend, x="Date", y="Sales", markers=True, title="Monthly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Products")
        top = filtered.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(10)
        fig2 = px.bar(top, orientation="h", title="Top 10 Products by Sales")
        st.plotly_chart(fig2, use_container_width=True)

    elif page == "Data Cleaning":
        st.subheader("Data Cleaning Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Raw dataset**")
            st.write(f"Rows: {len(raw_df):,}")
            st.write(f"Missing values: {int(raw_df.isnull().sum().sum())}")
            st.write(f"Duplicate rows: {int(raw_df.duplicated().sum())}")
        with c2:
            st.write("**Cleaned dataset**")
            st.write(f"Rows: {len(df):,}")
            st.write(f"Missing values: {int(df.isnull().sum().sum())}")
            st.write(f"Duplicate rows: {int(df.duplicated().sum())}")

        st.info(
            "Missing numeric values were imputed with the column median, missing "
            "categorical values with the column mode, exact duplicate rows were "
            "dropped, and outliers in Sales/Quantity/Profit/UnitPrice were capped "
            "using the IQR method."
        )

    elif page == "Exploratory Analysis":
        st.subheader("Statistical Summary")
        st.dataframe(filtered.describe().round(2), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(filtered, x="Sales", nbins=40, title="Sales Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(filtered, x="Category", y="Sales", title="Sales by Category")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.pie(filtered, names="Region", values="Sales", title="Sales Share by Region")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Correlation":
        st.subheader("Correlation Heatmap")
        numeric_df = filtered.select_dtypes(include=[np.number])
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Predictions":
        st.subheader("Model Prediction Section")
        pred_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "prediction_results.csv"
        )
        if os.path.exists(pred_path):
            preds = pd.read_csv(pred_path)
            st.write("Sample of test-set predictions (Sales regression + High-Value classification):")
            st.dataframe(preds.head(30), use_container_width=True)

            fig = px.scatter(
                preds, x="ActualSales", y="PredictedSales",
                title="Predicted vs Actual Sales", trendline="ols",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run `python run_pipeline.py` first to generate prediction results.")

    elif page == "Download Results":
        st.subheader("Download Results")
        st.download_button(
            "⬇️ Download Cleaned Dataset (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )
        pred_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "prediction_results.csv"
        )
        if os.path.exists(pred_path):
            with open(pred_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Prediction Results (CSV)",
                    data=f,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
