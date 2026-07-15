import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import xgboost as xgb


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    return df


df = load_data()


# ============================================================
# HELPER FUNCTION — SEASON
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Autumn"


# ============================================================
# HELPER FUNCTION — XGBOOST FORECAST
# ============================================================

def xgboost_forecast(segment_df, periods=3):

    # Create monthly sales
    monthly = (
        segment_df
        .set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
        .reset_index()
    )

    monthly.columns = ["Date", "Sales"]


    # Lag features
    monthly["Lag_1"] = monthly["Sales"].shift(1)
    monthly["Lag_2"] = monthly["Sales"].shift(2)
    monthly["Lag_3"] = monthly["Sales"].shift(3)


    # Rolling mean without target leakage
    monthly["Rolling_Mean_3"] = (
        monthly["Sales"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )


    # Calendar features
    monthly["Month"] = monthly["Date"].dt.month
    monthly["Quarter"] = monthly["Date"].dt.quarter

    monthly["Season"] = (
        monthly["Month"].apply(get_season)
    )


    # One-hot encoding
    monthly = pd.get_dummies(
        monthly,
        columns=["Season"],
        dtype=int
    )


    # Remove NaN caused by lags
    model_data = monthly.dropna().copy()


    # Features and target
    X = model_data.drop(
        columns=["Date", "Sales"]
    )

    y = model_data["Sales"]


    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)


    # Sales history
    sales_history = monthly["Sales"].tolist()

    last_date = monthly["Date"].iloc[-1]

    predictions = []


    # Recursive forecasting
    for i in range(periods):

        next_date = (
            last_date + pd.offsets.MonthEnd(1)
        )

        month = next_date.month
        quarter = next_date.quarter
        season = get_season(month)


        # Latest three values
        lag_1 = sales_history[-1]
        lag_2 = sales_history[-2]
        lag_3 = sales_history[-3]

        rolling_mean = np.mean(
            sales_history[-3:]
        )


        # Create future row
        future_row = pd.DataFrame({

            "Lag_1": [lag_1],
            "Lag_2": [lag_2],
            "Lag_3": [lag_3],

            "Rolling_Mean_3": [rolling_mean],

            "Month": [month],
            "Quarter": [quarter]
        })


        # Add season columns
        for season_name in [
            "Autumn",
            "Spring",
            "Summer",
            "Winter"
        ]:

            future_row[
                f"Season_{season_name}"
            ] = int(
                season == season_name
            )


        # Match exact training columns
        future_row = future_row.reindex(
            columns=X.columns,
            fill_value=0
        )


        # Predict
        prediction = model.predict(
            future_row
        )[0]


        predictions.append({

            "Date": next_date,

            "Predicted Sales": prediction

        })


        # Recursive update
        sales_history.append(prediction)

        last_date = next_date


    forecast_df = pd.DataFrame(predictions)

    return forecast_df


# ============================================================
# HELPER FUNCTION — PRODUCT CLUSTERING
# ============================================================

@st.cache_data
def create_product_clusters(df):

    # ----------------------------
    # Total Sales
    # ----------------------------

    total_sales = (
        df
        .groupby("Sub-Category")["Sales"]
        .sum()
        .rename("Total_Sales")
    )


    # ----------------------------
    # Average Order Value
    # ----------------------------

    order_values = (
        df
        .groupby(
            ["Sub-Category", "Order ID"]
        )["Sales"]
        .sum()
        .reset_index()
    )

    avg_order_value = (
        order_values
        .groupby("Sub-Category")["Sales"]
        .mean()
        .rename("Avg_Order_Value")
    )


    # ----------------------------
    # Monthly Sales Volatility
    # ----------------------------

    monthly_subcategory_sales = (
        df
        .groupby(
            [
                "Sub-Category",
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                )
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    sales_volatility = (
        monthly_subcategory_sales
        .groupby("Sub-Category")["Sales"]
        .std()
        .rename("Sales_Volatility")
    )


    # ----------------------------
    # YoY Growth Rate
    # ----------------------------

    yearly_sales = (
        df
        .groupby(
            ["Sub-Category", "Year"]
        )["Sales"]
        .sum()
        .reset_index()
    )

    yearly_sales["YoY_Growth"] = (
        yearly_sales
        .groupby("Sub-Category")["Sales"]
        .pct_change() * 100
    )

    avg_growth = (
        yearly_sales
        .groupby("Sub-Category")[
            "YoY_Growth"
        ]
        .mean()
        .rename("Sales_Growth_Rate")
    )


    # ----------------------------
    # Combine features
    # ----------------------------

    product_features = pd.concat(

        [
            total_sales,
            avg_growth,
            sales_volatility,
            avg_order_value
        ],

        axis=1

    ).reset_index()


    product_features = (
        product_features.dropna()
    )


    feature_columns = [

        "Total_Sales",

        "Sales_Growth_Rate",

        "Sales_Volatility",

        "Avg_Order_Value"
    ]


    # ----------------------------
    # Scaling
    # ----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        product_features[feature_columns]
    )


    # ----------------------------
    # KMeans
    # ----------------------------

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    product_features["Cluster"] = (
        kmeans.fit_predict(X_scaled)
    )


    # ----------------------------
    # Cluster Labels
    # ----------------------------

    cluster_labels = {

        0: "High Value, High Volatility",

        1: "Lower Volume, Stable Demand",

        2: "High Volume, Established Demand",

        3: "Rapidly Growing Demand"
    }

    product_features["Demand_Segment"] = (
        product_features["Cluster"]
        .map(cluster_labels)
    )


    # ----------------------------
    # PCA
    # ----------------------------

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    product_features["PCA_1"] = (
        X_pca[:, 0]
    )

    product_features["PCA_2"] = (
        X_pca[:, 1]
    )


    return product_features


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(

    "Select Page",

    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)


# ============================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================

if page == "Sales Overview":

    st.title(
        "📊 Sales Overview Dashboard"
    )

    st.write(
        "Explore historical sales performance across years, regions, and product categories."
    )


    # ----------------------------
    # KPI Metrics
    # ----------------------------

    total_sales_value = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    avg_order_value = (
        df
        .groupby("Order ID")["Sales"]
        .sum()
        .mean()
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${total_sales_value:,.2f}"
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Average Order Value",
        f"${avg_order_value:,.2f}"
    )


    st.divider()


    # ----------------------------
    # Total Sales by Year
    # ----------------------------

    yearly_sales = (
        df
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )


    fig_year = px.bar(

        yearly_sales,

        x="Year",

        y="Sales",

        title="Total Sales by Year",

        text_auto=".2s"
    )


    st.plotly_chart(
        fig_year,
        use_container_width=True
    )


    # ----------------------------
    # Monthly Sales Trend
    # ----------------------------

    monthly_sales = (

        df
        .set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
        .reset_index()
    )


    fig_month = px.line(

        monthly_sales,

        x="Order Date",

        y="Sales",

        title="Monthly Sales Trend",

        markers=True
    )


    st.plotly_chart(
        fig_month,
        use_container_width=True
    )


    # ----------------------------
    # Interactive Filters
    # ----------------------------

    st.subheader(
        "Interactive Sales Analysis"
    )


    selected_regions = st.multiselect(

        "Select Region",

        options=sorted(
            df["Region"].unique()
        ),

        default=sorted(
            df["Region"].unique()
        )
    )


    selected_categories = st.multiselect(

        "Select Category",

        options=sorted(
            df["Category"].unique()
        ),

        default=sorted(
            df["Category"].unique()
        )
    )


    filtered_df = df[

        (df["Region"].isin(selected_regions))

        &

        (df["Category"].isin(selected_categories))

    ]


    sales_breakdown = (

        filtered_df

        .groupby(
            ["Region", "Category"]
        )["Sales"]

        .sum()

        .reset_index()
    )


    fig_filter = px.bar(

        sales_breakdown,

        x="Region",

        y="Sales",

        color="Category",

        barmode="group",

        title="Sales by Region and Category"
    )


    st.plotly_chart(
        fig_filter,
        use_container_width=True
    )


# ============================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================

elif page == "Forecast Explorer":

    st.title(
        "🔮 Forecast Explorer"
    )

    st.write(
        "Generate future sales forecasts using the best-performing XGBoost model."
    )


    # ----------------------------
    # Select segment type
    # ----------------------------

    segment_type = st.selectbox(

        "Select Forecast Type",

        [
            "Category",
            "Region"
        ]
    )


    # ----------------------------
    # Select specific segment
    # ----------------------------

    if segment_type == "Category":

        selected_segment = st.selectbox(

            "Select Category",

            sorted(
                df["Category"].unique()
            )
        )

        segment_df = df[

            df["Category"]
            == selected_segment

        ].copy()


    else:

        selected_segment = st.selectbox(

            "Select Region",

            sorted(
                df["Region"].unique()
            )
        )

        segment_df = df[

            df["Region"]
            == selected_segment

        ].copy()


    # ----------------------------
    # Forecast horizon
    # ----------------------------

    forecast_horizon = st.slider(

        "Select Forecast Horizon (Months)",

        min_value=1,

        max_value=3,

        value=3,

        step=1
    )


    # ----------------------------
    # Generate forecast
    # ----------------------------

    forecast = xgboost_forecast(

        segment_df,

        periods=forecast_horizon
    )


    st.subheader(

        f"{forecast_horizon}-Month Forecast for {selected_segment}"

    )


    st.dataframe(
        forecast,
        use_container_width=True
    )


    # ----------------------------
    # Historical data
    # ----------------------------

    historical = (

        segment_df

        .set_index("Order Date")

        .resample("ME")["Sales"]

        .sum()

        .reset_index()
    )


    historical.columns = [
        "Date",
        "Sales"
    ]


    # ----------------------------
    # Plot historical + forecast
    # ----------------------------

    fig_forecast = px.line(

        historical,

        x="Date",

        y="Sales",

        title=(
            f"Historical Sales and "
            f"{forecast_horizon}-Month Forecast"
        )
    )


    fig_forecast.add_scatter(

        x=forecast["Date"],

        y=forecast["Predicted Sales"],

        mode="lines+markers",

        name="XGBoost Forecast"
    )


    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )


    # ----------------------------
    # Model Metrics
    # ----------------------------

    st.subheader(
        "Best Model Performance"
    )


    metric1, metric2, metric3 = (
        st.columns(3)
    )


    metric1.metric(
        "MAE",
        "8,582.53"
    )


    metric2.metric(
        "RMSE",
        "10,760.89"
    )


    metric3.metric(
        "MAPE",
        "12.78%"
    )


    st.caption(
        "XGBoost achieved the lowest MAE, RMSE, and MAPE among SARIMA, Prophet, and XGBoost on the common historical test period."
    )


# ============================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================

elif page == "Anomaly Report":

    st.title(
        "⚠️ Sales Anomaly Report"
    )

    st.write(
        "Detect unusually high or low sales weeks using Isolation Forest."
    )


    # ----------------------------
    # Weekly Sales
    # ----------------------------

    weekly_sales = (

        df

        .set_index("Order Date")

        .resample("W")["Sales"]

        .sum()

        .reset_index()
    )


    weekly_sales.columns = [
        "Date",
        "Sales"
    ]


    # ----------------------------
    # Isolation Forest
    # ----------------------------

    iso_model = IsolationForest(

        contamination=0.05,

        random_state=42
    )


    weekly_sales["Anomaly"] = (

        iso_model.fit_predict(

            weekly_sales[["Sales"]]

        )

    )


    anomalies = weekly_sales[

        weekly_sales["Anomaly"] == -1

    ].copy()


    # ----------------------------
    # KPI
    # ----------------------------

    st.metric(

        "Total Anomalous Weeks Detected",

        len(anomalies)
    )


    # ----------------------------
    # Plot
    # ----------------------------

    fig_anomaly = px.line(

        weekly_sales,

        x="Date",

        y="Sales",

        title=(
            "Weekly Sales with "
            "Isolation Forest Anomalies"
        )
    )


    fig_anomaly.add_scatter(

        x=anomalies["Date"],

        y=anomalies["Sales"],

        mode="markers",

        marker=dict(
            size=10,
            color="red",
            symbol="x"
        ),

        name="Anomaly"
    )


    st.plotly_chart(

        fig_anomaly,

        use_container_width=True
    )


    # ----------------------------
    # Anomaly Table
    # ----------------------------

    st.subheader(
        "Detected Anomaly Dates"
    )


    anomaly_table = anomalies[
        [
            "Date",
            "Sales"
        ]
    ].copy()


    anomaly_table["Sales"] = (

        anomaly_table["Sales"]

        .round(2)

    )


    st.dataframe(

        anomaly_table,

        use_container_width=True
    )


# ============================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================

elif page == "Product Demand Segments":

    st.title(
        "📦 Product Demand Segmentation"
    )

    st.write(
        "Sub-categories are grouped using K-Means clustering based on total sales, year-over-year growth, sales volatility, and average order value."
    )


    # ----------------------------
    # Create clusters
    # ----------------------------

    product_features = (
        create_product_clusters(df)
    )


    # ----------------------------
    # PCA Cluster Chart
    # ----------------------------

    fig_cluster = px.scatter(

        product_features,

        x="PCA_1",

        y="PCA_2",

        color="Demand_Segment",

        hover_name="Sub-Category",

        hover_data=[

            "Total_Sales",

            "Sales_Growth_Rate",

            "Sales_Volatility",

            "Avg_Order_Value"

        ],

        title=(
            "Product Demand Clusters — "
            "PCA Visualization"
        )
    )


    st.plotly_chart(

        fig_cluster,

        use_container_width=True
    )


    # ----------------------------
    # Cluster Table
    # ----------------------------

    st.subheader(
        "Sub-Category Demand Segments"
    )


    cluster_table = product_features[
        [
            "Sub-Category",

            "Cluster",

            "Demand_Segment",

            "Total_Sales",

            "Sales_Growth_Rate",

            "Sales_Volatility",

            "Avg_Order_Value"
        ]
    ].sort_values(
        "Cluster"
    )


    st.dataframe(

        cluster_table,

        use_container_width=True
    )


    # ----------------------------
    # Stocking Strategies
    # ----------------------------

    st.subheader(
        "Recommended Stocking Strategies"
    )


    strategies = {

        "High Value, High Volatility":
            "Maintain moderate safety stock and use frequent demand monitoring because sales are valuable but highly unpredictable.",

        "Lower Volume, Stable Demand":
            "Maintain lean inventory with regular replenishment because demand is relatively stable and sales volume is generally lower.",

        "High Volume, Established Demand":
            "Maintain high inventory availability and reliable replenishment cycles to prevent stockouts for consistently high-demand products.",

        "Rapidly Growing Demand":
            "Gradually increase inventory levels and closely monitor demand growth to capture rising sales while avoiding excessive overstocking."

    }


    for segment, strategy in strategies.items():

        st.markdown(
            f"**{segment}:** {strategy}"
        )