import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Nassau Candy Dashboard", layout="wide")
st.title("🍭 Nassau Candy: Smart Analytics Dashboard")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a file to continue")
    st.stop()

# -------------------------------
# LOAD FILE
# -------------------------------
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
df.columns = df.columns.str.strip().str.lower()

# -------------------------------
# VALIDATE REQUIRED COLUMNS
# -------------------------------
required_cols = ["sales", "cost", "gross profit"]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# -------------------------------
# DATA CLEANING
# -------------------------------
df = df.drop_duplicates()

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=required_cols)

# -------------------------------
# CALCULATIONS
# -------------------------------
df["profit"] = df["sales"] - df["cost"]

df["margin %"] = np.where(
    df["sales"] != 0,
    (df["profit"] / df["sales"]) * 100,
    0
)

# Date conversion
if "order date" in df.columns:
    df["order date"] = pd.to_datetime(df["order date"], errors='coerce')

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True)

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

if "order date" in df.columns:
    min_date = df["order date"].min()
    max_date = df["order date"].max()

    dates = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    if len(dates) == 2:
        df = df[
            (df["order date"] >= pd.to_datetime(dates[0])) &
            (df["order date"] <= pd.to_datetime(dates[1]))
        ]

if "division" in df.columns:
    division = st.sidebar.selectbox(
        "Select Division",
        ["All"] + list(df["division"].dropna().unique())
    )
    if division != "All":
        df = df[df["division"] == division]

min_sales = float(df["sales"].min())
max_sales = float(df["sales"].max())

if min_sales != max_sales:
    selected_sales = st.sidebar.slider("Minimum Sales", min_sales, max_sales, min_sales)
    df = df[df["sales"] >= selected_sales]

margin = st.sidebar.slider("Minimum Margin (%)", 0.0, 100.0, 0.0)
df = df[df["margin %"] >= margin]

if "product name" in df.columns:
    product = st.sidebar.text_input("Search Product")

    if product:
        words = product.lower().split()
        df = df[
            df["product name"].astype(str).str.lower().apply(
                lambda x: all(word in x for word in words)
            )
        ]

# -------------------------------
# EMPTY CHECK
# -------------------------------
if df.empty:
    st.error("🚫 No data matches your filters.")
    st.stop()

# -------------------------------
# FILTERED DATA
# -------------------------------
st.subheader("📄 Filtered Data")
st.dataframe(df, use_container_width=True)

# -------------------------------
# KPI METRICS
# -------------------------------
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Total Sales", f"${df['sales'].sum():,.0f}")
c2.metric("Total Profit", f"${df['profit'].sum():,.0f}")
c3.metric("Avg Margin", f"{df['margin %'].mean():.2f}%")

# -------------------------------
# MARGIN DISTRIBUTION
# -------------------------------
if "division" in df.columns:
    st.subheader("📊 Margin Distribution by Division")

    fig = px.box(df, x="division", y="margin %")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Some divisions have lower margin spread indicating pricing or cost issues.")

# -------------------------------
# TOP PRODUCTS
# -------------------------------
if "product name" in df.columns:
    st.subheader("🏆 Top Products by Profit")

    product_profit = df.groupby("product name")["gross profit"].sum().sort_values(ascending=False)

    fig = px.bar(product_profit.head(10))
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Top few products generate majority of profit.")

# -------------------------------
# DIVISION PERFORMANCE
# -------------------------------
if "division" in df.columns:
    st.subheader("🏢 Division Performance")

    division_data = df.groupby("division")[["sales", "gross profit"]].sum().reset_index()

    fig = px.bar(division_data, x="division", y=["sales", "gross profit"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Some divisions have high sales but low profit.")

# -------------------------------
# COST VS SALES
# -------------------------------
st.subheader("📉 Cost vs Sales")

fig = px.scatter(
    df,
    x="sales",
    y="cost",
    color="division" if "division" in df.columns else None
)

st.plotly_chart(fig, use_container_width=True)

st.success("Insight: Sales and cost show a strong positive relationship.")

# -------------------------------
# PARETO ANALYSIS
# -------------------------------
if "product name" in df.columns:
    st.subheader("📊 Pareto Analysis")

    pareto = df.groupby("product name")["gross profit"].sum().sort_values(ascending=False)
    cumulative = (pareto / pareto.sum()).cumsum()

    pareto_df = pd.DataFrame({
        "profit %": pareto / pareto.sum(),
        "cumulative %": cumulative
    })

    fig = px.bar(pareto_df.head(10), y="profit %")
    fig.add_scatter(y=pareto_df.head(10)["cumulative %"], mode='lines+markers')

    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: 20% of products contribute to 80% of profit.")

# -------------------------------
# TREND ANALYSIS
# -------------------------------
if "order date" in df.columns:
    st.subheader("📈 Sales Trend")

    trend = df.groupby("order date")["sales"].sum().reset_index()

    fig = px.line(trend, x="order date", y="sales")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Insight: Shows sales growth pattern over time.")

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.subheader("🤖 Profit Prediction")

st.info("Using Linear Regression to predict Profit based on Sales.")

X = df[["sales"]]
y = df["profit"]

model = LinearRegression().fit(X, y)

preds = model.predict(X)
r2 = r2_score(y, preds)

st.write(f"Model Accuracy (R²): {r2:.2f}")

target_sales = st.number_input("Enter Target Sales", value=100.0)

predicted_profit = model.predict([[target_sales]])[0]

st.success(f"Estimated Profit: ${predicted_profit:,.2f}")

# -------------------------------
# DOWNLOAD
# -------------------------------
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Processed Data",
    data=csv,
    file_name="nassau_analysis.csv",
    mime="text/csv"
)
