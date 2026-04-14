import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

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
file_name = uploaded_file.name

if file_name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)

elif file_name.endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

# -------------------------------
# DATA CLEANING
# -------------------------------
df = df.drop_duplicates()

for col in ["Sales", "Cost", "Gross Profit"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=["Sales", "Cost", "Gross Profit"])

# Derived columns
df["Profit"] = df["Sales"] - df["Cost"]
df["Margin %"] = (df["Profit"] / df["Sales"]) * 100

# Convert date
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

# -------------------------------
# SHOW CLEAN DATA (BEFORE FILTERS ✅)
# -------------------------------
df_original = df.copy()

st.subheader("📄 Dataset Preview(Before Filters)")
st.dataframe(df_original, use_container_width=True)

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

# Date filter
if "Order Date" in df.columns:
    min_date = df["Order Date"].min()
    max_date = df["Order Date"].max()
    dates = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )

    start_date, end_date = min_date, max_date
    if len(dates) == 2:
        start_date, end_date = dates
    df = df[
        (df["Order Date"] >= pd.to_datetime(start_date)) &
        (df["Order Date"] <= pd.to_datetime(end_date))
    ]

# Division filter
if "Division" in df.columns:
    division = st.sidebar.selectbox(
        "Select Division",
        ["All"] + list(df["Division"].dropna().unique())
    )
    if division != "All":
        df = df[df["Division"] == division]

# Sales filter (ADDED ✅)
min_val = df["Sales"].min()
max_val = df["Sales"].max()

if min_val == max_val:
    st.sidebar.write("Only one sales value available:", min_val)
    min_sales = min_val
else:
    min_sales = st.sidebar.slider(
        "Select minimum sales",
        float(min_val),
        float(max_val),
        float(min_val)
    )

# Margin filter
margin = st.sidebar.slider("Minimum Gross Margin (%)", 0.0, 100.0, 0.0)
df = df[df["Margin %"] >= margin]

# Product search (FINAL FIXED ✅)
product = st.sidebar.text_input("Search Product")

if product:
    search_words = product.lower().split()

    def match_all(text):
        text = str(text).lower()
        return all(word in text for word in search_words)

    df = df[df["Product Name"].apply(match_all)]

# -------------------------------
# SHOW FILTERED DATA
# -------------------------------
st.subheader("📄 Filtered Data")
st.dataframe(df, use_container_width=True)

st.write("Rows after filter:", df.shape)

# -------------------------------
# KPI METRICS
# -------------------------------
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
c2.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
c3.metric("Avg Margin", f"{df['Margin %'].mean():.2f}%")

# -------------------------------
# TOP PRODUCTS
# -------------------------------
st.subheader("🏆 Top Products by Profit")

product_profit = df.groupby("Product Name")["Gross Profit"].sum().sort_values(ascending=False)

fig1 = px.bar(product_profit.head(10), title="Top 10 Products by Profit")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# DIVISION PERFORMANCE
# -------------------------------
st.subheader("🏢 Division Performance")

if "Division" in df.columns:
    division_data = df.groupby("Division")[["Sales", "Gross Profit"]].sum().reset_index()

    fig2 = px.bar(division_data,
                  x="Division",
                  y=["Sales", "Gross Profit"],
                  barmode="group",
                  title="Division Sales vs Profit")

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# COST VS SALES
# -------------------------------
st.subheader("📉 Cost vs Sales")

color_col = "Division" if "Division" in df.columns else None

fig3 = px.scatter(df,
                  x="Sales",
                  y="Cost",
                  color=color_col,
                  title="Sales vs Cost")

st.plotly_chart(fig3, use_container_width=True)

st.success(
    "Insight: The scatter plot indicates a strong positive relationship between sales and cost. "
    "This suggests a stable cost structure with minor inefficiencies."
)

# -------------------------------
# PARETO ANALYSIS
# -------------------------------
st.subheader("📊 Pareto Analysis")

pareto = df.groupby("Product Name")["Gross Profit"].sum().sort_values(ascending=False)
pareto_pct = pareto / pareto.sum()
cumulative = pareto_pct.cumsum()

pareto_df = pareto_pct.to_frame(name="Profit %")
pareto_df["Cumulative %"] = cumulative

fig4 = px.bar(pareto_df.head(10), y="Profit %", title="Top Products Contribution")

fig4.add_scatter(y=pareto_df.head(10)["Cumulative %"],
                 mode='lines+markers',
                 name="Cumulative %")

st.plotly_chart(fig4, use_container_width=True)

top_80 = pareto_df[pareto_df["Cumulative %"] <= 0.8]
st.success(f"👉 {len(top_80)} products generate ~80% of total profit")

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.subheader("🤖 Profit Prediction")

X = df[["Sales"]]
y = df["Profit"]

model = LinearRegression().fit(X, y)

target_sales = st.number_input("Enter Target Sales", value=100.0)
predicted_profit = model.predict(np.array([[target_sales]]))[0]

st.info(f"If Sales = ${target_sales:,.2f}, Estimated Profit = ${predicted_profit:,.2f}")

fig5 = px.scatter(df, x="Sales", y="Profit", title="ML Trend Line")
fig5.add_scatter(x=df["Sales"], y=model.predict(X),
                 mode='lines', name="Trend")

st.plotly_chart(fig5, use_container_width=True)

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
