# streamlit_app.py
# Run:
#   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
#
# Expects:
#   ./data/gdp_data.csv

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Revenue Dashboard", layout="wide")

# ✅ Template-style data load
DATA_FILENAME = Path(__file__).parent / "data/gdp_data.csv"
raw_gdp_df = pd.read_csv(DATA_FILENAME)

# ✅ Clean column headers (handles BOM + whitespace)
raw_gdp_df.columns = raw_gdp_df.columns.str.replace("\ufeff", "", regex=False).str.strip()


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist (and show what we actually received if not)
    required = [
        "Month",
        "Total_Revenue_USD",
        "Subscription_Revenue_USD",
        "API_Revenue_USD",
        "Units",
        "New_Customers",
        "Churned_Customers",
        "Gross_Margin_%"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f"\n\nColumns found: {list(df.columns)}"
        )

    # Parse Month (YYYY-MM)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    if df["Month"].isna().any():
        bad = df.loc[df["Month"].isna(), ["Month"]].head(10)
        raise ValueError(
            "Some Month values could not be parsed as YYYY-MM.\n"
            f"Examples (first 10):\n{bad.to_string(index=False)}"
        )

    # Clean numeric columns (handles commas/$/% if present)
    numeric_cols = [c for c in df.columns if c != "Month"]
    for c in numeric_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort
    df = df.sort_values("Month").reset_index(drop=True)

    # Derived metrics
    df["Net_Customers"] = df["New_Customers"] - df["Churned_Customers"]
    df["Gross_Profit_USD"] = df["Total_Revenue_USD"] * (df["Gross_Margin_%"] / 100.0)

    df["Total_Revenue_MoM_%"] = df["Total_Revenue_USD"].pct_change() * 100.0
    df["Total_Revenue_YoY_%"] = df["Total_Revenue_USD"].pct_change(12) * 100.0

    # Revenue mix
    df["Subscription_Share_%"] = np.where(
        df["Total_Revenue_USD"] > 0,
        (df["Subscription_Revenue_USD"] / df["Total_Revenue_USD"]) * 100.0,
        np.nan,
    )
    df["API_Share_%"] = np.where(
        df["Total_Revenue_USD"] > 0,
        (df["API_Revenue_USD"] / df["Total_Revenue_USD"]) * 100.0,
        np.nan,
    )

    return df


df = clean_data(raw_gdp_df)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Filters")

min_date = df["Month"].min()
max_date = df["Month"].max()

date_range = st.sidebar.slider(
    "Month Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM",
)

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
fdf = df[(df["Month"] >= start) & (df["Month"] <= end)].copy()

metric_options = [
    "Total_Revenue_USD",
    "Subscription_Revenue_USD",
    "API_Revenue_USD",
    "Gross_Profit_USD",
    "Units",
    "New_Customers",
    "Churned_Customers",
    "Net_Customers",
    "Gross_Margin_%",
    "Subscription_Share_%",
    "API_Share_%",
    "Total_Revenue_MoM_%",
    "Total_Revenue_YoY_%"
]

selected_metrics = st.sidebar.multiselect(
    "Select Metrics to Plot",
    options=metric_options,
    default=["Total_Revenue_USD", "Subscription_Revenue_USD", "API_Revenue_USD"],
)

show_table = st.sidebar.checkbox("Show Data Table", value=True)

# -----------------------------
# Header + KPIs
# -----------------------------
st.title("Revenue & Customer Trends Dashboard")

latest = fdf.iloc[-1] if len(fdf) else df.iloc[-1]
prev = fdf.iloc[-2] if len(fdf) >= 2 else None


def money_fmt(x):
    return f"${x:,.0f}" if pd.notna(x) else "—"


def pct_fmt(x):
    return f"{x:,.2f}%" if pd.notna(x) else "—"


k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Revenue (Latest)",
    money_fmt(latest["Total_Revenue_USD"]),
    None
    if prev is None
    else pct_fmt((latest["Total_Revenue_USD"] / prev["Total_Revenue_USD"] - 1) * 100.0),
)

k2.metric(
    "Gross Margin (Latest)",
    pct_fmt(latest["Gross_Margin_%"]),
    None if prev is None else pct_fmt(latest["Gross_Margin_%"] - prev["Gross_Margin_%"]),
)

k3.metric(
    "New Customers (Latest)",
    f"{int(latest['New_Customers']):,}",
    None if prev is None else f"{int(latest['New_Customers'] - prev['New_Customers']):,}",
)

k4.metric(
    "Net Customers (Latest)",
    f"{int(latest['Net_Customers']):,}",
    None if prev is None else f"{int(latest['Net_Customers'] - prev['Net_Customers']):,}",
)

st.divider()

# -----------------------------
# Charts
# -----------------------------
st.subheader("Revenue Breakdown Over Time")
rev_df = fdf.set_index("Month")[["Subscription_Revenue_USD", "API_Revenue_USD"]]
st.area_chart(rev_df, use_container_width=True)

st.subheader("Selected Metrics (Line Chart)")
if selected_metrics:
    chart_df = fdf.set_index("Month")[selected_metrics]
    st.line_chart(chart_df, use_container_width=True)
else:
    st.info("Select at least one metric in the sidebar to display the line chart.")

st.subheader("Customer Movement")
cust_df = fdf.set_index("Month")[["New_Customers", "Churned_Customers", "Net_Customers"]]
st.bar_chart(cust_df, use_container_width=True)

# -----------------------------
# Data Table
# -----------------------------
if show_table:
    st.subheader("Filtered Data Table")
    st.dataframe(fdf, use_container_width=True, hide_index=True)
