# streamlit_app.py
# Run:
#   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
#
# Expects:
#   ./data/gdp_data.csv
#
# Add to requirements.txt:
#   streamlit
#   pandas
#   numpy
#   scikit-learn
#   altair

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(page_title="OpenAIgi | Revenue Dashboard", layout="wide")

# -----------------------------
# Color Pallette
# -----------------------------
OT_BLUE = "#003767"
OT_GOLD = "#FFD24F"
OT_LAVENDER = "#7474C1"
OT_SEA_GREEN = "#007B5F"
OT_PEACOCK = "#00677F"
OT_BRASS = "#AC8400"
OT_BURGUNDY = "#8F3237"

CHART_PALETTE = [OT_BLUE, OT_SEA_GREEN, OT_PEACOCK, OT_LAVENDER, OT_BRASS, OT_BURGUNDY, OT_GOLD]

# Apply Altair theme / palette
alt.themes.enable("default")
OT_SCALE = alt.Scale(range=CHART_PALETTE)

# -----------------------------
# Light CSS polish (optional)
# -----------------------------
st.markdown(
    f"""
    <style>
      /* Header accent */
      h1, h2, h3 {{
        color: {OT_BLUE};
      }}

      /* Sidebar headings */
      section[data-testid="stSidebar"] h2, 
      section[data-testid="stSidebar"] h3 {{
        color: {OT_BLUE};
      }}

      /* Metric label color */
      div[data-testid="stMetricLabel"] {{
        color: {OT_BLUE};
      }}

      /* Small gold accent line under title */
      .ot-accent {{
        height: 6px;
        width: 160px;
        background: {OT_GOLD};
        border-radius: 99px;
        margin: 0.25rem 0 1rem 0;
      }}
    </style>

    <div class="ot-accent"></div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data load (template style)
# -----------------------------
DATA_FILENAME = Path(__file__).parent / "data/gdp_data.csv"
raw_gdp_df = pd.read_csv(DATA_FILENAME)

# Clean column headers (BOM + whitespace)
raw_gdp_df.columns = raw_gdp_df.columns.str.replace("\ufeff", "", regex=False).str.strip()


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    if df["Month"].isna().any():
        bad = df.loc[df["Month"].isna(), ["Month"]].head(10)
        raise ValueError(
            "Some Month values could not be parsed as YYYY-MM.\n"
            f"Examples (first 10):\n{bad.to_string(index=False)}"
        )

    numeric_cols = [c for c in df.columns if c != "Month"]
    for c in numeric_cols:
        df[c] = (
            df[c].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("Month").reset_index(drop=True)

    # Derived metrics
    df["Net_Customers"] = df["New_Customers"] - df["Churned_Customers"]
    df["Gross_Profit_USD"] = df["Total_Revenue_USD"] * (df["Gross_Margin_%"] / 100.0)

    df["Total_Revenue_MoM_%"] = df["Total_Revenue_USD"].pct_change() * 100.0
    df["Total_Revenue_YoY_%"] = df["Total_Revenue_USD"].pct_change(12) * 100.0

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


def make_time_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy().sort_values("Month").reset_index(drop=True)
    out["t"] = np.arange(len(out))
    m = out["Month"].dt.month.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    return out


def fit_forecast(
    d: pd.DataFrame,
    target_col: str,
    horizon: int = 6,
    alpha: float = 1.0,
    test_months: int = 12,
) -> tuple[pd.DataFrame, dict]:
    d = d.dropna(subset=["Month", target_col]).copy()
    d = make_time_features(d)

    feature_cols = ["t", "month_sin", "month_cos"]
    X = d[feature_cols]
    y = d[target_col]

    n = len(d)
    test_size = min(test_months, max(0, n // 4))
    train_end = n - test_size if test_size > 0 else n

    model = Ridge(alpha=alpha)
    model.fit(X.iloc[:train_end], y.iloc[:train_end])

    fitted = model.predict(X)

    last_month = d["Month"].max()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future = pd.DataFrame({"Month": future_months})
    future["t"] = np.arange(n, n + horizon)
    m = future["Month"].dt.month.astype(int)
    future["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    future["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    y_fore = model.predict(future[feature_cols])

    hist_out = d[["Month", target_col]].rename(columns={target_col: "Actual"}).copy()
    hist_out["Fitted"] = fitted
    hist_out["Forecast"] = np.nan

    fut_out = pd.DataFrame(
        {"Month": future["Month"], "Actual": np.nan, "Fitted": np.nan, "Forecast": y_fore}
    )

    forecast_df = pd.concat([hist_out, fut_out], ignore_index=True)

    metrics: dict = {"Backtest_Months": int(test_size)}
    if test_size > 0:
        y_true = y.iloc[train_end:]
        y_pred = pd.Series(fitted[train_end:], index=y_true.index)
        metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))
        metrics["MAPE"] = None if (y_true == 0).any() else float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)

    return forecast_df, metrics


def money_fmt(x):
    return f"${x:,.0f}" if pd.notna(x) else "—"


def pct_fmt(x):
    return f"{x:,.2f}%" if pd.notna(x) else "—"


def altair_multiline(df_in: pd.DataFrame, x_col: str, y_cols: list[str], title: str, y_title: str = ""):
    long = df_in[[x_col] + y_cols].melt(id_vars=[x_col], var_name="Series", value_name="Value")
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:T", title="Month"),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color("Series:N", scale=OT_SCALE, legend=alt.Legend(title="")),
            tooltip=[alt.Tooltip(f"{x_col}:T", title="Month"), "Series:N", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )
    return chart


def altair_area_stacked(df_in: pd.DataFrame, x_col: str, cols: list[str], title: str, y_title: str = ""):
    long = df_in[[x_col] + cols].melt(id_vars=[x_col], var_name="Series", value_name="Value")
    chart = (
        alt.Chart(long)
        .mark_area(opacity=0.9)
        .encode(
            x=alt.X(f"{x_col}:T", title="Month"),
            y=alt.Y("Value:Q", stack="zero", title=y_title),
            color=alt.Color("Series:N", scale=OT_SCALE, legend=alt.Legend(title="")),
            tooltip=[alt.Tooltip(f"{x_col}:T", title="Month"), "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )
    return chart


def altair_bar_grouped(df_in: pd.DataFrame, x_col: str, cols: list[str], title: str, y_title: str = ""):
    long = df_in[[x_col] + cols].melt(id_vars=[x_col], var_name="Series", value_name="Value")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:T", title="Month"),
            xOffset=alt.XOffset("Series:N"),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color("Series:N", scale=OT_SCALE, legend=alt.Legend(title="")),
            tooltip=[alt.Tooltip(f"{x_col}:T", title="Month"), "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )
    return chart


# -----------------------------
# App
# -----------------------------
df = clean_data(raw_gdp_df)

st.title("OpenAI Revenue & Customer Trends Dashboard")

# Sidebar
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

# KPIs
latest = fdf.iloc[-1] if len(fdf) else df.iloc[-1]
prev = fdf.iloc[-2] if len(fdf) >= 2 else None

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Total Revenue (Latest)",
    money_fmt(latest["Total_Revenue_USD"]),
    None if prev is None else pct_fmt((latest["Total_Revenue_USD"] / prev["Total_Revenue_USD"] - 1) * 100.0),
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

# Charts (branded)
st.subheader("Revenue Breakdown Over Time")
st.altair_chart(
    altair_area_stacked(
        fdf, "Month",
        ["Subscription_Revenue_USD", "API_Revenue_USD"],
        "Subscription vs API Revenue",
        "Revenue (USD)"
    ),
    use_container_width=True
)

st.subheader("Selected Metrics (Line Chart)")
if selected_metrics:
    st.altair_chart(
        altair_multiline(fdf, "Month", selected_metrics, "Trends Over Time"),
        use_container_width=True
    )
else:
    st.info("Select at least one metric in the sidebar to display the line chart.")

st.subheader("Customer Movement")
st.altair_chart(
    altair_bar_grouped(
        fdf, "Month",
        ["New_Customers", "Churned_Customers", "Net_Customers"],
        "New vs Churned vs Net Customers",
        "Customers"
    ),
    use_container_width=True
)

# Table
if show_table:
    st.subheader("Filtered Data Table")
    st.dataframe(fdf, use_container_width=True, hide_index=True)

# Predictive modeling
st.divider()
st.header("Predictive Modeling (Forecast)")

target = st.selectbox(
    "Select a metric to forecast",
    options=[
        "Total_Revenue_USD",
        "Subscription_Revenue_USD",
        "API_Revenue_USD",
        "Units",
        "New_Customers",
        "Churned_Customers",
        "Net_Customers",
        "Gross_Profit_USD",
        "Gross_Margin_%",
    ],
    index=0,
)

horizon = st.slider("Forecast horizon (months)", 3, 24, 6)
alpha = st.slider("Model regularization (alpha)", 0.1, 50.0, 1.0)
test_months = st.slider("Backtest window (months)", 0, 24, 12)

forecast_df, metrics = fit_forecast(df, target_col=target, horizon=horizon, alpha=alpha, test_months=test_months)

m1, m2, m3 = st.columns(3)
m1.metric("Backtest months", f"{metrics.get('Backtest_Months', 0)}")
m2.metric("Backtest MAE", "—" if "MAE" not in metrics else f"{metrics['MAE']:,.0f}")
mape = metrics.get("MAPE")
m3.metric("Backtest MAPE", "—" if mape is None else f"{mape:,.2f}%")

st.subheader("Actual vs Fitted vs Forecast")

st.altair_chart(
    altair_multiline(
        forecast_df.drop(columns=[], errors="ignore"),
        "Month",
        ["Actual", "Fitted", "Forecast"],
        f"{target}: Actual / Fitted / Forecast",
        ""
    ),
    use_container_width=True
)

