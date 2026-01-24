import plotly.express as px
import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.data_loader import parquet_loader
from Favorita_TSA.utils.dataset import Dataset, PreDataset

st.set_page_config(layout="wide")
st.header("📦 Item Viability & Forecast Readiness")

# -------------------------------------------------
# Load data (cached by your loader)
# -------------------------------------------------
df = load_table(PreDataset.ITEM_DAILY)
items = parquet_loader(Dataset.ITEMS)  # item_nbr, item_name, family, class

# -------------------------------------------------
# Build ITEM summary (GLOBAL)
# -------------------------------------------------
item_summary = (
    df[df["unit_sales_sum"] > 0]
    .groupby("item_nbr")
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        active_days=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
    )
    .reset_index()
)

item_summary["active_span_days"] = (
    item_summary["last_date"] - item_summary["first_date"]
).dt.days + 1

item_summary["sales_density"] = (
    item_summary["active_days"] / item_summary["active_span_days"]
)

item_summary["avg_units_per_day"] = (
    item_summary["total_units"] / item_summary["active_days"]
)

# -------------------------------------------------
# Join item meta
# -------------------------------------------------
item_summary = item_summary.merge(items, on="item_nbr", how="left")


item_summary = item_summary.rename(
    columns={
        "family": "item_family",
        "class": "item_class",
    }
)

# -------------------------------------------------
# Threshold controls
# -------------------------------------------------
st.subheader("🎛️ Filter Thresholds")

col1, col2, col3 = st.columns(3)

min_days = int(item_summary["active_days"].min())
max_days = int(item_summary["active_days"].max())


with col1:
    min_active_days = st.slider(
        "Min active days (absolute)",
        min_days,
        max_days,
        min_days,
        1,
    )

with col2:
    min_density = st.slider(
        "Min sales density (percentile)",
        0.0,
        0.5,
        0.0,
        0.05,
    )

with col3:
    min_units = st.slider(
        "Min total units (percentile)",
        0.0,
        0.5,
        0.0,
        0.05,
    )

# -------------------------------------------------
# Compute cutoffs
# -------------------------------------------------
# cut_active = item_summary["active_days"].quantile(min_active_days)
cut_density = item_summary["sales_density"].quantile(min_density)
cut_units = item_summary["total_units"].quantile(min_units)

item_summary["forecast_viable"] = (
    (item_summary["active_days"] >= min_active_days)
    & (item_summary["sales_density"] >= cut_density)
    & (item_summary["total_units"] >= cut_units)
)

# -------------------------------------------------
# Scatter: Density vs Active Days
# -------------------------------------------------
st.subheader("📊 Item Distribution")

fig = px.scatter(
    item_summary,
    x="active_days",
    y="sales_density",
    color="forecast_viable",
    hover_data=[
        "item_nbr",
        "item_family",
        "item_class",
        "perishable",
        "total_units",
        "avg_units_per_day",
    ],
    title="Item Stability Map",
)


fig.update_xaxes(type="log", title="Active Days (log)")
fig.update_yaxes(title="Sales Density")
fig.update_traces(marker={"size": 6, "opacity": 0.6})

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Table: Decision-ready view
# -------------------------------------------------
st.subheader("🧾 Item Decision Table")

table_cols = [
    "item_nbr",
    "item_family",
    "item_class",
    "perishable",
    "active_days",
    "active_span_days",
    "sales_density",
    "total_units",
    "avg_units_per_day",
    "forecast_viable",
]


st.dataframe(
    item_summary[table_cols].sort_values("forecast_viable", ascending=False),
    use_container_width=True,
)

# -------------------------------------------------
# KPIs
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total items", len(item_summary))
col2.metric(
    "Forecast viable",
    int(item_summary["forecast_viable"].sum()),
)
col3.metric(
    "Share viable",
    f"{item_summary['forecast_viable'].mean():.1%}",
)

st.markdown(
    """
**How to use this page**
- Bottom-left = weak, sparse items → drop first  
- High density + many days = ideal forecast candidates  
- Use percentiles instead of fixed numbers to stay data-driven
"""
)
