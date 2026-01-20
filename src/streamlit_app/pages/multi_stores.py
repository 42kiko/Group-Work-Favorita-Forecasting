import plotly.express as px
import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.dataset import PreDataset

st.title("Store Analysis")

df_store_daily = load_table(PreDataset.STORE_DAILY)
df_store_weekly = load_table(PreDataset.STORE_WEEKLY)
df_store_monthly = load_table(PreDataset.STORE_MONTHLY)

# Make timeseries columns for plotting
df_store_weekly["week_ts"] = df_store_weekly["week"].dt.start_time
df_store_monthly["month_ts"] = df_store_monthly["month"].dt.to_timestamp()


store_ids = st.multiselect(
    "Select stores",
    options=sorted(df_store_daily["store_nbr"].unique()),
    default=[1],  # optional
)

if not store_ids:
    st.info("Please select at least one store.")
    st.stop()


df_store_daily_choice = df_store_daily[df_store_daily["store_nbr"].isin(store_ids)]
df_store_weekly_choice = df_store_weekly[df_store_weekly["store_nbr"].isin(store_ids)]
df_store_monthly_choice = df_store_monthly[
    df_store_monthly["store_nbr"].isin(store_ids)
]


fig_daily = px.line(
    df_store_daily_choice,
    x="date",
    y="unit_sales_sum",
    color="store_nbr",
    title="Daily Sales",
)
st.plotly_chart(fig_daily, use_container_width=True)


fig_weekly = px.line(
    df_store_weekly_choice,
    x="week_ts",
    y="unit_sales_sum",
    color="store_nbr",
    title="Weekly Sales",
)
st.plotly_chart(fig_weekly, use_container_width=True)


fig_monthly = px.line(
    df_store_monthly_choice,
    x="month_ts",
    y="unit_sales_sum",
    color="store_nbr",
    title="Monthly Sales",
)
st.plotly_chart(fig_monthly, use_container_width=True)
