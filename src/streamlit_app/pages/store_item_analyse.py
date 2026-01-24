import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.dataset import PreDataset

st.title("Store  Item Analysis")

df_store_daily = load_table(PreDataset.STORE_DAILY)
df_store_weekly = load_table(PreDataset.STORE_WEEKLY)
df_store_monthly = load_table(PreDataset.STORE_MONTHLY)

# Make timeseries columns for plotting
df_store_weekly["week_ts"] = df_store_weekly["week"].dt.start_time
df_store_monthly["month_ts"] = df_store_monthly["month"].dt.to_timestamp()
