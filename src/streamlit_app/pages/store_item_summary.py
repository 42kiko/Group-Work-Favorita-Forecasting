from __future__ import annotations

import pandas as pd
import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.dataset import PreDataset

st.set_page_config(page_title="Store Item Summary", layout="wide")
st.title("Stor Item Event Summary")
st.caption("Event-based metrics (no explicit zero-sales days in source data).")


@st.cache_data(show_spinner=False)
def build_store_item_event_summary() -> pd.DataFrame:
    df = load_table(PreDataset.STORE_ITEM_DAILY).copy()

    # Ensure expected columns exist
    required = {"date", "store_nbr", "item_nbr", "unit_sales_sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Only sold events (should already be true, but keep it robust)
    df = df.loc[
        df["unit_sales_sum"] > 0, ["date", "store_nbr", "item_nbr", "unit_sales_sum"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sort for gap calculation
    df = df.sort_values(["store_nbr", "item_nbr", "date"])

    # Gaps (days between sold events)
    df["gap_days"] = df.groupby(["store_nbr", "item_nbr"])["date"].diff().dt.days

    # Aggregate
    agg = df.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        first_sale_date=("date", "min"),
        last_sale_date=("date", "max"),
        days_sold=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
        mean_sales=("unit_sales_sum", "mean"),
        std_sales=("unit_sales_sum", "std"),
        median_gap_days=("gap_days", "median"),
        p90_gap_days=("gap_days", lambda s: s.quantile(0.90)),
    )

    # Derived metrics
    agg["active_span_days"] = (
        agg["last_sale_date"] - agg["first_sale_date"]
    ).dt.days + 1
    agg["sales_frequency"] = agg["days_sold"] / agg["active_span_days"]

    # CV (avoid divide-by-zero)
    agg["cv"] = agg["std_sales"] / agg["mean_sales"].replace(0, pd.NA)

    # Clean types for display
    agg["median_gap_days"] = agg["median_gap_days"].fillna(0).astype("int64")
    agg["p90_gap_days"] = agg["p90_gap_days"].fillna(0).astype("int64")

    return agg


df_summary = build_store_item_event_summary()

st.subheader("Preview")
st.dataframe(
    df_summary,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Rows: {len(df_summary):,} | Unique stores: {df_summary['store_nbr'].nunique():,} | Unique items: {df_summary['item_nbr'].nunique():,}"
)
