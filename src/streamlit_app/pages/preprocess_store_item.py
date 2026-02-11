import numpy as np
import pandas as pd
import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.data_loader import parquet_loader
from Favorita_TSA.utils.dataset import Dataset, PreDataset

# -------------------------------------------------
# Load data
# -------------------------------------------------
df_store_item_daily = load_table(PreDataset.STORE_ITEM_DAILY)


# -------------------------------------------------
# Cached item meta with emojis
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_item_meta():
    items = parquet_loader(Dataset.ITEMS)

    FAMILY_EMOJI = {
        "GROCERY": "🥫",
        "BEVERAGES": "🥤",
        "CLEANING": "🧽",
        "DAIRY": "🥛",
        "PRODUCE": "🥕",
        "MEATS": "🥩",
        "BREAD/BAKERY": "🍞",
        "DELI": "🧀",
        "PERSONAL CARE": "🧴",
        "HOME CARE": "🏠",
    }

    items["emoji"] = items["family"].map(FAMILY_EMOJI).fillna("📦")

    # nur schöner Anzeigename
    items["item_label"] = items["item_nbr"].astype(str)

    return items[
        [
            "item_nbr",
            "item_label",
            "family",
            "class",
            "emoji",
            "perishable",
        ]
    ]


# -------------------------------------------------
# Forecastability metrics (FAST + cached)
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def build_store_item_metrics(
    df: pd.DataFrame,
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    date_col: str = "date",
    sales_col: str = "unit_sales_sum",
) -> pd.DataFrame:
    """
    Creates forecastability metrics for each (store, item).

    Core idea:
    We only see days WITH sales → no explicit zero days.
    So we use the item lifecycle (first → last sale) as timeline.

    Metrics:
    -------
    active_span_days        = lifecycle length (days between first and last sale)
    days_sold               = days with sales
    sales_density           = days_sold / active_span_days
    zero_rate               = share of days without sales
    adi                     = average days between sales (gap length)
    cv2                     = demand volatility
    total_units             = total historical units sold
    """

    d = df[[store_col, item_col, date_col, sales_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()

    agg = (
        d.groupby([store_col, item_col])
        .agg(
            first_sale=(date_col, "min"),
            last_sale=(date_col, "max"),
            days_sold=(date_col, "count"),
            total_units=(sales_col, "sum"),
            mean_sales=(sales_col, "mean"),
            std_sales=(sales_col, "std"),
        )
        .reset_index()
    )

    # -----------------------------
    # Lifecycle
    # -----------------------------
    agg["active_span_days"] = (agg["last_sale"] - agg["first_sale"]).dt.days + 1

    # -----------------------------
    # Core metrics (same math as before)
    # -----------------------------
    agg["sales_density"] = agg["days_sold"] / agg["active_span_days"]

    agg["zero_rate"] = 1 - agg["sales_density"]

    agg["adi"] = agg["active_span_days"] / agg["days_sold"]

    agg["cv2"] = (agg["std_sales"] / agg["mean_sales"]) ** 2

    agg.replace([np.inf, -np.inf], np.nan, inplace=True)

    return agg[
        [
            store_col,
            item_col,
            "first_sale",
            "last_sale",
            "active_span_days",
            "days_sold",
            "sales_density",
            "zero_rate",
            "adi",
            "cv2",
            "total_units",
        ]
    ]


# -------------------------------------------------
# Build metrics
# -------------------------------------------------
metrics_df = build_store_item_metrics(df_store_item_daily)

item_meta = load_item_meta()

meta_cols = ["item_label", "family", "class", "emoji", "perishable"]

for col in meta_cols:
    metrics_df[col] = metrics_df["item_nbr"].map(
        dict(zip(item_meta["item_nbr"], item_meta[col], strict=False))
    )

st.subheader("🧾 Store - Item Forecastability Table")

col1, col2, col3 = st.columns(3)

with col1:
    fam_filter = st.multiselect(
        "Family",
        sorted(metrics_df["family"].dropna().unique()),
        default=None,
    )

with col2:
    perishable_filter = st.selectbox(
        "Perishable",
        options=["All", True, False],
        index=0,
    )

with col3:
    min_density = st.slider("Min Density", 0.0, 1.0, 0.1, 0.05)


df_view = metrics_df.copy()

if fam_filter:
    df_view = df_view[df_view["family"].isin(fam_filter)]

if perishable_filter != "All":
    df_view = df_view[df_view["perishable"] == perishable_filter]

df_view = df_view[df_view["sales_density"] >= min_density]
# -------------------------------------------------
# Guide (aligned to your column names)
# -------------------------------------------------
with st.expander("📘 Metric Guide"):
    st.markdown(
        """
### Column meanings

**active_span_days**
Days between first and last sale
→ item lifetime in this store

**days_sold**
Number of days the item sold

**sales_density**
days_sold / active_span_days
→ higher = more regular

**zero_rate**
share of days without sales
→ higher = more intermittent

**adi (Average Demand Interval)**
average days between sales
→ higher = rarer demand

**cv2 (Demand volatility)**
variance of demand
→ higher = noisy / unstable

**total_units**
total historical volume


### Rule of thumb

Low ADI + low CV² → very forecastable
High ADI → intermittent
High CV² → volatile
Both high → hard to forecast
"""
    )


# -------------------------------------------------
# Display
# -------------------------------------------------
display_cols = [
    "store_nbr",
    "item_label",
    "emoji",
    "family",
    "class",
    "perishable",
    "total_units",
    "active_span_days",
    "days_sold",
    "sales_density",
    "adi",
    "cv2",
]

st.dataframe(
    df_view[display_cols].sort_values("total_units", ascending=False),
    use_container_width=True,
)
