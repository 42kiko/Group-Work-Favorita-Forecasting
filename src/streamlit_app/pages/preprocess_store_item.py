import numpy as np
import pandas as pd
import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.data_loader import parquet_loader
from Favorita_TSA.utils.dataset import Dataset, PreDataset

st.set_page_config(layout="wide")
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

    # -----------------------------
    # Croston demand type (vectorized)
    # -----------------------------
    ADI_THR = 1.32
    CV2_THR = 0.49

    agg["pattern"] = np.select(
        [
            (agg["adi"] <= ADI_THR) & (agg["cv2"] <= CV2_THR),
            (agg["adi"] <= ADI_THR) & (agg["cv2"] > CV2_THR),
            (agg["adi"] > ADI_THR) & (agg["cv2"] <= CV2_THR),
            (agg["adi"] > ADI_THR) & (agg["cv2"] > CV2_THR),
        ],
        [
            "Smooth",
            "Erratic",
            "Intermittent",
            "Lumpy",
        ],
        default="Unknown",
    )

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
            "pattern",
        ]
    ]


# -------------------------------------------------
# Build metrics
# -------------------------------------------------
metrics_df = build_store_item_metrics(df_store_item_daily)

PATTERN_EMOJI = {
    "Smooth": "🟢",
    "Erratic": "🟡",
    "Intermittent": "🟠",
    "Lumpy": "🔴",
    "Unknown": "⚪",
}

metrics_df["pattern_emoji"] = metrics_df["pattern"].map(PATTERN_EMOJI)

metrics_df["pattern_label"] = metrics_df["pattern_emoji"] + " " + metrics_df["pattern"]

item_meta = load_item_meta()

meta_cols = ["item_label", "family", "class", "emoji", "perishable"]

for col in meta_cols:
    metrics_df[col] = metrics_df["item_nbr"].map(
        dict(zip(item_meta["item_nbr"], item_meta[col], strict=False))
    )


metrics_df["perishable"] = metrics_df["perishable"].astype(bool)

st.subheader("🧾 Store - Item Forecastability Table")

col1, col2, col3, col4 = st.columns(4)

with col1:
    fam_filter = st.multiselect(
        "Family",
        sorted(metrics_df["family"].dropna().unique()),
    )

with col2:
    pattern_filter = st.multiselect(
        "Demand Type",
        ["Smooth", "Erratic", "Intermittent", "Lumpy"],
    )

with col3:
    perishable_filter = st.selectbox(
        "Perishable",
        ["All", True, False],
    )

with col4:
    min_density = st.slider(
        "Min density (sales_density)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )

df_view = metrics_df.copy()

if fam_filter:
    df_view = df_view[df_view["family"].isin(fam_filter)]

if pattern_filter:
    df_view = df_view[df_view["pattern"].isin(pattern_filter)]

if perishable_filter != "All":
    df_view = df_view[df_view["perishable"] == perishable_filter]

# ✅ density filter
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
    "pattern_label",
    "perishable",
    "total_units",
    "active_span_days",
    "days_sold",
    "sales_density",
    "adi",
    "cv2",
]

st.data_editor(
    df_view[display_cols].sort_values("total_units", ascending=False),
    use_container_width=True,
    hide_index=True,
    disabled=display_cols,
    column_config={
        "total_units": st.column_config.NumberColumn(label="Units sold", format="{:,}"),
        "sales_density": st.column_config.NumberColumn(
            label="Density", format="{:.2f}"
        ),
        "adi": st.column_config.NumberColumn(label="ADI", format="{:.2f}"),
        "cv2": st.column_config.NumberColumn(label="CV²", format="{:.2f}"),
        "perishable": st.column_config.CheckboxColumn(label="Perishable"),
    },
)


col1, col2, col3 = st.columns(3)

col1.metric("Rows shown", f"{len(df_view):,}")
col2.metric("Unique stores", df_view["store_nbr"].nunique())
col3.metric("Unique items", df_view["item_nbr"].nunique())

st.divider()


st.metric("Smooth share", f"{(df_view['pattern']=='Smooth').mean():.1%}")
