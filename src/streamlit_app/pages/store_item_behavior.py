import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

st.set_page_config(layout="wide")

# =====================================================
# Load data
# =====================================================
df_daily = load_table(PreDataset.STORE_ITEM_DAILY)
df_weekly = load_table(PreDataset.STORE_ITEM_WEEKLY)


# =====================================================
# Helpers
# =====================================================
def to_dense_index(ts: pd.Series, freq: str) -> pd.Series:
    """
    Make the series dense by creating a complete time index and filling missing periods with 0.
    freq:
      - "D" for daily
      - "W-MON" for weekly (week starts Monday)
    """
    if ts.empty:
        return ts

    ts = ts.sort_index()

    start = ts.index.min()
    end = ts.index.max()

    full_idx = pd.date_range(start=start, end=end, freq=freq)
    return ts.reindex(full_idx, fill_value=0.0)


def get_series_dense(
    df: pd.DataFrame, store: int, item: int, date_col: str, value_col: str, freq: str
) -> pd.Series:
    """
    1) Filter store-item
    2) Aggregate in case there are duplicates
    3) Fill missing time steps with 0 (Notebook-like behavior)
    """
    d = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
    if d.empty:
        return pd.Series(dtype=float)

    # robust datetime / period handling
    if pd.api.types.is_period_dtype(d[date_col]):
        # period -> timestamp (start of period)
        d[date_col] = d[date_col].dt.start_time
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()

    ts = d.groupby(date_col)[value_col].sum().sort_index().astype(float)

    ts = to_dense_index(ts, freq=freq)
    return ts


def seasonality_strength(series: pd.Series, lag: int) -> float:
    """
    ACF at a given lag, on the DENSE series (including zeros).
    """
    if series.empty:
        return np.nan
    if len(series) < lag * 2:
        return np.nan
    vals = acf(series.values, nlags=lag, fft=True)
    return float(vals[lag])


def stl_decompose(series: pd.Series, period: int):
    """
    STL needs enough data and a sensible period.
    For weekly data with many zeros, robust=True helps.
    """
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid


def plot_decomposition(
    ts: pd.Series, trend: pd.Series, seasonal: pd.Series, resid: pd.Series, title: str
):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("unit_sales (dense)", "trend", "seasonal", "residual"),
    )

    fig.add_trace(go.Scatter(x=ts.index, y=ts, name="unit_sales"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=trend, name="trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=seasonal, name="seasonal"), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=ts.index, y=resid, mode="markers", name="residual"), row=4, col=1
    )

    fig.update_layout(height=900, showlegend=False, title=title)
    return fig


def interpret_series(
    mean_val: float, std_val: float, seasonality_val: float
) -> tuple[str, str]:
    """
    Fix for your previous bug:
    you must pass numbers (ts.mean(), ts.std()), not methods (ts.mean, ts.std).
    """
    cv = (std_val / mean_val) if mean_val > 0 else 0.0

    if mean_val < 1:
        demand_type = "Intermittent"
    elif cv > 0.8:
        demand_type = "Volatile"
    else:
        demand_type = "Smooth"

    if np.isnan(seasonality_val):
        seasonal = "seasonality not reliable (too short series)"
    elif seasonality_val > 0.4:
        seasonal = "strong seasonal pattern"
    elif seasonality_val > 0.15:
        seasonal = "moderate seasonal pattern"
    else:
        seasonal = "weak seasonal pattern"

    return demand_type, seasonal


# =====================================================
# UI
# =====================================================
st.title("🔎 Store-Item Behavior EDA (dense series incl. zeros)")

colA, colB, colC, colD = st.columns([1, 1, 1, 1])

with colA:
    level = st.selectbox("Level", ["Daily", "Weekly"])

with colB:
    store = int(st.number_input("Store", min_value=1, value=1, step=1))

with colC:
    item = int(st.number_input("Item", min_value=1, value=1000, step=1))

with colD:
    show_sparse = st.checkbox("Also show sparse series (sales-days only)", value=False)

# =====================================================
# Select dataset settings
# =====================================================
if level == "Daily":
    df = df_daily
    date_col = "date"
    value_col = "unit_sales_sum"
    freq = "D"
    period = 7
    seasonal_label = "Weekly"
    season_lag = 7
else:
    df = df_weekly
    # your weekly table might have either "week_start" or "week"
    date_col = "week_start" if "week_start" in df.columns else "week"
    value_col = "unit_sales_sum"
    freq = "W-MON"
    period = 52
    seasonal_label = "Yearly"
    season_lag = 52

# =====================================================
# Extract series (DENSE like notebook)
# =====================================================
ts_dense = get_series_dense(df, store, item, date_col, value_col, freq=freq)

if ts_dense.empty:
    st.warning("No data for this store-item combination")
    st.stop()

# Optional sparse series (only observed sales periods)
if show_sparse:
    d0 = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
    if not d0.empty:
        if pd.api.types.is_period_dtype(d0[date_col]):
            d0[date_col] = d0[date_col].dt.start_time
        d0[date_col] = pd.to_datetime(d0[date_col]).dt.normalize()
        ts_sparse = d0.groupby(date_col)[value_col].sum().sort_index().astype(float)
    else:
        ts_sparse = pd.Series(dtype=float)

# =====================================================
# STL on dense series
# =====================================================
# STL can fail if too short. Guard it.
if len(ts_dense) < period * 2:
    trend = ts_dense.copy() * np.nan
    seasonal = ts_dense.copy() * np.nan
    resid = ts_dense.copy() * np.nan
    season_strength = np.nan
else:
    trend, seasonal, resid = stl_decompose(ts_dense, period)
    season_strength = seasonality_strength(ts_dense, season_lag)

# =====================================================
# Stats
# =====================================================
mean_val = float(ts_dense.mean())
std_val = float(ts_dense.std(ddof=1)) if len(ts_dense) > 1 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Observations", f"{len(ts_dense):,}")
col2.metric("Mean", f"{mean_val:.2f}")
col3.metric("Std", f"{std_val:.2f}")
col4.metric(
    f"{seasonal_label} seasonality (ACF@{season_lag})",
    "—" if np.isnan(season_strength) else f"{season_strength:.3f}",
)
col5.metric("Zero share", f"{(ts_dense.eq(0).mean()):.1%}")

with st.expander("📊 How to interpret these time-series stats"):
    st.markdown(
        f"""
**Observations**
Number of time points in the dense series (missing periods filled with 0).

**Mean**
Average demand per period (including zeros).

**Std**
Demand variability across the whole timeline (including zeros).
A handy relative measure is **CV = Std / Mean**.

**{seasonal_label} seasonality (ACF@{season_lag})**
Autocorrelation at lag `{season_lag}` computed on the **dense** series.
Higher means a stronger repeating pattern at that cycle.

**Zero share**
Share of periods with exactly 0 demand (only meaningful with dense fill).
"""
    )

dtype, seasonal_txt = interpret_series(
    mean_val, std_val, season_strength if not np.isnan(season_strength) else np.nan
)
st.info(f"Series type: **{dtype}** · {seasonal_txt}")

# =====================================================
# Plot
# =====================================================
st.plotly_chart(
    plot_decomposition(
        ts_dense,
        trend,
        seasonal,
        resid,
        title=f"{level} decomposition (dense incl. zeros) · store={store} · item={item}",
    ),
    use_container_width=True,
)

if show_sparse and not ts_sparse.empty:
    st.subheader("🧾 Sparse series (sales periods only)")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=ts_sparse.index, y=ts_sparse.values, mode="lines+markers")
    )
    fig2.update_layout(height=300, title="Sparse series (no zero-filled periods)")
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# Interpretation hints
# =====================================================
with st.expander("Model interpretation"):
    st.markdown(
        """
**High seasonal component / high ACF at seasonal lag**
→ SARIMA / seasonal models likely useful

**Strong trend**
→ boosting / trend models useful

**High residual noise**
→ difficult series; consider aggregation (weekly) or stronger regularization

**High zero share + intermittent**
→ Croston-type approaches or intermittent-demand features
"""
    )
