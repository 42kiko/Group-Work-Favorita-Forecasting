import numpy as np
import pandas as pd
import streamlit as st

from Favorita_TSA.preprocess_eda import load_table
from Favorita_TSA.utils.dataset import PreDataset

# -------------------------------------------------
# Load data (cached by your loader)
# -------------------------------------------------
df_store_item_daily = load_table(PreDataset.STORE_ITEM_DAILY)


# def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
#     return df.groupby(group_cols)


# def store_item_daily(df: pd.DataFrame) -> pd.DataFrame:
#     return aggregate(df, ["store_nbr", "item_nbr", "date", "unit_sales"])


# store_item_daily_new = store_item_daily(load_fact_table())

# st.dataframe(store_item_daily_new.head())


st.subheader("🧾 Item Store PreProcess Decision Table")

# table_cols = [
#     "item_nbr",
#     "item_family",
#     "item_class",
#     "perishable",
#     "active_days",
#     "active_span_days",
#     "sales_density",
#     "total_units",
#     "avg_units_per_day",
# ]


# st.dataframe(
#     item_summary[table_cols],
#     use_container_width=True,
# )


def add_forecastability_metrics_daily(
    df: pd.DataFrame,
    store_col: str = "store_nbr",
    item_col: str = "item_nbr",
    date_col: str = "date",
    y_col: str = "unit_sales_sum",
) -> pd.DataFrame:
    """
    Erwartet: df enthält nur Tage mit Verkäufen (unit_sales>0). 0-Tage fehlen.
    Logik: Store ist 'offen', wenn an dem Tag irgendein Item verkauft wurde.
    Output: 1 Zeile pro (store,item) mit n_periods, n_nonzero, zero_rate, adi, cv2.
    """

    d = df[[store_col, item_col, date_col, y_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()

    # 1) Store-Open-Days: alle Tage, an denen im Store irgendein Verkauf stattfand
    store_days = (
        d[[store_col, date_col]].drop_duplicates().sort_values([store_col, date_col])
    )

    # Mapping store -> Index der offenen Tage
    store_to_days = {
        s: g[date_col].to_numpy() for s, g in store_days.groupby(store_col, sort=False)
    }

    # 2) Verkäufe auf Tagesebene pro store-item-date (falls mehrere Zeilen pro Tag existieren)
    sales = d.groupby([store_col, item_col, date_col], as_index=False)[y_col].sum()

    # 3) Metriken pro (store,item) berechnen (mit Reindex auf Store-Open-Days)
    rows = []
    for (s, it), g in sales.groupby([store_col, item_col], sort=False):
        open_days = store_to_days.get(s)
        if open_days is None or len(open_days) == 0:
            # sollte praktisch nicht vorkommen, aber robust halten
            y = np.array([], dtype=float)
        else:
            # Reindex auf open_days
            g2 = g.set_index(date_col).reindex(open_days)
            y = g2[y_col].fillna(0.0).to_numpy(dtype=float)

        n_periods = int(len(y))
        n_nonzero = int((y > 0).sum())
        total_units = float(y.sum())
        zero_rate = float((y == 0).mean()) if n_periods > 0 else np.nan
        adi = float(n_periods / n_nonzero) if n_nonzero > 0 else np.inf

        mean = float(y.mean()) if n_periods > 0 else np.nan
        std = float(y.std(ddof=1)) if n_periods > 1 else 0.0
        cv2 = float((std / mean) ** 2) if mean > 0 else np.inf

        rows.append(
            {
                store_col: s,
                item_col: it,
                "n_periods": n_periods,
                "n_nonzero": n_nonzero,
                "zero_rate": zero_rate,
                "adi": adi,
                "cv2": cv2,
                "total_units": total_units,  # ✅ HIER
            }
        )

    metrics = pd.DataFrame(rows)
    return metrics


# --- Beispiel ---
# metrics_df = add_forecastability_metrics_daily(df)


metrics_df = add_forecastability_metrics_daily(df_store_item_daily)


st.dataframe(
    metrics_df.head(20),
    use_container_width=True,
)
