"""
sarimax_model.py

Interaktive Streamlit-Seite für SARIMAX-Modellierung mit exogenen Features.
Manuelle Einstellung aller ARIMA-Parameter (p,d,q)(P,D,Q,s).
Feature-Gruppen können per Checkbox selektiert werden.
Tägliche/wöchentliche Features werden entsprechend aggregiert.

Voraussetzung:
  - data/metrics/*.parquet vorhanden (Forecastability-Matrizen)
  - data/processed/*.parquet vorhanden (Oil, Transactions, Stores, Items)
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd
import streamlit as st

from Favorita_TSA.features.sarima_features import load_sarimax_segment
from Favorita_TSA.models.sarimax import run_sarimax_plotly
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="SARIMAX Model")
set_plotly_theme()

os.chdir(MLRUNS_DIR.parent.parent)  # PROJECT_ROOT

EXPERIMENT = cfg.mlflow.sarimax_experiment

_pe = cfg.defaults.pattern_examples
PATTERN_DEFAULTS: dict[str, dict] = {
    "daily_smooth": vars(_pe.daily_smooth),
    "daily_erratic": vars(_pe.daily_erratic),
    "weekly_smooth": vars(_pe.weekly_smooth),
    "weekly_erratic": vars(_pe.weekly_erratic),
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature-Gruppen (Name → Liste der Spalten)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "Holidays": ["is_holiday_or_event", "pre_holiday", "post_holiday"],
    "Oil Price": [
        "oil_price",
        "oil_price_ma7",
        "oil_price_ma28",
        "oil_price_pct_change",
    ],
    "Calendar": [
        "is_weekend",  # nur täglich
        "is_payday",
        "is_month_start",
        "is_month_end",
        "days_to_next_holiday",
        "days_since_last_holiday",
    ],
    "Transactions": ["transactions", "transactions_ma7", "transactions_z_score"],
    "Promotion": ["onpromotion", "promo_streak", "promo_rate_7d"],
    "Store / Item": ["store_cluster", "perishable", "store_type", "family"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(
    show_spinner="Lade und enriche Segment-Daten (einmalig, kann ~60 s dauern) …"
)
def _load_segment(pattern: str) -> pd.DataFrame:
    return load_sarimax_segment(pattern)


@st.cache_data(ttl=15, show_spinner=False)
def _load_mlflow_runs() -> pd.DataFrame:
    setup_mlflow(EXPERIMENT)
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT],
            order_by=["start_time DESC"],
        )
    except Exception:
        return pd.DataFrame()

    wanted = [
        "run_id",
        "tags.mlflow.runName",
        "params.pattern",
        "params.store",
        "params.item",
        "params.freq",
        "params.p",
        "params.d",
        "params.q",
        "params.s_p",
        "params.s_d",
        "params.s_q",
        "params.s",
        "params.n_exog_features",
        "metrics.mae_primary",
        "metrics.mae_naive",
        "metrics.r2_primary",
        "metrics.improvement_pct",
        "start_time",
    ]
    existing = [c for c in wanted if c in runs.columns]
    return runs[existing].head(30)


# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("SARIMAX Model")
st.caption(
    "Manuelle SARIMA-Parameter + auswählbare exogene Features — "
    "alle Runs werden in MLflow und img/mlflow/ gespeichert."
)

st.divider()

# ── Zeile 1: Daten-Auswahl ───────────────────────────────────────────────────

st.subheader("Daten")

col_pat, col_store, col_item, col_test, col_trim = st.columns(5)

with col_pat:
    pattern = st.selectbox(
        "Demand Pattern",
        options=list(PATTERN_DEFAULTS.keys()),
        index=0,
    )

defaults = PATTERN_DEFAULTS[pattern]
is_daily = "daily" in pattern
is_weekly = not is_daily

with col_store:
    store = st.number_input(
        "Store Nr.",
        min_value=1,
        max_value=54,
        value=defaults["store"],
        step=1,
    )

with col_item:
    item = st.number_input(
        "Item Nr.",
        min_value=1,
        value=defaults["item"],
        step=1,
    )

with col_test:
    test_weeks = st.slider(
        "Test Weeks",
        min_value=1,
        max_value=52,
        value=defaults["test_weeks"],
        help="Anzahl der Wochen im Test-Set",
    )

with col_trim:
    trim_days = st.slider(
        "Trim Zero-Phase",
        min_value=0,
        max_value=90,
        value=0,
        step=5,
        format="%d Tage",
        disabled=is_weekly,
        help="Entfernt den letzten langen Null-Block am Ende der Zeitreihe. "
        "0 = deaktiviert. Nur für Daily-Muster relevant.",
    )
trailing_zero_min_days = int(trim_days) if is_daily else 0

st.divider()

# ── Zeile 2: SARIMAX Parameter ───────────────────────────────────────────────

st.subheader("SARIMAX Parameter (p,d,q)(P,D,Q,s)")

_default_season = defaults["season"]

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    p_val = st.number_input(
        "p", min_value=0, max_value=5, value=1, help="AR-Ordnung (nicht-saisonal)"
    )
with c2:
    d_val = st.number_input(
        "d", min_value=0, max_value=2, value=1, help="Differenzierungsordnung"
    )
with c3:
    q_val = st.number_input(
        "q", min_value=0, max_value=5, value=1, help="MA-Ordnung (nicht-saisonal)"
    )
with c4:
    P_val = st.number_input(
        "P", min_value=0, max_value=3, value=1, help="Saisonale AR-Ordnung"
    )
with c5:
    D_val = st.number_input(
        "D", min_value=0, max_value=2, value=0, help="Saisonale Differenzierungsordnung"
    )
with c6:
    Q_val = st.number_input(
        "Q", min_value=0, max_value=3, value=1, help="Saisonale MA-Ordnung"
    )
with c7:
    s_val = st.slider(
        "s (Season)",
        min_value=1,
        max_value=52,
        value=_default_season,
        help="Saisonalitäts-Periode: 7 für täglich, 52 für wöchentlich",
    )

order = (int(p_val), int(d_val), int(q_val))
seasonal_order = (int(P_val), int(D_val), int(Q_val), int(s_val))

st.divider()

# ── Zeile 3: Feature-Auswahl ─────────────────────────────────────────────────

st.subheader("Exogene Features")
st.caption(
    "Wähle die exogenen Variablen für das SARIMAX-Modell. "
    "Bei wöchentlichem Pattern wird `is_weekend` automatisch ausgeblendet."
)

selected_features: list[str] = []

with st.expander("Feature-Gruppen auswählen", expanded=True):
    for group_name, group_cols in FEATURE_GROUPS.items():
        # Bei weekly: is_weekend ausblenden
        display_cols = (
            group_cols if is_daily else [c for c in group_cols if c != "is_weekend"]
        )
        if not display_cols:
            continue

        # on_change Callback: setzt Session State aller Feature-Checkboxen der Gruppe
        def _make_select_all_callback(gname: str, cols: list[str]):
            def _cb():
                new_val = st.session_state[f"all_{gname}"]
                for feat in cols:
                    st.session_state[f"feat_{gname}_{feat}"] = new_val

            return _cb

        g_col1, g_col2 = st.columns([1, 5])
        with g_col1:
            st.checkbox(
                "Alle",
                key=f"all_{group_name}",
                value=False,
                on_change=_make_select_all_callback(group_name, display_cols),
                help=f"Alle Features der Gruppe '{group_name}' auswählen",
            )
        with g_col2:
            st.markdown(f"**{group_name}**")
            feat_cols_ui = st.columns(min(len(display_cols), 4))
            for i, feat in enumerate(display_cols):
                with feat_cols_ui[i % len(feat_cols_ui)]:
                    checked = st.checkbox(feat, key=f"feat_{group_name}_{feat}")
                    if checked:
                        selected_features.append(feat)

st.caption(
    f"Ausgewählte Features ({len(selected_features)}): "
    + (
        ", ".join(selected_features)
        if selected_features
        else "— keine (reines SARIMA) —"
    )
)

st.divider()

# ── Run-Button ───────────────────────────────────────────────────────────────

run_col, _ = st.columns([1, 4])
with run_col:
    run_button = st.button("Run SARIMAX", use_container_width=True, type="primary")

# ── Training ─────────────────────────────────────────────────────────────────

if run_button:
    setup_mlflow(EXPERIMENT)

    with st.spinner(
        f"Trainiere SARIMAX{order}x{seasonal_order} auf {pattern} "
        f"(store={store}, item={item}) …"
    ):
        try:
            df = _load_segment(pattern)

            results, fig = run_sarimax_plotly(
                df=df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(s_val),
                test_weeks=int(test_weeks),
                order=order,
                seasonal_order=seasonal_order,
                feature_cols=selected_features,
                trailing_zero_min_days=trailing_zero_min_days,
                img_dir=IMG_DIR,
            )

            st.session_state["sarimax_results"] = results
            st.session_state["sarimax_fig"] = fig
            _load_mlflow_runs.clear()

        except Exception as exc:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            st.error(f"Fehler beim SARIMAX-Training: {exc}")
            st.exception(exc)

# ── Ergebnisse ───────────────────────────────────────────────────────────────

if "sarimax_results" in st.session_state:
    r = st.session_state["sarimax_results"]
    fig = st.session_state["sarimax_fig"]

    st.divider()
    st.subheader("Ergebnisse")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("MAE SARIMAX", f"{r['mae_primary']:.2f}")
    m2.metric("MAE Naive", f"{r['mae_naive']:.2f}")
    m3.metric("R² SARIMAX", f"{r['r2_primary']:.3f}")
    m4.metric(
        "Verbesserung",
        f"{r['improvement_pct']:+.1f}%",
        delta=f"{r['improvement_pct']:.1f}%",
        delta_color="normal",
    )
    m5.metric("Train / Test", f"{r['train_size']} / {r['test_size']}")
    m6.metric("Exogene Features", r["n_exog"])

    st.plotly_chart(fig, use_container_width=True)

# ── MLflow Run History ────────────────────────────────────────────────────────

st.divider()
st.subheader("MLflow Run History — SARIMAX")

btn_col, _ = st.columns([1, 5])
with btn_col:
    if st.button("Aktualisieren"):
        _load_mlflow_runs.clear()

runs_df = _load_mlflow_runs()

if runs_df.empty:
    st.info("Noch keine SARIMAX-Runs gefunden. Starte ein Modell oben.")
else:
    rename_map = {
        "tags.mlflow.runName": "Run Name",
        "params.pattern": "Pattern",
        "params.store": "Store",
        "params.item": "Item",
        "params.freq": "Freq",
        "params.p": "p",
        "params.d": "d",
        "params.q": "q",
        "params.s_p": "P",
        "params.s_d": "D",
        "params.s_q": "Q",
        "params.s": "s",
        "params.n_exog_features": "Exog #",
        "metrics.mae_primary": "MAE (SARIMAX)",
        "metrics.mae_naive": "MAE (Naive)",
        "metrics.r2_primary": "R²",
        "metrics.improvement_pct": "Improvement %",
        "start_time": "Zeitpunkt",
    }
    display_df = runs_df.rename(columns=rename_map)

    for col in ["MAE (SARIMAX)", "MAE (Naive)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )
    if "R²" in display_df.columns:
        display_df["R²"] = display_df["R²"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
        )
    if "Improvement %" in display_df.columns:
        display_df["Improvement %"] = display_df["Improvement %"].map(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
        )
    if "Zeitpunkt" in display_df.columns:
        display_df["Zeitpunkt"] = pd.to_datetime(display_df["Zeitpunkt"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    cols_to_show = [c for c in display_df.columns if c != "run_id"]
    st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)
