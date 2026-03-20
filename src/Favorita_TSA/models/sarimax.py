"""
sarimax.py

Backend für SARIMAX-Modellierung mit exogenen Features.
Verwendet statsmodels.tsa.statespace.sarimax.SARIMAX mit manuell
eingestellten Parametern (p,d,q)(P,D,Q,s).

Funktionen
──────────
  run_sarimax_plotly(df, pattern, store, item, freq, season_length,
                     test_weeks, order, seasonal_order, feature_cols,
                     img_dir) -> (dict, go.Figure)

Alle Runs werden automatisch in MLflow geloggt.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Favorita_TSA.models.baseline import load_and_prepare, train_test_split

# Kategorische Spalten, die one-hot encodiert werden müssen
_CATEGORICAL_COLS = {"store_type", "family"}

# Konstante Spalten, die für SARIMAX ungeeignet sind
_EXCLUDE_COLS = {
    "store_nbr",
    "item_nbr",
    "date",
    "week_start",
    "unit_sales",
    "ds",
    "y",
    "unique_id",
    "year",
    "dow",
    "year_iso",
    "week",
    "month",
}


def _build_exog_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    is_weekly: bool,
) -> pd.DataFrame:
    """
    Baut die exogene Feature-Matrix aus dem angereicherten DataFrame.

    Kategorische Spalten (store_type, family) werden one-hot encodiert.
    Bei wöchentlicher Zeitachse wird ``is_weekend`` automatisch ausgeschlossen.

    Parameters
    ----------
    df : DataFrame
        Angereicherter Store-Item-Slice (bereits gefiltert auf store/item).
    feature_cols : list[str]
        Auswahl der Feature-Spalten.
    is_weekly : bool
        Falls True, wird ``is_weekend`` aus ``feature_cols`` entfernt.

    Returns
    -------
    DataFrame mit rein numerischen Spalten, Index kompatibel mit df.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if is_weekly:
        cols = [c for c in cols if c != "is_weekend"]

    if not cols:
        return pd.DataFrame(index=df.index)

    X = df[cols].copy()

    # One-hot encoding für kategorische Spalten
    cat_present = [c for c in _CATEGORICAL_COLS if c in X.columns]
    if cat_present:
        X = pd.get_dummies(X, columns=cat_present, drop_first=True)

    # Sicherstellen: alle Spalten numerisch
    # astype(object) auf dem gesamten DataFrame entfernt nullable BooleanDtype / IntDtype,
    # danach apply(to_numeric) + fillna + astype(float) ohne column-by-column Zuweisung.
    X = (
        X.astype(object)
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .fillna(0)
        .astype(float)
    )

    return X


def _next_sarimax_run_number(pattern: str) -> int:
    """Gibt die nächste Run-Nummer für das Pattern zurück."""
    try:
        existing = mlflow.search_runs(
            filter_string=f"params.pattern = '{pattern}'",
            max_results=1000,
        )
        return len(existing) + 1
    except Exception:
        return 1


def run_sarimax_plotly(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    test_weeks: int = 4,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    feature_cols: list[str] | None = None,
    trailing_zero_min_days: int = 0,
    img_dir: Path | str | None = None,
) -> tuple[dict, go.Figure]:
    """
    Vollständige SARIMAX-Pipeline: Vorbereitung → Split → SARIMAX fitten
    → Naive Benchmark → Metriken berechnen → Plotly-Chart → MLflow loggen.

    Parameters
    ----------
    df : DataFrame
        Angereicherter Store-Item-Segment-DataFrame (aus load_sarimax_segment()).
    pattern : str
        z.B. "daily_smooth", "weekly_erratic".
    store, item : int
    freq : str
        "D" für täglich, "W" für wöchentlich.
    season_length : int
        Saisonalitäts-Periode (s im seasonal_order).
    test_weeks : int
        Anzahl Wochen im Test-Set.
    order : tuple (p, d, q)
        Nicht-saisonale ARIMA-Ordnung.
    seasonal_order : tuple (P, D, Q, s)
        Saisonale ARIMA-Ordnung inkl. Periode s.
    feature_cols : list[str] | None
        Auswahl der exogenen Features. None oder leer = reines SARIMA.
    img_dir : Path | str | None
        Verzeichnis für HTML-Plot-Export.

    Returns
    -------
    (results_dict, fig)
    """
    is_weekly = freq == "W"
    if feature_cols is None:
        feature_cols = []

    # ── 1. Zeitreihe vorbereiten (ds/y/unique_id) ────────────────────────────
    ts = load_and_prepare(
        df,
        store=store,
        item=item,
        freq=freq,
        trailing_zero_min_days=trailing_zero_min_days,
    )

    # ── 2. Exogene Feature-Matrix aus angereichertem df ─────────────────────
    # Benötigt den gefilterten Slice aus df (gleiche Datumsrange wie ts)
    if "store_nbr" in df.columns:
        slice_df = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
        date_col = "week_start" if is_weekly else "date"
        slice_df[date_col] = pd.to_datetime(slice_df[date_col])
        slice_df = slice_df.sort_values(date_col).reset_index(drop=True)
    else:
        slice_df = df.copy()
        date_col = "week_start" if "week_start" in df.columns else "date"

    X_full = _build_exog_matrix(slice_df, feature_cols, is_weekly=is_weekly)

    ts["ds"] = pd.to_datetime(ts["ds"])

    # ── 3. Train/Test-Split ───────────────────────────────────────────────────
    train_ts, test_ts = train_test_split(ts, test_weeks=test_weeks)

    # X-Matrix auf gleiche Zeitpunkte filtern
    if X_full.shape[1] > 0:
        # Mapping über Position (slice_df ist bereits gleich sortiert wie ts)
        # Sicherheitshalber per merge
        ts_reset = ts.reset_index(drop=True)

        if len(ts_reset) == len(X_full):
            X_aligned = X_full.values
        else:
            # Merge über Datum
            ts_with_idx = ts_reset[["ds"]].copy()
            ts_with_idx["_pos"] = np.arange(len(ts_with_idx))
            slice_with_idx = pd.DataFrame(
                {date_col: pd.to_datetime(slice_df[date_col])}
            ).reset_index(drop=True)
            slice_with_idx["_slice_pos"] = np.arange(len(slice_with_idx))
            merged = ts_with_idx.merge(
                slice_with_idx, left_on="ds", right_on=date_col, how="left"
            )
            valid_pos = merged["_slice_pos"].fillna(-1).astype(int).values
            X_arr = np.zeros((len(ts_reset), X_full.shape[1]))
            for i, pos in enumerate(valid_pos):
                if 0 <= pos < len(X_full):
                    X_arr[i] = X_full.iloc[pos].values
            X_aligned = X_arr

        n_train = len(train_ts)
        X_train = X_aligned[:n_train]
        X_test = X_aligned[n_train:]

        use_exog = True
    else:
        X_train = None
        X_test = None
        use_exog = False

    # ── 4. SARIMAX fitten ─────────────────────────────────────────────────────
    print(
        f"Fitting SARIMAX{order}x{seasonal_order} "
        f"auf {pattern} (store={store}, item={item})"
    )
    print(f"  Exogene Features: {feature_cols if use_exog else 'keine'}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=train_ts["y"].values,
            exog=X_train if use_exog else None,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

    # ── 5. Forecast ───────────────────────────────────────────────────────────
    n_test = len(test_ts)
    forecast = result.forecast(steps=n_test, exog=X_test if use_exog else None)
    forecast = np.maximum(forecast, 0)  # keine negativen Verkäufe

    # ── 6. SeasonalNaive Benchmark ───────────────────────────────────────────
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=season_length)],
        freq=freq,
    )
    train_sf = train_ts[["unique_id", "ds", "y"]].copy()
    sf.fit(train_sf)
    naive_pred = sf.predict(h=n_test)
    naive_vals = np.maximum(naive_pred["SeasonalNaive"].values, 0)

    # ── 7. Metriken ───────────────────────────────────────────────────────────
    y_true = test_ts["y"].values
    mae_primary = float(mean_absolute_error(y_true, forecast))
    mae_naive = float(mean_absolute_error(y_true, naive_vals))
    r2_primary = float(r2_score(y_true, forecast)) if len(y_true) > 1 else float("nan")
    improvement_pct = (
        (mae_naive - mae_primary) / mae_naive * 100 if mae_naive > 0 else 0.0
    )

    print(
        f"  MAE SARIMAX={mae_primary:.3f}  MAE Naive={mae_naive:.3f}  "
        f"Verbesserung={improvement_pct:+.1f}%"
    )

    results = {
        "mae_primary": mae_primary,
        "mae_naive": mae_naive,
        "r2_primary": r2_primary,
        "improvement_pct": improvement_pct,
        "train_size": len(train_ts),
        "test_size": n_test,
        "n_exog": X_full.shape[1] if use_exog else 0,
    }

    # ── 8. Plotly-Chart ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Gesamtzeitreihe", "Test-Zeitraum (Zoom)"],
        vertical_spacing=0.12,
        shared_xaxes=False,
    )

    # Gesamte Trainingsdaten
    fig.add_trace(
        go.Scatter(
            x=train_ts["ds"],
            y=train_ts["y"],
            mode="lines",
            name="Train",
            line={"color": "#4C8BF5", "width": 1},
        ),
        row=1,
        col=1,
    )
    # Test-Ist-Werte
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=test_ts["y"],
            mode="lines",
            name="Test (Ist)",
            line={"color": "#34A853", "width": 2},
        ),
        row=1,
        col=1,
    )
    # SARIMAX-Forecast
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=forecast,
            mode="lines",
            name=f"SARIMAX{order}",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    # Zoom: Test-Zeitraum
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=test_ts["y"],
            mode="lines",
            name="Test (Ist)",
            line={"color": "#34A853", "width": 2},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=forecast,
            mode="lines",
            name=f"SARIMAX{order}",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=naive_vals,
            mode="lines",
            name="Seasonal Naive",
            line={"color": "#9E9E9E", "width": 1.5, "dash": "dot"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    p, d, q = order
    P, D, Q, s = seasonal_order
    feat_str = ", ".join(feature_cols[:3]) + ("…" if len(feature_cols) > 3 else "")
    title = (
        f"SARIMAX({p},{d},{q})({P},{D},{Q},{s}) — {pattern} "
        f"| Store {store} · Item {item} | "
        f"MAE={mae_primary:.2f} ({improvement_pct:+.1f}%)"
    )
    if feat_str:
        title += f" | Features: {feat_str}"

    fig.update_layout(
        title=title,
        height=700,
        legend={"orientation": "h", "y": -0.08},
    )

    # ── 9. MLflow loggen ──────────────────────────────────────────────────────
    run_n = _next_sarimax_run_number(pattern)
    run_name = f"{pattern}_{run_n:03d}_sarimax"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "pattern": pattern,
                "model_type": "sarimax",
                "store": store,
                "item": item,
                "freq": freq,
                "season_length": season_length,
                "test_weeks": test_weeks,
                "p": p,
                "d": d,
                "q": q,
                "s_p": P,
                "s_d": D,
                "s_q": Q,
                "s": s,
                "n_exog_features": X_full.shape[1] if use_exog else 0,
                "feature_cols": str(feature_cols),
            }
        )
        mlflow.log_metrics(
            {
                "mae_primary": mae_primary,
                "mae_naive": mae_naive,
                "r2_primary": r2_primary if not np.isnan(r2_primary) else -999.0,
                "improvement_pct": improvement_pct,
            }
        )

        if img_dir is not None:
            img_path = Path(img_dir) / f"{run_name}.html"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(img_path))
            mlflow.log_artifact(str(img_path))

    return results, fig
