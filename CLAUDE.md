# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time series forecasting project for Favorita supermarket stores, implementing demand pattern classification (Croston/Syntetos-Boylan), baseline statistical models (SARIMA, Theta), and an interactive Streamlit dashboard. Authors: Agus, Kiko, Patrick.

## Setup & Common Commands

```bash
# Full dev environment setup
make setup-dev          # creates .venv, installs deps, activates pre-commit hooks

# Or step by step
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
pre-commit install
```

```bash
# Run tests
pytest tests/

# Run a single test
pytest tests/test_forecastability.py::test_classify_demand_pattern

# Launch Streamlit dashboard
streamlit run src/streamlit_app/app.py
# or
make run app

# Linting & formatting
ruff check src/          # lint
ruff format src/         # format (or use black)
```

## Architecture

### Source Layout (`src/`)

**`Favorita_TSA/`** — main Python package:
- `utils/forecastability.py` — core demand classification logic (ADI, CV², Croston/Syntetos-Boylan thresholds: ADI=1.32, CV²=0.49); classifies store-item combinations into Smooth/Erratic/Intermittent/Lumpy
- `utils/data_loader.py` — parquet I/O with caching
- `utils/preprocess_data.py` — fact table creation with temporal features (DOW, ISO week, month, year)
- `models/data_preparation.py` — splits fact table by demand pattern into 4 DataFrames (smooth/erratic × daily/weekly)
- `features/holidays.py` / `preprocess/holiday_parquets.py` — holiday feature engineering
- `viz/` — Plotly dark-mode theme, color management from `configs/COLORS.yaml`

**`streamlit_app/`** — multi-page dashboard:
- `app.py` — entry point, full-screen dark layout
- `pages/forecastability_store_item.py` — forecastability matrix
- `pages/store_item_behavior.py` — individual store-item time series
- `pages/multi_stores.py` — cross-store comparison

### Data Pipeline

Raw CSV → `fact_table.parquet` (874 MB) → aggregated parquets (daily/weekly/monthly at store, item, store-item levels) → forecastability metrics → baseline model inputs.

All processed data lives in `data/processed/preprocessed/`; forecastability metrics in `data/metrics/`; baseline results in `data/baseline_results/`.

### MLflow Experiment Tracking

MLflow uses a **local filesystem backend** at `mlruns/`. Notebooks log to experiment `"favorita_baseline_store_item"`.

```python
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("favorita_baseline_store_item")
```

### Key Concepts

**Demand Pattern Classification (Croston/Syntetos-Boylan):**
- ADI (Average Demand Interval) = n_periods / n_nonzero
- CV² (Coefficient of Variation Squared) = (σ/μ)²
- Smooth: ADI ≤ 1.32 AND CV² ≤ 0.49
- Erratic: ADI ≤ 1.32 AND CV² > 0.49
- Intermittent: ADI > 1.32 AND CV² ≤ 0.49
- Lumpy: ADI > 1.32 AND CV² > 0.49

Models are trained and evaluated separately per demand pattern.

### Notebooks

Notebooks in `notebooks/` are the primary workspace for EDA and modeling. Key ones:
- `baseline_modeling_sarima.ipynb` / `baseline_modeling_theta.ipynb` — MLflow-tracked baseline models
- `02_eda_prepocess.ipynb` — main EDA and preprocessing (57 MB, heavy)
- `pattern_recognition.ipynb` — demand pattern analysis

### Code Quality

Pre-commit hooks run `ruff` (with auto-fix) and `black` on every commit. Line length: 88 chars. Ruff rules: E, F, I, B, UP, SIM, C4, ARG, RUF.
