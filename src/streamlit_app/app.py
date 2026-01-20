import sys
from pathlib import Path

import plotly.io as pio
import streamlit as st

from Favorita_TSA.viz.ploty_theme import set_plotly_theme

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))


set_plotly_theme()
pio.templates.default = "favorita_dark"

st.set_page_config(
    page_title="Favorita Analytics",
    layout="wide",
)

st.title("Favorita Grocery Sales - Analytics Dashboard")

st.markdown(
    """
    This dashboard provides exploratory insights into:
    - Store performance
    - Item behavior
    - Seasonality
    - Outliers

    Use the sidebar to navigate.
    """
)
