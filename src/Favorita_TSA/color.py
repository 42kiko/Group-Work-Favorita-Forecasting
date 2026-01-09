import os
from typing import Any, cast

import plotly.graph_objects as go
import yaml


def load_global_colors(
    file_path: str = os.path.join("configs", "COLORS.yaml"),
) -> dict[str, str] | None:
    """
    Loads colors from YAML.
    Typisierung: file_path ist ein String (str), Rückgabe ist ein Dictionary oder None.
    """
    try:
        with open(file_path) as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None


def apply_plotly_theme(fig: go.Figure, colors: dict[str, Any]) -> None:
    """
    Wendet das Thema an. Nutzt Literale {} statt dict() für Ruff C408.
    """
    if not colors:
        return

    # 'cast' verhindert Pylance-Fehler bei verschachtelten Zugriffen
    dark = cast(dict[str, Any], colors.get("theme_dark", {}))

    # RAM-Schonung: Lokale Extraktion der benötigten Werte
    bg = cast(dict[str, str], dark.get("background", {}))
    text = cast(dict[str, str], dark.get("text", {}))
    ui = cast(dict[str, str], dark.get("ui", {}))

    fig.update_layout(
        plot_bgcolor=bg.get("surface"),
        paper_bgcolor=bg.get("main"),
        font_color=text.get("primary"),
        # RUFF C408 FIX: {} statt dict()
        xaxis={"gridcolor": bg.get("grid"), "linecolor": ui.get("border")},
        yaxis={"gridcolor": bg.get("grid"), "linecolor": ui.get("border")},
        template="plotly_dark",
    )


# --- Execution ---
colors_data: dict[str, Any] | None = load_global_colors()
fig_instance: go.Figure = go.Figure()

if colors_data:
    # Zugriff auf Analyse-Farben
    analysis = cast(dict[str, Any], colors_data.get("analysis", {}))
    lines = cast(dict[str, str], analysis.get("lines", {}))

    fig_instance.add_trace(
        go.Scatter(
            x=[1, 2, 3],
            y=[10, 15, 13],
            # RUFF C408 FIX: line={"color": ...} statt line=dict(color=...)
            line={"color": lines.get("observed")},
            name="Observed",
        )
    )

    apply_plotly_theme(fig_instance, colors_data)

# --- Execution ---
colors = load_global_colors()
fig = go.Figure()

if colors:
    analysis = cast(dict[str, Any], colors.get("analysis", {}))
    lines = cast(dict[str, str], analysis.get("lines", {}))

    observed_color = lines.get("observed")

    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3],
            y=[10, 15, 13],
            # Ruff C408: {} statt dict()
            line={"color": observed_color},
            name="Observed",
        )
    )

    apply_plotly_theme(fig, colors)
    fig.show()
