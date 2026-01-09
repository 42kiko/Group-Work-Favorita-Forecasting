import os
from dataclasses import dataclass

import plotly.graph_objects as go
import yaml


@dataclass(frozen=True)
class ThemeColors:
    """Stark typisierte Farbdefinitionen für IDE-Support."""

    main_bg: str
    surface: str
    grid: str
    text_primary: str
    text_secondary: str
    border: str
    accent: str
    observed: str
    trend: str
    forecast: str
    uncertainty: str
    anomaly: str
    event: str
    heat_min: str
    heat_mid: str
    heat_max: str
    top20: list[str]


class ColorManager:
    """Verwaltet das Laden und Bereitstellen der globalen Farben."""

    _colors: ThemeColors | None = None

    @classmethod
    def get_colors(
        cls, file_path: str = os.path.join("configs", "COLORS.yaml")
    ) -> ThemeColors:
        if cls._colors is None:
            try:
                with open(file_path) as f:
                    raw = yaml.safe_load(f)

                cls._colors = ThemeColors(
                    main_bg=raw["theme_dark"]["background"]["main"],
                    surface=raw["theme_dark"]["background"]["surface"],
                    grid=raw["theme_dark"]["background"]["grid"],
                    text_primary=raw["theme_dark"]["text"]["primary"],
                    text_secondary=raw["theme_dark"]["text"]["secondary"],
                    border=raw["theme_dark"]["ui"]["border"],
                    accent=raw["theme_dark"]["ui"]["accent"],
                    observed=raw["analysis"]["lines"]["observed"],
                    trend=raw["analysis"]["lines"]["trend"],
                    forecast=raw["analysis"]["lines"]["forecast"],
                    uncertainty=raw["analysis"]["lines"]["uncertainty"],
                    anomaly=raw["analysis"]["markers"]["anomaly"],
                    event=raw["analysis"]["markers"]["event"],
                    heat_min=raw["visualizations"]["heatmap"]["min"],
                    heat_mid=raw["visualizations"]["heatmap"]["mid"],
                    heat_max=raw["visualizations"]["heatmap"]["max"],
                    top20=raw["visualizations"]["top20"],
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"Fehler beim Laden der Farben: {e}")
                raise
        return cls._colors


def apply_modern_theme(fig: go.Figure) -> None:
    """Wendet das globale Design auf eine Plotly-Figur an."""
    c = ColorManager.get_colors()
    fig.update_layout(
        plot_bgcolor=c.surface,
        paper_bgcolor=c.main_bg,
        font_color=c.text_primary,
        xaxis={"gridcolor": c.grid, "linecolor": c.border, "zeroline": False},
        yaxis={"gridcolor": c.grid, "linecolor": c.border, "zeroline": False},
        template="plotly_dark",
    )


# =====================================
# ANLEITUNG: SO NUTZT DU DIE FARBEN
# =====================================
# 1. IMPORT:    from color import ColorManager, apply_modern_theme
# 2. SETUP:     color = ColorManager.get_colors()
# 3. NUTZUNG:   Verwende 'color.<name>' ohne Anführungszeichen.
#
# BEISPIEL PLOTLY:
# fig.add_trace(go.Scatter(..., line={"color": color.forecast}))
#
# BEISPIEL TABELLE (dein Standard):
# fig.add_trace(go.Table(header={"fill_color": color.surface}, ...))
#
# VORTEIL: Deine IDE (Cursor/VS Code) schlägt dir die Namen nach dem Punkt automatisch vor.
# =====================================

# Dieser Block sorgt dafür, dass der darin enthaltene Code nur dann ausgeführt wird, wenn du die Datei color.py direkt startest (z. B. um zu testen, ob die Farben richtig geladen werden).
if __name__ == "__main__":
    color = ColorManager.get_colors()
    fig = go.Figure()
    # Test der Top-20 Liste
    for i, c in enumerate(color.top20[:5]):
        fig.add_trace(
            go.Scatter(x=[1, 2], y=[i, i + 1], line={"color": c}, name=f"Farbe {i+1}")
        )
    apply_modern_theme(fig)
    fig.show()
