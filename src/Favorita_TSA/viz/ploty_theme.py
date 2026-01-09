import plotly.io as pio

from Favorita_TSA.color import ColorManager


def set_plotly_theme() -> None:
    """
    Registers and activates the global Favorita dark Plotly theme.
    Colors are loaded via ColorManager from COLORS.yaml.
    """
    c = ColorManager.get_colors()

    pio.templates["favorita_dark"] = {
        "layout": {
            # Backgrounds
            "paper_bgcolor": c.main_bg,
            "plot_bgcolor": c.surface,
            # Typography
            "font": {
                "color": c.text_primary,
                "family": "Inter, system-ui, sans-serif",
                "size": 13,
            },
            # Axes
            "xaxis": {
                "gridcolor": c.grid,
                "linecolor": c.border,
                "zeroline": False,
            },
            "yaxis": {
                "gridcolor": c.grid,
                "linecolor": c.border,
                "zeroline": False,
            },
            # Default color cycle (Top-20 Palette aus YAML)
            "colorway": c.top20,
            # General UI
            "legend": {
                "bgcolor": c.surface,
                "bordercolor": c.border,
            },
        }
    }

    # Activate globally
    pio.templates.default = "favorita_dark"
