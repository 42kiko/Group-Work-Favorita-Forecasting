from pathlib import Path

import plotly.graph_objects as go

# Default output directory for all Plotly exports
PLOTLY_REPORT_DIR = Path("reports") / "plotly"


def _resolve_path(path: str | Path, suffix: str) -> Path:
    """
    Resolves relative paths against the default Plotly report directory
    and ensures the correct file suffix.
    """
    path = Path(path)

    if not path.suffix:
        path = path.with_suffix(suffix)

    if path.is_absolute():
        return path

    return PLOTLY_REPORT_DIR / path


def save_html(fig: go.Figure, name: str | Path) -> None:
    output_path = _resolve_path(name, ".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cd n")


def save_png(fig: go.Figure, name: str | Path, scale: int = 2) -> None:
    output_path = _resolve_path(name, ".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, scale=scale)


def save_all(fig: go.Figure, name: str | Path, scale: int = 2) -> None:
    save_html(fig, name)
    save_png(fig, name, scale=scale)
