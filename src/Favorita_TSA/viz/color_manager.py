import os
from types import SimpleNamespace
from typing import Any

import plotly.graph_objects as go
import yaml


class ColorManager:
    """
    Verwaltet das dynamische Laden der globalen Farben mit intelligenter Automatisierung.
    Unterstützt 2-Wege-Zugriff:
    1. Flach (Shortcut): color.main oder color.forecast
    2. Strukturiert: color.theme_dark.background.main
    """
    _colors: SimpleNamespace | None = None
    _raw_dict: dict[str, Any] | None = None

    @classmethod
    def _dict_to_namespace(cls, data: Any) -> Any:
        """Konvertiert Dictionary rekursiv in SimpleNamespace."""
        if isinstance(data, dict):
            return SimpleNamespace(**{k: cls._dict_to_namespace(v) for k, v in data.items()})
        return data

    @classmethod
    def _flatten_dict(cls, data: Any, result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extrahiert alle Farben auf eine flache Ebene für ultraschnellen Zugriff."""
        if result is None:
            result = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    cls._flatten_dict(v, result)
                else:
                    result[k] = v
        return result

    @classmethod
    def get_colors(cls, file_path: str | None = None) -> SimpleNamespace:
        """
        Lädt die COLORS.yaml und erstellt automatische UI-Mappings.
        Verbesserte Pfad-Logik verhindert FileNotFoundError in Notebooks.
        """
        if cls._colors is None:
            # ROBUSTE PFAD-SUCHE: Prüft erst Root, dann eine Ebene höher
            if file_path is None:
                potential_paths = [
                    os.path.join("configs", "COLORS.yaml"),
                    os.path.join("..", "configs", "COLORS.yaml")
                ]
                for p in potential_paths:
                    if os.path.exists(p):
                        file_path = p
                        break
            
            if file_path is None or not os.path.exists(file_path):
                raise FileNotFoundError(f"COLORS.yaml konnte nicht gefunden werden. Pfade geprüft: {['configs/COLORS.yaml', '../configs/COLORS.yaml']}")

            try:
                with open(file_path, encoding="utf-8") as f:
                    cls._raw_dict = yaml.safe_load(f)
                
                # Weg 1: Flache Shortcuts
                flat_data = cls._flatten_dict(cls._raw_dict)
                
                # AUTOMATISIERUNG: Intelligentes Mapping für Theme-Farben
                theme = cls._raw_dict.get("theme_dark", {})
                bg = theme.get("background", {})
                txt = theme.get("text", {})
                ui = theme.get("ui", {})

                auto_mapping = {
                    "ui_paper": bg.get("main") or bg.get("main_bg") or "#0B0E14",
                    "ui_plot": bg.get("surface") or bg.get("plot_bg") or "#161B22",
                    "ui_grid": bg.get("grid") or "#1F242C",
                    "ui_text": txt.get("primary") or txt.get("text_primary") or "#E6EDF3",
                    "ui_border": ui.get("border") or "#30363D"
                }
                
                # Weg 2: Strukturierte Daten
                structured_ns = cls._dict_to_namespace(cls._raw_dict)
                
                # Alles zusammenfügen
                cls._colors = SimpleNamespace(**{**flat_data, **auto_mapping, **vars(structured_ns)})
                
            except Exception as e:
                print(f"Fehler beim Laden: {e}")
                raise
        return cls._colors

    @classmethod
    def get_raw_dict(cls):
        if cls._raw_dict is None:
            cls.get_colors()
        return cls._raw_dict

def apply_modern_theme(fig: go.Figure) -> None:
    """
    Wendet das Design vollautomatisch an. Nutzt die intelligenten 'ui_'-Mappings,
    damit der Code nicht bricht, wenn Namen in der YAML geändert werden.
    """
    c = ColorManager.get_colors()
    
    fig.update_layout(
        paper_bgcolor=c.ui_paper,
        plot_bgcolor=c.ui_plot,
        font_color=c.ui_text,
        xaxis={"gridcolor": c.ui_grid, "linecolor": c.ui_border, "zeroline": False},
        yaxis={"gridcolor": c.ui_grid, "linecolor": c.ui_border, "zeroline": False},
        template="plotly_dark",
    )

# =============================================================================
# ANLEITUNG: SO NUTZT DU DAS DESIGN-SYSTEM (2 VARIANTEN)
# =============================================================================
#
# 1. DER SCHNELLE WEG (Flach / Shortcut):
#    Ideal für tägliches Coding. Alle Farben sind direkt erreichbar.
#    -> color.forecast, color.main, color.observed, color.top20
#
# 2. DER STRUKTURIERTE WEG (Pfad):
#    Ideal für Ordnung und den Audit-Plot. Folgt der YAML-Hierarchie.
#    -> color.theme_dark.background.main
#    -> color.analysis.lines.observed
#
# 3. DAS AUTOMATISCHE THEME:
#    Einfach 'apply_modern_theme(fig)' aufrufen. Es zieht sich die korrekten
#    Farben für den Hintergrund und Achsen selbstständig aus der YAML.
#
# =============================================================================

if __name__ == "__main__":
    color = ColorManager.get_colors()
    fig = go.Figure()
    
    # Beispiel für Shortcut-Nutzung
    top_colors = color.top20 
    for i, col in enumerate(top_colors[:5]):
        fig.add_trace(go.Scatter(x=[1, 2], y=[i, i+1], line={"color": col}, name=f"Farbe {i+1}"))
    
    apply_modern_theme(fig)
    print("System-Check: Farben erfolgreich geladen und Theme automatisch angewendet.")
    fig.show()