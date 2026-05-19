"""theme.py — palette, fonts, and small helpers for the Trust dashboard.

Single source of truth for colors and font choices. Imported by widgets.py,
panels.py, and main.py. Change a hex here and the whole app reflects it.
"""

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtCore import Qt


# ─── Palette (cool slate, light theme — resolved from oklch) ───────────────
BG          = "#f7f8fa"
BG_DEEP     = "#eef0f4"
PANEL       = "#ffffff"
PANEL_2     = "#f5f6f9"
LINE        = "#d9dce3"
LINE_SOFT   = "#e7e9ef"
TEXT        = "#2d3340"
TEXT_DIM    = "#5f6675"
TEXT_FAINT  = "#8a91a1"
TEXT_GHOST  = "#b0b6c4"

# Channel hues (spread across the wheel so the four bars are unambiguously
# distinguishable at a glance — the failure mode of the warm design was that
# coral/mauve/peach all looked alike).
C_FACIAL    = "#1a8aa3"   # teal
C_VOCAL     = "#6e3fce"   # violet
C_GAZE      = "#b88318"   # gold
C_HRV       = "#cd4734"   # coral
C_WORKLOAD  = "#2da46a"   # green (cognitive-load indicator)

ACCENT      = "#2872c4"   # primary blue (logo mark, focus states)
DANGER      = "#c93a3a"   # End Session, error states


# ─── Trust bands ─ same hex values trust_engine.py emits ────────────────────
TRUST_BANDS = [
    (82, "Very High Trust", "#4ade80"),
    (64, "High Trust",      "#34d399"),
    (46, "Neutral",         "#60a5fa"),
    (28, "Low Trust",       "#fb923c"),
    (0,  "Very Low Trust",  "#f87171"),
]

def trust_band(score: int):
    """Return (label, hex_color) for a 0–100 trust score."""
    for threshold, label, color in TRUST_BANDS:
        if score >= threshold:
            return label, color
    return TRUST_BANDS[-1][1], TRUST_BANDS[-1][2]


# ─── Fonts ─ Inter + JetBrains Mono with graceful system fallbacks ─────────
# Qt picks the first installed family from each list.
UI_FAMILIES   = ["Inter", "SF Pro Display", "Segoe UI", "Helvetica Neue", "Arial"]
MONO_FAMILIES = ["JetBrains Mono", "SF Mono", "Menlo", "Consolas", "Courier New"]


def ui_font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Inter (or fallback) at the given pt size."""
    f = QFont()
    f.setFamilies(UI_FAMILIES)
    f.setPointSize(size)
    f.setWeight(weight)
    return f


def mono_font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """JetBrains Mono (or fallback) at the given pt size, tabular numerics."""
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setPointSize(size)
    f.setWeight(weight)
    f.setStyleHint(QFont.StyleHint.TypeWriter)
    return f


def load_packaged_fonts() -> None:
    """Optional: ship Inter/JetBrainsMono .ttf files in a fonts/ folder
    next to main.py and they'll be loaded at startup. Falls back silently
    if the files aren't present."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(here, "fonts")
    if not os.path.isdir(fonts_dir):
        return
    for f in os.listdir(fonts_dir):
        if f.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(fonts_dir, f))


# ─── Common QSS snippets ────────────────────────────────────────────────────
def panel_qss(name: str = "panel") -> str:
    """Rounded white panel with a 1px line border. Use objectName trick to
    stop the radius from leaking onto child widgets."""
    return f"""
        #{name} {{
            background: {PANEL};
            border: 1px solid {LINE};
            border-radius: 8px;
        }}
    """


def head_qss(name: str = "panelHead") -> str:
    return f"""
        #{name} {{
            border-bottom: 1px solid {LINE_SOFT};
            background: {PANEL};
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }}
    """
