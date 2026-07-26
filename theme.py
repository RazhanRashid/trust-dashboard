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


# DISPLAY-ONLY mapping: the colour + label the UI shows for a given 0–100 score.
# NOT part of scoring/weighting — trust_engine.py does not read this.
TRUST_BANDS = [
    (82, "Calm + Engaged", "#3b9edd"),   # clear blue
    (64, "Relaxed",        "#4fc4a0"),   # teal-green (perceptually distinct)
    (46, "Baseline",       "#f5c842"),   # amber
    (28, "Activated",      "#f07d2a"),   # orange
    (0,  "Heightened",     "#e03e3e"),   # red
]

HRV_IS_PLACEHOLDER = True  # initial state before any BLE connection attempt;
                            # ScorePanel.set_hrv_connected() toggles this live once a sensor connects

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

# Sizes here are PIXELS, not points, and the difference is load-bearing.
#
# Qt reports a logical DPI of 72 on macOS but 96 on Windows, so the same point
# size renders 96/72 = 1.33x larger there. Every panel in this app is laid out
# in hard-coded pixels (setFixedHeight and friends), and those don't grow to
# match — so on Windows the type outgrew its boxes and rows overlapped and
# clipped, while macOS looked fine. Pixel sizing removes the DPI term: 1 px is
# 1 px on both, so text and layout stay in proportion.
#
# This is not the same as ignoring HiDPI. Qt6 scales the whole UI by the
# screen's devicePixelRatio, which magnifies pixel-sized fonts and pixel-sized
# layout together — a 150% display still gets 150% type. What it no longer
# does is inflate one and not the other.
#
# Because 1 pt == 1 px at 72 DPI, these numbers render identically to the old
# point sizes on macOS; only Windows changes.


# ─── Adaptive scale ────────────────────────────────────────────────────────
# Every dimension in this app is a pixel number picked against one screen. On
# a shorter one the stage needs more vertical space than exists and panels
# clip. Windows makes this routine: at 125% or 150% display scaling Qt divides
# the logical window by that factor, so a 1080p laptop presents 801 or 667
# logical pixels to a layout that wants ~870.
#
# So measure the screen once at startup and scale the design to fit. sp()
# converts a design pixel to a device pixel; fonts scale by the same factor so
# type and boxes stay in proportion. Scale is capped at 1.0 — a big display
# gets the design at its intended size, never inflated — and floored so the
# UI cannot shrink into illegibility on something tiny.
DESIGN_HEIGHT = 980      # the window height this layout was drawn against
_CHROME_H     = 40       # title bar + frame that availableGeometry doesn't remove
# 0.65 not 0.70: at 150% Windows scaling a 1080p screen leaves a 648px window,
# and 0.70 needs exactly 648 — no margin at all. 0.65 leaves ~40px.
MIN_UI_SCALE  = 0.65
MAX_UI_SCALE  = 1.00

UI_SCALE = 1.0           # replaced by init_ui_scale() before any widget is built


def init_ui_scale(app) -> float:
    """Measure the primary screen and set the global UI scale. Call once, from
    main(), before constructing any widget — sizes are read at construction."""
    global UI_SCALE
    try:
        screen = app.primaryScreen()
        usable = screen.availableGeometry().height() - _CHROME_H
        UI_SCALE = max(MIN_UI_SCALE, min(MAX_UI_SCALE, usable / DESIGN_HEIGHT))
    except Exception:
        UI_SCALE = 1.0   # never let a display query stop the app from starting
    return UI_SCALE


def sp(px: float) -> int:
    """Scale a design pixel to this device.

    Zero stays zero — a flush margin must stay flush. Anything else floors at
    1 so a hairline rule stays a visible rule instead of vanishing.
    """
    if px <= 0:
        return 0
    return max(1, round(px * UI_SCALE))


def ui_font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Inter (or fallback) at the given design pixel size, scaled to the device."""
    f = QFont()
    f.setFamilies(UI_FAMILIES)
    f.setPixelSize(max(8, round(size * UI_SCALE)))   # 8px is the legibility floor
    f.setWeight(weight)
    return f


def mono_font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """JetBrains Mono (or fallback) at the given design pixel size, scaled."""
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setPixelSize(max(8, round(size * UI_SCALE)))
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
