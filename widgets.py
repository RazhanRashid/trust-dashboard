"""widgets.py — custom-painted primitives and reusable composites.

Everything here is independent of the analyzers — pure UI building blocks.
Each widget exposes a small, explicit public API so panels.py can wire them
to backend data with a single method call.
"""

from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (QWidget, QLabel, QFrame, QHBoxLayout, QVBoxLayout,
                              QSizePolicy)

from theme import (LINE, LINE_SOFT, TEXT, TEXT_DIM, TEXT_FAINT, TEXT_GHOST,
                    PANEL, PANEL_2, C_FACIAL, C_VOCAL, C_GAZE, C_HRV,
                    ui_font, mono_font, trust_band, head_qss)


# ─── Custom-painted gauge ───────────────────────────────────────────────────
class GaugeWidget(QWidget):
    """Semicircle arc + huge mono numeral + tiny '/ 100' subtitle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = 50
        self._color = QColor("#60a5fa")
        self.setMinimumSize(280, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setScore(self, score: int, color_hex: str):
        self._score = max(0, min(100, int(score)))
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()

        # Fill widget background so it blends with PANEL (no raw Qt grey)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PANEL))
        p.drawRect(0, 0, w, h)

        thickness = 10
        margin = 18
        diameter = min(w - margin * 2, (h - 20) * 2)
        radius = diameter / 2
        cx = w / 2
        cy = h - 14
        arc_rect = QRectF(cx - radius, cy - radius, diameter, diameter)

        # Background arc (full top half)
        p.setPen(QPen(QColor(LINE), thickness, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(arc_rect, 0 * 16, 180 * 16)

        # Active arc — sweeps clockwise from 180° (west)
        active_span = -180 * (self._score / 100)
        p.setPen(QPen(self._color, thickness, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(arc_rect, 180 * 16, int(active_span * 16))

        # Big number
        num_font = mono_font(48, QFont.Weight.DemiBold)
        num_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 94)
        p.setFont(num_font)
        p.setPen(self._color)
        num_rect = QRectF(0, cy - radius * 0.78, w, radius * 0.78)
        p.drawText(num_rect, Qt.AlignmentFlag.AlignCenter, str(self._score))

        # "/ 100" subtitle
        sub_font = mono_font(8, QFont.Weight.Medium)
        sub_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 130)
        p.setFont(sub_font)
        p.setPen(QColor(TEXT_GHOST))
        sub_rect = QRectF(0, cy - 14, w, 18)
        p.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, "/ 100")


# ─── Bar track ──────────────────────────────────────────────────────────────
class BarTrack(QWidget):
    """Pill-shaped track + fill, anti-aliased."""

    def __init__(self, color_hex: str, is_stub: bool = False, parent=None):
        super().__init__(parent)
        self._color = QColor(color_hex)
        self._value = 50
        self._is_stub = is_stub
        self.setFixedHeight(7)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setValue(self, v: float):
        self._value = max(0, min(100, float(v)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        radius = h / 2

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(LINE_SOFT))
        p.drawRoundedRect(QRectF(0, 0, w, h), radius, radius)

        fill_w = max(h, w * (self._value / 100))   # at least h so cap stays round
        fill_w = min(fill_w, w)
        color = QColor(LINE) if self._is_stub else self._color
        p.setBrush(color)
        p.drawRoundedRect(QRectF(0, 0, fill_w, h), radius, radius)


# ─── Channel bar row ────────────────────────────────────────────────────────
class ChannelBar(QWidget):
    """label | track | numeric value | weight%"""

    def __init__(self, label: str, color_hex: str, weight_pct: int,
                 is_stub: bool = False, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        lbl_color = TEXT_FAINT if is_stub else color_hex
        lbl = QLabel(label)
        lbl.setFixedWidth(58)
        lbl.setFont(ui_font(10, QFont.Weight.Medium))
        lbl.setStyleSheet(f"color: {lbl_color};")

        self._track = BarTrack(color_hex, is_stub=is_stub)

        self._num = QLabel("50")
        self._num.setFixedWidth(32)
        self._num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._num.setFont(mono_font(10, QFont.Weight.Medium))
        self._num.setStyleSheet(f"color: {TEXT_FAINT if is_stub else TEXT};")

        w_lbl = QLabel(f"{weight_pct}%")
        w_lbl.setFixedWidth(36)
        w_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        w_lbl.setFont(mono_font(8))
        w_lbl.setStyleSheet(f"color: {TEXT_GHOST}; letter-spacing: 0.5px;")

        layout.addWidget(lbl)
        layout.addWidget(self._track, 1)
        layout.addWidget(self._num)
        layout.addWidget(w_lbl)

    def setValue(self, value: float):
        self._track.setValue(value)
        self._num.setText(str(int(round(value))))


# ─── Trust band badge ───────────────────────────────────────────────────────
class TrustBadge(QLabel):
    """Pill that recolors with the trust band."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(28)
        self.setFont(mono_font(10, QFont.Weight.DemiBold))
        self.setBand("Calibrating…", TEXT_DIM)

    def setBand(self, label: str, color_hex: str):
        self.setText(label.upper())
        c = QColor(color_hex)
        bg = f"rgba({c.red()}, {c.green()}, {c.blue()}, 26)"  # ~10% opacity
        self.setStyleSheet(f"""
            QLabel {{
                color: {color_hex};
                border: 1px solid {color_hex};
                background-color: {bg};
                border-radius: 14px;
                padding: 6px 16px;
                letter-spacing: 1px;
            }}
        """)


# ─── Status dot (header) ────────────────────────────────────────────────────
class StatusDot(QWidget):
    """8px round indicator with named states (active / loading / idle / off)."""

    STATES = {
        "active":  "#2da46a",
        "loading": "#c9a23a",
        "idle":    TEXT_GHOST,
        "off":     "#b3b8c4",
    }

    def __init__(self, state: str = "loading", parent=None):
        super().__init__(parent)
        self._state = state
        self.setFixedSize(QSize(10, 10))

    def setState(self, state: str):
        if state != self._state:
            self._state = state
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.STATES.get(self._state, TEXT_GHOST)))
        p.drawEllipse(1, 1, 8, 8)


# ─── Metric box (the cream-tile pattern translated to cool slate) ───────────
class MetricBox(QFrame):
    """Small two-row tile: tiny uppercase label, mono value."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("metricBox")
        self.setStyleSheet(f"""
            #metricBox {{
                background: {PANEL_2};
                border: 1px solid {LINE_SOFT};
                border-radius: 6px;
            }}
        """)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 9, 12, 9)
        v.setSpacing(2)

        self._label = QLabel(label.upper())
        self._label.setFont(ui_font(8, QFont.Weight.DemiBold))
        self._label.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.0px;")

        self._value = QLabel("—")
        self._value.setFont(mono_font(13, QFont.Weight.Medium))
        self._value.setStyleSheet(f"color: {TEXT};")

        v.addWidget(self._label)
        v.addWidget(self._value)

    def setValue(self, text: str):
        self._value.setText(text)


# ─── Panel header ───────────────────────────────────────────────────────────
class PanelHead(QFrame):
    """Uppercase title left, faint mono identifier right."""

    def __init__(self, title: str, identifier: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("panelHead")
        self.setStyleSheet(head_qss())
        self.setFixedHeight(42)

        h = QHBoxLayout(self)
        h.setContentsMargins(22, 0, 22, 0)

        self._title = QLabel(title.upper())
        self._title.setFont(ui_font(8, QFont.Weight.DemiBold))
        self._title.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px;")

        self._id = QLabel(identifier)
        self._id.setFont(mono_font(8))
        self._id.setStyleSheet(f"color: {TEXT_GHOST};")

        h.addWidget(self._title)
        h.addStretch()
        h.addWidget(self._id)

    def setIdentifier(self, text: str):
        self._id.setText(text)


# ─── Waveform widget (custom-painted audio buffer) ──────────────────────────
class WaveformWidget(QWidget):
    """Compact polyline of audio samples — uses violet (vocal channel hue)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setSamples(self, samples):
        # Downsample to widget width for cheap drawing
        self._samples = list(samples)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PANEL_2))
        p.drawRoundedRect(QRectF(0, 0, w, h), 6, 6)

        if len(self._samples) < 2:
            return

        # Downsample
        n_target = min(w, 200)
        stride = max(1, len(self._samples) // n_target)
        pts = self._samples[::stride]

        p.setPen(QPen(QColor(C_VOCAL), 1.4, Qt.PenStyle.SolidLine))
        cy = h / 2
        amp = h / 2 - 4
        x_step = w / max(1, len(pts) - 1)

        prev_x = 0
        prev_y = cy
        for i, s in enumerate(pts):
            x = i * x_step
            y = cy - max(-1.0, min(1.0, float(s))) * amp
            if i > 0:
                p.drawLine(int(prev_x), int(prev_y), int(x), int(y))
            prev_x, prev_y = x, y


# ─── Spectrum bars (FFT) ────────────────────────────────────────────────────
class SpectrumWidget(QWidget):
    """Lo-fi vertical bars driven by an externally-computed magnitude array."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bins = []
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setBins(self, bins):
        self._bins = list(bins)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PANEL_2))
        p.drawRoundedRect(QRectF(0, 0, w, h), 6, 6)

        if not self._bins:
            return

        n = min(len(self._bins), 64)
        bin_w = (w - 12) / n
        gap = max(1, bin_w * 0.2)
        bar_w = bin_w - gap

        p.setBrush(QColor(C_VOCAL))
        for i in range(n):
            mag = max(0.0, min(1.0, float(self._bins[i])))
            bar_h = max(1, mag * (h - 8))
            x = 6 + i * bin_w
            y = h - 4 - bar_h
            p.drawRoundedRect(QRectF(x, y, bar_w, bar_h), 1.5, 1.5)
