"""
TEMPORARY — live BPM readout for checking the strap against a watch.

A small always-on-top window showing the heart rate the Polar H10 is
reporting right now, so it can be compared side by side with a watch.
Not part of the dashboard's measurement pipeline: it only reads what
HRVAnalyzer has already received and changes nothing about the score.

Deliberately kept in its own file with a three-line integration in
main.py (construct, update, close) so the whole thing can be deleted
without touching anything that matters. Toggle with the H key.

Reads two numbers because they answer different questions:

  BPM       — the strap's own heart-rate field. Polar averages this over
              several beats, and it is what a watch shows, so this is the
              number to compare.
  inst      — 60000 / the last R-R interval, i.e. the rate implied by one
              single beat. Much jumpier. Useful as a sanity check that the
              beats themselves are being detected cleanly; a wildly
              swinging inst next to a steady BPM means dropped or spurious
              beats, usually poor electrode contact.

The age indicator matters more than it looks. `heart_rate` holds its last
value indefinitely after the strap stops sending, so a dropped connection
looks exactly like a resting heart rate. Anything older than two seconds
is stale and the number greys out.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from theme import (C_HRV, LINE, PANEL, TEXT, TEXT_DIM, TEXT_FAINT, TEXT_GHOST,
                   mono_font, sp, ui_font)

# Beats older than this are treated as stale rather than current.
STALE_AFTER_S = 2.0


class BpmMonitor(QWidget):
    """Frameless, always-on-top BPM readout. Drag anywhere on it to move."""

    def __init__(self, parent=None):
        # Tool + FramelessWindowHint keeps it out of the taskbar/dock and off
        # the window list; it is scaffolding, not a second app window.
        super().__init__(None, Qt.WindowType.Tool
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        card = QFrame(self)
        card.setObjectName("bpmCard")
        card.setStyleSheet(
            f"#bpmCard {{ background: {PANEL}; border: {sp(1)}px solid {LINE};"
            f" border-radius: {sp(10)}px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)

        v = QVBoxLayout(card)
        v.setContentsMargins(sp(16), sp(10), sp(16), sp(12))
        v.setSpacing(sp(2))

        title = QLabel("HEART RATE · TEMP")
        title.setFont(ui_font(8, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {TEXT_GHOST}; letter-spacing: 1.3px;")
        v.addWidget(title)

        # The number, and its unit sitting on the same baseline.
        num_row = QHBoxLayout()
        num_row.setSpacing(sp(6))
        self._bpm = QLabel("--")
        self._bpm.setFont(mono_font(46, QFont.Weight.DemiBold))
        self._bpm.setStyleSheet(f"color: {C_HRV};")
        unit = QLabel("bpm")
        unit.setFont(ui_font(11))
        unit.setStyleSheet(f"color: {TEXT_FAINT};")
        num_row.addWidget(self._bpm)
        num_row.addWidget(unit, 0, Qt.AlignmentFlag.AlignBottom)
        num_row.addStretch()
        v.addLayout(num_row)

        self._detail = QLabel("waiting for strap")
        self._detail.setFont(mono_font(9))
        self._detail.setStyleSheet(f"color: {TEXT_DIM};")
        v.addWidget(self._detail)

        self._status = QLabel("")
        self._status.setFont(ui_font(9))
        self._status.setStyleSheet(f"color: {TEXT_FAINT};")
        v.addWidget(self._status)

        hint = QLabel("H to hide · drag to move")
        hint.setFont(ui_font(8))
        hint.setStyleSheet(f"color: {TEXT_GHOST};")
        v.addWidget(hint)

        self._drag_offset: QPoint | None = None
        self.resize(sp(210), sp(150))

    # ── Positioning ─────────────────────────────────────────────────────────

    def place_near(self, other: QWidget) -> None:
        """Sit just inside the top-right corner of *other*."""
        geo = other.frameGeometry()
        self.move(geo.right() - self.width() - sp(24), geo.top() + sp(64))

    # Frameless windows get no title bar, so dragging is done by hand.
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        event.accept()

    # ── Update ──────────────────────────────────────────────────────────────

    def update_from(self, display: dict) -> None:
        """Refresh from one HRVAnalyzer.get_display() dict."""
        hr = display.get("heart_rate")
        rr = display.get("last_rr_ms")
        rmssd = display.get("rmssd_ms")
        status = str(display.get("status", "unknown"))
        last_at = display.get("last_beat_at")

        age = (time.time() - float(last_at)) if last_at else None
        fresh = age is not None and age <= STALE_AFTER_S

        self._bpm.setText(str(int(hr)) if hr else "--")
        # Grey the number out rather than blanking it: seeing the last value
        # alongside "stale" is more informative than seeing it disappear.
        self._bpm.setStyleSheet(f"color: {C_HRV if fresh else TEXT_GHOST};")

        bits = []
        if rr:
            bits.append(f"inst {60000 / float(rr):5.1f}")
            bits.append(f"rr {int(rr)}ms")
        if rmssd is not None:
            bits.append(f"rmssd {float(rmssd):.0f}ms")
        self._detail.setText("  ".join(bits) if bits else "no beats yet")

        if fresh:
            self._status.setText(f"live · {age:.1f}s ago")
            self._status.setStyleSheet(f"color: {TEXT};")
        elif age is not None:
            self._status.setText(f"STALE · {age:.0f}s ago ({status})")
            self._status.setStyleSheet(f"color: {C_HRV};")
        else:
            self._status.setText(status)
            self._status.setStyleSheet(f"color: {TEXT_FAINT};")
