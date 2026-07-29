"""baseline_dialog.py — what the calibration actually measured.

Shown once, between the end of the calibration window and the start of the
session, so the researcher sees the numbers the rest of the run will be scored
against *before* any data is recorded. A bad window — participant leaning out
of frame, nobody speaking, strap not paired — is cheap to redo at this point
and expensive to discover afterwards.

Every row is marked measured or default. TrustEngine.apply_calibration skips
any signal that was never captured, leaving it on the population reference
value, and an unmarked default reads as if it were this participant's
measurement. The dialog therefore leads with a count of what was missed and
only then lists values.

Usage:
    dlg = BaselineDialog(report, coverage_pct=91.4, parent=win)
    dlg.exec()          # modal — the session starts when it closes
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from theme import (ACCENT, BG, DANGER, LINE_SOFT, PANEL, PANEL_2, TEXT,
                   TEXT_DIM, TEXT_FAINT, TEXT_GHOST, mono_font, sp, ui_font)

# Coverage below this means the face was missing for much of the window, so the
# facial and gaze baselines rest on comparatively few frames.
LOW_COVERAGE_PCT = 80.0


def _fmt(value, unit: str) -> str:
    """Baselines span 0.005 (spectral flux) to 1000 (pupil), so a fixed number
    of decimals either rounds the small ones to nothing or pads the large ones
    with noise. Pick the precision from the magnitude."""
    if value is None:
        return "—"
    v = float(value)
    if abs(v) >= 100:
        text = f"{v:.0f}"
    elif abs(v) >= 10:
        text = f"{v:.1f}"
    elif abs(v) >= 1:
        text = f"{v:.2f}"
    else:
        text = f"{v:.4f}".rstrip("0").rstrip(".") or "0"
    return f"{text} {unit}".strip() if unit else text


class BaselineDialog(QDialog):
    def __init__(self, report, coverage_pct: float | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration baseline")
        self.setModal(True)
        self.setStyleSheet(f"QDialog {{ background: {BG}; }}")

        defaults = [(sensor, r) for sensor, rws in report
                    for r in rws if not r["measured"]]

        root = QVBoxLayout(self)
        root.setContentsMargins(sp(24), sp(20), sp(24), sp(18))
        root.setSpacing(sp(10))

        title = QLabel("Calibration complete")
        title.setFont(ui_font(20, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {TEXT};")
        root.addWidget(title)

        sub = QLabel("These resting values are what the whole session is scored "
                     "against. Everything below is measured from this "
                     "participant unless marked otherwise.")
        sub.setFont(ui_font(11))
        sub.setStyleSheet(f"color: {TEXT_DIM};")
        sub.setWordWrap(True)
        root.addWidget(sub)

        warning = self._build_warning(defaults, coverage_pct)
        if warning is not None:
            root.addWidget(warning)

        # Scrolled, because the row count grows with every sensor and the
        # dialog must not push its buttons off a small laptop screen.
        inner = QWidget()
        inner.setStyleSheet(f"background: {BG};")
        grid = QVBoxLayout(inner)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(sp(8))
        for sensor, rws in report:
            grid.addWidget(self._build_group(sensor, rws))
        grid.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {BG}; border: none; }}")
        scroll.viewport().setStyleSheet(f"background: {BG};")
        root.addWidget(scroll, 1)

        note = QLabel("Also written to the session Excel: a baseline row on each "
                      "sensor sheet, and the full set under Score Config.")
        note.setFont(ui_font(9))
        note.setStyleSheet(f"color: {TEXT_GHOST};")
        note.setWordWrap(True)
        root.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        start = QPushButton("Start session")
        start.setFont(ui_font(12, QFont.Weight.DemiBold))
        start.setCursor(Qt.CursorShape.PointingHandCursor)
        start.setMinimumHeight(sp(36))
        start.setStyleSheet(
            f"QPushButton {{ background: {ACCENT}; color: #ffffff; border: none;"
            f" border-radius: {sp(6)}px; padding: {sp(6)}px {sp(22)}px; }}"
        )
        start.clicked.connect(self.accept)
        start.setDefault(True)
        btn_row.addWidget(start)
        root.addLayout(btn_row)

        self.resize(sp(560), sp(620))

    # ── Pieces ──────────────────────────────────────────────────────────────

    def _build_warning(self, defaults, coverage_pct):
        """Banner naming what was not measured, or None when all is well."""
        lines = []
        if defaults:
            names = ", ".join(f"{s} · {r['label']}" for s, r in defaults)
            lines.append(f"Not measured, using population defaults: {names}. "
                         f"Those signals are not personalised to this participant.")
        if coverage_pct is not None and coverage_pct < LOW_COVERAGE_PCT:
            lines.append(f"A face was detected in only {coverage_pct:.0f} % of "
                         f"calibration frames, so the facial and gaze baselines "
                         f"rest on relatively few samples.")
        if not lines:
            return None

        box = QFrame()
        box.setStyleSheet(
            f"background: #fdf3f0; border: {sp(1)}px solid {DANGER};"
            f" border-radius: {sp(6)}px;"
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(sp(12), sp(9), sp(12), sp(9))
        v.setSpacing(sp(4))
        for text in lines:
            lbl = QLabel(text)
            lbl.setFont(ui_font(10))
            lbl.setStyleSheet(f"color: {DANGER}; background: transparent; border: none;")
            lbl.setWordWrap(True)
            v.addWidget(lbl)
        return box

    def _build_group(self, sensor: str, rows) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"background: {PANEL}; border: {sp(1)}px solid {LINE_SOFT};"
            f" border-radius: {sp(8)}px;"
        )
        v = QVBoxLayout(card)
        v.setContentsMargins(sp(14), sp(10), sp(14), sp(11))
        v.setSpacing(sp(6))

        head = QLabel(sensor.upper())
        head.setFont(ui_font(9, QFont.Weight.DemiBold))
        head.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.2px;"
                           f" background: transparent; border: none;")
        v.addWidget(head)

        g = QGridLayout()
        g.setHorizontalSpacing(sp(12))
        g.setVerticalSpacing(sp(3))
        g.setColumnStretch(0, 0)
        g.setColumnStretch(1, 0)
        g.setColumnStretch(2, 1)

        for i, r in enumerate(rows):
            name = QLabel(r["label"])
            name.setFont(ui_font(11))
            name.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
            name.setMinimumWidth(sp(120))

            val = QLabel(_fmt(r["value"], r["unit"]))
            val.setFont(mono_font(12, QFont.Weight.DemiBold))
            val.setStyleSheet(f"color: {TEXT if r['measured'] else TEXT_GHOST};"
                              f" background: transparent; border: none;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setMinimumWidth(sp(96))

            # The note explains the unit; "default" replaces it, because which
            # participant a number describes matters more than what it means.
            tail = "default — not measured" if not r["measured"] else r["note"]
            hint = QLabel(tail)
            hint.setFont(ui_font(9))
            hint.setStyleSheet(
                f"color: {DANGER if not r['measured'] else TEXT_GHOST};"
                f" background: transparent; border: none;"
            )
            hint.setWordWrap(True)

            g.addWidget(name, i, 0)
            g.addWidget(val,  i, 1)
            g.addWidget(hint, i, 2)

        v.addLayout(g)
        return card
