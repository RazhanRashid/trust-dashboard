"""overlays.py — full-window screens that sit on top of the main dashboard.

Three overlays:
  * OverviewScreen      — landing page with past sessions and a Start button
  * CalibrationOverlay  — 30-second calibration with live camera preview
  * SessionSummary      — post-session stats + history chart + export
"""

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import (QPainter, QPen, QColor, QFont, QImage, QPixmap,
                          QPainterPath)
from PyQt6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QVBoxLayout,
                              QHBoxLayout, QGridLayout, QScrollArea, QSizePolicy,
                              QMessageBox)
from PyQt6.QtGui import QDesktopServices

from theme import (BG, BG_DEEP, PANEL, PANEL_2, LINE, LINE_SOFT,
                    TEXT, TEXT_DIM, TEXT_FAINT, TEXT_GHOST,
                    C_FACIAL, C_VOCAL, C_GAZE, C_HRV, ACCENT, DANGER,
                    ui_font, mono_font, trust_band, panel_qss, TRUST_BANDS)
from widgets import BarTrack, ChannelBar, MetricBox, PanelHead


# ═══════════════════════════════════════════════════════════════════════════
# Custom countdown arc widget
# ═══════════════════════════════════════════════════════════════════════════
class CountdownArc(QWidget):
    """Full circle that fills clockwise from 12 o'clock as time elapses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0      # 0 → 1
        self._remaining = 30
        self.setFixedSize(QSize(180, 180))

    def setProgress(self, progress: float, remaining_sec: int):
        self._progress = max(0.0, min(1.0, float(progress)))
        self._remaining = int(remaining_sec)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()
        thickness = 13
        margin = thickness + 4
        rect = QRectF(margin, margin, w - margin * 2, h - margin * 2)

        # Background ring
        p.setPen(QPen(QColor(LINE_SOFT), thickness, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 0, 360 * 16)

        # Active arc — starts at 12 o'clock (90°), sweeps clockwise
        sweep = -360 * self._progress
        p.setPen(QPen(QColor(ACCENT), thickness, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 90 * 16, int(sweep * 16))

        # Center number
        num_font = mono_font(36, QFont.Weight.DemiBold)
        p.setFont(num_font)
        p.setPen(QColor(ACCENT))
        num_rect = QRectF(0, h / 2 - 28, w, 38)
        p.drawText(num_rect, Qt.AlignmentFlag.AlignCenter, str(self._remaining))

        # Subtitle
        sub_font = ui_font(9)
        p.setFont(sub_font)
        p.setPen(QColor(TEXT_FAINT))
        sub_rect = QRectF(0, h / 2 + 8, w, 18)
        p.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, "seconds left")


# ═══════════════════════════════════════════════════════════════════════════
# Post-hoc waiting screen
# ═══════════════════════════════════════════════════════════════════════════

class _SpinnerArc(QWidget):
    """Rotating arc that spins indefinitely while OpenFace is processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0          # current start angle of the arc head
        self.setFixedSize(QSize(120, 120))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)    # ~60 fps rotation

    def _tick(self):
        self._angle = (self._angle - 6) % 360   # rotate 6° per frame clockwise
        self.update()

    def stop(self):
        self._timer.stop()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        thickness = 10
        margin    = thickness + 4
        rect = QRectF(margin, margin,
                      self.width() - margin * 2, self.height() - margin * 2)
        # Background ring
        p.setPen(QPen(QColor(LINE_SOFT), thickness,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 0, 360 * 16)
        # Spinning arc — 270° sweep so there is always a visible gap
        p.setPen(QPen(QColor(ACCENT), thickness,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, self._angle * 16, 270 * 16)


class PosthocWaitingScreen(QWidget):
    """
    Full-window screen shown while OpenFace post-hoc analysis runs.
    Automatically dismissed by main.py once the background thread finishes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(2)

        # Centre card
        card = QFrame()
        card.setObjectName("waitCard")
        card.setStyleSheet(f"""
            #waitCard {{
                background: {PANEL};
                border: 1px solid {LINE};
                border-radius: 12px;
            }}
        """)
        card.setFixedWidth(440)
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(48, 48, 48, 48)
        card_l.setSpacing(20)
        card_l.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Spinner
        self._spinner = _SpinnerArc()
        card_l.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        # Heading
        heading = QLabel("Analysing session…")
        heading.setFont(ui_font(16, QFont.Weight.Bold))
        heading.setStyleSheet(f"color: {TEXT}; background: transparent;")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_l.addWidget(heading)

        # Subtitle
        sub = QLabel("OpenFace is processing the recording.\nThis usually takes 20–60 seconds.")
        sub.setFont(ui_font(11))
        sub.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_l.addWidget(sub)

        # Elapsed time counter
        self._elapsed_lbl = QLabel("0 s")
        self._elapsed_lbl.setFont(mono_font(10))
        self._elapsed_lbl.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent;")
        self._elapsed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_l.addWidget(self._elapsed_lbl)

        self._start_time = time.time()
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._update_elapsed)
        self._tick_timer.start(1000)

        # Centre the card horizontally
        h = QHBoxLayout()
        h.addStretch()
        h.addWidget(card)
        h.addStretch()
        root.addLayout(h)
        root.addStretch(3)

    def _update_elapsed(self):
        secs = int(time.time() - self._start_time)
        self._elapsed_lbl.setText(f"{secs} s")

    def stop_spinner(self):
        """Call before removing the screen so the timers are cleaned up."""
        self._spinner.stop()
        self._tick_timer.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Calibration overlay (full-window)
# ═══════════════════════════════════════════════════════════════════════════
class CalibrationOverlay(QWidget):
    """30-second calibration screen — covers the entire window."""

    start_clicked = pyqtSignal()    # user clicked Start Calibration
    skip_clicked  = pyqtSignal()    # user clicked Skip

    SENTENCES = [
        '"The meeting is scheduled for Thursday at three in the afternoon."',
        '"I usually take the main road when the weather allows it."',
        '"She mentioned the project would wrap up by the end of the month."',
        '"Can you send me the details when you get a chance?"',
        '"The quarterly report includes updated figures from all four regions."',
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG};")
        self._intro_visible = True

        v = QVBoxLayout(self)
        v.setContentsMargins(60, 50, 60, 50)
        v.setSpacing(28)

        # Title
        title = QLabel("Calibration")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(ui_font(22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT};")

        sub = QLabel("Sit comfortably, look at the camera, and read aloud naturally for 30 seconds.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setFont(ui_font(11))
        sub.setStyleSheet(f"color: {TEXT_FAINT};")

        head = QVBoxLayout()
        head.setSpacing(8)
        head.addWidget(title)
        head.addWidget(sub)
        v.addLayout(head)

        # Two-column body
        body = QHBoxLayout()
        body.setSpacing(20)

        # ── Left: camera preview card ──
        cam_card = QFrame()
        cam_card.setObjectName("calCam")
        cam_card.setStyleSheet(panel_qss("calCam"))
        cam_l = QVBoxLayout(cam_card)
        cam_l.setContentsMargins(22, 18, 22, 22)
        cam_l.setSpacing(12)

        cam_title = QLabel("LIVE PREVIEW")
        cam_title.setFont(ui_font(8, QFont.Weight.DemiBold))
        cam_title.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px;")
        cam_l.addWidget(cam_title)

        self._video = QLabel()
        self._video.setMinimumHeight(280)
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setStyleSheet(f"""
            QLabel {{
                background: #2a3142;
                border: 1px solid {LINE};
                border-radius: 6px;
                color: {TEXT_GHOST};
            }}
        """)
        self._video.setText("waiting for camera…")
        cam_l.addWidget(self._video, 1)

        self._sentence = QLabel(self.SENTENCES[0])
        self._sentence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sentence.setFont(ui_font(13))
        self._sentence.setWordWrap(True)
        self._sentence.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")
        cam_l.addWidget(self._sentence)

        body.addWidget(cam_card, 3)

        # ── Right: status card ──
        stat_card = QFrame()
        stat_card.setObjectName("calStat")
        stat_card.setStyleSheet(panel_qss("calStat"))
        stat_l = QVBoxLayout(stat_card)
        stat_l.setContentsMargins(22, 18, 22, 22)
        stat_l.setSpacing(14)

        stat_title = QLabel("CALIBRATING")
        stat_title.setFont(ui_font(8, QFont.Weight.DemiBold))
        stat_title.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px;")
        stat_l.addWidget(stat_title)

        # Countdown arc
        self._arc = CountdownArc()
        arc_wrap = QHBoxLayout()
        arc_wrap.addStretch()
        arc_wrap.addWidget(self._arc)
        arc_wrap.addStretch()
        stat_l.addLayout(arc_wrap)

        # Progress bar
        prog_wrap = QVBoxLayout()
        prog_wrap.setSpacing(6)
        prog_label = QLabel("Progress")
        prog_label.setFont(ui_font(9))
        prog_label.setStyleSheet(f"color: {TEXT_FAINT};")
        self._prog = BarTrack(ACCENT)
        prog_wrap.addWidget(prog_label)
        prog_wrap.addWidget(self._prog)
        stat_l.addLayout(prog_wrap)

        # Divider
        sep = QFrame()
        sep.setStyleSheet(f"background-color: {LINE_SOFT}; border: none;")
        sep.setFixedHeight(1)
        stat_l.addWidget(sep)

        # Face + Voice indicators
        self._face_dot = self._make_dot()
        self._face_lbl = QLabel("Looking for face…")
        self._face_lbl.setFont(ui_font(10))
        self._face_lbl.setStyleSheet(f"color: {TEXT_FAINT};")
        face_row = QHBoxLayout()
        face_row.setSpacing(10)
        face_row.addWidget(self._face_dot)
        face_name = QLabel("Face")
        face_name.setFixedWidth(48)
        face_name.setFont(ui_font(10, QFont.Weight.DemiBold))
        face_name.setStyleSheet(f"color: {TEXT};")
        face_row.addWidget(face_name)
        face_row.addWidget(self._face_lbl)
        face_row.addStretch()
        stat_l.addLayout(face_row)

        self._voice_dot = self._make_dot()
        self._voice_lbl = QLabel("Speak to calibrate…")
        self._voice_lbl.setFont(ui_font(10))
        self._voice_lbl.setStyleSheet(f"color: {TEXT_FAINT};")
        voice_row = QHBoxLayout()
        voice_row.setSpacing(10)
        voice_row.addWidget(self._voice_dot)
        voice_name = QLabel("Voice")
        voice_name.setFixedWidth(48)
        voice_name.setFont(ui_font(10, QFont.Weight.DemiBold))
        voice_name.setStyleSheet(f"color: {TEXT};")
        voice_row.addWidget(voice_name)
        voice_row.addWidget(self._voice_lbl)
        voice_row.addStretch()
        stat_l.addLayout(voice_row)

        stat_l.addStretch()
        body.addWidget(stat_card, 2)
        v.addLayout(body, 1)

        # Footer: start button + skip link
        foot = QHBoxLayout()
        foot.addStretch()

        self._start_btn = QPushButton("Start Calibration")
        self._start_btn.setFont(ui_font(11, QFont.Weight.DemiBold))
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 10px 24px;
            }}
            QPushButton:hover {{ background: #1f5fa3; }}
        """)
        self._start_btn.clicked.connect(self._on_start)

        skip = QLabel("<a href='#skip' style='color: " + TEXT_FAINT +
                      "; text-decoration: underline;'>Skip calibration</a>")
        skip.setFont(ui_font(10))
        skip.setCursor(Qt.CursorShape.PointingHandCursor)
        skip.setOpenExternalLinks(False)
        skip.linkActivated.connect(lambda _: self.skip_clicked.emit())

        foot.addWidget(self._start_btn)
        foot.addSpacing(20)
        foot.addWidget(skip)
        foot.addStretch()
        v.addLayout(foot)

    def _make_dot(self) -> QFrame:
        dot = QFrame()
        dot.setFixedSize(QSize(10, 10))
        dot.setStyleSheet(f"background: {TEXT_GHOST}; border-radius: 5px;")
        return dot

    def _set_dot(self, dot: QFrame, color: str):
        dot.setStyleSheet(f"background: {color}; border-radius: 5px;")

    def _on_start(self):
        self._start_btn.setEnabled(False)
        self._start_btn.setText("Calibrating…")
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {LINE_SOFT};
                color: {TEXT_FAINT};
                border: 0;
                border-radius: 6px;
                padding: 10px 24px;
            }}
        """)
        self.start_clicked.emit()

    # ── Public update API ──
    def update_progress(self, elapsed_sec: float, total_sec: float):
        progress = elapsed_sec / total_sec
        remaining = max(0, int(math.ceil(total_sec - elapsed_sec)))
        self._arc.setProgress(progress, remaining)
        self._prog.setValue(progress * 100)
        # Rotate sentence every 6 seconds
        idx = int(elapsed_sec // 6) % len(self.SENTENCES)
        self._sentence.setText(self.SENTENCES[idx])

    def update_preview(self, frame_bgr, face_data: dict | None):
        if frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._video.width(), self._video.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video.setPixmap(pix)

    def update_indicators(self, face_detected: bool, voice_samples: int):
        if face_detected:
            self._set_dot(self._face_dot, "#2da46a")
            self._face_lbl.setText("Detected ✓")
            self._face_lbl.setStyleSheet(f"color: {TEXT};")
        else:
            self._set_dot(self._face_dot, TEXT_GHOST)
            self._face_lbl.setText("Looking for face…")
            self._face_lbl.setStyleSheet(f"color: {TEXT_FAINT};")

        if voice_samples > 20:
            self._set_dot(self._voice_dot, "#2da46a")
            self._voice_lbl.setText(f"Captured ({voice_samples} samples) ✓")
            self._voice_lbl.setStyleSheet(f"color: {TEXT};")
        else:
            self._set_dot(self._voice_dot, TEXT_GHOST)
            self._voice_lbl.setText("Speak to calibrate…")
            self._voice_lbl.setStyleSheet(f"color: {TEXT_FAINT};")


# ─── Pixmap helper ──────────────────────────────────────────────────────────
def _rounded_pixmap(pixmap: QPixmap, radius: int = 8) -> QPixmap:
    """Return a copy of *pixmap* with rounded corners (anti-aliased clip)."""
    out = QPixmap(pixmap.size())
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(out.rect()), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pixmap)
    p.end()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Session summary
# ═══════════════════════════════════════════════════════════════════════════
class SessionSummary(QWidget):
    """Full-window summary screen after End Session."""

    back_clicked   = pyqtSignal()
    export_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG};")

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 32, 40, 32)
        v.setSpacing(18)

        # Header
        head = QFrame()
        head.setObjectName("sumHead")
        head.setStyleSheet(panel_qss("sumHead"))
        head_l = QHBoxLayout(head)
        head_l.setContentsMargins(24, 20, 24, 20)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("Session complete")
        title.setFont(ui_font(18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        self._meta = QLabel("—")
        self._meta.setFont(mono_font(10))
        self._meta.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent;")
        title_col.addWidget(title)
        title_col.addWidget(self._meta)
        head_l.addLayout(title_col)
        head_l.addStretch()
        v.addWidget(head)

        # Three cards in a row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        # Card 1 — Overall composure + thumbnail
        c1 = self._make_card()
        c1_l = c1.layout()
        c1_l.addWidget(self._tile_title("OVERALL COMPOSURE"))


        # Thumbnail (hidden until populate() loads a real image)
        self._thumb_lbl = QLabel()
        self._thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_lbl.setFixedSize(240, 135)
        self._thumb_lbl.setStyleSheet(f"""
            QLabel {{
                background: {BG_DEEP};
                border: 1px solid {LINE};
                border-radius: 8px;
                color: {TEXT_GHOST};
                font-family: 'JetBrains Mono', monospace;
                font-size: 9pt;
            }}
        """)
        self._thumb_lbl.setText("no face captured")
        thumb_wrap = QHBoxLayout()
        thumb_wrap.addStretch()
        thumb_wrap.addWidget(self._thumb_lbl)
        thumb_wrap.addStretch()
        c1_l.addLayout(thumb_wrap)
        c1_l.addSpacing(12)

        self._big_num = QLabel("—")
        self._big_num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._big_num.setFont(mono_font(48, QFont.Weight.DemiBold))
        self._big_num.setStyleSheet(f"color: {TEXT}; background: transparent;")
        self._big_label = QLabel("—")
        self._big_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._big_label.setFont(mono_font(11, QFont.Weight.DemiBold))
        self._big_label.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1px; background: transparent;")
        c1_l.addWidget(self._big_num)
        c1_l.addWidget(self._big_label)
        c1_l.addSpacing(2)
        hint = QLabel("session average")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFont(ui_font(9))
        hint.setStyleSheet(f"color: {TEXT_GHOST}; background: transparent;")
        c1_l.addWidget(hint)
        c1_l.addSpacing(8)

        # Play + Delete buttons (hidden by default)
        media_row = QHBoxLayout()
        media_row.setSpacing(0)

        self._play_btn = QPushButton("▶  Play recording")
        self._play_btn.setFont(ui_font(9, QFont.Weight.Medium))
        self._play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._play_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {ACCENT};
                border: 1px solid {ACCENT}; border-radius: 6px;
                padding: 7px 14px;
            }}
            QPushButton:hover {{ background: rgba(42, 99, 163, 0.12); }}
        """)
        self._play_btn.hide()

        self._del_btn = QPushButton("Delete recording")
        self._del_btn.setFont(ui_font(9))
        self._del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._del_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_FAINT};
                border: 1px solid {LINE}; border-radius: 6px;
                padding: 7px 12px;
            }}
            QPushButton:hover {{ color: {DANGER}; border-color: {DANGER}; }}
        """)
        self._del_btn.hide()

        media_row.addStretch()
        media_row.addWidget(self._play_btn)
        media_row.addSpacing(24)
        media_row.addWidget(self._del_btn)
        media_row.addStretch()
        c1_l.addLayout(media_row)
        c1.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        cards_row.addWidget(c1, 1)

        # Card 2 — channel breakdown
        c2 = self._make_card()
        c2_l = c2.layout()
        c2_l.addWidget(self._tile_title("CHANNEL BREAKDOWN"))
        c2_l.addSpacing(8)
        self._sb = {}
        for key, label, color in [("facial", "Facial", C_FACIAL),
                                    ("vocal",  "Vocal",  C_VOCAL),
                                    ("gaze",   "Gaze",   C_GAZE),
                                    ("hrv",    "HRV",    C_HRV)]:
            is_hrv = (key == "hrv")
            lbl_color = TEXT_FAINT if is_hrv else color
            num_color = TEXT_FAINT if is_hrv else TEXT
            outer_v = QVBoxLayout()
            outer_v.setSpacing(2)
            row = QHBoxLayout()
            row.setSpacing(12)
            lbl = QLabel(label)
            lbl.setFixedWidth(58)
            lbl.setFont(ui_font(10, QFont.Weight.Medium))
            lbl.setStyleSheet(f"color: {lbl_color}; background: transparent;")
            track = BarTrack(color, is_stub=is_hrv)
            num = QLabel("—")
            num.setFixedWidth(32)
            num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            num.setFont(mono_font(10, QFont.Weight.Medium))
            num.setStyleSheet(f"color: {num_color}; background: transparent;")
            row.addWidget(lbl)
            row.addWidget(track)
            row.addWidget(num)
            outer_v.addLayout(row)
            if is_hrv:
                sub = QLabel("Sensor not connected")
                sub.setFont(ui_font(8))
                sub.setStyleSheet(f"color: {TEXT_FAINT}; padding-left: 70px; background: transparent;")
                outer_v.addWidget(sub)
            c2_l.addLayout(outer_v)
            self._sb[key] = (track, num)
        c2.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        cards_row.addWidget(c2, 2)

        # Card 3 — session info
        c3 = self._make_card()
        c3_l = c3.layout()
        c3_l.addWidget(self._tile_title("SESSION INFO"))
        c3_l.addSpacing(8)
        self._stats = {}
        for key, label in [("peak",     "Peak composure"),
                            ("low",      "Lowest composure")]:
            row = QHBoxLayout()
            row.setSpacing(8)
            k = QLabel(label)
            k.setFont(ui_font(10))
            k.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent;")
            v_lbl = QLabel("—")
            v_lbl.setFont(mono_font(10, QFont.Weight.Medium))
            v_lbl.setStyleSheet(f"color: {TEXT}; background: transparent;")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(k)
            row.addStretch()
            row.addWidget(v_lbl)
            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background-color: {LINE_SOFT}; border: none;")
            c3_l.addLayout(row)
            c3_l.addWidget(sep)
            self._stats[key] = v_lbl
        c3.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        cards_row.addWidget(c3, 1)

        v.addLayout(cards_row)

        # Trust history chart
        chart_card = self._make_card()
        chart_l = chart_card.layout()
        chart_l.addWidget(self._tile_title("COMPOSURE OVER SESSION"))
        chart_l.addSpacing(4)
        pg.setConfigOption("background", PANEL)
        self._chart = pg.PlotWidget()
        self._chart.setBackground(PANEL)
        self._chart.setYRange(0, 100, padding=0)
        self._chart.showGrid(x=False, y=True, alpha=0.12)
        for ax in ("top", "right"):
            self._chart.getAxis(ax).hide()
        self._chart.getAxis("bottom").setPen(pg.mkPen(LINE_SOFT))
        self._chart.getAxis("bottom").setTextPen(pg.mkPen(TEXT_GHOST))
        self._chart.getAxis("bottom").setStyle(tickLength=-4, showValues=True)
        self._chart.getAxis("bottom").setLabel(
            "Time (s)", **{"color": TEXT_GHOST, "font-size": "9pt"})
        self._chart.getAxis("left").setPen(pg.mkPen(LINE_SOFT))
        self._chart.getAxis("left").setTextPen(pg.mkPen(TEXT_GHOST))
        self._chart.getAxis("left").setStyle(tickLength=-4, showValues=True)
        self._chart.getPlotItem().getViewBox().setBorder(None)
        self._chart.setMouseEnabled(x=False, y=False)
        self._chart.setMenuEnabled(False)
        self._chart.hideButtons()
        self._chart.setFixedHeight(180)

        # Composure-band background zones
        prev_top = 100
        self._band_label_items = []  # list of (TextItem, mid_y)
        for thr, _lbl, hexc in TRUST_BANDS:
            cc = QColor(hexc)
            region = pg.LinearRegionItem(values=(thr, prev_top),
                                         orientation="horizontal", movable=False)
            region.setBrush(pg.mkBrush(cc.red(), cc.green(), cc.blue(), 35))
            for ln in region.lines:
                ln.setPen(pg.mkPen(None))
            region.setZValue(-10)
            self._chart.addItem(region)
            if thr > 0:
                div = pg.InfiniteLine(pos=thr, angle=0,
                    pen=pg.mkPen(LINE_SOFT, width=0.8))
                div.setZValue(-9)
                self._chart.addItem(div)
            mid_y = (thr + prev_top) / 2
            text_item = pg.TextItem(text=_lbl, anchor=(1, 0.5),
                color=QColor(TEXT_GHOST))
            text_item.setFont(ui_font(7))
            text_item.setPos(0, mid_y)
            text_item.setZValue(5)
            self._chart.addItem(text_item)
            self._band_label_items.append((text_item, mid_y))
            prev_top = thr

        c = QColor(ACCENT)
        self._chart_curve = self._chart.plot([], [],
            pen=pg.mkPen(ACCENT, width=2.4),
            fillLevel=0,
            brush=pg.mkBrush(c.red(), c.green(), c.blue(), 30))
        self._avg_line = None
        self._markers = None
        self._dot_labels = []

        self._chart.setStyleSheet("border: none;")
        chart_l.addWidget(self._chart)
        v.addWidget(chart_card)

        # Notable events card
        flags_card = self._make_card()
        flags_l = flags_card.layout()
        flags_l.addWidget(self._tile_title("NOTABLE EVENTS"))
        flags_l.addSpacing(4)
        self._flags_holder = QWidget()
        self._flags_holder.setStyleSheet("background: transparent;")
        self._flags_holder_l = QHBoxLayout(self._flags_holder)
        self._flags_holder_l.setContentsMargins(0, 0, 0, 0)
        self._flags_holder_l.setSpacing(8)
        flags_l.addWidget(self._flags_holder)
        v.addWidget(flags_card)

        # Buttons
        btn_row = QHBoxLayout()
        back_btn = QPushButton("⌂  Back to overview")
        back_btn.setFont(ui_font(10, QFont.Weight.Medium))
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PANEL}; color: {TEXT_DIM};
                border: 1px solid {LINE}; border-radius: 6px;
                padding: 10px 22px;
            }}
            QPushButton:hover {{ border-color: {TEXT_FAINT}; }}
        """)
        back_btn.clicked.connect(self.back_clicked.emit)

        exp_btn = QPushButton("↓  Export CSV")
        exp_btn.setFont(ui_font(10, QFont.Weight.Medium))
        exp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        exp_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; color: white;
                border: 0; border-radius: 6px;
                padding: 10px 22px;
            }}
            QPushButton:hover {{ background: #1f5fa3; }}
        """)
        exp_btn.clicked.connect(self.export_clicked.emit)

        btn_row.addWidget(back_btn)
        btn_row.addStretch()
        btn_row.addWidget(exp_btn)
        v.addLayout(btn_row)

    def _make_card(self) -> QFrame:
        f = QFrame()
        f.setObjectName("sumCard")
        f.setStyleSheet(panel_qss("sumCard"))
        l = QVBoxLayout(f)
        l.setContentsMargins(22, 18, 22, 22)
        l.setSpacing(10)
        return f

    def _tile_title(self, text: str) -> QLabel:
        t = QLabel(text)
        t.setFont(ui_font(8, QFont.Weight.DemiBold))
        t.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px; background: transparent;")
        return t

    def populate(self, stats: dict):
        total = int(stats.get("trust_total", 50))
        label, color = trust_band(total)
        self._meta.setText(
            f"Duration {stats.get('duration_str', '00:00')}  ·  "
            f"{stats.get('n_samples', 0)} samples recorded"
        )
        self._big_num.setText(str(total))
        self._big_num.setStyleSheet(f"color: {color}; background: transparent;")
        self._big_label.setText(label.upper())
        self._big_label.setStyleSheet(f"color: {color}; letter-spacing: 1px; background: transparent;")

        for key in ("facial", "vocal", "gaze"):
            v = int(stats.get(f"trust_{key}", 50))
            track, num = self._sb[key]
            track.setValue(v)
            num.setText(str(v))
        # HRV is inactive — show stub state
        hrv_track, hrv_num = self._sb["hrv"]
        hrv_track.setValue(0)
        hrv_num.setText("—")

        self._stats["peak"].setText(str(int(stats.get("peak_trust", 0))))
        self._stats["low"].setText(str(int(stats.get("low_trust", 0))))

        hist = stats.get("trust_history", [])
        if len(hist) >= 2:
            xs = list(range(len(hist)))
            # Recolour curve to session band colour
            _, curve_color = trust_band(total)
            self._chart_curve.setPen(pg.mkPen(curve_color, width=2.4))
            cc = QColor(curve_color)
            try:
                self._chart_curve.setFillBrush(pg.mkBrush(cc.red(), cc.green(), cc.blue(), 45))
            except Exception:
                pass
            self._chart_curve.setData(xs, hist)
            self._chart.setXRange(0, len(hist) - 1, padding=0)

            # Update band label x positions to right edge
            for item, mid_y in self._band_label_items:
                item.setPos(len(hist) - 1, mid_y)

            # Labelled average line
            if self._avg_line is not None:
                self._chart.removeItem(self._avg_line)
            self._avg_line = pg.InfiniteLine(
                pos=total, angle=0,
                pen=pg.mkPen(TEXT_FAINT, width=1.2, style=Qt.PenStyle.DashLine),
                label=f"avg {total}",
                labelOpts={"position": 0.04, "color": TEXT_FAINT, "fill": (255, 255, 255, 200)})
            self._chart.addItem(self._avg_line)

            # Peak/low marker dots
            if self._markers is not None:
                self._chart.removeItem(self._markers)
            peak_i = max(range(len(hist)), key=lambda i: hist[i])
            low_i  = min(range(len(hist)), key=lambda i: hist[i])
            self._markers = pg.ScatterPlotItem(
                [peak_i, low_i], [hist[peak_i], hist[low_i]],
                size=9, pen=pg.mkPen("white", width=1.5),
                brush=[pg.mkBrush(QColor(trust_band(int(hist[peak_i]))[1])),
                       pg.mkBrush(QColor(trust_band(int(hist[low_i]))[1]))])
            self._chart.addItem(self._markers)

            # Remove old dot labels and add new ones
            for dl in self._dot_labels:
                self._chart.removeItem(dl)
            self._dot_labels = []

            peak_color = trust_band(int(hist[peak_i]))[1]
            peak_lbl = pg.TextItem(text=str(int(hist[peak_i])), anchor=(0.5, 1.4),
                color=QColor(peak_color))
            peak_lbl.setFont(mono_font(8, QFont.Weight.DemiBold))
            peak_lbl.setPos(peak_i, hist[peak_i])
            peak_lbl.setZValue(10)
            self._chart.addItem(peak_lbl)
            self._dot_labels.append(peak_lbl)

            low_color = trust_band(int(hist[low_i]))[1]
            low_lbl = pg.TextItem(text=str(int(hist[low_i])), anchor=(0.5, -0.4),
                color=QColor(low_color))
            low_lbl.setFont(mono_font(8, QFont.Weight.DemiBold))
            low_lbl.setPos(low_i, hist[low_i])
            low_lbl.setZValue(10)
            self._chart.addItem(low_lbl)
            self._dot_labels.append(low_lbl)

        self._render_flags(stats.get("flags", []))

        # ── Thumbnail ──────────────────────────────────────────────────────
        thumb_path = stats.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            pix = QPixmap(thumb_path)
            if not pix.isNull():
                pix = pix.scaled(
                    240, 135,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._thumb_lbl.setPixmap(_rounded_pixmap(pix, 8))
                self._thumb_lbl.setText("")
            self._thumb_lbl.show()
        else:
            self._thumb_lbl.setText("no face captured")
            self._thumb_lbl.setPixmap(QPixmap())
            self._thumb_lbl.show()

        # ── Play button ────────────────────────────────────────────────────
        rec_path = stats.get("recording_path")
        if rec_path and Path(rec_path).exists():
            self._play_btn.show()
            # Disconnect any previous connections before reconnecting
            try:
                self._play_btn.clicked.disconnect()
            except Exception:
                pass
            self._play_btn.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(rec_path))
            )
        else:
            self._play_btn.hide()

        # ── Delete button ──────────────────────────────────────────────────
        if rec_path or thumb_path:
            self._del_btn.show()
            try:
                self._del_btn.clicked.disconnect()
            except Exception:
                pass
            self._del_btn.clicked.connect(
                lambda: self._delete_recording(rec_path, thumb_path, stats)
            )
        else:
            self._del_btn.hide()

    def _render_flags(self, flags):
        from collections import Counter
        while self._flags_holder_l.count():
            item = self._flags_holder_l.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not flags:
            none = QLabel("No behavioural flags — composure stayed stable.")
            none.setFont(ui_font(10))
            none.setStyleSheet(f"color: {TEXT_FAINT};")
            self._flags_holder_l.addWidget(none)
            self._flags_holder_l.addStretch()
            return
        counts = Counter(f[1] for f in flags if f[1])
        color_for = {f[1]: f[2] for f in flags if f[1]}
        if not counts:
            none = QLabel("No behavioural flags — composure stayed stable.")
            none.setFont(ui_font(10))
            none.setStyleSheet(f"color: {TEXT_FAINT};")
            self._flags_holder_l.addWidget(none)
            self._flags_holder_l.addStretch()
            return
        for text, n in counts.most_common():
            hexc = color_for.get(text, TEXT_DIM)
            cc = QColor(hexc)
            chip = QLabel(f"{n}×  {text}")
            chip.setFont(ui_font(9, QFont.Weight.Medium))
            chip.setStyleSheet(
                f"color: {hexc}; border: 1px solid {hexc}; border-radius: 11px;"
                f" padding: 4px 12px;"
                f" background: rgba({cc.red()}, {cc.green()}, {cc.blue()}, 20);")
            self._flags_holder_l.addWidget(chip)
        self._flags_holder_l.addStretch()

    def _delete_recording(self, rec_path, thumb_path, stats: dict):
        """Ask for confirmation then delete both files."""
        reply = QMessageBox.question(
            self, "Delete recording",
            "Permanently delete the recording and thumbnail for this session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for p in (rec_path, thumb_path):
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass
        # Update sessions.json to clear the paths
        sessions_file = Path.home() / "Desktop" / "trust-dashboard" / "session-data" / "sessions.json"
        try:
            with open(sessions_file) as f:
                sessions = json.load(f)
            sid = stats.get("session_id", "")
            for s in sessions:
                if s.get("session_id") == sid:
                    s["recording_path"] = None
                    s["thumbnail_path"] = None
            with open(sessions_file, "w") as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"[rec] Could not update sessions.json after delete: {e}")
        # Hide UI elements
        self._play_btn.hide()
        self._del_btn.hide()
        self._thumb_lbl.setPixmap(QPixmap())
        self._thumb_lbl.setText("no face captured")


# ═══════════════════════════════════════════════════════════════════════════
# Overview / landing page
# ═══════════════════════════════════════════════════════════════════════════
class OverviewScreen(QWidget):
    """Landing page — past sessions list + Start Session button."""

    start_clicked = pyqtSignal()

    def __init__(self, sessions_file: Path, parent=None):
        super().__init__(parent)
        self._sessions_file = sessions_file
        self.setStyleSheet(f"background: {BG};")

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ── Header band ───────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("ovHead")
        header.setStyleSheet(f"""
            #ovHead {{
                background: {BG_DEEP};
                border-bottom: 1px solid {LINE};
            }}
        """)
        header.setFixedHeight(72)
        h = QHBoxLayout(header)
        h.setContentsMargins(28, 0, 28, 0)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        t = QLabel("Trust Level Dashboard")
        t.setFont(ui_font(17, QFont.Weight.Bold))
        t.setStyleSheet(f"color: {TEXT}; letter-spacing: -0.2px; background: transparent;")
        s = QLabel("Facial · vocal · gaze · HRV · workload analysis")
        s.setFont(ui_font(10))
        s.setStyleSheet(f"color: {TEXT_FAINT}; background: transparent;")
        title_col.addWidget(t)
        title_col.addWidget(s)
        h.addLayout(title_col)
        h.addStretch()

        start_btn = QPushButton("▶   Start Session")
        start_btn.setFont(ui_font(11, QFont.Weight.DemiBold))
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; color: white;
                border: 0; border-radius: 6px;
                padding: 11px 26px;
            }}
            QPushButton:hover {{ background: #1f5fa3; }}
        """)
        start_btn.clicked.connect(self.start_clicked.emit)
        h.addWidget(start_btn)
        v.addWidget(header)

        # ── Load all sessions once ────────────────────────────────────────────
        sessions_all = self._load_sessions()
        recent = sessions_all[-10:] if len(sessions_all) >= 2 else []

        # ── Analytics section (trend chart + stat strip + legend) ─────────────
        analytics_wrap = QWidget()
        analytics_wrap.setStyleSheet(f"background: {BG_DEEP}; border-bottom: 1px solid {LINE};")
        av = QVBoxLayout(analytics_wrap)
        av.setContentsMargins(28, 12, 28, 14)
        av.setSpacing(10)

        if recent:
            ys = [float(s.get("trust_total", 50)) for s in recent]
            n = len(ys)

            # ── 4-up stat strip ───────────────────────────────────────────────
            stat_row = QHBoxLayout()
            stat_row.setSpacing(0)

            def _stat_box(value_text, value_color, top_label, bottom_text, bottom_color=None):
                box = QWidget()
                box.setStyleSheet("background: transparent;")
                bl = QVBoxLayout(box)
                bl.setContentsMargins(0, 0, 0, 0)
                bl.setSpacing(1)
                cap = QLabel(top_label.upper())
                cap.setFont(ui_font(7, QFont.Weight.DemiBold))
                cap.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.2px;")
                val_lbl = QLabel(value_text)
                val_lbl.setFont(mono_font(20, QFont.Weight.DemiBold))
                val_lbl.setStyleSheet(f"color: {value_color};")
                sub_lbl = QLabel(bottom_text)
                sub_lbl.setFont(mono_font(8))
                sub_lbl.setStyleSheet(f"color: {bottom_color or TEXT_FAINT};")
                bl.addWidget(cap)
                bl.addWidget(val_lbl)
                bl.addWidget(sub_lbl)
                return box

            latest_score = int(ys[-1])
            latest_label, latest_color = trust_band(latest_score)
            prev_delta_text = ""
            if n >= 2:
                delta = latest_score - int(ys[-2])
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                sign = "+" if delta > 0 else ""
                prev_delta_text = f"{arrow} {sign}{delta} vs prev"
            else:
                prev_delta_text = "first session"

            avg_score = sum(ys) / n
            avg_label, avg_color = trust_band(int(avg_score))

            min_score = int(min(ys))
            max_score = int(max(ys))
            spread = max_score - min_score

            trend_val = int(ys[-1]) - int(ys[0])
            trend_arrow = "↑" if trend_val > 0 else ("↓" if trend_val < 0 else "→")
            trend_color = "#3aaa6e" if trend_val > 0 else (DANGER if trend_val < 0 else TEXT_FAINT)
            sign2 = "+" if trend_val > 0 else ""

            stat_row.addWidget(_stat_box(
                str(latest_score), latest_color,
                "Latest", prev_delta_text,
            ))
            _stat_divider = lambda: (lambda w: (w.setFixedWidth(1), w.setStyleSheet(f"background: {LINE_SOFT}; margin: 4px 18px;"), w)[-1])(QFrame())
            stat_row.addWidget(_stat_divider())
            stat_row.addWidget(_stat_box(
                f"{avg_score:.0f}", avg_color,
                f"{n}-session avg", avg_label,
            ))
            stat_row.addWidget(_stat_divider())
            stat_row.addWidget(_stat_box(
                f"{min_score}–{max_score}", TEXT,
                "Range", f"{spread} pt spread",
            ))
            stat_row.addWidget(_stat_divider())
            stat_row.addWidget(_stat_box(
                f"{trend_arrow} {sign2}{trend_val}", trend_color,
                "Trend", "latest vs first", trend_color,
            ))
            stat_row.addStretch()
            av.addLayout(stat_row)

            # ── Trend chart ───────────────────────────────────────────────────
            trend_title = QLabel(f"Composure trend · last {n} sessions")
            trend_title.setFont(ui_font(8, QFont.Weight.DemiBold))
            trend_title.setStyleSheet(
                f"color: {TEXT_FAINT}; letter-spacing: 1.2px; background: transparent;"
            )
            av.addWidget(trend_title)

            pg.setConfigOption("background", BG_DEEP)
            trend_plot = pg.PlotWidget()
            trend_plot.setBackground(BG_DEEP)
            trend_plot.setFixedHeight(160)
            trend_plot.setMouseEnabled(x=False, y=False)
            trend_plot.setMenuEnabled(False)
            trend_plot.hideButtons()
            trend_plot.setStyleSheet("border: none;")

            y_min = max(0, min(ys) - 8)
            y_max = min(100, max(ys) + 10)
            trend_plot.setYRange(y_min, y_max, padding=0)
            trend_plot.showGrid(x=False, y=True, alpha=0.10)

            for ax in ("top", "right"):
                trend_plot.getAxis(ax).hide()
            for ax in ("left", "bottom"):
                trend_plot.getAxis(ax).setPen(pg.mkPen(LINE_SOFT))
                trend_plot.getAxis(ax).setTextPen(pg.mkPen(TEXT_GHOST))
            trend_plot.getAxis("left").setStyle(tickLength=-4, showValues=True)
            trend_plot.getAxis("bottom").setStyle(tickLength=-4)
            trend_plot.getPlotItem().getViewBox().setBorder(None)

            # X-axis ticks: use short date/time strings from session data
            import datetime as _dt
            tick_labels = []
            for i, sess in enumerate(recent):
                raw = sess.get("date", "")
                try:
                    # Try to parse common formats and shorten
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d %b %Y %H:%M", "%d %b %Y"):
                        try:
                            parsed = _dt.datetime.strptime(raw, fmt)
                            tick_labels.append((i, parsed.strftime("%b %d\n%H:%M")))
                            break
                        except ValueError:
                            continue
                    else:
                        tick_labels.append((i, raw[:10] if raw else str(i + 1)))
                except Exception:
                    tick_labels.append((i, str(i + 1)))

            trend_plot.getAxis("bottom").setTicks([tick_labels])

            # Band background regions — each band covers [lower_bound, upper_bound]
            from PyQt6.QtGui import QColor as _QColor
            band_ranges = []
            sorted_bands = sorted(TRUST_BANDS, key=lambda b: b[0])  # ascending by threshold
            for idx, (thresh, b_label, b_color) in enumerate(sorted_bands):
                lo = 0 if idx == 0 else sorted_bands[idx - 1][0]
                hi = thresh
                band_ranges.append((lo, hi, b_label, b_color))
            # Add the top band (above highest threshold)
            top = sorted_bands[-1]
            band_ranges.append((top[0], 100, top[1], top[2]))

            for lo, hi, b_label, b_color in band_ranges:
                vis_lo = max(y_min, lo)
                vis_hi = min(y_max, hi)
                if vis_hi <= vis_lo:
                    continue
                c = _QColor(b_color)
                region = pg.LinearRegionItem(
                    values=(vis_lo, vis_hi),
                    orientation="horizontal",
                    brush=pg.mkBrush(c.red(), c.green(), c.blue(), 28),
                    pen=pg.mkPen(None),
                    movable=False,
                )
                region.setZValue(-10)
                trend_plot.addItem(region)
                # Right-edge band label
                mid_y = (vis_lo + vis_hi) / 2
                txt = pg.TextItem(
                    text=b_label,
                    color=(_QColor(b_color).lighter(80).name()),
                    anchor=(1.0, 0.5),
                )
                txt.setFont(ui_font(7))
                trend_plot.addItem(txt)
                txt.setPos(n - 0.5, mid_y)

            # Average line
            avg_val = sum(ys) / len(ys)
            avg_line = pg.InfiniteLine(
                pos=avg_val, angle=0, movable=False,
                pen=pg.mkPen(TEXT_GHOST, width=1, style=Qt.PenStyle.DashLine),
                label=f"avg {avg_val:.0f}",
                labelOpts={
                    "color": TEXT_FAINT,
                    "position": 0.05,
                    "font": mono_font(7),
                },
            )
            trend_plot.addItem(avg_line)

            # Main trend line (no symbols — we'll add custom colored scatter)
            xs = list(range(n))
            from PyQt6.QtGui import QColor as _QC2
            _ac = _QC2(ACCENT)
            trend_plot.plot(
                xs, ys,
                pen=pg.mkPen(ACCENT, width=2.0),
                fillLevel=y_min,
                brush=pg.mkBrush(_ac.red(), _ac.green(), _ac.blue(), 18),
            )

            # Colored dots + value labels per point
            for i, y_val in enumerate(ys):
                _, dot_color = trust_band(int(y_val))
                dot = pg.ScatterPlotItem(
                    [i], [y_val],
                    symbol="o", size=9,
                    brush=pg.mkBrush(dot_color),
                    pen=pg.mkPen(BG_DEEP, width=1.5),
                )
                trend_plot.addItem(dot)
                val_txt = pg.TextItem(
                    text=str(int(y_val)),
                    color=dot_color,
                    anchor=(0.5, 1.3),
                )
                val_txt.setFont(mono_font(7, QFont.Weight.Medium))
                trend_plot.addItem(val_txt)
                val_txt.setPos(i, y_val)

            av.addWidget(trend_plot)

            # ── Band legend ───────────────────────────────────────────────────
            legend_row = QHBoxLayout()
            legend_row.setSpacing(16)
            legend_row.setContentsMargins(0, 2, 0, 0)

            # Legend entries: TRUST_BANDS is already high→low
            legend_entries = []
            for idx, (thresh, b_label, b_color) in enumerate(TRUST_BANDS):
                if idx == 0:
                    lo, hi = thresh, 100
                else:
                    lo = thresh
                    hi = TRUST_BANDS[idx - 1][0] - 1
                legend_entries.append((b_label, b_color, lo, hi))

            for b_label, b_color, lo, hi in legend_entries:
                entry = QHBoxLayout()
                entry.setSpacing(5)
                dot_lbl = QLabel("●")
                dot_lbl.setFont(ui_font(9))
                dot_lbl.setStyleSheet(f"color: {b_color}; background: transparent;")
                name_lbl = QLabel(b_label)
                name_lbl.setFont(ui_font(8))
                name_lbl.setStyleSheet(f"color: {TEXT}; background: transparent;")
                range_lbl = QLabel(f"{lo}–{hi}")
                range_lbl.setFont(mono_font(7))
                range_lbl.setStyleSheet(f"color: {TEXT_GHOST}; background: transparent;")
                entry.addWidget(dot_lbl)
                entry.addWidget(name_lbl)
                entry.addWidget(range_lbl)
                legend_row.addLayout(entry)

            legend_row.addStretch()
            av.addLayout(legend_row)

        else:
            # Fewer than 2 sessions — show placeholder note
            note = QLabel("Not enough sessions yet to show a trend (need ≥ 2).")
            note.setFont(ui_font(10))
            note.setStyleSheet(f"color: {TEXT_FAINT}; padding: 8px 0;")
            av.addWidget(note)

        v.addWidget(analytics_wrap)

        # ── Section label ─────────────────────────────────────────────────────
        section = QLabel("PREVIOUS SESSIONS")
        section.setFont(ui_font(8, QFont.Weight.DemiBold))
        section.setStyleSheet(
            f"color: {TEXT_FAINT}; letter-spacing: 1.3px;"
            f" padding: 14px 28px 6px 28px;"
        )
        v.addWidget(section)

        # ── Sessions list (scrollable) ────────────────────────────────────────
        sessions = self._load_sessions()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {BG}; }}")

        inner = QWidget()
        inner.setStyleSheet(f"background: {BG};")
        inner_l = QVBoxLayout(inner)
        inner_l.setContentsMargins(24, 0, 24, 24)
        inner_l.setSpacing(8)

        if not sessions:
            empty = QLabel("No sessions recorded yet.\nPress  ▶ Start Session  to begin.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setFont(ui_font(13))
            empty.setStyleSheet(f"color: {TEXT_FAINT}; padding: 60px;")
            inner_l.addWidget(empty)
            inner_l.addStretch()
        else:
            reversed_sessions = list(reversed(sessions))
            for i, sess in enumerate(reversed_sessions):
                prev_sess = reversed_sessions[i + 1] if i + 1 < len(reversed_sessions) else None
                inner_l.addWidget(self._make_session_card(sess, prev_sess))
            inner_l.addStretch()

        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

    def _load_sessions(self) -> list:
        try:
            with open(self._sessions_file, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _make_session_card(self, sess: dict, prev_sess: dict | None = None) -> QFrame:
        total = int(sess.get("trust_total", 50))
        label, color = trust_band(total)

        card = QFrame()
        card.setObjectName("sessCard")
        card.setStyleSheet(panel_qss("sessCard") + """
            #sessCard:hover { background: #f0f3f8; }
        """)
        card.setCursor(Qt.CursorShape.PointingHandCursor)

        h = QHBoxLayout(card)
        h.setContentsMargins(16, 12, 16, 12)
        h.setSpacing(0)

        # ── Left: score anchor ────────────────────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        left_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        score_lbl = QLabel(str(total))
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        score_lbl.setFont(mono_font(36, QFont.Weight.Bold))
        score_lbl.setStyleSheet(f"color: {color};")
        left_col.addWidget(score_lbl)

        # Band chip: colored pill with dot + label
        from PyQt6.QtGui import QColor as _QC
        chip_c = _QC(color)
        chip_bg = f"rgba({chip_c.red()},{chip_c.green()},{chip_c.blue()},30)"
        chip = QLabel(f"● {label}")
        chip.setFont(ui_font(8, QFont.Weight.DemiBold))
        chip.setStyleSheet(f"""
            color: {color};
            background: {chip_bg};
            border-radius: 8px;
            padding: 2px 8px;
        """)
        chip.setMaximumWidth(160)
        left_col.addWidget(chip)

        # Delta vs previous session
        if prev_sess is not None:
            prev_total = int(prev_sess.get("trust_total", total))
            delta = total - prev_total
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            sign = "+" if delta > 0 else ""
            d_color = "#3aaa6e" if delta > 0 else (DANGER if delta < 0 else TEXT_FAINT)
            delta_lbl = QLabel(f"{arrow} {sign}{delta} vs prev")
        else:
            delta_lbl = QLabel("first session")
            d_color = TEXT_GHOST
        delta_lbl.setFont(mono_font(8))
        delta_lbl.setStyleSheet(f"color: {d_color};")
        left_col.addWidget(delta_lbl)

        left_wrap = QWidget()
        left_wrap.setFixedWidth(130)
        left_wrap.setLayout(left_col)
        left_wrap.setStyleSheet("background: transparent;")
        h.addWidget(left_wrap)

        # Thin divider
        sep1 = QFrame()
        sep1.setFixedWidth(1)
        sep1.setStyleSheet(f"background-color: {LINE_SOFT}; border: none; margin: 4px 14px;")
        h.addWidget(sep1)

        # ── Middle: meta + channel bars ───────────────────────────────────────
        mid_col = QVBoxLayout()
        mid_col.setSpacing(6)
        mid_col.setContentsMargins(14, 0, 14, 0)

        date_lbl = QLabel(sess.get("date", "—"))
        date_lbl.setFont(ui_font(11, QFont.Weight.DemiBold))
        date_lbl.setStyleSheet(f"color: {TEXT};")
        mid_col.addWidget(date_lbl)

        meta_lbl = QLabel(
            f"Duration  {sess.get('duration_str', '—')}  ·  "
            f"{sess.get('n_samples', 0)} samples"
        )
        meta_lbl.setFont(mono_font(8))
        meta_lbl.setStyleSheet(f"color: {TEXT_FAINT};")
        mid_col.addWidget(meta_lbl)

        for ch_label, ch_key, ch_color in [
            ("Facial", "trust_facial", C_FACIAL),
            ("Vocal",  "trust_vocal",  C_VOCAL),
            ("Gaze",   "trust_gaze",   C_GAZE),
        ]:
            val = int(sess.get(ch_key, 50))
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(ch_label)
            lbl.setFixedWidth(44)
            lbl.setFont(ui_font(8))
            lbl.setStyleSheet(f"color: {TEXT_FAINT};")

            # Bar track with a 50-baseline tick overlay
            track_wrap = QWidget()
            track_wrap.setStyleSheet("background: transparent;")
            tw_l = QHBoxLayout(track_wrap)
            tw_l.setContentsMargins(0, 0, 0, 0)
            tw_l.setSpacing(0)
            track = _BarTrackWithTick(ch_color)
            track.setValue(val)
            tw_l.addWidget(track)

            num = QLabel(str(val))
            num.setFixedWidth(26)
            num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            num.setFont(mono_font(8, QFont.Weight.Medium))
            num.setStyleSheet(f"color: {TEXT};")
            row.addWidget(lbl)
            row.addWidget(track_wrap, 1)
            row.addWidget(num)
            mid_col.addLayout(row)

        mid_wrap = QWidget()
        mid_wrap.setLayout(mid_col)
        mid_wrap.setStyleSheet("background: transparent;")
        h.addWidget(mid_wrap, 1)

        # ── Right: thumbnail ──────────────────────────────────────────────────
        thumb_rel = sess.get("thumbnail_path")
        if thumb_rel:
            thumb_abs = Path.home() / "Desktop" / "trust-dashboard" / thumb_rel
            if thumb_abs.exists():
                pix = QPixmap(str(thumb_abs))
                if not pix.isNull():
                    pix = pix.scaled(48, 48,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation)
                    if pix.width() > 48 or pix.height() > 48:
                        x = (pix.width()  - 48) // 2
                        y = (pix.height() - 48) // 2
                        pix = pix.copy(x, y, 48, 48)
                    thumb_lbl = QLabel()
                    thumb_lbl.setFixedSize(48, 48)
                    thumb_lbl.setPixmap(_rounded_pixmap(pix, 6))
                    thumb_lbl.setStyleSheet("border: none; margin-left: 12px;")
                    h.addWidget(thumb_lbl)

        return card


class _BarTrackWithTick(BarTrack):
    """BarTrack subclass that draws a faint 1px baseline tick at the 50-mark."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        tick_x = int(self.width() * 0.5)
        from PyQt6.QtGui import QColor as _QCT
        tick_color = _QCT(TEXT_GHOST)
        tick_color.setAlpha(140)
        painter.setPen(QPen(tick_color, 1))
        painter.drawLine(tick_x, 2, tick_x, self.height() - 2)
        painter.end()
