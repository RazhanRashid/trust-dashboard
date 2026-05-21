import os
import csv
import io
import logging
import json
import threading
import time
import math
from datetime import datetime
from pathlib import Path

logging.getLogger("root").setLevel(logging.ERROR)

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
        QPushButton, QGridLayout, QHBoxLayout, QVBoxLayout, QStackedWidget, QSizePolicy)
    from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QPointF
    from PySide6.QtGui import (QColor, QPainter, QPen, QFont, QPixmap, QImage, QPainterPath)
except ImportError:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
        QPushButton, QGridLayout, QHBoxLayout, QVBoxLayout, QStackedWidget, QSizePolicy)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal, QRectF, QPointF
    from PyQt6.QtGui import (QColor, QPainter, QPen, QFont, QPixmap, QImage, QPainterPath)

import cv2
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('QtAgg')                           # Use Qt backend instead of TkAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from Physio_analysis.face_analyzer import FaceAnalyzer
from Physio_analysis.vocal_analyzer import VocalAnalyzer
from trust_engine import TrustEngine
from hrv_analyzer import HRVAnalyzer
from workload_engine import WorkloadEngine
from nasa_tlx import NasaTLX

# ── Colour palette ─────────────────────────────────────────────────────────────
# Exact Coolors palette as specified: Light Coral / Light Bronze / Rosewood /
# Dusty Lavender / Dusk Blue — with a warm cream background derived from Bronze.
BG     = '#fdf0e6'   # Warm cream (lighter derivative of BRONZE, user-requested)
SURFACE= '#ffffff'
BORDER = '#e8d5c4'
CORAL  = '#E56B6F'   # Light Coral
BRONZE = '#EAAC8B'   # Light Bronze
MAUVE  = '#B56576'   # Rosewood
GRAPE  = '#6D597A'   # Dusty Lavender
BLUE   = '#355070'   # Dusk Blue (darkest; used for low-trust states & accents)
T1     = '#2a1a24'
T2     = '#6b4a5e'
T3     = '#9a7285'


def qc(h):
    """Convert a hex colour string to a QColor."""
    return QColor(h)


def trust_color(s):
    """Map a 0–100 trust score to the appropriate palette colour."""
    if s >= 72: return BRONZE    # Light Bronze  — high trust
    if s >= 50: return CORAL     # Light Coral   — medium-high
    if s >= 32: return MAUVE     # Rosewood      — medium-low
    return BLUE                  # Dusk Blue     — very low trust


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR ndarray to a QPixmap for display."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(
        QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    )


# ── Custom widgets ─────────────────────────────────────────────────────────────

class GaugeWidget(QWidget):
    """Semicircular arc gauge that shows a 0–100 trust score."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = 0
        self._calibrating = False
        self.setMinimumSize(180, 110)

    def set_score(self, score, calibrating=False):
        """Update the displayed score and repaint."""
        self._score = score
        self._calibrating = calibrating
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx   = w / 2
        cy   = h * 0.58                          # Slightly below centre for the arc
        r    = min(w, h) * 0.44

        rect = QRectF(cx - r, cy - r, r * 2, r * 2)

        # Grey background arc: 180° (left) → 0° (right) in Qt coords
        pen_bg = QPen(qc(BORDER), 14, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap)
        p.setPen(pen_bg)
        p.drawArc(rect, 0 * 16, 180 * 16)       # Qt uses 1/16th degree units

        # Colour fill arc proportional to score
        score    = max(0, min(100, self._score))
        span_deg = int(180 * score / 100)
        if span_deg > 0:
            color = MAUVE if self._calibrating else trust_color(score)
            pen_fg = QPen(qc(color), 14, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap)
            p.setPen(pen_fg)
            # Arc starts at 180° (left) and sweeps clockwise (negative span)
            p.drawArc(rect, 180 * 16, -span_deg * 16)

        # Score number centred
        color = MAUVE if self._calibrating else trust_color(score)
        p.setPen(qc(color))
        font = QFont('Segoe UI', 28)
        font.setBold(True)
        p.setFont(font)
        text_rect = QRectF(0, cy - r * 0.5, w, r)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                   str(score))

        # "TRUST LEVEL" label below the number
        p.setPen(qc(T3))
        font2 = QFont('Segoe UI', 7)
        p.setFont(font2)
        lbl_rect = QRectF(0, cy + r * 0.18, w, 20)
        p.drawText(lbl_rect, Qt.AlignmentFlag.AlignHCenter, 'TRUST LEVEL')

        p.end()


class BarWidget(QWidget):
    """Single horizontal progress bar, height 8px."""

    def __init__(self, color=CORAL, parent=None):
        super().__init__(parent)
        self._value = 0
        self._color = color
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v):
        """Set value 0-100 and repaint."""
        self._value = max(0, min(100, v))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Grey rounded track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(BORDER))
        p.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)

        # Coloured rounded fill
        fill_w = w * self._value / 100
        if fill_w > 0:
            p.setBrush(qc(self._color))
            p.drawRoundedRect(QRectF(0, 0, fill_w, h), 4, 4)

        p.end()


class WaveformWidget(QWidget):
    """Live audio waveform display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buf      = np.zeros(4096)
        self._speaking = False
        self.setMinimumHeight(70)

    def update_data(self, buf, speaking):
        """Accept new audio buffer and speaking flag."""
        self._buf      = buf
        self._speaking = speaking
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background fill
        p.fillRect(0, 0, w, h, qc('#f5e6d8'))

        color = GRAPE if self._speaking else BORDER
        pen   = QPen(qc(color), 1.5)
        p.setPen(pen)

        buf  = self._buf
        step = max(1, len(buf) // w)
        path = QPainterPath()
        for i in range(w):
            sample = float(buf[min(i * step, len(buf) - 1)])
            y      = h / 2 + sample * h * 0.42
            if i == 0:
                path.moveTo(QPointF(i, y))
            else:
                path.lineTo(QPointF(i, y))
        p.drawPath(path)
        p.end()


class SpectrumWidget(QWidget):
    """FFT frequency spectrum bar display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buf = np.zeros(4096)
        self.setMinimumHeight(50)

    def update_data(self, buf):
        """Accept new audio buffer."""
        self._buf = buf
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, qc('#f5e6d8'))

        buf    = self._buf
        window = np.hanning(len(buf))
        fft    = np.abs(np.fft.rfft(buf * window))
        fft    = fft[:len(fft) // 4]          # Use first quarter (lower frequencies)
        fft    = fft / (fft.max() + 1e-10)
        bars   = min(w // 4, 64)
        if bars == 0:
            p.end()
            return
        bw   = w / bars
        step = max(1, len(fft) // bars)

        # Gradient colour: GRAPE (#6D597A) → CORAL (#E56B6F) by intensity
        r1, g1, b1 = 0x6D, 0x59, 0x7A   # GRAPE  (Dusty Lavender)
        r2, g2, b2 = 0xE5, 0x6B, 0x6F   # CORAL  (Light Coral)

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(bars):
            val = float(np.mean(fft[i * step: i * step + step]))
            bh  = val * h
            r   = int(r1 + (r2 - r1) * val)
            g   = int(g1 + (g2 - g1) * val)
            b   = int(b1 + (b2 - b1) * val)
            p.setBrush(QColor(r, g, b))
            p.drawRect(QRectF(i * bw, h - bh, bw - 1, bh))
        p.end()


class StatusIconWidget(QWidget):
    """Small camera or microphone status icon."""

    def __init__(self, kind='camera', parent=None):
        super().__init__(parent)
        self._kind   = kind           # 'camera' or 'mic'
        self._active = False
        self.setFixedSize(28, 22)

    def set_active(self, active: bool):
        """Toggle active state and repaint."""
        self._active = active
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        if self._active:
            color = qc(BRONZE) if self._kind == 'camera' else qc(GRAPE)
        else:
            color = qc(T3)

        if self._kind == 'camera':
            # Camera body
            p.setBrush(color)
            p.drawRoundedRect(QRectF(1, 5, 20, 12), 2, 2)
            # White lens circle
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QRectF(5, 7, 8, 8))
            # Coloured inner lens
            p.setBrush(color)
            p.drawEllipse(QRectF(7, 9, 4, 4))
            # Viewfinder notch
            p.setBrush(color)
            p.drawRoundedRect(QRectF(18, 2, 6, 5), 1, 1)
        else:
            # Mic capsule body
            p.setBrush(color)
            p.drawRoundedRect(QRectF(7, 1, 8, 11), 4, 4)
            # Stand arc (bottom half of circle from 0° to -180°)
            pen = QPen(color, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(3, 6, 16, 10), 0, -180 * 16)   # 0→−180° sweeps through bottom
            # Vertical stand line
            p.drawLine(QPointF(11, 16), QPointF(11, 19))
            # Base line
            p.drawLine(QPointF(7, 19), QPointF(15, 19))

        p.end()


class CountdownWidget(QWidget):
    """Circular countdown ring for the calibration screen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._remaining = 30
        self._total     = 30
        self.setMinimumSize(160, 160)

    def set_time(self, remaining):
        """Update remaining seconds and repaint."""
        self._remaining = remaining
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r      = min(w, h) * 0.42
        rect   = QRectF(cx - r, cy - r, r * 2, r * 2)
        lw     = 13

        # Full grey background ring
        pen_bg = QPen(qc(BORDER), lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen_bg)
        p.drawArc(rect, 0, 360 * 16)

        # Coral progress arc, sweeps clockwise from top (90°)
        progress = 1 - self._remaining / self._total
        span_deg = int(360 * progress)
        if span_deg > 0:
            pen_fg = QPen(qc(CORAL), lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen_fg)
            p.drawArc(rect, 90 * 16, -span_deg * 16)   # Negative = clockwise

        # Large countdown number
        p.setPen(qc(CORAL))
        font = QFont('Segoe UI', 30)
        font.setBold(True)
        p.setFont(font)
        num_rect = QRectF(0, cy - r * 0.5, w, r * 0.9)
        p.drawText(num_rect,
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                   str(int(math.ceil(self._remaining))))

        # "seconds left" label
        p.setPen(qc(T3))
        p.setFont(QFont('Segoe UI', 8))
        lbl_rect = QRectF(0, cy + r * 0.28, w, 20)
        p.drawText(lbl_rect, Qt.AlignmentFlag.AlignHCenter, 'seconds left')

        p.end()


# ── Main window ────────────────────────────────────────────────────────────────

class TrustDashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Trust Level Dashboard')
        self.setMinimumSize(1100, 680)

        # Shared state protected by _lock
        self._lock         = threading.Lock()
        self._last_frame   = None              # (frame_bgr, face_data) from camera thread
        self._last_vocal   = None              # vocal_data dict from audio callback
        self._audio_buffer = np.zeros(4096)   # Ring buffer of latest audio samples
        self._sample_rate  = 44100
        self._running      = True
        self._history      = {k: [] for k in ('total', 'facial', 'vocal', 'gaze')}

        # Camera state
        self._cam_ok            = False
        self._mic_ok            = False
        self._available_cameras = []
        self._camera_idx_pos    = 0
        self._pending_frame     = None

        # Calibration state
        self._calibrating            = True
        self._calibration_seconds    = 30
        self._calibration_started_at = time.time()
        self._calibration_face  = {"eye_ar": [], "blink_rate": [],
                                   "gaze_deviation": [], "pupil_norm": []}
        self._calibration_vocal = {"pitch_stability": [], "energy_level": [],
                                   "tremor_index": []}
        self._calibration_baseline = {}

        # Analyser objects
        self.face     = FaceAnalyzer()
        self.vocal    = VocalAnalyzer()
        self.trust    = TrustEngine()
        self.hrv      = HRVAnalyzer()
        self.workload = WorkloadEngine()
        self.workload.set_tlx_callback(self._on_workload_spike)

        # Last NASA TLX result (displayed in score panel after completion)
        self._last_tlx        = None
        self._last_wl_state   = {}          # Latest WorkloadEngine snapshot

        self._build_window()
        self._start_camera()
        self._start_audio()

        # 33 ms update timer (~30 fps) drives rendering
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(33)

        # 100 ms calibration timer updates the calibration overlay
        self._cal_timer = QTimer()
        self._cal_timer.timeout.connect(self._update_cal)
        self._cal_timer.start(100)

    # ── Window construction ───────────────────────────────────────────────────

    def _build_window(self):
        """Build the stacked widget holding calibration (0) and main (1) pages."""
        central = QWidget()
        central.setStyleSheet(f'background-color: {BG};')
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        cal_page  = self._build_calibration_page()
        main_page = self._build_main_page()
        self._stack.addWidget(cal_page)    # index 0
        self._stack.addWidget(main_page)   # index 1
        self._stack.setCurrentIndex(0)

        self.setCentralWidget(central)

    def _card(self, title):
        """Return (QFrame, QVBoxLayout) for a labelled surface card."""
        frame = QFrame()
        frame.setStyleSheet(
            f'background-color: {SURFACE}; border-radius: 8px;'
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        lbl = QLabel(title)
        lbl.setStyleSheet(f'color: {T3}; font: bold 7pt "Segoe UI";')
        layout.addWidget(lbl)
        return frame, layout

    def _metric_box(self, label, parent):
        """Return (QFrame, value_QLabel) for a small metric tile."""
        frame = QFrame(parent)
        frame.setStyleSheet(
            f'background-color: {BG}; border-radius: 6px;'
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        title_lbl = QLabel(label)
        title_lbl.setStyleSheet(f'color: {T3}; font: 7pt "Segoe UI";')
        layout.addWidget(title_lbl)

        value_lbl = QLabel('—')
        value_lbl.setStyleSheet(f'color: {CORAL}; font: bold 10pt "Segoe UI";')
        layout.addWidget(value_lbl)
        return frame, value_lbl

    # ── Main page ─────────────────────────────────────────────────────────────

    def _build_main_page(self):
        """Construct the live-monitoring page."""
        page = QWidget()
        page.setStyleSheet(f'background-color: {BG};')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        layout.addWidget(self._build_header())

        grid_layout = self._build_grid()
        layout.addLayout(grid_layout, stretch=1)

        layout.addWidget(self._build_chart_panel())
        return page

    def _build_header(self):
        """Top banner with title, subtitle, and status icons."""
        frame = QFrame()
        frame.setStyleSheet(f'background-color: {SURFACE}; border-radius: 8px;')
        h_layout = QHBoxLayout(frame)
        h_layout.setContentsMargins(14, 10, 14, 10)

        # Left: title block
        left = QVBoxLayout()
        title_lbl = QLabel('Trust Level Dashboard')
        title_lbl.setStyleSheet(f'color: {CORAL}; font: bold 16pt "Segoe UI";')
        sub_lbl   = QLabel('Real-time facial · vocal · gaze analysis')
        sub_lbl.setStyleSheet(f'color: {T3}; font: 9pt "Segoe UI";')
        left.addWidget(title_lbl)
        left.addWidget(sub_lbl)
        h_layout.addLayout(left)
        h_layout.addStretch()

        # Right: camera + mic status icons
        right = QHBoxLayout()
        right.setSpacing(8)

        self._cam_icon = StatusIconWidget('camera')
        cam_lbl        = QLabel('Camera')
        cam_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        right.addWidget(self._cam_icon)
        right.addWidget(cam_lbl)

        right.addSpacing(8)

        self._mic_icon = StatusIconWidget('mic')
        mic_lbl        = QLabel('Mic')
        mic_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        right.addWidget(self._mic_icon)
        right.addWidget(mic_lbl)

        h_layout.addLayout(right)
        return frame

    def _build_grid(self):
        """Three-panel horizontal layout: camera | score | voice."""
        layout = QHBoxLayout()
        layout.setSpacing(6)
        layout.addWidget(self._build_camera_panel(), stretch=2)
        layout.addWidget(self._build_score_panel(),  stretch=2)
        layout.addWidget(self._build_voice_panel(),  stretch=2)
        return layout

    def _build_camera_panel(self):
        """Camera feed card with face metrics below."""
        card, layout = self._card('CAMERA FEED')

        # Live camera label
        self._cam_label = QLabel()
        self._cam_label.setFixedSize(320, 240)
        self._cam_label.setStyleSheet('background-color: #1a0f18; border-radius: 4px;')
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._cam_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Camera switch row
        btn_row = QHBoxLayout()
        self._switch_btn = QPushButton('⇄  Switch Camera')
        self._switch_btn.setStyleSheet(
            f'QPushButton {{ background-color: {BG}; color: {T2}; border: none; '
            f'padding: 4px 10px; font: 8pt "Segoe UI"; border-radius: 4px; }}'
            f'QPushButton:hover {{ background-color: {BORDER}; }}'
        )
        self._switch_btn.clicked.connect(self._switch_camera)
        self._cam_idx_label = QLabel('')
        self._cam_idx_label.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        btn_row.addWidget(self._switch_btn)
        btn_row.addWidget(self._cam_idx_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 2×2 face metric boxes
        grid = QGridLayout()
        grid.setSpacing(4)
        self._face_labels = {}
        for i, (label, key) in enumerate([
            ('Expression', 'expr'), ('Eye Openness', 'ear'),
            ('Blink Rate', 'blink'), ('Gaze Deviation', 'gaze_dev')
        ]):
            box, val_lbl = self._metric_box(label, card)
            grid.addWidget(box, i // 2, i % 2)
            self._face_labels[key] = val_lbl
        layout.addLayout(grid)
        return card

    def _build_score_panel(self):
        """Trust score card with gauge and channel bars."""
        card, layout = self._card('TRUST SCORE')

        self._gauge = GaugeWidget()
        layout.addWidget(self._gauge, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._trust_label = QLabel('Calibrating…')
        self._trust_label.setStyleSheet(
            f'color: {MAUVE}; font: bold 11pt "Segoe UI";'
        )
        self._trust_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._trust_label)

        # Channel bars: facial / vocal / gaze / hrv
        self._bars     = {}
        self._bar_nums = {}
        channels = [
            ('Facial', 'facial', CORAL),
            ('Vocal',  'vocal',  GRAPE),
            ('Gaze',   'gaze',   BRONZE),
            ('HRV',    'hrv',    BLUE),
        ]
        for label, key, color in channels:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f'color: {T2}; font: 9pt "Segoe UI";')
            lbl.setFixedWidth(48)
            bar = BarWidget(color)
            num = QLabel('50')
            num.setStyleSheet(f'color: {T1}; font: bold 9pt "Segoe UI";')
            num.setFixedWidth(30)
            num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            row.addWidget(bar)
            row.addWidget(num)
            layout.addLayout(row)
            self._bars[key]     = bar
            self._bar_nums[key] = num

        # ── Cognitive workload section ─────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f'color: {BORDER};')
        layout.addWidget(divider)

        wl_title = QLabel('COGNITIVE LOAD')
        wl_title.setStyleSheet(f'color: {T3}; font: bold 7pt "Segoe UI";')
        layout.addWidget(wl_title)

        # Spike progress bar
        spike_row = QHBoxLayout()
        spike_lbl = QLabel('Spike')
        spike_lbl.setStyleSheet(f'color: {T2}; font: 9pt "Segoe UI";')
        spike_lbl.setFixedWidth(48)
        self._spike_bar = BarWidget(MAUVE)
        self._spike_pct = QLabel('0%')
        self._spike_pct.setStyleSheet(f'color: {T1}; font: bold 9pt "Segoe UI";')
        self._spike_pct.setFixedWidth(30)
        self._spike_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        spike_row.addWidget(spike_lbl)
        spike_row.addWidget(self._spike_bar)
        spike_row.addWidget(self._spike_pct)
        layout.addLayout(spike_row)

        # Status label (normal / high load / TLX score)
        self._wl_status_lbl = QLabel('Normal load')
        self._wl_status_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        self._wl_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._wl_status_lbl)

        return card

    def _build_voice_panel(self):
        """Voice analysis card with waveform, spectrum, and metrics."""
        card, layout = self._card('VOICE ANALYSIS')

        self._waveform = WaveformWidget()
        self._waveform.setMinimumHeight(70)
        layout.addWidget(self._waveform)

        self._spectrum = SpectrumWidget()
        self._spectrum.setMinimumHeight(50)
        layout.addWidget(self._spectrum)

        # 2×2 + 1 full-width metric boxes
        grid = QGridLayout()
        grid.setSpacing(4)
        self._vocal_labels = {}
        items = [
            ('Pitch Stability', 'pitch'), ('Voice Energy', 'energy'),
            ('Tremor Index', 'tremor'),   ('Dominant Hz', 'hz'),
        ]
        for i, (label, key) in enumerate(items):
            box, val_lbl = self._metric_box(label, card)
            grid.addWidget(box, i // 2, i % 2)
            self._vocal_labels[key] = val_lbl
        layout.addLayout(grid)

        # Full-width speaking box
        spk_box, spk_lbl = self._metric_box('Speaking', card)
        layout.addWidget(spk_box)
        self._vocal_labels['speaking'] = spk_lbl

        return card

    def _build_chart_panel(self):
        """Scrolling trust-history line chart at the bottom."""
        card, layout = self._card('TRUST HISTORY')

        fig = plt.Figure(figsize=(11, 1.8))
        fig.patch.set_facecolor(SURFACE)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        ax.set_ylim(0, 100)
        ax.tick_params(colors=T3, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(color=BORDER, linewidth=0.5, alpha=0.6)
        ax.xaxis.set_visible(False)

        # Four chart lines
        self._chart_lines = {
            'total':  ax.plot([], [], color=CORAL,  lw=2.5,          label='Total')[0],
            'facial': ax.plot([], [], color=BRONZE, lw=1.2, ls='--', label='Facial')[0],
            'vocal':  ax.plot([], [], color=GRAPE,  lw=1.2, ls='--', label='Vocal')[0],
            'gaze':   ax.plot([], [], color=MAUVE,  lw=1.2, ls='--', label='Gaze')[0],
        }
        ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=T2,
                  loc='upper left')
        self._chart_ax = ax

        fig.tight_layout(pad=0.8)
        self._fig_canvas = FigureCanvas(fig)
        layout.addWidget(self._fig_canvas)
        return card

    # ── Calibration page ──────────────────────────────────────────────────────

    def _build_calibration_page(self):
        """Full-screen calibration page shown before monitoring begins."""
        page = QWidget()
        page.setStyleSheet(f'background-color: {BG};')
        outer = QVBoxLayout(page)
        outer.setContentsMargins(40, 36, 40, 20)
        outer.setSpacing(10)

        # Title
        title = QLabel('Trust Level Dashboard')
        title.setStyleSheet(f'color: {CORAL}; font: bold 22pt "Segoe UI";')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        subtitle = QLabel('Sit comfortably, look at the camera and speak a few sentences naturally.')
        subtitle.setStyleSheet(f'color: {T2}; font: 10pt "Segoe UI";')
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(subtitle)

        # Content area
        content = QHBoxLayout()
        content.setSpacing(12)

        # Left card: live preview
        left_card, left_layout = self._card('LIVE PREVIEW')
        self._cal_cam = QLabel()
        self._cal_cam.setFixedSize(420, 315)
        self._cal_cam.setStyleSheet('background-color: #1a0f18; border-radius: 4px;')
        self._cal_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self._cal_cam, alignment=Qt.AlignmentFlag.AlignHCenter)

        hint = QLabel('Position your face in the centre of the frame.')
        hint.setStyleSheet(f'color: {T2}; font: 9pt "Segoe UI";')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(hint)
        left_layout.addStretch()
        content.addWidget(left_card, stretch=3)

        # Right card: countdown + status
        right_card, right_layout = self._card('CALIBRATING')

        self._cal_countdown = CountdownWidget()
        right_layout.addWidget(self._cal_countdown, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Progress bar row
        prog_lbl = QLabel('Progress')
        prog_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        right_layout.addWidget(prog_lbl)

        self._cal_progress = BarWidget(CORAL)
        right_layout.addWidget(self._cal_progress)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f'color: {BORDER};')
        right_layout.addWidget(divider)

        # Face status row
        self._cal_face_dot, self._cal_face_status = self._status_row('Face', right_layout)
        # Voice status row
        self._cal_voice_dot, self._cal_voice_status = self._status_row('Voice', right_layout)

        right_layout.addStretch()
        content.addWidget(right_card, stretch=2)

        outer.addLayout(content, stretch=1)

        # Footer
        footer = QLabel('Calibration personalises trust scoring to your natural resting state.')
        footer.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(footer)

        return page

    def _status_row(self, name, layout):
        """Add a status indicator row (dot + name + status text) to layout.
        Returns (dot_label, status_label)."""
        row = QHBoxLayout()
        dot = QLabel('●')
        dot.setStyleSheet(f'color: {T3}; font: 10pt "Segoe UI";')
        dot.setFixedWidth(16)
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(f'color: {T2}; font: bold 9pt "Segoe UI";')
        name_lbl.setFixedWidth(42)
        status_lbl = QLabel('Waiting…')
        status_lbl.setStyleSheet(f'color: {T3}; font: 9pt "Segoe UI";')
        row.addWidget(dot)
        row.addWidget(name_lbl)
        row.addWidget(status_lbl)
        row.addStretch()
        layout.addLayout(row)
        return dot, status_lbl

    # ── Camera management ────────────────────────────────────────────────────

    def _pick_camera(self):
        """Probe indices 0–5 silently and store available cameras. Returns first found index."""
        available = []
        for i in range(6):
            old_err = os.dup(2)                          # Suppress noisy AVFoundation stderr
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                cap   = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                found = cap.isOpened()
                cap.release()
            finally:
                os.dup2(old_err, 2)
                os.close(old_err)
            if found:
                available.append(i)

        self._available_cameras = available if available else [0]
        return self._available_cameras[0]

    def _start_camera(self):
        """Open the camera and launch background threads."""
        idx = self._pick_camera()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(0.5)                                  # Give AVFoundation time to warm up
        self._pending_frame = None
        self._update_switch_btn()
        threading.Thread(target=self._camera_loop,   daemon=True).start()
        threading.Thread(target=self._analysis_loop, daemon=True).start()

    def _switch_camera(self):
        """Cycle to the next available camera index."""
        if len(self._available_cameras) <= 1:
            return
        self._camera_idx_pos = (self._camera_idx_pos + 1) % len(self._available_cameras)
        next_idx = self._available_cameras[self._camera_idx_pos]
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(next_idx, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cam_ok = False                             # Reset until new camera delivers a frame
        self._update_switch_btn()

    def _update_switch_btn(self):
        """Enable/disable switch button and update camera index label."""
        n = len(self._available_cameras)
        self._switch_btn.setEnabled(n > 1)
        idx = self._available_cameras[self._camera_idx_pos] if self._available_cameras else 0
        if n > 1:
            self._cam_idx_label.setText(f'cam {idx}  ({self._camera_idx_pos + 1}/{n})')
        else:
            self._cam_idx_label.setText(f'cam {idx}')

    def _camera_loop(self):
        """Background thread: reads raw frames from the webcam."""
        while self._running:
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.mean() > 1.0:
                frame = cv2.flip(frame, 1)               # Mirror so it feels natural
                self._cam_ok = True
                with self._lock:
                    self._pending_frame = frame
                    if self._last_frame is None:
                        self._last_frame = (frame, {"detected": False})
                    else:
                        self._last_frame = (frame, self._last_frame[1])
            time.sleep(0.033)

    def _analysis_loop(self):
        """Background thread: runs face analysis on the latest pending frame."""
        while self._running:
            with self._lock:
                frame = self._pending_frame
            if frame is not None:
                small     = cv2.resize(frame, (640, 360))   # Downscale for faster analysis
                face_data = self.face.analyze(small)
                with self._lock:
                    self._last_frame = (frame, face_data)   # Pair result with its source frame
            time.sleep(0.033)

    # ── Audio ────────────────────────────────────────────────────────────────

    def _start_audio(self):
        """Open a sounddevice input stream and start the audio callback."""
        try:
            self._sample_rate = int(sd.query_devices(kind='input')['default_samplerate'])
        except Exception:
            self._sample_rate = 44100

        def callback(indata, frames, time_info, status):
            samples = indata[:, 0].copy()
            result  = self.vocal.analyze(samples, self._sample_rate)
            n       = min(len(samples), len(self._audio_buffer))
            with self._lock:
                self._last_vocal   = result
                self._audio_buffer = np.roll(self._audio_buffer, -n)
                self._audio_buffer[-n:] = samples[:n]
            self._mic_ok = True                          # First callback proves mic is live

        try:
            self._audio_stream = sd.InputStream(channels=1, blocksize=4096,
                                                callback=callback)
            self._audio_stream.start()
        except Exception as e:
            print(f'Microphone unavailable: {e}')

    # ── Calibration update ───────────────────────────────────────────────────

    def _update_cal(self):
        """100 ms timer: refresh the calibration overlay."""
        if not self._calibrating:
            self._cal_timer.stop()
            return

        elapsed   = time.time() - self._calibration_started_at
        remaining = max(0.0, self._calibration_seconds - elapsed)
        progress  = min(1.0, elapsed / self._calibration_seconds)

        self._cal_countdown.set_time(remaining)
        self._cal_progress.set_value(int(progress * 100))

        # Camera preview with optional face box
        with self._lock:
            frame_data = self._last_frame

        face_detected = False
        if frame_data is not None:
            frame, fd   = frame_data
            face_detected = bool(fd and fd.get('detected'))
            overlay       = frame.copy()
            fh, fw_       = overlay.shape[:2]
            if face_detected:
                bx, by, bw_, bh_ = fd['box_norm']
                cv2.rectangle(
                    overlay,
                    (int(bx * fw_),         int(by * fh)),
                    (int((bx + bw_) * fw_), int((by + bh_) * fh)),
                    (82, 107, 201), 2,
                )
            small = cv2.resize(overlay, (420, 315))
            self._cal_cam.setPixmap(frame_to_pixmap(small))

        # Face dot + status
        face_color = BRONZE if face_detected else T3
        self._cal_face_dot.setStyleSheet(
            f'color: {face_color}; font: 10pt "Segoe UI";'
        )
        self._cal_face_status.setText('Detected ✓' if face_detected else 'Looking for face…')
        self._cal_face_status.setStyleSheet(
            f'color: {face_color}; font: 9pt "Segoe UI";'
        )

        # Voice dot + status
        voice_samples = len(self._calibration_vocal['pitch_stability'])
        voice_ok      = voice_samples > 3
        voice_color   = GRAPE if voice_ok else T3
        self._cal_voice_dot.setStyleSheet(
            f'color: {voice_color}; font: 10pt "Segoe UI";'
        )
        status_text = f'Captured ({voice_samples} samples) ✓' if voice_ok else 'Speak to calibrate…'
        self._cal_voice_status.setText(status_text)
        self._cal_voice_status.setStyleSheet(
            f'color: {voice_color}; font: 9pt "Segoe UI";'
        )

    # ── Main update loop ─────────────────────────────────────────────────────

    def _update(self):
        """33 ms timer: collect data, run trust engine, render all widgets."""
        try:
            with self._lock:
                frame_data = self._last_frame
                vocal_data = self._last_vocal
                audio_buf  = self._audio_buffer.copy()

            face_data = frame_data[1] if frame_data else None

            if self._calibrating:
                self._collect_calibration_samples(face_data, vocal_data)
                elapsed = time.time() - self._calibration_started_at
                if elapsed >= self._calibration_seconds:
                    self._finish_calibration()
                return                                   # Don't render main UI during calibration

            hrv_score = self.hrv.get_score()
            scores    = self.trust.update(face_data, vocal_data, hrv_score)
            for k, v in scores.items():
                if k in self._history:
                    self._history[k].append(v)
                    if len(self._history[k]) > 120:     # Keep rolling 120-sample window
                        self._history[k].pop(0)

            # Feed workload engine with latest pupil measurement
            pupil_norm = (face_data.get('pupil_norm')
                          if face_data and face_data.get('detected') else None)
            self._last_wl_state = self.workload.update(pupil_norm)

            self._render_camera(frame_data)
            self._render_gauge(scores)
            self._render_bars(scores)
            self._render_face_metrics(face_data)
            self._render_vocal_metrics(vocal_data)
            self._render_workload()
            self._waveform.update_data(audio_buf,
                                       bool(vocal_data and vocal_data.get('is_speaking')))
            self._spectrum.update_data(audio_buf)
            self._render_status_icons()
            self._render_chart()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f'[update error] {exc}', flush=True)

    # ── Calibration helpers ──────────────────────────────────────────────────

    def _collect_calibration_samples(self, face_data, vocal_data):
        """Accumulate baseline samples during the calibration window."""
        if face_data and face_data.get('detected'):
            self._calibration_face["eye_ar"].append(
                float(face_data.get("eye_ar", 0.27)))
            self._calibration_face["blink_rate"].append(
                float(face_data.get("blink_rate", 15.0)))
            self._calibration_face["gaze_deviation"].append(
                float(face_data.get("gaze_deviation", 0.0)))
            pn = face_data.get("pupil_norm")
            if pn is not None:
                self._calibration_face["pupil_norm"].append(float(pn))
        if vocal_data:
            self._calibration_vocal["pitch_stability"].append(
                float(vocal_data.get("pitch_stability", 0.5)))
            self._calibration_vocal["energy_level"].append(
                float(vocal_data.get("energy_level", 0.0)))
            self._calibration_vocal["tremor_index"].append(
                float(vocal_data.get("tremor_index", 0.0)))

    @staticmethod
    def _mean_or(values, fallback):
        """Return the mean of values, or fallback if the list is empty."""
        return sum(values) / len(values) if values else fallback

    def _finish_calibration(self):
        """Compute baseline, reset engine, switch to main monitoring page."""
        baseline_pupil = self._mean_or(self._calibration_face["pupil_norm"], None)

        self._calibration_baseline = {
            "face_eye_ar":           self._mean_or(self._calibration_face["eye_ar"], 0.27),
            "face_blink_rate":       self._mean_or(self._calibration_face["blink_rate"], 15.0),
            "face_gaze_deviation":   self._mean_or(self._calibration_face["gaze_deviation"], 0.0),
            "face_pupil_norm":       baseline_pupil,
            "voice_pitch_stability": self._mean_or(self._calibration_vocal["pitch_stability"], 0.5),
            "voice_energy_level":    self._mean_or(self._calibration_vocal["energy_level"], 0.0),
            "voice_tremor_index":    self._mean_or(self._calibration_vocal["tremor_index"], 0.0),
        }

        # Give the workload engine its personalised pupil baseline
        if baseline_pupil is not None:
            self.workload.set_baseline(baseline_pupil)

        self.trust    = TrustEngine()                   # Fresh engine using personalised baseline
        self._history = {k: [] for k in ('total', 'facial', 'vocal', 'gaze')}
        self._calibrating = False
        self._stack.setCurrentIndex(1)                  # Show the main monitoring page

    # ── Render methods ───────────────────────────────────────────────────────

    def _render_camera(self, frame_data):
        """Draw the face overlay on the current frame and update the camera label."""
        if frame_data is None:
            return
        frame, fd = frame_data
        overlay   = frame.copy()
        h, w      = overlay.shape[:2]

        if fd and fd.get('detected'):
            bx, by, bw, bh_ = fd['box_norm']
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh_) * h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (111, 107, 229), 2)
            cv2.putText(overlay, fd['dominant'], (x1 + 4, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (111, 107, 229), 1, cv2.LINE_AA)
            if fd.get('eye_norm'):
                for pts_norm, ear in [(fd['eye_norm']['l'], fd.get('l_ear', 0.3)),
                                      (fd['eye_norm']['r'], fd.get('r_ear', 0.3))]:
                    pts   = np.array([[int(p[0] * w), int(p[1] * h)]
                                      for p in pts_norm], dtype=np.int32)
                    color = (139, 172, 234) if ear > 0.21 else (118, 89, 178)
                    cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)

        display = cv2.resize(overlay, (320, 240))
        self._cam_label.setPixmap(frame_to_pixmap(display))

    def _render_gauge(self, scores):
        """Update gauge widget and trust label."""
        self._gauge.set_score(scores['total'])
        color = trust_color(scores['total'])
        label = TrustEngine.trust_label(scores['total'])
        self._trust_label.setText(label['text'] if isinstance(label, dict) else str(label))
        self._trust_label.setStyleSheet(f'color: {color}; font: bold 11pt "Segoe UI";')

    def _render_bars(self, scores):
        """Update channel bar widgets and their numeric labels."""
        for key, bar in self._bars.items():
            bar.set_value(scores.get(key, 50))
            self._bar_nums[key].setText(str(scores.get(key, 50)))

    def _render_face_metrics(self, fd):
        """Populate the face metric tiles."""
        if fd and fd.get('detected'):
            self._face_labels['expr'].setText(fd['dominant'])
            self._face_labels['ear'].setText(f"{fd['eye_ar'] * 100:.0f}%")
            self._face_labels['blink'].setText(f"{fd['blink_rate']:.0f}/min")
            self._face_labels['gaze_dev'].setText(f"{fd['gaze_deviation'] * 100:.0f}%")
        else:
            for lbl in self._face_labels.values():
                lbl.setText('—')

    def _render_vocal_metrics(self, vd):
        """Populate the vocal metric tiles."""
        if not vd:
            return
        self._vocal_labels['pitch'].setText(f"{vd['pitch_stability'] * 100:.0f}%")
        self._vocal_labels['energy'].setText(f"{vd['energy_level'] * 100:.0f}%")
        self._vocal_labels['tremor'].setText(f"{vd['tremor_index'] * 100:.0f}%")
        hz = vd.get('dominant_hz', 0)
        self._vocal_labels['hz'].setText(f"{hz:.0f} Hz" if hz else '—')
        speaking = vd.get('is_speaking', False)
        self._vocal_labels['speaking'].setText('Yes' if speaking else 'No')
        spk_color = BRONZE if speaking else T2
        self._vocal_labels['speaking'].setStyleSheet(
            f'color: {spk_color}; font: bold 10pt "Segoe UI";'
        )

    def _render_workload(self):
        """Update the cognitive load / workload widgets in the score panel."""
        wl = self._last_wl_state
        if not wl:
            return

        progress_pct = int(wl.get('spike_progress', 0.0) * 100)
        self._spike_bar.set_value(progress_pct)
        self._spike_pct.setText(f'{progress_pct}%')

        if wl.get('is_high_workload'):
            self._wl_status_lbl.setText('⚠ High cognitive load')
            self._wl_status_lbl.setStyleSheet(
                f'color: {MAUVE}; font: bold 8pt "Segoe UI";')
        elif self._last_tlx is not None:
            tlx = self._last_tlx.get('weighted_tlx', 0)
            self._wl_status_lbl.setText(f'Last TLX: {tlx:.0f}')
            self._wl_status_lbl.setStyleSheet(
                f'color: {GRAPE}; font: 8pt "Segoe UI";')
        else:
            self._wl_status_lbl.setText('Normal load')
            self._wl_status_lbl.setStyleSheet(
                f'color: {T3}; font: 8pt "Segoe UI";')

    def _render_status_icons(self):
        """Update the header camera and mic status icons."""
        self._cam_icon.set_active(self._cam_ok)
        self._mic_icon.set_active(self._mic_ok)

    def _render_chart(self):
        """Update matplotlib history chart lines and redraw."""
        hist = self._history
        n    = max(len(v) for v in hist.values()) if hist else 0
        if n == 0:
            return
        xs = list(range(n))
        for key, line in self._chart_lines.items():
            data = hist.get(key, [])
            if data:
                line.set_data(list(range(len(data))), data)
        self._chart_ax.set_xlim(0, max(n - 1, 1))
        self._fig_canvas.draw_idle()                    # Non-blocking redraw

    # ── Workload spike → NASA TLX ─────────────────────────────────────────────

    def _on_workload_spike(self):
        """Called from the WorkloadEngine background thread when a 60 s spike ends.
        Must hand off to the Qt main thread before touching any Qt objects."""
        QTimer.singleShot(0, self._show_tlx_dialog)

    def _show_tlx_dialog(self):
        """Open the NASA TLX dialog on the Qt main thread."""
        dlg = NasaTLX(self, trigger_ts=time.time())
        dlg.completed.connect(self._on_tlx_complete)
        dlg.open()                                      # Non-blocking; emits completed when done

    def _on_tlx_complete(self, result):
        """Receive NASA TLX result (dict) or None if the user dismissed."""
        if result is not None:
            self._last_tlx = result
            print(f'[TLX] weighted={result["weighted_tlx"]:.1f}  '
                  f'raw={result["raw_tlx"]:.1f}', flush=True)

    # ── Window close ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Cleanly shut down all background threads and streams."""
        self._running = False
        self._timer.stop()
        self._cal_timer.stop()

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        if hasattr(self, '_audio_stream'):
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass

        if hasattr(self.face, '_of_running'):
            self.face._of_running = False              # Signal OpenFace thread to exit

        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TrustDashboard()
    window.show()
    sys.exit(app.exec())
