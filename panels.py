"""panels.py — composite panels assembled from widgets.py primitives.

Each panel is a QFrame with a stable public API (setXxx / updateXxx).
main.py instantiates them once and pumps data via those methods on a QTimer
tick. No panel reaches into Trust analyzer code directly — that wiring is
all in main.py so this file stays UI-only.
"""

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap, QColor
from PyQt6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QVBoxLayout,
                              QHBoxLayout, QGridLayout, QSizePolicy)

import pyqtgraph as pg

from theme import (BG, BG_DEEP, PANEL, PANEL_2, LINE, LINE_SOFT,
                    TEXT, TEXT_DIM, TEXT_FAINT, TEXT_GHOST,
                    C_FACIAL, C_VOCAL, C_GAZE, C_HRV, C_WORKLOAD,
                    ACCENT, DANGER,
                    ui_font, mono_font, trust_band, panel_qss, head_qss)
from widgets import (GaugeWidget, BarTrack, ChannelBar, TrustBadge, StatusDot,
                      MetricBox, PanelHead, WaveformWidget, SpectrumWidget,
                      AttributionStrip)


# ═══════════════════════════════════════════════════════════════════════════
# Top strip — header
# ═══════════════════════════════════════════════════════════════════════════
class TopStrip(QFrame):
    end_session_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topStrip")
        self.setStyleSheet(f"""
            #topStrip {{
                background: {BG_DEEP};
                border-bottom: 1px solid {LINE};
            }}
        """)
        self.setFixedHeight(60)

        h = QHBoxLayout(self)
        h.setContentsMargins(22, 0, 22, 0)
        h.setSpacing(20)

        # Logo
        logo_wrap = QHBoxLayout()
        logo_wrap.setSpacing(10)
        mark = QFrame()
        mark.setFixedSize(QSize(22, 22))
        mark.setStyleSheet(f"""
            QFrame {{
                background: {ACCENT};
                border-radius: 5px;
            }}
        """)
        logo_text = QLabel("TRUST")
        logo_text.setFont(ui_font(11, QFont.Weight.Bold))
        logo_text.setStyleSheet(f"color: {TEXT}; letter-spacing: 1.5px; background: transparent;")
        logo_wrap.addWidget(mark)
        logo_wrap.addWidget(logo_text)
        h.addLayout(logo_wrap)

        # Separator
        h.addWidget(self._sep())

        # Subject + session info
        meta_wrap = QHBoxLayout()
        meta_wrap.setSpacing(18)
        meta_wrap.addLayout(self._pair("SUBJECT", "LOCAL"))
        meta_wrap.addWidget(self._sep())

        rec_dot = QFrame()
        rec_dot.setFixedSize(QSize(8, 8))
        rec_dot.setStyleSheet(f"background: {DANGER}; border-radius: 4px;")
        rec_label = QLabel("REC")
        rec_label.setFont(ui_font(9, QFont.Weight.Bold))
        rec_label.setStyleSheet(f"color: {DANGER}; letter-spacing: 1.2px; background: transparent;")
        rec_row = QHBoxLayout()
        rec_row.setSpacing(6)
        rec_row.addWidget(rec_dot)
        rec_row.addWidget(rec_label)
        meta_wrap.addLayout(rec_row)
        meta_wrap.addWidget(self._sep())

        # Status dots — face / gaze / voice
        self.dot_face  = StatusDot("loading")
        self.dot_gaze  = StatusDot("loading")
        self.dot_voice = StatusDot("loading")
        for dot, label in [(self.dot_face, "FACE"),
                           (self.dot_gaze, "GAZE"),
                           (self.dot_voice, "VOICE")]:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(label)
            lbl.setFont(ui_font(8, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.0px; background: transparent;")
            row.addWidget(dot)
            row.addWidget(lbl)
            meta_wrap.addLayout(row)

        h.addLayout(meta_wrap)
        h.addStretch()

        # End session button
        self.end_btn = QPushButton("◼  End session")
        self.end_btn.setFont(ui_font(10, QFont.Weight.Medium))
        self.end_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.end_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PANEL};
                color: {DANGER};
                border: 1px solid #e8b3ac;
                border-radius: 6px;
                padding: 7px 16px;
            }}
            QPushButton:hover {{
                background: #fff6f4;
                border-color: {DANGER};
            }}
        """)
        self.end_btn.clicked.connect(self.end_session_clicked.emit)
        h.addWidget(self.end_btn)

    def _pair(self, key: str, val: str):
        row = QHBoxLayout()
        row.setSpacing(7)
        k = QLabel(key)
        k.setFont(ui_font(8, QFont.Weight.DemiBold))
        k.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.0px; background: transparent;")
        v = QLabel(val)
        v.setFont(mono_font(9, QFont.Weight.Medium))
        v.setStyleSheet(f"color: {TEXT}; background: transparent;")
        row.addWidget(k)
        row.addWidget(v)
        return row

    def _sep(self) -> QFrame:
        s = QFrame()
        # QFrame VLine uses the palette's midlight — override with background-color
        # (the 'color' property doesn't drive the line colour reliably on macOS)
        s.setFixedSize(QSize(1, 16))
        s.setStyleSheet(f"background-color: {LINE}; border: none;")
        return s

    def set_status(self, face: str, gaze: str, voice: str):
        self.dot_face.setState(face)
        self.dot_gaze.setState(gaze)
        self.dot_voice.setState(voice)

    def setWorkloadProgress(self, progress: float):
        """Glow the bottom border from transparent → workload-green as progress → 1."""
        if progress <= 0.05:
            self.setStyleSheet(f"""
                #topStrip {{
                    background: {BG_DEEP};
                    border-bottom: 1px solid {LINE};
                }}
            """)
        else:
            intensity = min(1.0, progress)
            thickness = max(1, int(1 + intensity * 3))
            alpha = int(80 + intensity * 175)
            self.setStyleSheet(f"""
                #topStrip {{
                    background: {BG_DEEP};
                    border-bottom: {thickness}px solid rgba(45, 164, 106, {alpha});
                }}
            """)


# ═══════════════════════════════════════════════════════════════════════════
# Camera panel
# ═══════════════════════════════════════════════════════════════════════════
class CameraPanel(QFrame):
    switch_camera_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("camPanel")
        self.setStyleSheet(panel_qss("camPanel"))

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Custom head with switch camera icon button
        cam_head = QFrame()
        cam_head.setObjectName("panelHead")
        cam_head.setStyleSheet(head_qss())
        cam_head.setFixedHeight(42)
        cam_head_h = QHBoxLayout(cam_head)
        cam_head_h.setContentsMargins(22, 0, 22, 0)
        cam_title = QLabel("CAMERA · REFERENCE")
        cam_title.setFont(ui_font(8, QFont.Weight.DemiBold))
        cam_title.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px; background: transparent;")
        self._cam_label = QLabel("CAM_00")
        self._cam_label.setFont(mono_font(8))
        self._cam_label.setStyleSheet(f"color: {TEXT_GHOST}; background: transparent;")
        self._switch_btn = QPushButton("⇄")
        self._switch_btn.setFixedSize(QSize(24, 24))
        self._switch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._switch_btn.setToolTip("Switch camera")
        self._switch_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_FAINT};
                border: 1px solid {LINE_SOFT}; border-radius: 4px;
                font-size: 14px;
            }}
            QPushButton:hover {{ border-color: {TEXT_FAINT}; color: {TEXT}; }}
            QPushButton:disabled {{ color: {TEXT_GHOST}; border-color: {LINE_SOFT}; }}
        """)
        self._switch_btn.clicked.connect(self.switch_camera_clicked.emit)
        cam_head_h.addWidget(cam_title)
        cam_head_h.addStretch()
        cam_head_h.addWidget(self._switch_btn)
        cam_head_h.addSpacing(8)
        cam_head_h.addWidget(self._cam_label)
        v.addWidget(cam_head)

        body = QWidget()
        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(20, 16, 20, 18)
        body_l.setSpacing(14)

        # Video feed
        self._video = QLabel()
        self._video.setFixedHeight(200)
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

        # 2×2 metric grid (face metrics)
        grid = QGridLayout()
        grid.setSpacing(8)
        self._metrics = {
            "expr":     MetricBox("Expression"),
            "ear":      MetricBox("Eye Openness"),
            "blink":    MetricBox("Blink Rate"),
            "gaze_dev": MetricBox("Gaze Deviation"),
        }
        grid.addWidget(self._metrics["expr"],     0, 0)
        grid.addWidget(self._metrics["ear"],      0, 1)
        grid.addWidget(self._metrics["blink"],    1, 0)
        grid.addWidget(self._metrics["gaze_dev"], 1, 1)
        body_l.addLayout(grid)

        v.addWidget(body, 1)

    def update_frame(self, frame_bgr: np.ndarray, face_data: dict | None):
        """Take a BGR cv2 frame, draw face overlay, push to QLabel."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        # Optional face overlay (drawn directly on the cv2 array — simplest path)
        if face_data and face_data.get("detected"):
            self._draw_overlay(frame_bgr, face_data)

        # BGR → RGB → QImage → QPixmap (scaled to label, preserving aspect)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._video.width(), self._video.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video.setPixmap(pix)

    def _draw_overlay(self, frame_bgr: np.ndarray, fd: dict):
        h, w = frame_bgr.shape[:2]
        box = fd.get("box_norm")
        if box:
            bx, by, bw, bh = box
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            # Cool slate accent box (BGR-ordered hex for cv2: 196, 114, 40 for #2872c4)
            color = (196, 114, 40)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            # Top-left expression label
            dom = str(fd.get("dominant", "")).upper()
            if dom:
                cv2.putText(frame_bgr, dom, (x1, max(14, y1 - 8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)

    def update_metrics(self, fd: dict | None, baseline: dict | None = None):
        if fd and fd.get("detected"):
            self._metrics["expr"].setValue(str(fd.get("dominant", "—")))
            ear = fd.get("eye_ar", 0)
            self._metrics["ear"].setValue(f"{ear * 100:.0f}%")
            blink = fd.get("blink_rate", 0)
            self._metrics["blink"].setValue(f"{blink:.0f}/min")
            gd = fd.get("gaze_deviation", 0)
            self._metrics["gaze_dev"].setValue(f"{gd * 100:.0f}%")

            if baseline:
                b_ear = baseline.get("face_eye_ar")
                if b_ear is not None:
                    d = (ear - b_ear) * 100
                    # lower eye openness is stressful
                    self._metrics["ear"].setDelta(f"{'▲' if d > 0 else '▼'} {abs(d):.0f}%",
                                                   "good" if d > 0 else "bad")
                b_blink = baseline.get("face_blink_rate")
                if b_blink is not None:
                    d = blink - b_blink
                    # higher blink rate is stressful
                    self._metrics["blink"].setDelta(f"{'▲' if d > 0 else '▼'} {abs(d):.0f}/min",
                                                     "bad" if d > 0 else "good")
                b_gd = baseline.get("face_gaze_deviation")
                if b_gd is not None:
                    d = (gd - b_gd) * 100
                    # higher deviation is worse
                    self._metrics["gaze_dev"].setDelta(f"{'▲' if d > 0 else '▼'} {abs(d):.0f}%",
                                                        "bad" if d > 0 else "good")
        else:
            for k in self._metrics:
                self._metrics[k].setValue("—")
                self._metrics[k].clearDelta()

    def set_camera_info(self, index: int, total: int):
        self._switch_btn.setEnabled(total > 1)
        self._cam_label.setText(f"CAM_{index:02d}")


# ═══════════════════════════════════════════════════════════════════════════
# Score panel (with cognitive-load sub-section)
# ═══════════════════════════════════════════════════════════════════════════
class ScorePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("scorePanel")
        self.setStyleSheet(panel_qss("scorePanel"))

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        v.addWidget(PanelHead("Composure index", "α 0.20 · ema"))

        body_l = QVBoxLayout()
        body_l.setContentsMargins(22, 14, 22, 20)
        body_l.setSpacing(14)

        # Gauge
        self.gauge = GaugeWidget()
        gw = QHBoxLayout()
        gw.addStretch()
        gw.addWidget(self.gauge)
        gw.addStretch()
        body_l.addLayout(gw)

        # Badge
        self.badge = TrustBadge()
        bw = QHBoxLayout()
        bw.addStretch()
        bw.addWidget(self.badge)
        bw.addStretch()
        body_l.addLayout(bw)

        # Attribution strip
        self._attribution = AttributionStrip()
        body_l.addWidget(self._attribution)

        # Channel bars
        bars = QVBoxLayout()
        bars.setSpacing(10)
        self.bar_facial = ChannelBar("Facial", C_FACIAL, 35)
        self.bar_vocal  = ChannelBar("Vocal",  C_VOCAL,  25)
        self.bar_gaze   = ChannelBar("Gaze",   C_GAZE,   25)
        self.bar_hrv    = ChannelBar("HRV",    C_HRV,    15, is_stub=True)
        for b in (self.bar_facial, self.bar_vocal, self.bar_gaze, self.bar_hrv):
            bars.addWidget(b)
        body_l.addLayout(bars)

        # Divider
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {LINE_SOFT}; border: none;")
        body_l.addWidget(sep)

        # Cognitive load sub-section
        wl_title = QLabel("COGNITIVE LOAD")
        wl_title.setFont(ui_font(8, QFont.Weight.DemiBold))
        wl_title.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px;")
        body_l.addWidget(wl_title)

        wl_row = QHBoxLayout()
        wl_row.setSpacing(12)
        spike_lbl = QLabel("Spike")
        spike_lbl.setFixedWidth(58)
        spike_lbl.setFont(ui_font(10, QFont.Weight.Medium))
        spike_lbl.setStyleSheet(f"color: {C_WORKLOAD};")
        self._wl_track = BarTrack(C_WORKLOAD)
        self._wl_num = QLabel("0%")
        self._wl_num.setFixedWidth(40)
        self._wl_num.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._wl_num.setFont(mono_font(10, QFont.Weight.Medium))
        self._wl_num.setStyleSheet(f"color: {TEXT};")
        wl_row.addWidget(spike_lbl)
        wl_row.addWidget(self._wl_track, 1)
        wl_row.addWidget(self._wl_num)
        body_l.addLayout(wl_row)

        # Workload state label (e.g. "Normal load" / "⚠ High cognitive load")
        self._wl_state = QLabel("Normal load")
        self._wl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._wl_state.setFont(ui_font(10))
        self._wl_state.setStyleSheet(f"color: {TEXT_FAINT};")
        body_l.addWidget(self._wl_state)

        body_l.addStretch()
        v.addLayout(body_l)

    def update_scores(self, total: int, facial: int, vocal: int, gaze: int, hrv: int):
        label, color = trust_band(int(total))
        self.gauge.setScore(int(total), color)
        self.badge.setBand(label, color)
        self.bar_facial.setValue(facial)
        self.bar_vocal.setValue(vocal)
        self.bar_gaze.setValue(gaze)
        self.bar_hrv.setValue(hrv)

    def update_attribution(self, delta: float, contributions: dict):
        self._attribution.update(delta, contributions)

    def update_workload(self, wl_state: dict | None):
        if not wl_state:
            return
        progress = float(wl_state.get("spike_progress", 0.0)) * 100
        high = bool(wl_state.get("is_high_workload", False))
        self._wl_track.setValue(progress)
        self._wl_num.setText(f"{progress:.0f}%")
        if high:
            self._wl_state.setText("⚠ High cognitive load")
            self._wl_state.setStyleSheet(f"color: {C_HRV};")
        else:
            self._wl_state.setText("Normal load")
            self._wl_state.setStyleSheet(f"color: {TEXT_FAINT};")


# ═══════════════════════════════════════════════════════════════════════════
# Voice panel
# ═══════════════════════════════════════════════════════════════════════════
class VoicePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("voicePanel")
        self.setStyleSheet(panel_qss("voicePanel"))

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Head with spectrum toggle
        head_frame = QFrame()
        head_frame.setObjectName("panelHead")
        head_frame.setStyleSheet(head_qss())
        head_frame.setFixedHeight(42)
        head_h = QHBoxLayout(head_frame)
        head_h.setContentsMargins(22, 0, 22, 0)
        title_lbl = QLabel("VOICE ANALYSIS")
        title_lbl.setFont(ui_font(8, QFont.Weight.DemiBold))
        title_lbl.setStyleSheet(f"color: {TEXT_FAINT}; letter-spacing: 1.3px; background: transparent;")
        self._spec_btn = QPushButton("spectrum")
        self._spec_btn.setCheckable(True)
        self._spec_btn.setChecked(False)
        self._spec_btn.setFont(mono_font(7))
        self._spec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._spec_btn.setStyleSheet(f"""
            QPushButton {{
                color: {TEXT_GHOST}; background: transparent;
                border: 1px solid {LINE_SOFT}; border-radius: 3px;
                padding: 2px 6px;
            }}
            QPushButton:checked {{
                color: {TEXT_FAINT}; border-color: {TEXT_FAINT};
            }}
        """)
        self._spec_btn.toggled.connect(self._toggle_spectrum)
        id_lbl = QLabel("MIC_00 · 44.1k")
        id_lbl.setFont(mono_font(8))
        id_lbl.setStyleSheet(f"color: {TEXT_GHOST}; background: transparent;")
        head_h.addWidget(title_lbl)
        head_h.addStretch()
        head_h.addWidget(self._spec_btn)
        head_h.addSpacing(8)
        head_h.addWidget(id_lbl)
        v.addWidget(head_frame)

        body_l = QVBoxLayout()
        body_l.setContentsMargins(20, 16, 20, 18)
        body_l.setSpacing(10)

        # Waveform
        self._wave = WaveformWidget()
        body_l.addWidget(self._wave)

        # Spectrum (hidden by default)
        self._spec = SpectrumWidget()
        self._spec.hide()
        body_l.addWidget(self._spec)

        # Metric grid (2×2)
        grid = QGridLayout()
        grid.setSpacing(8)
        self._metrics = {
            "pitch":  MetricBox("Pitch Stability"),
            "energy": MetricBox("Voice Energy"),
            "tremor": MetricBox("Tremor Index"),
            "hz":     MetricBox("Dominant Hz"),
        }
        grid.addWidget(self._metrics["pitch"],  0, 0)
        grid.addWidget(self._metrics["energy"], 0, 1)
        grid.addWidget(self._metrics["tremor"], 1, 0)
        grid.addWidget(self._metrics["hz"],     1, 1)
        body_l.addLayout(grid)

        body_l.addStretch()
        v.addLayout(body_l)

    def _toggle_spectrum(self, checked: bool):
        self._spec.setVisible(checked)

    def set_waveform(self, samples):
        self._wave.setSamples(samples)

    def set_spectrum(self, bins):
        self._spec.setBins(bins)

    def update_metrics(self, vd: dict | None, baseline: dict | None = None):
        speaking = bool(vd and vd.get("is_speaking"))
        self._wave.setSpeaking(speaking)
        if not vd:
            for k in self._metrics:
                self._metrics[k].setValue("—")
                self._metrics[k].clearDelta()
            return
        ps = vd.get("pitch_stability", 0)
        tr = vd.get("tremor_index", 0)
        self._metrics["pitch"].setValue(f"{ps * 100:.0f}%")
        self._metrics["energy"].setValue(f"{vd.get('energy_level', 0) * 100:.0f}%")
        self._metrics["tremor"].setValue(f"{tr * 100:.0f}%")
        hz = vd.get("dominant_hz", 0) or 0
        self._metrics["hz"].setValue(f"{hz:.0f} Hz" if hz > 0 else "—")

        if baseline:
            b_ps = baseline.get("voice_pitch_stability")
            if b_ps is not None:
                d = (ps - b_ps) * 100
                # higher pitch stability is better
                self._metrics["pitch"].setDelta(f"{'▲' if d > 0 else '▼'} {abs(d):.0f}%",
                                                 "good" if d > 0 else "bad")
            b_tr = baseline.get("voice_tremor_index")
            if b_tr is not None:
                d = (tr - b_tr) * 100
                # higher tremor is worse
                self._metrics["tremor"].setDelta(f"{'▲' if d > 0 else '▼'} {abs(d):.0f}%",
                                                  "bad" if d > 0 else "good")


# ═══════════════════════════════════════════════════════════════════════════
# History chart (pyqtgraph PlotWidget)
# ═══════════════════════════════════════════════════════════════════════════
class HistoryChart(QFrame):
    """Live 60-sample chart with Trust + Facial + Vocal + Gaze traces."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("histPanel")
        self.setStyleSheet(panel_qss("histPanel"))

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(PanelHead("Composure history", "live"))

        body_l = QVBoxLayout()
        body_l.setContentsMargins(20, 12, 20, 18)
        body_l.setSpacing(6)

        # Legend row (custom — pyqtgraph's built-in is ugly)
        legend = QHBoxLayout()
        legend.setSpacing(20)
        for name, color in [("Composure", ACCENT),
                             ("Facial", C_FACIAL),
                             ("Vocal", C_VOCAL),
                             ("Gaze", C_GAZE)]:
            sw = QFrame()
            sw.setFixedSize(QSize(14, 3))
            sw.setStyleSheet(f"background: {color}; border-radius: 2px;")
            lbl = QLabel(name)
            lbl.setFont(ui_font(9, QFont.Weight.Medium))
            lbl.setStyleSheet(f"color: {TEXT_DIM};")
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(sw)
            row.addWidget(lbl)
            legend.addLayout(row)
        legend.addStretch()
        body_l.addLayout(legend)

        # PlotWidget — styled to match the cool slate palette
        pg.setConfigOption("background", PANEL)
        pg.setConfigOption("foreground", TEXT_FAINT)
        self._plot = pg.PlotWidget()
        self._plot.setBackground(PANEL)
        self._plot.setYRange(0, 100, padding=0)
        self._plot.showGrid(x=False, y=True, alpha=0.18)
        # Hide every axis frame that pyqtgraph renders by default
        for ax in ("bottom", "top", "right"):
            self._plot.getAxis(ax).hide()
        self._plot.getAxis("left").setPen(pg.mkPen(LINE_SOFT))
        self._plot.getAxis("left").setTextPen(pg.mkPen(TEXT_GHOST))
        self._plot.getAxis("left").setStyle(tickLength=-4, showValues=True)
        # Remove the rectangle pyqtgraph draws around the ViewBox
        self._plot.getPlotItem().getViewBox().setBorder(None)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()

        # Curves — Total solid + filled, channels with distinct dash patterns
        self._curve_total = self._plot.plot([], [],
            pen=pg.mkPen(ACCENT, width=2.4),
            fillLevel=0,
            brush=pg.mkBrush(self._rgba_with_alpha(ACCENT, 30)))
        self._curve_facial = self._plot.plot([], [],
            pen=pg.mkPen(C_FACIAL, width=1.4, dash=[5, 4]))
        self._curve_vocal = self._plot.plot([], [],
            pen=pg.mkPen(C_VOCAL,  width=1.4, dash=[2, 3]))
        self._curve_gaze = self._plot.plot([], [],
            pen=pg.mkPen(C_GAZE,   width=1.4, dash=[8, 3]))

        # Remove the Qt frame pyqtgraph inherits (QFrame subclass)
        self._plot.setStyleSheet("border: none;")
        body_l.addWidget(self._plot, 1)
        v.addLayout(body_l)

    @staticmethod
    def _rgba_with_alpha(hex_color: str, alpha: int) -> tuple:
        c = QColor(hex_color)
        return (c.red(), c.green(), c.blue(), alpha)

    def update_traces(self, history: dict):
        """history = {'total': [...], 'facial': [...], 'vocal': [...], 'gaze': [...]}"""
        if not history.get("total"):
            return
        n = len(history["total"])
        xs = list(range(n))
        self._curve_total.setData(xs, history.get("total", []))
        self._curve_facial.setData(xs, history.get("facial", []))
        self._curve_vocal.setData(xs, history.get("vocal", []))
        self._curve_gaze.setData(xs, history.get("gaze", []))
        self._plot.setXRange(0, max(60, n - 1), padding=0)


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════
class Footer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("footer")
        self.setStyleSheet(f"""
            #footer {{
                background: {BG_DEEP};
                border-top: 1px solid {LINE};
            }}
        """)
        self.setFixedHeight(36)
        h = QHBoxLayout(self)
        h.setContentsMargins(22, 0, 22, 0)

        left = QLabel("v0.9 · local · no telemetry leaves device")
        left.setFont(mono_font(8))
        left.setStyleSheet(f"color: {TEXT_GHOST}; letter-spacing: 0.5px;")
        right = QLabel("Cool slate · instrument panel")
        right.setFont(mono_font(8))
        right.setStyleSheet(f"color: {TEXT_GHOST}; letter-spacing: 0.5px;")

        h.addWidget(left)
        h.addStretch()
        h.addWidget(right)
