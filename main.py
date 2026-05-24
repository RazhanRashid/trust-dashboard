"""main.py — Trust Level Dashboard (PyQt6 / cool slate)

Drop-in replacement for the warm-aesthetic main.py. Same analyzer code,
same threading model, same session persistence — only the UI layer is
rewritten on top of PyQt6 + pyqtgraph.

Run:
    source /Users/razhanr/trust-dashboard/.venv/bin/activate
    pip install -r requirements.txt          # PyQt6 + pyqtgraph
    python main.py
"""

import os
import sys
import openpyxl
import json
import math
import time
import signal
import threading
import logging
from datetime import datetime
from pathlib import Path

logging.getLogger("root").setLevel(logging.ERROR)

import cv2
import numpy as np
import sounddevice as sd

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QStackedWidget,
                              QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox,
                              QFileDialog)

# ── Analyzer modules (unchanged) ─────────────────────────────────────────────
from Physio_analysis.face_analyzer    import FaceAnalyzer
from Physio_analysis.vocal_analyzer   import VocalAnalyzer
from Physio_analysis.trust_engine     import TrustEngine
from Physio_analysis.workload_engine  import WorkloadEngine
from Physio_analysis.hrv_analyzer     import HRVAnalyzer
try:
    from Physio_analysis.nasa_tlx import NasaTLX  # noqa: F401
    HAS_TLX = True
except Exception:
    HAS_TLX = False

# ── UI modules ───────────────────────────────────────────────────────────────
from theme import (BG, BG_DEEP, PANEL, LINE, LINE_SOFT, TEXT, TEXT_FAINT, TEXT_GHOST,
                    ui_font, load_packaged_fonts)
from panels import TopStrip, CameraPanel, ScorePanel, VoicePanel, HistoryChart, Footer
from overlays import OverviewScreen, CalibrationOverlay, SessionSummary


# ═══════════════════════════════════════════════════════════════════════════
class TrustDashboard(QMainWindow):
    """Top-level window. Hosts a QStackedWidget that swaps between
    Overview → Calibration → Live → Summary."""

    CAM_W, CAM_H = 320, 240

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trust Level Dashboard")
        self.setStyleSheet(f"background: {BG};")
        self.resize(1400, 980)
        self.setMinimumSize(1200, 880)

        # ── Analyzers (same instances as before) ─────────────────────────────
        self.face     = FaceAnalyzer()
        self.vocal    = VocalAnalyzer()
        self.trust    = TrustEngine()
        self.workload = WorkloadEngine()
        self.hrv      = HRVAnalyzer()
        self.workload.set_tlx_callback(self._on_workload_spike)

        # ── Threading + shared state ─────────────────────────────────────────
        self._lock = threading.Lock()
        self._pending_frame = None
        self._last_frame    = None   # (frame_bgr, face_data)
        self._last_vocal    = None
        self._audio_buffer  = np.zeros(4096)
        self._sample_rate   = 44100
        self._running       = True
        self._cap = None

        # ── Session / history state ─────────────────────────────────────────
        self._history = {k: [] for k in ("total", "facial", "vocal", "gaze", "hrv")}
        self._workload_state: dict = {}
        self._tlx_open = False
        self._session_rows: list = []
        self._session_start: float = 0.0
        self._last_record_time: float = 0.0
        self._session_ended = False

        # ── Calibration state ───────────────────────────────────────────────
        self._calibrating = False
        self._calibration_started_at = None
        self._calibration_seconds = 30
        self._calibration_pupil:  list[float] = []
        self._calibration_face = {"eye_ar": [], "blink_rate": [], "gaze_deviation": []}
        self._calibration_vocal = {"pitch_stability": [], "energy_level": [], "tremor_index": []}
        self._calibration_baseline: dict = {}

        # ── Camera bookkeeping ──────────────────────────────────────────────
        self._available_cameras: list[int] = []
        self._camera_idx_pos = 0
        self._cam_ok = False
        self._mic_ok = False

        # ── Data directories ─────────────────────────────────────────────────
        self._data_dir      = Path.home() / "Desktop" / "trust-dashboard"
        self._session_dir   = self._data_dir / "session-data"   # JSON + Excel exports
        self._recordings_dir = self._data_dir / "recordings"    # video + thumbnails
        for d in (self._data_dir, self._session_dir, self._recordings_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Recording pipeline ───────────────────────────────────────────────
        self._writer: cv2.VideoWriter | None = None
        self._writer_lock = threading.Lock()
        self._recording_path: Path | None = None
        self._session_id: str = ""
        # Best-thumbnail tracking (highest eye_ar = clearest open-eye frame)
        self._best_thumb_frame = None
        self._best_thumb_conf: float = -1.0

        # ── Latest scores (read by camera loop for recording overlay) ────────
        self._last_scores: dict = {}   # Written by _update_body; read by _camera_loop

        # ── Post-hoc OpenFace analysis ───────────────────────────────────────
        # After a session ends, OpenFace is run on the .mp4 recording in a
        # background thread. Results are stored here and written into the Excel.
        self._postprocess_rows: list | None = None   # Per-frame AU dicts from OpenFace; None = not yet done
        self._postprocess_thread: threading.Thread | None = None
        self._auto_excel_path: Path | None = None    # Auto-saved Excel path for the current session

        # ── Persistence ─────────────────────────────────────────────────────
        self._sessions_file = self._session_dir / "sessions.json"

        # ── Build the UI ────────────────────────────────────────────────────
        self._build_ui()
        self._show_overview()

        # ── Main UI tick (60 ms — comfortable for eye, plenty fast for data)
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._update_body)
        self._tick.start(60)

    # ════════════════════════════════════════════════════════════════════════
    # UI assembly
    # ════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Live dashboard screen — built once, reused
        self._live = self._build_live_dashboard()
        self._stack.addWidget(self._live)   # index 0

        # The other screens (overview/calibration/summary) are constructed on
        # demand and swapped in via the stack.
        self._overview = None
        self._cal = None
        self._sum = None

    def _build_live_dashboard(self) -> QWidget:
        """Top strip + 3-column main row + history chart + footer."""
        root = QWidget()
        root.setStyleSheet(f"background: {BG};")
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # Top strip
        self.top = TopStrip()
        self.top.end_session_clicked.connect(self._end_session)
        rl.addWidget(self.top)

        # Stage — padded content area
        stage = QWidget()
        stage.setStyleSheet(f"background: {BG};")
        sl = QVBoxLayout(stage)
        sl.setContentsMargins(20, 18, 20, 18)
        sl.setSpacing(14)

        # Row 1: camera | score | voice
        row1 = QHBoxLayout()
        row1.setSpacing(14)

        self.cam_panel = CameraPanel()
        self.cam_panel.switch_camera_clicked.connect(self._switch_camera)
        self.cam_panel.setFixedWidth(310)

        self.score_panel = ScorePanel()
        self.score_panel.setFixedWidth(560)

        self.voice_panel = VoicePanel()
        self.voice_panel.setMaximumWidth(360)

        # Cap height so panels don't over-stretch on tall windows
        for _p in (self.cam_panel, self.score_panel, self.voice_panel):
            _p.setMaximumHeight(600)

        row1.addWidget(self.cam_panel)
        row1.addWidget(self.score_panel)
        row1.addWidget(self.voice_panel, 1)
        sl.addLayout(row1, 0)

        # Row 2: history chart full-width
        self.history_chart = HistoryChart()
        self.history_chart.setMinimumHeight(220)
        sl.addWidget(self.history_chart, 1)

        rl.addWidget(stage, 1)

        # Footer
        rl.addWidget(Footer())
        return root

    # ════════════════════════════════════════════════════════════════════════
    # Screen routing
    # ════════════════════════════════════════════════════════════════════════
    def _show_overview(self):
        """Build (or rebuild) and show the overview/landing page."""
        if self._overview is not None:
            self._stack.removeWidget(self._overview)
            self._overview.deleteLater()
        self._overview = OverviewScreen(self._sessions_file)
        self._overview.start_clicked.connect(self._start_session)
        self._stack.addWidget(self._overview)
        self._stack.setCurrentWidget(self._overview)

    def _show_live(self):
        self._stack.setCurrentWidget(self._live)

    def _show_calibration(self):
        if self._cal is not None:
            self._stack.removeWidget(self._cal)
            self._cal.deleteLater()
        self._cal = CalibrationOverlay()
        self._cal.start_clicked.connect(self._begin_calibration)
        self._cal.skip_clicked.connect(self._finish_calibration_now)
        self._stack.addWidget(self._cal)
        self._stack.setCurrentWidget(self._cal)

    def _show_summary(self, stats: dict):
        if self._sum is not None:
            self._stack.removeWidget(self._sum)
            self._sum.deleteLater()
        self._sum = SessionSummary()
        self._sum.populate(stats)
        self._sum.back_clicked.connect(self._back_to_overview)
        self._sum.export_clicked.connect(self._export_csv)
        self._stack.addWidget(self._sum)
        self._stack.setCurrentWidget(self._sum)

    # ════════════════════════════════════════════════════════════════════════
    # Session lifecycle
    # ════════════════════════════════════════════════════════════════════════
    def _start_session(self):
        """User clicked Start on the overview — open calibration overlay."""
        self._show_calibration()
        # Start camera + audio in background so the calibration preview
        # already has a live feed when the user clicks Start Calibration.
        self._start_camera()
        self._start_audio()

    def _begin_calibration(self):
        """User clicked Start Calibration inside the overlay."""
        self._calibration_started_at = time.time()
        self._calibrating = True
        self._session_ended = False

    def _finish_calibration_now(self):
        """User clicked Skip."""
        self._calibrating = False
        self._calibration_started_at = None
        self._enter_live_session()

    def _collect_calibration_samples(self, face_data, vocal_data):
        if face_data and face_data.get("detected"):
            self._calibration_face["eye_ar"].append(float(face_data.get("eye_ar", 0.27)))
            self._calibration_face["blink_rate"].append(float(face_data.get("blink_rate", 15.0)))
            self._calibration_face["gaze_deviation"].append(float(face_data.get("gaze_deviation", 0.0)))
            p = face_data.get("pupil_norm")
            if p is not None:
                self._calibration_pupil.append(float(p))
        if vocal_data:
            self._calibration_vocal["pitch_stability"].append(float(vocal_data.get("pitch_stability", 0.5)))
            self._calibration_vocal["energy_level"].append(float(vocal_data.get("energy_level", 0.0)))
            self._calibration_vocal["tremor_index"].append(float(vocal_data.get("tremor_index", 0.0)))

    @staticmethod
    def _mean_or(values, fallback):
        return sum(values) / len(values) if values else fallback

    def _enter_live_session(self):
        """Calibration complete (or skipped) — switch to live dashboard."""
        self._calibration_baseline = {
            "face_eye_ar":            self._mean_or(self._calibration_face["eye_ar"], 0.27),
            "face_blink_rate":        self._mean_or(self._calibration_face["blink_rate"], 15.0),
            "face_gaze_deviation":    self._mean_or(self._calibration_face["gaze_deviation"], 0.0),
            "voice_pitch_stability":  self._mean_or(self._calibration_vocal["pitch_stability"], 0.5),
            "voice_energy_level":     self._mean_or(self._calibration_vocal["energy_level"], 0.0),
            "voice_tremor_index":     self._mean_or(self._calibration_vocal["tremor_index"], 0.0),
        }
        self.trust = TrustEngine()
        self._history = {k: [] for k in ("total", "facial", "vocal", "gaze", "hrv")}
        if self._calibration_pupil:
            self.workload.set_baseline(sum(self._calibration_pupil) / len(self._calibration_pupil))
        self._calibrating = False
        self._session_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self._best_thumb_frame = None
        self._best_thumb_conf = -1.0
        self._session_start = time.time()
        self._session_rows = []
        self._start_recording()

        # Compute a baseline composure score from calibration samples, then
        # reset the engine so live scores start fresh from the neutral 50.
        self._baseline_total: int | None = None
        if self._calibration_face["eye_ar"]:
            probe_engine = TrustEngine()
            face_probe = {
                "detected": True,
                "expressions": {"happy": 0, "neutral": 1, "surprised": 0,
                                 "fearful": 0, "angry": 0, "disgusted": 0, "sad": 0},
                "aus": {},
                "duchenne": 0,
                "eye_ar":         self._calibration_baseline["face_eye_ar"],
                "blink_rate":     self._calibration_baseline["face_blink_rate"],
                "gaze_deviation": self._calibration_baseline["face_gaze_deviation"],
            }
            vocal_probe = {
                "is_speaking": True,
                "pitch_stability": self._calibration_baseline["voice_pitch_stability"],
                "energy_level":    self._calibration_baseline["voice_energy_level"],
                "tremor_index":    self._calibration_baseline["voice_tremor_index"],
            }
            for _ in range(20):
                probe_engine.update(face_probe, vocal_probe)
            result = probe_engine.update(face_probe, vocal_probe)
            self._baseline_total = int(result["total"])
            self.score_panel.gauge.setBaseline(self._baseline_total)

        self._show_live()

    def _end_session(self):
        if len(self._session_rows) < 2:
            QMessageBox.information(
                self, "Not enough data",
                "Not enough data yet — wait a few seconds before ending the session.",
            )
            return
        self._session_ended = True
        # Stop recording before computing stats so the writer is flushed
        rec_path, thumb_path = self._stop_recording()
        stats = self._compute_session_stats()
        def _rel(p):
            """Store recording paths relative to data_dir for portability."""
            if not p:
                return None
            try:
                return str(Path(p).relative_to(self._data_dir))
            except Exception:
                return str(p)

        stats["recording_path"] = _rel(rec_path)
        stats["thumbnail_path"] = _rel(thumb_path)
        stats["session_id"]     = self._session_id
        self._save_session(stats)

        # Auto-save a base Excel to the session directory immediately.
        # This file is updated automatically with accurate AU data once
        # post-hoc OpenFace analysis completes.
        self._postprocess_rows = None   # Clear any result from a previous session
        auto_excel = self._session_dir / f"trust-session-{self._session_id}.xlsx"
        try:
            self._build_excel(str(auto_excel))
            self._auto_excel_path = auto_excel
            print(f"[export] Auto-saved base Excel → {auto_excel}", flush=True)
        except Exception as e:
            print(f"[export] Auto-save failed: {e}", flush=True)
            self._auto_excel_path = None

        # Launch OpenFace post-hoc analysis in a background thread if a recording exists.
        if rec_path and rec_path.exists():
            self._launch_postprocess(rec_path)

        # Give the summary absolute paths so it can open/display files directly
        summary_stats = dict(stats)
        summary_stats["recording_path"] = str(rec_path) if rec_path else None
        summary_stats["thumbnail_path"] = str(thumb_path) if thumb_path else None
        self._show_summary(summary_stats)

    def _back_to_overview(self):
        self._session_ended = False
        self._session_rows = []
        self._history = {k: [] for k in ("total", "facial", "vocal", "gaze", "hrv")}
        self._show_overview()

    # ════════════════════════════════════════════════════════════════════════
    # Post-hoc OpenFace analysis
    # ════════════════════════════════════════════════════════════════════════

    def _launch_postprocess(self, video_path: Path):
        """
        Start OpenFace post-hoc analysis on the session recording in a daemon
        background thread. The main UI thread is never blocked.

        When analysis completes, _on_postprocess_done() is called which:
          1. Stores the per-frame AU rows in self._postprocess_rows
          2. Re-saves the auto Excel with the accurate AU sheet included
        """
        # If a previous post-hoc thread is still running, leave it — it will finish
        # harmlessly in the background. self._postprocess_rows will be overwritten
        # only when this new session's thread completes.
        def _worker(path: Path):
            rows = FaceAnalyzer.analyze_video(str(path))
            self._on_postprocess_done(rows)

        self._postprocess_thread = threading.Thread(
            target=_worker, args=(video_path,), daemon=True
        )
        self._postprocess_thread.start()
        print("[post-hoc] Background OpenFace thread started.", flush=True)

    def _on_postprocess_done(self, au_rows: list):
        """
        Called from the background thread when OpenFace finishes.
        Stores the AU rows and rewrites the auto-saved Excel with the accurate AU sheet.
        This runs on the background thread so no Qt UI calls are made here.
        """
        self._postprocess_rows = au_rows if au_rows else []
        print(f"[post-hoc] Analysis complete — {len(self._postprocess_rows)} frames.", flush=True)

        # Re-save the Excel now that accurate AU data is available.
        if self._auto_excel_path:
            try:
                self._build_excel(str(self._auto_excel_path))
                print(f"[post-hoc] Excel updated with AU data → {self._auto_excel_path}", flush=True)
            except Exception as e:
                print(f"[post-hoc] Excel update failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # Recording helpers
    # ════════════════════════════════════════════════════════════════════════
    def _start_recording(self):
        """Open a VideoWriter for the current session. Called from the main
        thread after calibration; must run while self._cap is already open."""
        if self._cap is None or not self._cap.isOpened():
            print("[rec] No camera — recording disabled")
            return
        self._recordings_dir.mkdir(exist_ok=True)
        path = self._recordings_dir / f"{self._session_id}.mp4"
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w < 1 or h < 1:
            w, h = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
        if writer.isOpened():
            with self._writer_lock:
                self._writer = writer
            self._recording_path = path
            print(f"[rec] Recording → {path}  ({w}×{h})")
        else:
            print("[rec] VideoWriter failed to open — no recording")

    def _stop_recording(self) -> tuple["Path | None", "Path | None"]:
        """Release the writer and save the best-thumbnail JPEG.
        Returns (recording_path, thumbnail_path), both may be None."""
        # ── 1. Grab and clear the writer atomically ──────────────────────────
        with self._writer_lock:
            writer = self._writer
            self._writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception as e:
                print(f"[rec] writer.release() error: {e}")

        rec_path = self._recording_path
        self._recording_path = None

        # ── 2. Grab the best thumbnail frame ────────────────────────────────
        with self._lock:
            thumb_frame = self._best_thumb_frame
            self._best_thumb_frame = None
        self._best_thumb_conf = -1.0

        if thumb_frame is None or not self._session_id:
            return rec_path, None

        # ── 3. Letterbox to 640×360 and write JPEG ──────────────────────────
        self._recordings_dir.mkdir(exist_ok=True)
        thumb_path = self._recordings_dir / f"{self._session_id}.jpg"
        try:
            h, w = thumb_frame.shape[:2]
            tgt_w, tgt_h = 640, 360
            scale = min(tgt_w / w, tgt_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(thumb_frame, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
            yo, xo = (tgt_h - nh) // 2, (tgt_w - nw) // 2
            canvas[yo:yo + nh, xo:xo + nw] = resized
            cv2.imwrite(str(thumb_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[rec] Thumbnail → {thumb_path}")
            return rec_path, thumb_path
        except Exception as e:
            print(f"[rec] Thumbnail save failed: {e}")
            return rec_path, None

    # ════════════════════════════════════════════════════════════════════════
    # Recording overlay
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _draw_recording_overlay(frame: np.ndarray, face_data: dict | None,
                                 scores: dict | None) -> np.ndarray:
        """
        Draw a blendshape emotion panel onto a copy of the recording frame.
        Called once per camera frame (~30 fps) before writing to the video file.
        The live display is unaffected — this only annotates the saved recording.
        """
        h, w = frame.shape[:2]

        # ── Emotion definitions: label, BGR colour ────────────────────────────
        EMOTIONS = [
            ("happy",     (86,  211,  86)),   # Green
            ("neutral",   (180, 180, 180)),   # Grey
            ("surprised", ( 50, 210, 210)),   # Yellow
            ("sad",       (180, 130,  80)),   # Steel blue
            ("angry",     ( 60,  60, 220)),   # Red
            ("fearful",   ( 60, 150, 220)),   # Orange
            ("disgusted", (160,  60, 160)),   # Purple
            ("contempt",  ( 50,  50, 170)),   # Dark red
        ]

        # ── Panel geometry ────────────────────────────────────────────────────
        PAD        = 10    # Inner padding
        ROW_H      = 26    # Height of each emotion row
        BAR_MAX_W  = 110   # Maximum bar width in pixels
        LABEL_W    = 72    # Width reserved for the emotion label
        VAL_W      = 34    # Width reserved for the numeric value
        PANEL_W    = PAD + LABEL_W + 6 + BAR_MAX_W + 6 + VAL_W + PAD   # ~248 px
        HEADER_H   = 38    # Title row height
        FOOTER_H   = 34    # Trust score row height
        PANEL_H    = PAD + HEADER_H + len(EMOTIONS) * ROW_H + FOOTER_H + PAD

        px = w - PANEL_W - 16   # Right-align with a small margin
        py = 16                  # Top margin

        # ── Semi-transparent dark background ──────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + PANEL_W, py + PANEL_H),
                      (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # ── Thin border ───────────────────────────────────────────────────────
        cv2.rectangle(frame, (px, py), (px + PANEL_W, py + PANEL_H),
                      (80, 80, 80), 1)

        # ── Header: "BLENDSHAPE EMOTIONS" ─────────────────────────────────────
        cv2.putText(frame, "BLENDSHAPE EMOTIONS",
                    (px + PAD, py + PAD + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        # Thin separator line under header
        sep_y = py + PAD + HEADER_H - 4
        cv2.line(frame, (px + PAD, sep_y), (px + PANEL_W - PAD, sep_y), (70, 70, 70), 1)

        # ── Emotion bars ──────────────────────────────────────────────────────
        expressions = (face_data or {}).get("expressions", {})
        dominant    = (face_data or {}).get("dominant", "")

        bar_x  = px + PAD + LABEL_W + 6   # Left edge of the bar area
        val_x  = bar_x + BAR_MAX_W + 4    # Left edge of the value column

        for i, (emotion, colour) in enumerate(EMOTIONS):
            row_y = py + PAD + HEADER_H + i * ROW_H
            score = float(expressions.get(emotion, 0.0))

            # Label — bold-ish by drawing twice for weight
            label_colour = colour if emotion == dominant else (160, 160, 160)
            cv2.putText(frame, emotion,
                        (px + PAD, row_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_colour, 1, cv2.LINE_AA)

            # Grey track
            cv2.rectangle(frame,
                          (bar_x, row_y + 6),
                          (bar_x + BAR_MAX_W, row_y + 19),
                          (55, 55, 55), cv2.FILLED)

            # Coloured fill — width proportional to score
            fill_w = int(BAR_MAX_W * score)
            if fill_w > 0:
                cv2.rectangle(frame,
                              (bar_x, row_y + 6),
                              (bar_x + fill_w, row_y + 19),
                              colour, cv2.FILLED)

            # Highlight bar for dominant emotion
            if emotion == dominant:
                cv2.rectangle(frame,
                              (bar_x, row_y + 6),
                              (bar_x + BAR_MAX_W, row_y + 19),
                              colour, 1)

            # Numeric value
            cv2.putText(frame, f"{score:.2f}",
                        (val_x, row_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Footer: trust score ───────────────────────────────────────────────
        foot_y = py + PAD + HEADER_H + len(EMOTIONS) * ROW_H + 6
        cv2.line(frame, (px + PAD, foot_y), (px + PANEL_W - PAD, foot_y), (70, 70, 70), 1)

        total = int((scores or {}).get("total", 0))
        # Pick a colour that matches the trust_label bands
        if   total >= 82: tc = ( 80, 222, 74)    # Green
        elif total >= 64: tc = ( 57, 211, 52)     # Teal
        elif total >= 46: tc = (250, 165, 96)     # Blue
        elif total >= 28: tc = ( 50, 147, 251)    # Orange
        else:             tc = (113, 129, 248)    # Red

        cv2.putText(frame, f"TRUST  {total}",
                    (px + PAD, foot_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, tc, 1, cv2.LINE_AA)

        # Dominant emotion label next to trust score
        if dominant:
            cv2.putText(frame, dominant.upper(),
                        (px + PAD + 100, foot_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

        return frame

    # ════════════════════════════════════════════════════════════════════════
    # Camera + analysis threads
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _avf_camera_names() -> tuple[dict[int, str], set[int]]:
        """Return (names, phone_indices) via direct PyObjC import in the main process.

        names        — {avf_index: display_name}
        phone_indices — set of indices that are Continuity / iPhone cameras,
                        detected via AVCaptureDevice.deviceType and
                        isContinuityCamera (macOS 13 / 14 APIs) rather than
                        relying solely on the display name.

        Must run in the main app process — macOS only grants camera permission
        to the application, not to subprocesses.
        """
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
            devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
            names: dict[int, str] = {}
            phone_indices: set[int] = set()
            for i, d in enumerate(devices):
                name = str(d.localizedName())
                names[i] = name
                is_phone = False
                # ① deviceType string contains "Continuity" on macOS 13+
                try:
                    dt = str(d.deviceType()).lower()
                    if "continuity" in dt:
                        is_phone = True
                except Exception:
                    pass
                # ② isContinuityCamera property, macOS 14+
                try:
                    if d.isContinuityCamera():
                        is_phone = True
                except Exception:
                    pass
                # ③ name-based fallback
                name_lc = name.lower()
                if any(kw in name_lc for kw in ("iphone", "ipad", "continuity", "desk view")):
                    is_phone = True
                if is_phone:
                    phone_indices.add(i)
            print(f"[camera] AVFoundation devices : {names}")
            print(f"[camera] Phone/continuity indices: {phone_indices}")
            return names, phone_indices
        except Exception as e:
            print(f"[camera] AVFoundation query failed: {e}")
            return {}, set()

    def _pick_camera(self) -> int:
        # ── Step 0: warm-up open to trigger macOS camera-permission grant ────
        _old = os.dup(2); os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        try:
            _w = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION); _w.release()
        except Exception:
            pass
        finally:
            os.dup2(_old, 2); os.close(_old)

        # ── Step 1: authoritative device list from AVFoundation ─────────────
        camera_names, phone_indices = self._avf_camera_names()

        # ── Step 2: scan all indices; validate with a frame ─────────────────
        available: list[int] = []
        for i in range(10):
            old_err = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                        print(f"[camera] Index {i} ({camera_names.get(i, '?')}) → OK")
                cap.release()
            except Exception:
                pass
            finally:
                os.dup2(old_err, 2)
                os.close(old_err)

        self._available_cameras = available if available else [0]
        print(f"[camera] Available cameras: {self._available_cameras}")

        # ── Step 3: prefer Continuity Camera (iPhone) when present ──────────
        # Check by AVFoundation device-type flags first (most reliable),
        # then fall back to name keywords.
        for i in self._available_cameras:
            if i in phone_indices:
                print(f"[camera] Selected index {i} ({camera_names.get(i, '?')}) — Continuity Camera")
                return i

        chosen = self._available_cameras[0]
        print(f"[camera] Selected index {chosen} ({camera_names.get(chosen, '?')}) — first available")
        return chosen

    def _start_camera(self):
        if self._cap is not None:
            return  # already running
        idx = self._pick_camera()
        self._cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(0.3)
        self.cam_panel.set_camera_info(idx, len(self._available_cameras))
        threading.Thread(target=self._camera_loop,   daemon=True).start()
        threading.Thread(target=self._analysis_loop, daemon=True).start()

    def _switch_camera(self):
        if len(self._available_cameras) <= 1:
            return
        self._camera_idx_pos = (self._camera_idx_pos + 1) % len(self._available_cameras)
        next_idx = self._available_cameras[self._camera_idx_pos]
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(next_idx, cv2.CAP_AVFOUNDATION)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cam_ok = False
        self.cam_panel.set_camera_info(next_idx, len(self._available_cameras))

    def _camera_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.mean() > 1.0:
                frame = cv2.flip(frame, 1)
                self._cam_ok = True
                with self._lock:
                    self._pending_frame = frame
                    if self._last_frame is None:
                        self._last_frame = (frame, {"detected": False})
                    else:
                        self._last_frame = (frame, self._last_frame[1])
                # Write to video file if recording is active.
                # Annotate a copy so the live display stays clean.
                with self._writer_lock:
                    w = self._writer
                if w is not None:
                    try:
                        with self._lock:
                            fd = self._last_frame[1] if self._last_frame else None
                        rec_frame = self._draw_recording_overlay(
                            frame.copy(), fd, self._last_scores
                        )
                        w.write(rec_frame)
                    except Exception:
                        pass
            time.sleep(0.033)

    def _analysis_loop(self):
        while self._running:
            with self._lock:
                frame = self._pending_frame
            if frame is not None:
                small = cv2.resize(frame, (640, 360))
                face_data = self.face.analyze(small)
                with self._lock:
                    self._last_frame = (frame, face_data)
                # Track best thumbnail: use eye_ar as quality proxy
                # (more open eyes → higher value → clearer capture)
                if face_data.get("detected"):
                    quality = float(face_data.get("eye_ar", 0.0))
                    if quality > self._best_thumb_conf:
                        self._best_thumb_conf = quality
                        with self._lock:
                            self._best_thumb_frame = frame.copy()
            time.sleep(0.033)

    # ════════════════════════════════════════════════════════════════════════
    # Audio thread
    # ════════════════════════════════════════════════════════════════════════
    def _start_audio(self):
        # Guard against re-entry when a second session starts
        if hasattr(self, "_audio_stream") and self._audio_stream is not None:
            return
        try:
            self._sample_rate = int(sd.query_devices(kind="input")["default_samplerate"])
        except Exception:
            self._sample_rate = 44100

        def callback(indata, frames, time_info, status):
            samples = indata[:, 0].copy()
            result  = self.vocal.analyze(samples, self._sample_rate)
            n = min(len(samples), len(self._audio_buffer))
            with self._lock:
                self._last_vocal = result
                self._audio_buffer = np.roll(self._audio_buffer, -n)
                self._audio_buffer[-n:] = samples[:n]
            self._mic_ok = True

        try:
            self._audio_stream = sd.InputStream(channels=1, blocksize=4096, callback=callback)
            self._audio_stream.start()
        except Exception as e:
            print(f"Microphone unavailable: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # Main UI tick
    # ════════════════════════════════════════════════════════════════════════
    def _update_body(self):
        # Skip when nothing dashboard-related is visible (overview / summary)
        cur = self._stack.currentWidget()
        if cur is None or cur is self._overview or cur is self._sum:
            return
        if self._session_ended:
            return

        with self._lock:
            frame_data = self._last_frame
            vocal_data = self._last_vocal
            audio_buf  = self._audio_buffer.copy()

        face_data = frame_data[1] if frame_data else None
        frame_bgr = frame_data[0] if frame_data else None

        # ── Calibration screen path ──
        if self._cal is not None and self._stack.currentWidget() is self._cal:
            # Push preview frame + indicators
            if frame_bgr is not None:
                self._cal.update_preview(frame_bgr, face_data)
            self._cal.update_indicators(
                face_detected=bool(face_data and face_data.get("detected")),
                voice_samples=len(self._calibration_vocal["pitch_stability"]),
            )

            if self._calibrating and self._calibration_started_at is not None:
                self._collect_calibration_samples(face_data, vocal_data)
                elapsed = time.time() - self._calibration_started_at
                self._cal.update_progress(elapsed, self._calibration_seconds)
                if elapsed >= self._calibration_seconds:
                    self._enter_live_session()
            return

        # ── Live dashboard path ──
        # Compute scores via the trust engine
        hrv_score = self.hrv.get_score()
        scores    = self.trust.update(face_data, vocal_data, hrv_score)
        self._last_scores = scores   # Make latest scores available to the recording overlay
        pupil_now = face_data.get("pupil_norm") if face_data else None
        wl_state  = self.workload.update(pupil_now)
        self._workload_state = wl_state

        # Roll histories (skip the contributions metadata key)
        for k, v in scores.items():
            if k not in self._history:
                continue
            self._history[k].append(v)
            if len(self._history[k]) > 120:
                self._history[k].pop(0)

        # Record one row per second
        now = time.time()
        if now - self._last_record_time >= 1.0:
            self._record_row(scores, face_data, vocal_data, wl_state)
            self._last_record_time = now

        # ── Push to widgets ──
        if frame_bgr is not None:
            self.cam_panel.update_frame(frame_bgr, face_data)
        baseline = self._calibration_baseline if self._calibration_baseline else None
        self.cam_panel.update_metrics(face_data, baseline)

        self.score_panel.update_scores(
            scores["total"], scores["facial"], scores["vocal"],
            scores["gaze"], scores["hrv"],
        )
        self.score_panel.update_workload(wl_state)

        # Attribution strip — 6s rolling delta (~100 ticks at 60ms)
        hist_total = self._history["total"]
        if len(hist_total) >= 6:
            delta_6s = float(hist_total[-1]) - float(hist_total[-min(100, len(hist_total))])
            self.score_panel.update_attribution(delta_6s, scores.get("contributions", {}))

        self.voice_panel.update_metrics(vocal_data, baseline)
        # Waveform downsampled, spectrum from a small FFT
        self.voice_panel.set_waveform(audio_buf[::32])
        spec = self._compute_spectrum(audio_buf)
        self.voice_panel.set_spectrum(spec)

        # History chart — keep last 60 samples for the rolling view
        h = {k: self._history[k][-60:] for k in ("total", "facial", "vocal", "gaze")}
        self.history_chart.update_traces(h)

        # Workload glow on top strip
        if wl_state:
            self.top.setWorkloadProgress(float(wl_state.get("spike_progress", 0.0)))

        # Status dots
        self.top.set_status(
            face  = "active"  if face_data and face_data.get("detected") else "loading",
            gaze  = "active"  if face_data and face_data.get("detected") else "loading",
            voice = "active"  if vocal_data and vocal_data.get("is_speaking") else
                    "idle"    if self._mic_ok else "off",
        )

    # ════════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _compute_spectrum(audio_buf: np.ndarray, n_bins: int = 48) -> list[float]:
        """Tiny FFT → log-magnitude bins, normalized 0..1."""
        if len(audio_buf) < 256:
            return [0.0] * n_bins
        # Take a power-of-two segment for FFT cleanliness
        segment = audio_buf[-1024:]
        # Windowed FFT
        window = np.hanning(len(segment))
        mag = np.abs(np.fft.rfft(segment * window))
        # Bin into log-spaced groups
        if mag.size == 0:
            return [0.0] * n_bins
        # Log-scale + normalize
        mag = np.log1p(mag)
        # Trim to lower-half spectrum (speech is sub-4k Hz typically)
        mag = mag[: max(1, len(mag) // 3)]
        # Resample down to n_bins
        idx = np.linspace(0, len(mag) - 1, n_bins).astype(int)
        bins = mag[idx]
        peak = float(bins.max()) if bins.size else 1.0
        if peak <= 0:
            return [0.0] * n_bins
        return (bins / peak).tolist()

    def _record_row(self, scores, face_data, vocal_data, wl_state):
        """Capture one second of data.  All %-based fields are stored
        already multiplied by 100 so the export needs no conversion."""
        now     = datetime.now()
        elapsed = round(time.time() - self._session_start, 1)

        fd = face_data or {}
        vd = vocal_data or {}
        wd = wl_state   or {}

        row = {
            # ── Timestamps ─────────────────────────────────────────────────
            "timestamp":      now.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":      elapsed,
            # ── Composure scores ────────────────────────────────────────────
            "total":          int(scores.get("total", 50)),
            "facial":         int(scores.get("facial", 50)),
            "vocal":          int(scores.get("vocal", 50)),
            "gaze":           int(scores.get("gaze", 50)),
            "hrv":            int(scores.get("hrv", 65)),
            # ── Facial ──────────────────────────────────────────────────────
            "face_det":       bool(fd.get("detected")),
            "expression":     str(fd.get("dominant", "—")),
            "eye_openness":   round(float(fd.get("eye_ar", 0)) * 100, 1),
            "blink_rate":     round(float(fd.get("blink_rate", 0)), 1),
            "gaze_dev":       round(float(fd.get("gaze_deviation", 0)) * 100, 1),
            "pupil_norm":     round(float(fd.get("pupil_norm") or 0), 4),
            "duchenne":       int(fd.get("duchenne", 0)),
            # ── Vocal ───────────────────────────────────────────────────────
            "speaking":       bool(vd.get("is_speaking")),
            "pitch_stab":     round(float(vd.get("pitch_stability", 0.5)) * 100, 1),
            "energy_level":   round(float(vd.get("energy_level",    0.0)) * 100, 1),
            "tremor":         round(float(vd.get("tremor_index",     0.0)) * 100, 1),
            "dominant_hz":    round(float(vd.get("dominant_hz",      0.0)), 1),
            # ── Cognitive load ──────────────────────────────────────────────
            "high_workload":  bool(wd.get("is_high_workload")),
            "pcps":           round(float(wd.get("pcps",           1000.0)), 2),
            "wiv":            round(float(wd.get("wiv",            1000.0)), 2),
            "spike_progress": round(float(wd.get("spike_progress",    0.0)) * 100, 1),
            # ── Action Units (OpenFace, normalized 0–1 from 0–5 scale) ─────
            # aus dict keys: AU01 AU02 AU04 AU05 AU06 AU07 AU09 AU10
            #                AU12 AU14 AU15 AU17 AU20 AU23 AU25 AU26 AU45
            "aus":            {au: round(v, 3)
                               for au, v in fd.get("aus", {}).items()},
        }
        self._session_rows.append(row)

    def _compute_session_stats(self) -> dict:
        rows = self._session_rows
        if not rows:
            return {}
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        pct = lambda k: 100 * sum(1 for r in rows if r[k]) / n

        durSecs = int(time.time() - self._session_start)
        durStr  = f"{durSecs // 60:02d}:{durSecs % 60:02d}"

        return {
            "duration_str":        durStr,
            "n_samples":           n,
            "trust_total":         int(round(avg("total"))),
            "trust_facial":        int(round(avg("facial"))),
            "trust_vocal":         int(round(avg("vocal"))),
            "trust_gaze":          int(round(avg("gaze"))),
            "trust_hrv":           int(round(avg("hrv"))),
            "peak_trust":          max(r["total"] for r in rows),
            "low_trust":           min(r["total"] for r in rows),
            "pct_face_detected":   pct("face_det"),
            "pct_speaking":        pct("speaking"),
            "pct_high_workload":   pct("high_workload"),
            # pitch_stab / tremor / gaze_dev already stored as %, no × 100
            "avg_pitch_stability": avg("pitch_stab"),
            "avg_tremor":          avg("tremor"),
            "avg_blink_rate":      avg("blink_rate"),
            "avg_gaze_deviation":  avg("gaze_dev"),
            "trust_history":       [r["total"] for r in rows],
        }

    # ════════════════════════════════════════════════════════════════════════
    # Persistence
    # ════════════════════════════════════════════════════════════════════════
    def _load_sessions(self) -> list:
        try:
            with open(self._sessions_file, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_session(self, stats: dict):
        sessions = self._load_sessions()
        sessions.append({
            "date":             datetime.now().strftime("%Y-%m-%d %H:%M"),
            "session_id":       stats.get("session_id", ""),
            "duration_str":     stats.get("duration_str", "00:00"),
            "n_samples":        stats.get("n_samples", 0),
            "trust_total":      stats.get("trust_total", 50),
            "trust_facial":     stats.get("trust_facial", 50),
            "trust_vocal":      stats.get("trust_vocal", 50),
            "trust_gaze":       stats.get("trust_gaze", 50),
            "trust_hrv":        stats.get("trust_hrv", 65),
            "recording_path":   stats.get("recording_path"),   # relative or None
            "thumbnail_path":   stats.get("thumbnail_path"),   # relative or None
        })
        try:
            with open(self._sessions_file, "w") as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"[sessions] Could not save: {e}")

    def _export_csv(self):
        if not self._session_rows:
            return
        default_name = str(
            self._session_dir / f"trust-session-{datetime.now():%Y-%m-%d_%H-%M-%S}.xlsx"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export session as Excel", default_name, "Excel (*.xlsx)",
        )
        if not path:
            return
        try:
            self._build_excel(path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # ── Excel builder ────────────────────────────────────────────────────────
    def _build_excel(self, path: str):
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        HDR_FILL   = PatternFill("solid", fgColor="2563EB")   # accent blue
        HDR_FONT   = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
        BODY_FONT  = Font(name="Calibri", size=10)
        ALT_FILL   = PatternFill("solid", fgColor="F1F5F9")   # very light slate
        LEG_TITLE  = Font(bold=True, name="Calibri", size=10, color="1E3A5F")
        LEG_KEY    = Font(bold=True, name="Calibri", size=9)
        LEG_VAL    = Font(name="Calibri", size=9, color="475569")
        LEG_FILL   = PatternFill("solid", fgColor="EFF6FF")
        CENTER     = Alignment(horizontal="center", vertical="center")
        LEFT       = Alignment(horizontal="left",   vertical="center", wrap_text=True)
        thin       = Side(style="thin", color="CBD5E1")
        BORDER     = Border(bottom=thin)

        rows = self._session_rows

        def _yn(v):
            return "Yes" if v else "No"

        def _auto_width(ws, min_w=10, max_w=48):
            for col in ws.columns:
                best = min_w
                for cell in col:
                    if cell.value is not None:
                        best = max(best, min(max_w, len(str(cell.value)) + 2))
                ws.column_dimensions[get_column_letter(col[0].column)].width = best

        def _write_sheet(ws, columns, legend):
            """columns: list of (header_label, row_key, formatter_fn | None)
               legend:  list of (field_name, description)"""
            # ── header ──────────────────────────────────────────────────────
            for c, (label, _, _) in enumerate(columns, 1):
                cell = ws.cell(row=1, column=c, value=label)
                cell.font      = HDR_FONT
                cell.fill      = HDR_FILL
                cell.alignment = CENTER
                cell.border    = BORDER
            ws.freeze_panes = "A2"
            ws.row_dimensions[1].height = 20

            # ── data rows ───────────────────────────────────────────────────
            for r_idx, row in enumerate(rows, 2):
                fill = ALT_FILL if r_idx % 2 == 0 else None
                for c, (_, key, fmt) in enumerate(columns, 1):
                    raw = row.get(key, "")
                    val = fmt(raw) if fmt else raw
                    cell = ws.cell(row=r_idx, column=c, value=val)
                    cell.font      = BODY_FONT
                    cell.alignment = CENTER
                    if fill:
                        cell.fill = fill

            # ── legend ──────────────────────────────────────────────────────
            leg_start = len(rows) + 3   # blank row gap
            title_cell = ws.cell(row=leg_start, column=1, value="LEGEND")
            title_cell.font      = LEG_TITLE
            title_cell.fill      = LEG_FILL
            title_cell.alignment = LEFT
            ws.merge_cells(start_row=leg_start, start_column=1,
                           end_row=leg_start,   end_column=len(columns))

            for i, (field, desc) in enumerate(legend, leg_start + 1):
                k = ws.cell(row=i, column=1, value=field)
                k.font      = LEG_KEY
                k.alignment = LEFT
                d = ws.cell(row=i, column=2, value=desc)
                d.font      = LEG_VAL
                d.alignment = LEFT
                if len(columns) > 2:
                    ws.merge_cells(start_row=i, start_column=2,
                                   end_row=i,   end_column=len(columns))

            _auto_width(ws)

        wb = openpyxl.Workbook()

        # ════════════════════════════════════════════════════════════════════
        # Sheet 1 — Trust Session  (matches reference format exactly)
        # ════════════════════════════════════════════════════════════════════
        ws1 = wb.active
        ws1.title = "Trust Session"
        _write_sheet(ws1, [
            ("Timestamp",        "timestamp",      None),
            ("Elapsed (s)",      "elapsed_s",      None),
            ("Trust Total",      "total",          None),
            ("Facial",           "facial",         None),
            ("Vocal",            "vocal",          None),
            ("Gaze",             "gaze",           None),
            ("HRV",              "hrv",            None),
            ("Face Detected",    "face_det",       _yn),
            ("Expression",       "expression",     None),
            ("Eye Openness %",   "eye_openness",   None),
            ("Blink Rate /min",  "blink_rate",     None),
            ("Gaze Deviation %", "gaze_dev",       None),
            ("Pupil (norm.)",    "pupil_norm",     None),
            ("Duchenne Smile",   "duchenne",       None),
            ("Speaking",         "speaking",       _yn),
            ("Pitch Stability %","pitch_stab",     None),
            ("Voice Energy %",   "energy_level",   None),
            ("Tremor Index %",   "tremor",         None),
            ("Vocal Hz",         "dominant_hz",    None),
            ("High Workload",    "high_workload",  _yn),
            ("PCPS",             "pcps",           None),
            ("WIV",              "wiv",            None),
            ("Spike Progress %", "spike_progress", None),
        ], legend=[
            ("Trust Total",      "Weighted composure index (0–100). "
                                 "35% Facial + 25% Vocal + 25% Gaze + 15% HRV, "
                                 "smoothed with α=0.20 exponential moving average."),
            ("Facial",           "Facial composure sub-score (0–100)."),
            ("Vocal",            "Vocal composure sub-score (0–100)."),
            ("Gaze",             "Gaze / head-pose composure sub-score (0–100)."),
            ("HRV",              "Heart-rate variability composure sub-score (0–100)."),
        ])

        # ════════════════════════════════════════════════════════════════════
        # Sheet 2 — Facial Analysis
        # ════════════════════════════════════════════════════════════════════
        ws2 = wb.create_sheet("Facial Analysis")
        _write_sheet(ws2, [
            ("Timestamp",        "timestamp",    None),
            ("Elapsed (s)",      "elapsed_s",    None),
            ("Facial Score",     "facial",       None),
            ("Face Detected",    "face_det",     _yn),
            ("Expression",       "expression",   None),
            ("Eye Openness %",   "eye_openness", None),
            ("Blink Rate /min",  "blink_rate",   None),
            ("Gaze Deviation %", "gaze_dev",     None),
            ("Pupil (norm.)",    "pupil_norm",   None),
            ("Duchenne Smile",   "duchenne",     None),
        ], legend=[
            ("Facial Score",     "Composure sub-score derived from eye openness stability, "
                                 "blink regularity, gaze deviation, and expression neutrality (0–100)."),
            ("Face Detected",    "Whether MediaPipe detected and tracked a face in the frame."),
            ("Expression",       "Dominant facial expression inferred from OpenFace Action Units "
                                 "(neutral · happy · sad · angry · surprised · fearful · disgusted)."),
            ("Eye Openness %",   "Eye Aspect Ratio × 100. Typical open-eye range: 25–45 %. "
                                 "Values below ~15 % indicate a blink in progress."),
            ("Blink Rate /min",  "Rolling blinks-per-minute count. "
                                 "Normal rest: 15–20 /min. Elevated rates may indicate fatigue or stress."),
            ("Gaze Deviation %", "Head-pose deviation from camera centre (yaw + 0.5 × pitch, normalised). "
                                 "100 % = looking 40° away. Values above 20 % indicate the subject is looking away."),
            ("Pupil (norm.)",    "Iris radius normalised to inter-ocular distance via MediaPipe iris landmarks. "
                                 "Larger values indicate pupil dilation (higher cognitive arousal)."),
            ("Duchenne Smile",   "Binary flag. 1 = genuine (Duchenne) smile detected: "
                                 "AU06 (cheek raiser) active simultaneously with AU12 (lip corner puller)."),
        ])

        # ════════════════════════════════════════════════════════════════════
        # Sheet 3 — Vocal Analysis
        # ════════════════════════════════════════════════════════════════════
        ws3 = wb.create_sheet("Vocal Analysis")
        _write_sheet(ws3, [
            ("Timestamp",          "timestamp",   None),
            ("Elapsed (s)",        "elapsed_s",   None),
            ("Vocal Score",        "vocal",       None),
            ("Speaking",           "speaking",    _yn),
            ("Pitch Stability %",  "pitch_stab",  None),
            ("Voice Energy %",     "energy_level",None),
            ("Tremor Index %",     "tremor",      None),
            ("Vocal Hz",           "dominant_hz", None),
        ], legend=[
            ("Vocal Score",        "Composure sub-score derived from pitch stability, "
                                   "energy consistency, and tremor index (0–100)."),
            ("Speaking",           "Active speech detected: RMS energy above the silence threshold (0.018)."),
            ("Pitch Stability %",  "Inverse coefficient of variation of fundamental frequency (F0). "
                                   "100 % = perfectly stable pitch. Low values suggest vocal stress or emotion."),
            ("Voice Energy %",     "Normalised RMS loudness (0–100 %). "
                                   "100 % corresponds to ≈ 0.09 RMS (comfortable speaking volume)."),
            ("Tremor Index %",     "High-frequency amplitude modulation of the voice (0–100 %). "
                                   "Values above 30 % suggest significant vocal tremor."),
            ("Vocal Hz",           "Fundamental frequency (F0) in Hz via autocorrelation. "
                                   "Typical speech range: 80–450 Hz. 0 = not speaking."),
        ])

        # ════════════════════════════════════════════════════════════════════
        # Sheet 4 — Gaze Analysis
        # ════════════════════════════════════════════════════════════════════
        ws4 = wb.create_sheet("Gaze Analysis")
        _write_sheet(ws4, [
            ("Timestamp",          "timestamp", None),
            ("Elapsed (s)",        "elapsed_s", None),
            ("Gaze Score",         "gaze",      None),
            ("Gaze Deviation %",   "gaze_dev",  None),
            ("Pupil (norm.)",      "pupil_norm",None),
        ], legend=[
            ("Gaze Score",         "Composure sub-score based on sustained head-pose stability (0–100). "
                                   "Frequent or large deviations lower the score."),
            ("Gaze Deviation %",   "Angular deviation computed from MediaPipe 3-D head-pose landmarks: "
                                   "(|yaw| + 0.5 × |pitch|) ÷ 40°, clamped to 100 %. "
                                   "Values above 20–25 % typically indicate deliberate look-away."),
            ("Pupil (norm.)",      "Included here because iris size encodes arousal "
                                   "and correlates with sustained attention."),
        ])

        # ════════════════════════════════════════════════════════════════════
        # Sheet 5 — Cognitive Load
        # ════════════════════════════════════════════════════════════════════
        ws5 = wb.create_sheet("Cognitive Load")
        _write_sheet(ws5, [
            ("Timestamp",          "timestamp",      None),
            ("Elapsed (s)",        "elapsed_s",      None),
            ("High Workload",      "high_workload",  _yn),
            ("PCPS",               "pcps",           None),
            ("WIV",                "wiv",            None),
            ("Spike Progress %",   "spike_progress", None),
        ], legend=[
            ("PCPS",               "Pupil Change Per Second — baseline-corrected real-time pupil dilation. "
                                   "Baseline pupil = 1000. Values > 1000 indicate dilation above baseline."),
            ("WIV",                "Within-session Inertia Value — 60-second rolling mean of PCPS. "
                                   "Adapts to the subject's typical pupil level throughout the session."),
            ("High Workload",      "'Yes' when PCPS > WIV × 1.015 "
                                   "(pupil at least 1.5 % above the rolling average)."),
            ("Spike Progress %",   "Progress (0–100 %) toward a confirmed high-workload spike. "
                                   "A spike is declared when High Workload persists for 60 continuous seconds. "
                                   "Resets to 0 % on any low-workload moment."),
        ])

        # ════════════════════════════════════════════════════════════════════
        # Sheet 6 — HRV
        # ════════════════════════════════════════════════════════════════════
        ws6 = wb.create_sheet("HRV")
        _write_sheet(ws6, [
            ("Timestamp",          "timestamp", None),
            ("Elapsed (s)",        "elapsed_s", None),
            ("HRV Score",          "hrv",       None),
        ], legend=[
            ("HRV Score",          "Heart-rate variability composure sub-score (0–100). "
                                   "Currently supplied by a stub analyzer returning a stable baseline (65). "
                                   "Future integration with a wearable HRV sensor will populate "
                                   "this column with real beat-to-beat interval data."),
        ])

        # ════════════════════════════════════════════════════════════════════
        # Sheet 7 — Action Units
        # ════════════════════════════════════════════════════════════════════
        # AU descriptions (FACS label + what elevated intensity means)
        AU_META = {
            "AU01": ("Inner Brow Raise",       "Raises the inner corners of the eyebrows. Active in sadness, fear, and worry."),
            "AU02": ("Outer Brow Raise",       "Raises the outer corners of the eyebrows. Seen in surprise and fear."),
            "AU04": ("Brow Lowerer",           "Pulls the brows together and down. Key marker of anger, concentration, and confusion."),
            "AU05": ("Upper Lid Raiser",       "Widens the eye aperture. Strongly associated with surprise and fear."),
            "AU06": ("Cheek Raiser",           "Raises the cheeks, forming crow's feet. Required component of a genuine (Duchenne) smile."),
            "AU07": ("Lid Tightener",          "Tightens the lower eyelid. Present in anger, disgust, and focus."),
            "AU09": ("Nose Wrinkler",          "Wrinkles the nose. Primary marker of disgust."),
            "AU10": ("Upper Lip Raiser",       "Raises the upper lip. Present in disgust and mild contempt."),
            "AU12": ("Lip Corner Puller",      "Pulls lip corners outward and upward. Core component of smiling."),
            "AU14": ("Dimpler",                "Creates dimples by pulling lip corners. Seen in suppressed smiles and smirks."),
            "AU15": ("Lip Corner Depressor",   "Pulls lip corners downward. Associated with sadness and disappointment."),
            "AU17": ("Chin Raiser",            "Raises the chin boss. Seen in sadness, doubt, and pouting."),
            "AU20": ("Lip Stretcher",          "Stretches lips horizontally. Common in fear and nervous tension."),
            "AU23": ("Lip Tightener",          "Tightens the lips. Present in anger and determination."),
            "AU25": ("Lips Part",              "Parts the lips. Accompanies many expressions; elevated in surprise, disgust, speech."),
            "AU26": ("Jaw Drop",               "Opens the jaw. Strong indicator of surprise, shock, or open-mouth speech."),
            "AU45": ("Blink / Eye Closure",    "Eye closure intensity. High sustained values indicate fatigue or discomfort."),
        }
        AU_ORDER = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10",
                    "AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU45"]

        # Decide AU data source.
        # Post-hoc OpenFace rows (accurate) are preferred over the blendshape-derived
        # approximate AUs stored in session_rows during the live session.
        postproc = self._postprocess_rows   # None = still running; [] = ran but no face; list = success
        au_source_label = "OpenFace post-hoc (accurate)" if postproc else "MediaPipe blendshapes (approximate)"

        # Build a timestamp_s → postproc_row lookup for fast nearest-frame matching.
        # Each session row records at 1 fps (elapsed_s = 0, 1, 2 …).
        # Each postproc row records at video frame rate (~30 fps, timestamp_s = 0.00, 0.033 …).
        # For each 1-fps session row we find the closest postproc frame by timestamp.
        def _nearest_postproc_aus(elapsed_s: float) -> dict:
            if not postproc:
                return {}
            best = min(postproc, key=lambda r: abs(r["timestamp_s"] - elapsed_s))
            return best["aus"] if best.get("success") else {}

        ws7 = wb.create_sheet("Action Units")
        # ── header ─────────────────────────────────────────────────────────
        fixed_cols = [("Timestamp", "timestamp"), ("Elapsed (s)", "elapsed_s"),
                      ("Face Detected", "face_det"), ("Expression", "expression")]
        all_headers = ([h for h, _ in fixed_cols] +
                       [f"{au} – {AU_META[au][0]}" for au in AU_ORDER if au in AU_META])

        for c, h in enumerate(all_headers, 1):
            cell = ws7.cell(row=1, column=c, value=h)
            cell.font      = HDR_FONT
            cell.fill      = HDR_FILL
            cell.alignment = CENTER
            cell.border    = BORDER
        ws7.freeze_panes = "A2"
        ws7.row_dimensions[1].height = 28

        # ── data rows ──────────────────────────────────────────────────────
        for r_idx, row in enumerate(rows, 2):
            fill = ALT_FILL if r_idx % 2 == 0 else None

            # Use accurate post-hoc AUs if available, otherwise blendshape approximations.
            if postproc is not None:
                aus = _nearest_postproc_aus(row.get("elapsed_s", 0))
            else:
                aus = row.get("aus", {})

            vals = (
                [row.get("timestamp", ""), row.get("elapsed_s", ""),
                 _yn(row.get("face_det", False)), row.get("expression", "—")] +
                [aus.get(au, "") for au in AU_ORDER]
            )
            for c, v in enumerate(vals, 1):
                cell = ws7.cell(row=r_idx, column=c, value=v)
                cell.font      = BODY_FONT
                cell.alignment = CENTER
                if fill:
                    cell.fill = fill

        # ── AU Detail sheet (full per-frame post-hoc data) ──────────────────
        # Only added when post-hoc analysis has completed successfully.
        if postproc:
            ws_detail = wb.create_sheet("AU Detail (post-hoc)")
            detail_headers = (["Frame", "Time (s)", "Face Found"] +
                              [f"{au} – {AU_META[au][0]}" for au in AU_ORDER if au in AU_META])
            for c, h in enumerate(detail_headers, 1):
                cell = ws_detail.cell(row=1, column=c, value=h)
                cell.font = HDR_FONT; cell.fill = HDR_FILL
                cell.alignment = CENTER; cell.border = BORDER
            ws_detail.freeze_panes = "A2"
            ws_detail.row_dimensions[1].height = 28
            for r_idx, prow in enumerate(postproc, 2):
                fill = ALT_FILL if r_idx % 2 == 0 else None
                aus_p = prow.get("aus", {})
                vals = ([prow.get("frame_idx", ""), round(prow.get("timestamp_s", 0), 3),
                         _yn(bool(prow.get("success")))] +
                        [round(aus_p.get(au, 0), 4) for au in AU_ORDER])
                for c, v in enumerate(vals, 1):
                    cell = ws_detail.cell(row=r_idx, column=c, value=v)
                    cell.font = BODY_FONT; cell.alignment = CENTER
                    if fill: cell.fill = fill

        # ── legend ─────────────────────────────────────────────────────────
        leg_start = len(rows) + 3
        title_cell = ws7.cell(row=leg_start, column=1, value="LEGEND")
        title_cell.font      = LEG_TITLE
        title_cell.fill      = LEG_FILL
        title_cell.alignment = LEFT
        ws7.merge_cells(start_row=leg_start, start_column=1,
                        end_row=leg_start, end_column=4)

        ws7.cell(row=leg_start, column=1).value = (
            f"LEGEND  —  AU source: {au_source_label}.  "
            "All AU values are normalised to 0–1 (0 = absent, 1 = maximum intensity).  "
            "Post-hoc values come from OpenFace regression intensities (AU_r ÷ 5); "
            "blendshape-derived values are approximate anatomical equivalents from MediaPipe.  "
            "See 'AU Detail (post-hoc)' sheet for full per-frame OpenFace output when available."
        )

        for i, au in enumerate(AU_ORDER, leg_start + 1):
            if au not in AU_META:
                continue
            facs_name, desc = AU_META[au]
            k = ws7.cell(row=i, column=1, value=f"{au}  {facs_name}")
            k.font      = LEG_KEY
            k.alignment = LEFT
            d = ws7.cell(row=i, column=2, value=desc)
            d.font      = LEG_VAL
            d.alignment = LEFT
            ws7.merge_cells(start_row=i, start_column=2, end_row=i, end_column=4)

        # auto-width: fixed cols narrow, AU cols fixed ~14
        from openpyxl.utils import get_column_letter as gcl
        ws7.column_dimensions[gcl(1)].width = 22   # Timestamp
        ws7.column_dimensions[gcl(2)].width = 12   # Elapsed
        ws7.column_dimensions[gcl(3)].width = 14   # Face Detected
        ws7.column_dimensions[gcl(4)].width = 14   # Expression
        for c in range(5, 5 + len(AU_ORDER)):
            ws7.column_dimensions[gcl(c)].width = 28

        wb.save(path)

    # ════════════════════════════════════════════════════════════════════════
    # Workload spike → NASA TLX dialog
    # ════════════════════════════════════════════════════════════════════════
    def _on_workload_spike(self):
        """Called from the workload engine thread — hand off to the Qt main thread."""
        if self._tlx_open or not HAS_TLX:
            return
        self._tlx_open = True
        QTimer.singleShot(0, self._show_tlx_dialog)

    def _show_tlx_dialog(self):
        """Open the NASA TLX QDialog on the main thread."""
        dlg = NasaTLX(self, trigger_ts=time.time())
        dlg.completed.connect(self._on_tlx_complete)
        dlg.open()   # non-blocking; fires completed signal when done

    def _on_tlx_complete(self, result):
        self._tlx_open = False
        if result is not None:
            print(f"[TLX] weighted={result['weighted_tlx']:.1f}  "
                  f"raw={result['raw_tlx']:.1f}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # Window lifecycle
    # ════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        self._running = False
        # Flush and release writer before anything else
        with self._writer_lock:
            writer = self._writer
            self._writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        try:
            if hasattr(self, "_audio_stream"):
                self._audio_stream.stop()
                self._audio_stream.close()
        except Exception:
            pass
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Trust")
    app.setOrganizationName("Trust")

    # Ship Inter / JetBrainsMono .ttf files in a fonts/ folder for pixel parity
    # with the design preview. Falls back to system fonts if not present.
    load_packaged_fonts()
    app.setFont(ui_font(10))

    # ── Global QSS: neutralise macOS native chrome that bleeds through ────────
    app.setStyleSheet(f"""
        /* Slim, on-brand scrollbars */
        QScrollBar:vertical {{
            background: {BG_DEEP}; width: 8px; margin: 0; border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {LINE}; border-radius: 4px; min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover  {{ background: {TEXT_FAINT}; }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical      {{ height: 0; border: none; }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical      {{ background: none; }}

        QScrollBar:horizontal {{
            background: {BG_DEEP}; height: 8px; margin: 0; border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: {LINE}; border-radius: 4px; min-width: 28px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {TEXT_FAINT}; }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal    {{ width: 0; border: none; }}
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal    {{ background: none; }}

        /* Tooltip */
        QToolTip {{
            background: {PANEL}; color: {TEXT};
            border: 1px solid {LINE}; border-radius: 4px;
            padding: 4px 8px;
        }}

        /* Suppress the native macOS focus rectangle on buttons */
        QPushButton:focus {{ outline: none; }}
        QPushButton {{ outline: none; }}
    """)

    # Allow Ctrl+C to quit: Qt's event loop blocks Python signal delivery,
    # so a no-op timer forces the interpreter to wake up periodically and
    # check for pending signals.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _sigint_timer = QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    w = TrustDashboard()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
