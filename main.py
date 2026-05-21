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

        # ── Persistence ─────────────────────────────────────────────────────
        self._sessions_file = Path(__file__).parent / "sessions.json"

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
        self._session_start = time.time()
        self._session_rows = []

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
        stats = self._compute_session_stats()
        self._save_session(stats)
        self._show_summary(stats)

    def _back_to_overview(self):
        self._session_ended = False
        self._session_rows = []
        self._history = {k: [] for k in ("total", "facial", "vocal", "gaze", "hrv")}
        self._show_overview()

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
        row = {
            "t":          datetime.now().strftime("%H:%M:%S"),
            "total":      int(scores.get("total", 50)),
            "facial":     int(scores.get("facial", 50)),
            "vocal":      int(scores.get("vocal", 50)),
            "gaze":       int(scores.get("gaze", 50)),
            "hrv":        int(scores.get("hrv", 65)),
            "speaking":   bool(vocal_data and vocal_data.get("is_speaking")),
            "pitch_stab": float(vocal_data.get("pitch_stability", 0.5)) if vocal_data else 0.5,
            "tremor":     float(vocal_data.get("tremor_index", 0.0)) if vocal_data else 0.0,
            "face_det":   bool(face_data and face_data.get("detected")),
            "blink_rate": float(face_data.get("blink_rate", 0.0)) if face_data else 0.0,
            "gaze_dev":   float(face_data.get("gaze_deviation", 0.0)) if face_data else 0.0,
            "high_workload": bool(wl_state.get("is_high_workload", False)) if wl_state else False,
        }
        self._session_rows.append(row)

    def _compute_session_stats(self) -> dict:
        rows = self._session_rows
        if not rows:
            return {}
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        pct = lambda k: 100 * sum(1 for r in rows if r[k]) / n

        durMs = (time.time() - self._session_start)
        durSecs = int(durMs)
        durStr = f"{durSecs // 60:02d}:{durSecs % 60:02d}"

        history = [r["total"] for r in rows]

        return {
            "duration_str":          durStr,
            "n_samples":             n,
            "trust_total":           int(round(avg("total"))),
            "trust_facial":          int(round(avg("facial"))),
            "trust_vocal":           int(round(avg("vocal"))),
            "trust_gaze":            int(round(avg("gaze"))),
            "trust_hrv":             int(round(avg("hrv"))),
            "peak_trust":            max(r["total"] for r in rows),
            "low_trust":             min(r["total"] for r in rows),
            "pct_face_detected":     pct("face_det"),
            "pct_speaking":          pct("speaking"),
            "pct_high_workload":     pct("high_workload"),
            "avg_pitch_stability":   100 * avg("pitch_stab"),
            "avg_tremor":            100 * avg("tremor"),
            "avg_blink_rate":        avg("blink_rate"),
            "avg_gaze_deviation":    100 * avg("gaze_dev"),
            "trust_history":         history,
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
            "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration_str": stats.get("duration_str", "00:00"),
            "n_samples":    stats.get("n_samples", 0),
            "trust_total":  stats.get("trust_total", 50),
            "trust_facial": stats.get("trust_facial", 50),
            "trust_vocal":  stats.get("trust_vocal", 50),
            "trust_gaze":   stats.get("trust_gaze", 50),
            "trust_hrv":    stats.get("trust_hrv", 65),
        })
        try:
            with open(self._sessions_file, "w") as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"[sessions] Could not save: {e}")

    def _export_csv(self):
        if not self._session_rows:
            return
        default_name = f"trust-session-{datetime.now():%Y-%m-%d_%H-%M-%S}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export session as Excel", default_name, "Excel (*.xlsx)",
        )
        if not path:
            return
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Session"
            headers = list(self._session_rows[0].keys())
            ws.append(headers)
            for row in self._session_rows:
                ws.append([row[h] for h in headers])
            wb.save(path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

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
