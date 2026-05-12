import os
import csv
import io
import logging
from datetime import datetime
from tkinter import filedialog
logging.getLogger("root").setLevel(logging.ERROR)

import tkinter as tk                    # Python's built-in GUI toolkit — creates the desktop window and all widgets
from tkinter import messagebox
import threading                        # Runs camera capture and audio capture in background threads so the UI stays responsive
import time                             # Used in the camera loop to throttle analysis to ~15 fps
import math                             # Used for sin/cos when drawing the waveform line

import cv2                              # Captures frames from the webcam and draws the face overlay (box, eye outlines, label)
import numpy as np                      # Buffers audio samples and computes the FFT spectrum for the voice visualiser
import sounddevice as sd                # Streams microphone input without needing PyAudio or system drivers
from PIL import Image, ImageTk          # Converts OpenCV BGR frames into a format Tkinter can display on a Canvas
import matplotlib                       # Core matplotlib import; backend must be set before importing pyplot
matplotlib.use('TkAgg')                 # Forces matplotlib to render inside a Tkinter widget instead of opening its own window
import matplotlib.pyplot as plt         # Creates the history chart figure and axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embeds the matplotlib figure directly inside the Tkinter window

from Physio_analysis.face_analyzer import FaceAnalyzer
from Physio_analysis.vocal_analyzer import VocalAnalyzer
from trust_engine    import TrustEngine
from workload_engine import WorkloadEngine
from hrv_analyzer    import HRVAnalyzer
from nasa_tlx        import NasaTLX

# ── Colour palette (professional dark) ───────────────────────────────────────
BG        = '#0f1117'
SURFACE   = '#1a1d27'
BORDER    = '#2a2e3f'
CORAL     = '#4f8ef7'   # primary accent (blue)
BRONZE    = '#34d399'   # high-trust green
MAUVE     = '#fbbf24'   # mid-trust amber
GRAPE     = '#818cf8'   # indigo (vocal)
RED       = '#f87171'   # low-trust red
T1        = '#e2e8f0'
T2        = '#94a3b8'
T3        = '#64748b'
HEADER_BG = '#141824'
PANEL_SOFT = '#1f2232'
WAVE_BG   = '#141824'


def _trust_color(score: int) -> str:
    """Maps a 0–100 trust score to a palette colour."""
    if score >= 72: return BRONZE
    if score >= 50: return CORAL
    if score >= 32: return MAUVE
    return RED                      # Low trust → red


class TrustDashboard:
    CAM_W, CAM_H = 320, 240   # Width and height (pixels) at which the camera feed is displayed in the panel

    def __init__(self):
        self.root = tk.Tk()                         # Creates the main application window
        self.root.title('Trust Level Dashboard')    # Sets the title bar text
        self.root.configure(bg=BG)                  # Applies the dark background colour to the window
        self.root.minsize(1200, 1000)
        self.root.geometry('1340x1080')

        self.face     = FaceAnalyzer()
        self.vocal    = VocalAnalyzer()
        self.trust    = TrustEngine()
        self.workload = WorkloadEngine()
        self.hrv      = HRVAnalyzer()
        self.workload.set_tlx_callback(self._on_workload_spike)

        self._lock         = threading.Lock()       # Mutex that protects shared data written by background threads and read by the UI thread
        self._last_frame   = None                   # Holds the most recent (frame_bgr, face_data) tuple from the camera thread
        self._last_vocal   = None                   # Holds the most recent vocal_data dict from the audio callback
        self._audio_buffer = np.zeros(4096)         # Ring buffer that stores the latest audio samples for the waveform display
        self._sample_rate  = 44100                  # Default sample rate; overwritten with the actual device rate at startup
        self._running      = True                   # Flag that tells background threads to keep looping; set to False on window close
        self._history      = {k: [] for k in ('total', 'facial', 'vocal', 'gaze', 'hrv')}
        self._workload_state: dict = {}
        self._tlx_open     = False   # prevent re-entrant TLX dialogs
        self._session_rows:    list  = []   # one dict per recorded second
        self._session_start:   float = 0.0
        self._last_record_time:float = 0.0
        self._session_ended:   bool  = False
        self._calibration_pupil: list[float] = []   # baseline pupil samples
        self._calibration_seconds = 30
        self._calibrating = False
        self._calibration_started_at = None
        self._calibration_face = {"eye_ar": [], "blink_rate": [], "gaze_deviation": []}
        self._calibration_vocal = {"pitch_stability": [], "energy_level": [], "tremor_index": []}
        self._calibration_baseline = {}

        # Camera switching state
        self._available_cameras: list[int] = []    # Populated by _pick_camera; holds all detected camera indices
        self._camera_idx_pos: int = 0              # Index into _available_cameras pointing at the currently active camera

        # Status flags updated by background threads and read by the icon renderer
        self._cam_ok  = False                      # True once the first non-black frame has been captured successfully
        self._mic_ok  = False                      # True once the audio stream has opened and the first callback has fired

        self._build_ui()                    # Constructs all widgets and lays out the window
        self._build_calibration_overlay()   # Lays a full-window calibration screen on top of the main UI
        self._start_camera()                # Launches the camera capture and face analysis thread
        self._start_audio()                 # Starts the microphone input stream
        self._update()                      # Main UI update loop runs in the background during calibration

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_main_grid()
        self._build_workload_strip()
        self._build_chart_panel()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HEADER_BG)
        hdr.pack(fill='x', padx=12, pady=(12, 6))

        # Left: title block
        title_f = tk.Frame(hdr, bg=HEADER_BG)
        title_f.pack(side='left', padx=14, pady=10)
        tk.Label(title_f, text='Trust Level Dashboard',
                 bg=HEADER_BG, fg='#e2e8f0',
                 font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        tk.Label(title_f, text='Real-time facial · vocal · gaze · HRV · workload analysis',
                 bg=HEADER_BG, fg=T3,
                 font=('Segoe UI', 9)).pack(anchor='w')

        # Right: end session button + camera icon + mic icon
        icons_f = tk.Frame(hdr, bg=HEADER_BG)
        icons_f.pack(side='right', padx=16)

        # End Session button
        end_btn = tk.Button(
            icons_f, text='■  End Session',
            command=self._end_session,
            bg=CORAL, fg=SURFACE,
            font=('Segoe UI', 9, 'bold'),
            relief='flat', bd=0, padx=14, pady=6,
            cursor='hand2', activebackground='#3a7bd5',
            activeforeground=SURFACE,
        )
        end_btn.pack(side='left', padx=(0, 20))
        end_btn.bind('<Enter>', lambda e: end_btn.configure(bg='#3a7bd5'))
        end_btn.bind('<Leave>', lambda e: end_btn.configure(bg=CORAL))

        # Camera status icon
        cam_wrap = tk.Frame(icons_f, bg=HEADER_BG)
        cam_wrap.pack(side='left', padx=10)
        self._cam_icon = tk.Canvas(cam_wrap, width=28, height=22,
                                   bg=HEADER_BG, highlightthickness=0)
        self._cam_icon.pack(side='left')
        tk.Label(cam_wrap, text='Camera', bg=HEADER_BG, fg=T3,
                 font=('Segoe UI', 8)).pack(side='left', padx=(4, 0))

        # Mic status icon
        mic_wrap = tk.Frame(icons_f, bg=HEADER_BG)
        mic_wrap.pack(side='left', padx=10)
        self._mic_icon = tk.Canvas(mic_wrap, width=20, height=22,
                                   bg=HEADER_BG, highlightthickness=0)
        self._mic_icon.pack(side='left')
        tk.Label(mic_wrap, text='Mic', bg=HEADER_BG, fg=T3,
                 font=('Segoe UI', 8)).pack(side='left', padx=(4, 0))

        # Draw icons in their initial (inactive) state
        self._draw_camera_icon(T3)
        self._draw_mic_icon(T3)

    def _draw_camera_icon(self, color: str):
        """Redraws the camera status icon in the given colour."""
        c = self._cam_icon
        c.delete('all')
        # Camera body
        c.create_rectangle(1, 5, 25, 19, fill=color, outline='', tags='icon')
        # Lens — circle centred in body
        c.create_oval(8, 7, 18, 17, fill=SURFACE, outline=color, width=2, tags='icon')
        c.create_oval(10, 9, 16, 15, fill=color, outline='', tags='icon')
        # Viewfinder notch on top-right of body
        c.create_rectangle(18, 2, 24, 6, fill=color, outline='', tags='icon')

    def _draw_mic_icon(self, color: str):
        """Redraws the microphone status icon in the given colour."""
        c = self._mic_icon
        c.delete('all')
        # Mic capsule (rounded via oval + rect combination)
        c.create_rectangle(6, 4, 14, 12, fill=color, outline='', tags='icon')
        c.create_oval(6, 1, 14, 7, fill=color, outline='', tags='icon')
        c.create_oval(6, 9, 14, 15, fill=color, outline='', tags='icon')
        # Stand arc (bottom half of a circle)
        c.create_arc(2, 7, 18, 18, start=180, extent=180,
                     style='arc', outline=color, width=2, tags='icon')
        # Vertical stand and base
        c.create_line(10, 18, 10, 21, fill=color, width=2, tags='icon')
        c.create_line(6, 21, 14, 21, fill=color, width=2, tags='icon')

    def _build_main_grid(self):
        grid = tk.Frame(self.root, bg=BG)
        grid.pack(fill='x', expand=False, padx=12, pady=(0, 6))
        grid.columnconfigure(0, weight=6)
        grid.columnconfigure(1, weight=5)
        grid.rowconfigure(0, weight=1)

        # Left: camera card. Right: score + voice stacked for lower visual density.
        self._build_camera_panel(grid, row=0, col=0)

        right = tk.Frame(grid, bg=BG)
        right.grid(row=0, column=1, sticky='nsew', padx=(6, 0), pady=4)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)
        self._build_score_panel(right, row=0, col=0)
        self._build_voice_panel(right, row=1, col=0)

    def _card(self, parent, row, col, title, padx=(0, 0), pady=(0, 0)):
        f = tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        f.grid(row=row, column=col, sticky='nsew', padx=padx, pady=pady)
        tk.Label(f, text=title, bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=14, pady=(10, 4))
        return f

    def _build_camera_panel(self, grid, row=0, col=0):
        card = self._card(grid, row, col, 'CAMERA FEED', padx=(0, 6), pady=(4, 4))

        self.cam_canvas = tk.Canvas(card, width=self.CAM_W, height=self.CAM_H,
                                    bg='#141824', highlightthickness=0)
        self.cam_canvas.pack(padx=12, pady=(0, 6))

        # Switch Camera button — only fully visible when multiple cameras exist
        btn_row = tk.Frame(card, bg=SURFACE)
        btn_row.pack(fill='x', padx=12, pady=(0, 6))
        self._switch_btn = tk.Button(
            btn_row,
            text='⇄  Switch Camera',
            command=self._switch_camera,
            bg=PANEL_SOFT, fg=T1,
            font=('Segoe UI', 8),
            relief='flat', bd=0, padx=10, pady=4,
            activebackground=PANEL_SOFT, activeforeground=T1,
            cursor='hand2',
        )
        self._switch_btn.pack(side='left')
        # Label showing which camera index is active
        self._cam_label = tk.Label(btn_row, text='', bg=SURFACE, fg=T3,
                                   font=('Segoe UI', 8))
        self._cam_label.pack(side='left', padx=(8, 0))

        metrics = tk.Frame(card, bg=SURFACE)
        metrics.pack(fill='x', padx=12, pady=(0, 10))
        for r in range(2): metrics.rowconfigure(r, weight=1)
        for c in range(2): metrics.columnconfigure(c, weight=1)

        self._face_labels = {}
        for i, (label, key) in enumerate([('Expression', 'expr'), ('Eye Openness', 'ear'),
                                           ('Blink Rate', 'blink'), ('Gaze Deviation', 'gaze_dev')]):
            box = tk.Frame(metrics, bg=PANEL_SOFT, highlightthickness=1, highlightbackground=BORDER)
            box.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky='nsew')
            tk.Label(box, text=label, bg=PANEL_SOFT, fg=T3,
                     font=('Segoe UI', 7)).pack(anchor='w', padx=8, pady=(5, 0))
            lbl = tk.Label(box, text='—', bg=PANEL_SOFT, fg=CORAL,
                           font=('Segoe UI', 10, 'bold'))
            lbl.pack(anchor='w', padx=8, pady=(0, 5))
            self._face_labels[key] = lbl

    def _build_score_panel(self, grid, row=0, col=0):
        card = self._card(grid, row, col, 'TRUST SCORE', pady=(0, 6))

        self.gauge_canvas = tk.Canvas(card, width=200, height=115,
                                      bg=SURFACE, highlightthickness=0)
        self.gauge_canvas.pack(pady=(0, 4))

        self.trust_label_var = tk.StringVar(value='Calibrating…')
        self.trust_badge = tk.Label(card, textvariable=self.trust_label_var,
                                    bg=PANEL_SOFT, fg=CORAL,
                                    font=('Segoe UI', 11, 'bold'),
                                    padx=18, pady=6)
        self.trust_badge.pack(pady=(0, 10))

        bars_frame = tk.Frame(card, bg=SURFACE)
        bars_frame.pack(fill='x', padx=16, pady=(0, 12))

        self._bars = {}
        self._bar_nums = {}
        HRV_COLOR = '#818cf8'
        channels = [
            ('facial', CORAL,      'Facial', False),
            ('vocal',  GRAPE,      'Vocal',  False),
            ('gaze',   BRONZE,     'Gaze',   False),
            ('hrv',    HRV_COLOR,  'HRV',    True),   # True = stub channel
        ]
        for key, color, label, is_stub in channels:
            row = tk.Frame(bars_frame, bg=SURFACE)
            row.pack(fill='x', pady=3)
            lbl_color = T3 if is_stub else T2
            tk.Label(row, text=label, bg=SURFACE, fg=lbl_color,
                     font=('Segoe UI', 9), width=7, anchor='w').pack(side='left')
            track = tk.Canvas(row, height=7, bg=BORDER, highlightthickness=0)
            track.pack(side='left', fill='x', expand=True, padx=(4, 0))
            num = tk.Label(row, text='50', bg=SURFACE,
                           fg=T3 if is_stub else T1,
                           font=('Segoe UI', 9, 'bold'), width=4, anchor='e')
            num.pack(side='left', padx=(4, 0))
            self._bars[key] = (track, color if not is_stub else BORDER)
            self._bar_nums[key] = num

    def _build_voice_panel(self, grid, row=0, col=0):
        card = self._card(grid, row, col, 'VOICE ANALYSIS')

        self.wave_canvas = tk.Canvas(card, height=54, bg=WAVE_BG, highlightthickness=0)
        self.wave_canvas.pack(fill='x', padx=12, pady=(0, 6))

        self.spec_canvas = tk.Canvas(card, height=34, bg=WAVE_BG, highlightthickness=0)
        self.spec_canvas.pack(fill='x', padx=12, pady=(0, 10))

        metrics = tk.Frame(card, bg=SURFACE)
        metrics.pack(fill='x', padx=12, pady=(0, 10))
        for r in range(3): metrics.rowconfigure(r, weight=1)
        for c in range(2): metrics.columnconfigure(c, weight=1)

        self._vocal_labels = {}
        items = [('Pitch Stability', 'pitch'), ('Voice Energy', 'energy'),
                 ('Tremor Index', 'tremor'), ('Dominant Hz', 'hz'), ('Speaking', 'speaking')]
        for i, (label, key) in enumerate(items):
            box = tk.Frame(metrics, bg=PANEL_SOFT, highlightthickness=1, highlightbackground=BORDER)
            if i == 4:
                box.grid(row=2, column=0, columnspan=2, padx=3, pady=3, sticky='nsew')
            else:
                box.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky='nsew')
            tk.Label(box, text=label, bg=PANEL_SOFT, fg=T3,
                     font=('Segoe UI', 7)).pack(anchor='w', padx=8, pady=(5, 0))
            lbl = tk.Label(box, text='—', bg=PANEL_SOFT, fg=CORAL,
                           font=('Segoe UI', 10, 'bold'))
            lbl.pack(anchor='w', padx=8, pady=(0, 5))
            self._vocal_labels[key] = lbl

    def _build_workload_strip(self):
        strip = tk.Frame(self.root, bg=SURFACE)
        strip.pack(fill='x', padx=12, pady=(0, 6))

        inner = tk.Frame(strip, bg=SURFACE)
        inner.pack(fill='x', padx=20, pady=10)

        # ── Donut canvas ───────────────────────────────────────────────────────
        self._wl_canvas = tk.Canvas(inner, width=80, height=80,
                                     bg=SURFACE, highlightthickness=0)
        self._wl_canvas.pack(side='left', padx=(0, 18))

        # ── Labels stacked next to donut ───────────────────────────────────────
        label_col = tk.Frame(inner, bg=SURFACE)
        label_col.pack(side='left', padx=(0, 30))

        tk.Label(label_col, text='COGNITIVE LOAD', bg=SURFACE, fg=T3,
                 font=('Segoe UI', 7, 'bold')).pack(anchor='w')

        self._wl_state_lbl = tk.Label(label_col, text='LOW', bg=SURFACE,
                                       fg='#34d399', font=('Segoe UI', 13, 'bold'))
        self._wl_state_lbl.pack(anchor='w', pady=(2, 0))

        # ── Right: status text ─────────────────────────────────────────────────
        right_col = tk.Frame(inner, bg=SURFACE)
        right_col.pack(side='left', fill='x', expand=True)

        self._wl_bar_label = tk.Label(right_col, text='Monitoring', bg=SURFACE,
                                       fg=T3, font=('Segoe UI', 8))
        self._wl_bar_label.pack(anchor='w')

        self._spike_lbl = tk.Label(right_col, text='', bg=SURFACE,
                                    fg=T3, font=('Segoe UI', 8))
        self._spike_lbl.pack(anchor='w', pady=(3, 0))

        self._wl_status_lbl = tk.Label(right_col, text='', bg=SURFACE,
                                        fg=T3, font=('Segoe UI', 8))
        self._wl_status_lbl.pack(anchor='w', pady=(3, 0))

        # Hidden stub — _render_workload_strip references this for compat
        self._wl_dot = tk.Canvas(inner, width=0, height=0, highlightthickness=0)
        self._spike_track = tk.Canvas(inner, width=0, height=0, highlightthickness=0)

    def _build_chart_panel(self):
        chart_frame = tk.Frame(self.root, bg=SURFACE)
        chart_frame.pack(fill='both', expand=True, padx=12, pady=(0, 12))

        tk.Label(chart_frame, text='TRUST HISTORY', bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=14, pady=(10, 2))

        fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor(SURFACE)

        for ax in (self.ax1, self.ax2):
            ax.set_facecolor(SURFACE)
            ax.set_ylim(0, 100)
            ax.tick_params(colors=T2, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.grid(color=BORDER, linewidth=0.5, alpha=0.6)
            ax.xaxis.set_visible(False)

        # Left chart: Facial, Vocal, HRV
        self.ax1.set_title('Facial · Vocal · HRV', color=T2, fontsize=10, pad=8)
        self._chart_lines = {
            'facial': self.ax1.plot([], [], color=BRONZE, lw=2,   ls='--', label='Facial')[0],
            'vocal':  self.ax1.plot([], [], color=GRAPE,  lw=2,   ls='--', label='Vocal')[0],
            'hrv':    self.ax1.plot([], [], color='#5b8fa8', lw=2, ls=':',  label='HRV')[0],
        }
        self.ax1.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER,
                        labelcolor=T2, loc='upper left')

        # Right chart: Trust total, Gaze
        self.ax2.set_title('Trust · Gaze', color=T2, fontsize=10, pad=8)
        self._chart_lines2 = {
            'total': self.ax2.plot([], [], color=CORAL, lw=2.5, label='Trust')[0],
            'gaze':  self.ax2.plot([], [], color=MAUVE, lw=2,   ls='--', label='Gaze')[0],
        }
        self.ax2.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER,
                        labelcolor=T2, loc='upper left')

        fig.tight_layout(pad=1.2)
        fig.subplots_adjust(wspace=0.25)

        widget = FigureCanvasTkAgg(fig, master=chart_frame)
        widget.get_tk_widget().pack(fill='both', expand=True, padx=12, pady=(0, 10))
        self._fig_canvas = widget

    # ── Camera thread ──────────────────────────────────────────────────────────

    def _pick_camera(self) -> int:
        available = []
        for i in range(6):
            old_err = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                found = cap.isOpened()
                cap.release()
            finally:
                os.dup2(old_err, 2)
                os.close(old_err)
            if found:
                available.append(i)

        self._available_cameras = available if available else [0]  # Store for the switch button

        if len(available) <= 1:
            return available[0] if available else 0

        # More than one camera — show a small picker dialog
        chosen = tk.IntVar(value=available[0])
        dialog = tk.Toplevel(self.root)
        dialog.title('Select Camera')
        dialog.configure(bg=SURFACE)
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog, text='Multiple cameras detected.\nChoose the one to use:',
                 bg=SURFACE, fg=T1, font=('Segoe UI', 10),
                 pady=10, padx=20).pack()

        for idx in available:
            cap = cv2.VideoCapture(idx)
            ok, frame = cap.read()
            cap.release()
            label = f'Camera {idx}'
            if ok and frame is not None:
                h, w = frame.shape[:2]
                label += f'  ({w}×{h})'
            tk.Radiobutton(dialog, text=label, variable=chosen, value=idx,
                           bg=SURFACE, fg=T1, selectcolor=BG,
                           activebackground=SURFACE, activeforeground=CORAL,
                           font=('Segoe UI', 10)).pack(anchor='w', padx=24, pady=2)

        tk.Button(dialog, text='Use this camera', command=dialog.destroy,
                  bg=CORAL, fg=T1, font=('Segoe UI', 10, 'bold'),
                  relief='flat', padx=14, pady=6).pack(pady=12)

        self.root.wait_window(dialog)
        chosen_idx = chosen.get()
        self._camera_idx_pos = available.index(chosen_idx) if chosen_idx in available else 0
        return chosen_idx

    def _start_camera(self):
        idx = self._pick_camera()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(0.5)
        self._pending_frame = None
        self._update_switch_btn()                                              # Show/hide switch button based on camera count
        threading.Thread(target=self._camera_loop,   daemon=True).start()
        threading.Thread(target=self._analysis_loop, daemon=True).start()

    def _switch_camera(self):
        """Cycles to the next available camera index."""
        if len(self._available_cameras) <= 1:
            return
        self._camera_idx_pos = (self._camera_idx_pos + 1) % len(self._available_cameras)
        next_idx = self._available_cameras[self._camera_idx_pos]
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()                                                 # Releases the current camera before opening the next one
        self.cap = cv2.VideoCapture(next_idx, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cam_ok = False                                                   # Resets the camera status flag until the new camera delivers a valid frame
        self._update_switch_btn()

    def _update_switch_btn(self):
        """Updates the switch button label and visibility."""
        n = len(self._available_cameras)
        if n <= 1:
            self._switch_btn.configure(state='disabled', fg=T3)
        else:
            self._switch_btn.configure(state='normal', fg=T2)
        idx = self._available_cameras[self._camera_idx_pos] if self._available_cameras else 0
        self._cam_label.configure(
            text=f'cam {idx}  ({self._camera_idx_pos + 1}/{n})' if n > 1 else f'cam {idx}'
        )

    def _camera_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.mean() > 1.0:
                frame = cv2.flip(frame, 1)
                self._cam_ok = True                                            # Marks the camera as successfully delivering frames
                with self._lock:
                    self._pending_frame = frame
                    if self._last_frame is None:
                        self._last_frame = (frame, {"detected": False})
                    else:
                        self._last_frame = (frame, self._last_frame[1])
            time.sleep(0.033)

    def _analysis_loop(self):
        # MediaPipe runs ~10 ms per frame so we can afford to analyse every
        # pending frame without the 100 ms sleep that was needed for OpenFace.
        # We pair face_data with the SAME frame it was analysed on so the
        # bounding box coordinates always match the displayed camera image.
        while self._running:
            with self._lock:
                frame = self._pending_frame
            if frame is not None:
                small     = cv2.resize(frame, (640, 360))    # Downscale for analysis
                face_data = self.face.analyze(small)         # ~10 ms (MediaPipe) + async OpenFace
                with self._lock:
                    # Store the analysis frame alongside its result so the
                    # bounding box is always drawn on the correct image.
                    self._last_frame = (frame, face_data)
            time.sleep(0.033)                                # ~30 fps analysis rate

    # ── Audio thread ────────────────────────────────────────────────────────────

    def _start_audio(self):
        try:
            self._sample_rate = int(sd.query_devices(kind='input')['default_samplerate'])
        except Exception:
            self._sample_rate = 44100

        def callback(indata, frames, time_info, status):
            samples = indata[:, 0].copy()
            result  = self.vocal.analyze(samples, self._sample_rate)
            n = min(len(samples), len(self._audio_buffer))
            with self._lock:
                self._last_vocal   = result
                self._audio_buffer = np.roll(self._audio_buffer, -n)
                self._audio_buffer[-n:] = samples[:n]
            self._mic_ok = True                                                # Marks the mic as successfully receiving audio

        try:
            self._audio_stream = sd.InputStream(channels=1, blocksize=4096, callback=callback)
            self._audio_stream.start()
        except Exception as e:
            print(f'Microphone unavailable: {e}')

    # ── Main UI update loop ────────────────────────────────────────────────────

    def _update(self):
        try:
            self._update_body()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[update error] {exc}", flush=True)
        finally:
            if self._running:
                self.root.after(33, self._update)

    def _update_body(self):
        # Session summary is on screen — skip all UI renders so the matplotlib
        # canvas draw_idle() call can't repaint over the overlay content.
        if self._session_ended:
            return

        with self._lock:
            frame_data = self._last_frame
            vocal_data = self._last_vocal
            audio_buf  = self._audio_buffer.copy()

        face_data = frame_data[1] if frame_data else None
        if self._calibrating:
            self._collect_calibration_samples(face_data, vocal_data)
            elapsed   = time.time() - self._calibration_started_at
            remaining = max(0, int(math.ceil(self._calibration_seconds - elapsed)))
            self.trust_label_var.set(f'Calibrating… {remaining}s')
            self.trust_badge.configure(fg=MAUVE)
            scores = {"total": 50, "facial": 50, "vocal": 50, "gaze": 50, "hrv": 65}
            wl_state = {}
            if elapsed >= self._calibration_seconds:
                self._finish_calibration()
        else:
            hrv_score = self.hrv.get_score()
            scores    = self.trust.update(face_data, vocal_data, hrv_score)
            pupil_now = face_data.get("pupil_norm") if face_data else None
            wl_state  = self.workload.update(pupil_now)
            # Record one row per second after calibration (stop when session ended)
            if not self._session_ended:
                now = time.time()
                if now - self._last_record_time >= 1.0:
                    self._record_row(scores, face_data, vocal_data, wl_state)
                    self._last_record_time = now

        for k, v in scores.items():
            self._history[k].append(v)
            if len(self._history[k]) > 120: self._history[k].pop(0)

        self._workload_state = wl_state
        self._render_camera(frame_data)
        self._render_gauge(scores)
        self._render_bars(scores)
        self._render_face_metrics(face_data)
        self._render_vocal_metrics(vocal_data)
        self._render_waveform(audio_buf, vocal_data)
        self._render_spectrum(audio_buf)
        self._render_status_icons()
        self._render_workload_strip(wl_state)
        self._render_chart()

    def _collect_calibration_samples(self, face_data, vocal_data):
        if face_data and face_data.get('detected'):
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

    def _finish_calibration(self):
        self._calibration_baseline = {
            "face_eye_ar":            self._mean_or(self._calibration_face["eye_ar"], 0.27),
            "face_blink_rate":        self._mean_or(self._calibration_face["blink_rate"], 15.0),
            "face_gaze_deviation":    self._mean_or(self._calibration_face["gaze_deviation"], 0.0),
            "voice_pitch_stability":  self._mean_or(self._calibration_vocal["pitch_stability"], 0.5),
            "voice_energy_level":     self._mean_or(self._calibration_vocal["energy_level"], 0.0),
            "voice_tremor_index":     self._mean_or(self._calibration_vocal["tremor_index"], 0.0),
        }
        self.trust = TrustEngine()
        self._history = {k: [] for k in ('total', 'facial', 'vocal', 'gaze', 'hrv')}
        if self._calibration_pupil:
            baseline = sum(self._calibration_pupil) / len(self._calibration_pupil)
            self.workload.set_baseline(baseline)
        self._calibrating = False
        self._session_start = time.time()
        self._session_rows  = []          # fresh recording for this session
        # Destroy the calibration overlay — the main dashboard is now visible
        if hasattr(self, '_cal_overlay') and self._cal_overlay.winfo_exists():
            self._cal_overlay.destroy()

    # ── Calibration overlay ────────────────────────────────────────────────────

    def _build_calibration_overlay(self):
        """Builds a full-window calibration screen placed on top of the main UI."""
        f = tk.Frame(self.root, bg=BG)
        f.place(relx=0, rely=0, relwidth=1, relheight=1)   # Covers the entire window
        self._cal_overlay = f

        # ── Instructions screen (shown first) ─────────────────────────────────
        self._cal_intro = tk.Frame(f, bg=BG)
        self._cal_intro.place(relx=0, rely=0, relwidth=1, relheight=1)

        intro_inner = tk.Frame(self._cal_intro, bg=BG)
        intro_inner.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(intro_inner, text='Trust Level Dashboard',
                 bg=BG, fg=CORAL, font=('Segoe UI', 24, 'bold')).pack(pady=(0, 6))
        tk.Label(intro_inner, text='Voice & Face Calibration',
                 bg=BG, fg=T2, font=('Segoe UI', 13)).pack(pady=(0, 28))

        steps_frame = tk.Frame(intro_inner, bg=SURFACE, padx=36, pady=24)
        steps_frame.pack(pady=(0, 32))
        instructions = [
            ('1', 'Sit comfortably and look directly at the camera.'),
            ('2', 'Read the sentences shown on screen aloud, naturally.'),
            ('3', 'Speak in your normal conversational tone — no need to perform.'),
            ('4', 'The calibration takes 30 seconds to complete.'),
        ]
        for num, text in instructions:
            row = tk.Frame(steps_frame, bg=SURFACE)
            row.pack(anchor='w', pady=5)
            tk.Label(row, text=num, bg=CORAL, fg=BG,
                     font=('Segoe UI', 9, 'bold'),
                     width=2, height=1).pack(side='left', padx=(0, 12))
            tk.Label(row, text=text, bg=SURFACE, fg=T1,
                     font=('Segoe UI', 11)).pack(side='left')

        # ── Active calibration screen (hidden until button clicked) ───────────
        self._cal_active = tk.Frame(f, bg=BG)
        self._cal_active.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._cal_active.lower(self._cal_intro)   # Keep intro on top until button click

        def _start():
            self._calibration_started_at = time.time()
            self._calibrating = True
            self._cal_intro.destroy()
            self._update_cal()

        btn = tk.Label(intro_inner, text='Start Calibration',
                       bg=CORAL, fg='#e2e8f0', font=('Segoe UI', 12, 'bold'),
                       padx=32, pady=12, cursor='hand2')
        btn.pack()
        btn.bind('<Button-1>', lambda e: _start())
        btn.bind('<Enter>',    lambda e: btn.configure(bg='#141824'))
        btn.bind('<Leave>',    lambda e: btn.configure(bg=CORAL))


        # ── Title bar ─────────────────────────────────────────────────────────
        hdr = tk.Frame(self._cal_active, bg=BG)
        hdr.pack(fill='x', pady=(36, 0))
        tk.Label(hdr, text='Trust Level Dashboard',
                 bg=BG, fg=CORAL, font=('Segoe UI', 22, 'bold')).pack()
        tk.Label(hdr,
                 text='Sit comfortably, look at the camera and speak a few sentences naturally.',
                 bg=BG, fg=T2, font=('Segoe UI', 10)).pack(pady=(5, 0))

        # ── Two-column content area ────────────────────────────────────────────
        content = tk.Frame(self._cal_active, bg=BG)
        content.pack(expand=True, fill='both', padx=60, pady=20)
        content.columnconfigure(0, weight=3)   # Camera column gets more space
        content.columnconfigure(1, weight=2)   # Status column
        content.rowconfigure(0, weight=1)

        # Left card: live camera preview ──────────────────────────────────────
        cam_card = tk.Frame(content, bg=SURFACE)
        cam_card.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        tk.Label(cam_card, text='LIVE PREVIEW', bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=14, pady=(12, 6))
        self._cal_cam_canvas = tk.Canvas(cam_card, width=420, height=315,
                                         bg='#141824', highlightthickness=0)
        self._cal_cam_canvas.pack(padx=12, pady=(0, 8))
        tk.Label(cam_card,
                 text='Position your face in the centre of the frame.',
                 bg=SURFACE, fg=T2, font=('Segoe UI', 9)).pack(pady=(0, 6))

        self._cal_sentences = [
            '"The meeting is scheduled for Thursday at three in the afternoon."',
            '"I usually take the main road when the weather allows it."',
            '"She mentioned the project would wrap up by the end of the month."',
            '"Can you send me the details when you get a chance?"',
            '"The quarterly report includes updated figures from all four regions."',
        ]
        tk.Label(cam_card, text='Read aloud naturally:',
                 bg=SURFACE, fg=T3, font=('Segoe UI', 8, 'bold')).pack(pady=(0, 4))
        self._cal_sentence_lbl = tk.Label(
            cam_card, text=self._cal_sentences[0],
            bg=SURFACE, fg=T2, font=('Segoe UI', 15, 'italic'),
            wraplength=400, justify='center')
        self._cal_sentence_lbl.pack(pady=(0, 14))

        # Right card: countdown + status ──────────────────────────────────────
        stat_card = tk.Frame(content, bg=SURFACE)
        stat_card.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        tk.Label(stat_card, text='CALIBRATING', bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=14, pady=(12, 0))

        # Countdown arc
        self._cal_gauge_canvas = tk.Canvas(stat_card, width=160, height=160,
                                            bg=SURFACE, highlightthickness=0)
        self._cal_gauge_canvas.pack(pady=(8, 4))

        # Progress bar
        prog_wrap = tk.Frame(stat_card, bg=SURFACE)
        prog_wrap.pack(fill='x', padx=20, pady=(0, 16))
        tk.Label(prog_wrap, text='Progress', bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8)).pack(anchor='w')
        self._cal_prog_track = tk.Canvas(prog_wrap, height=8, bg=BORDER,
                                          highlightthickness=0)
        self._cal_prog_track.pack(fill='x', pady=(3, 0))

        # Thin divider
        tk.Frame(stat_card, bg=BORDER, height=1).pack(fill='x', padx=20, pady=(0, 14))

        # Status rows (face + voice)
        rows = tk.Frame(stat_card, bg=SURFACE)
        rows.pack(fill='x', padx=20, pady=(0, 10))

        face_row = tk.Frame(rows, bg=SURFACE)
        face_row.pack(fill='x', pady=7)
        self._cal_face_dot = tk.Canvas(face_row, width=12, height=12,
                                        bg=SURFACE, highlightthickness=0)
        self._cal_face_dot.pack(side='left')
        self._cal_face_dot.create_oval(1, 1, 11, 11, fill=BORDER, outline='', tags='dot')
        tk.Label(face_row, text='Face', bg=SURFACE, fg=T2,
                 font=('Segoe UI', 9, 'bold'), width=5, anchor='w').pack(side='left', padx=(6, 0))
        self._cal_face_lbl = tk.Label(face_row, text='Looking for face…',
                                       bg=SURFACE, fg=T3, font=('Segoe UI', 9))
        self._cal_face_lbl.pack(side='left')

        voice_row = tk.Frame(rows, bg=SURFACE)
        voice_row.pack(fill='x', pady=7)
        self._cal_voice_dot = tk.Canvas(voice_row, width=12, height=12,
                                         bg=SURFACE, highlightthickness=0)
        self._cal_voice_dot.pack(side='left')
        self._cal_voice_dot.create_oval(1, 1, 11, 11, fill=BORDER, outline='', tags='dot')
        tk.Label(voice_row, text='Voice', bg=SURFACE, fg=T2,
                 font=('Segoe UI', 9, 'bold'), width=5, anchor='w').pack(side='left', padx=(6, 0))
        self._cal_voice_lbl = tk.Label(voice_row, text='Speak to calibrate…',
                                        bg=SURFACE, fg=T3, font=('Segoe UI', 9))
        self._cal_voice_lbl.pack(side='left')

        # Bottom footnote + skip link
        foot = tk.Frame(self._cal_active, bg=BG)
        foot.pack(pady=(0, 18))
        tk.Label(foot,
                 text='Calibration personalises trust scoring to your natural resting state.',
                 bg=BG, fg=T3, font=('Segoe UI', 8)).pack()
        skip = tk.Label(foot, text='Skip calibration',
                        bg=BG, fg=T3, font=('Segoe UI', 9, 'underline'), cursor='hand2')
        skip.pack(pady=(6, 0))
        skip.bind('<Button-1>', lambda e: self._finish_calibration())
        skip.bind('<Enter>',    lambda e: skip.configure(fg=T2))
        skip.bind('<Leave>',    lambda e: skip.configure(fg=T3))

    def _update_cal(self):
        """Refreshes the calibration overlay every 100 ms; stops when calibration ends."""
        if not self._calibrating:
            return   # Overlay already destroyed by _finish_calibration; stop rescheduling

        elapsed   = time.time() - self._calibration_started_at
        remaining = max(0.0, self._calibration_seconds - elapsed)
        progress  = min(1.0, elapsed / self._calibration_seconds)

        # ── Rotating sentence ─────────────────────────────────────────────────
        idx = int(elapsed // 6) % len(self._cal_sentences)
        self._cal_sentence_lbl.configure(text=self._cal_sentences[idx])

        # ── Countdown arc ─────────────────────────────────────────────────────
        c = self._cal_gauge_canvas
        c.delete('all')
        cx, cy, r, lw = 80, 80, 58, 13
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        c.create_arc(x1, y1, x2, y2, start=0, extent=360,
                     style='arc', outline=BORDER, width=lw)         # Full grey ring
        if progress > 0:
            c.create_arc(x1, y1, x2, y2, start=90,
                         extent=-(360 * progress),                  # Coral arc sweeps clockwise as time passes
                         style='arc', outline=CORAL, width=lw)
        c.create_text(cx, cy - 10, text=str(int(math.ceil(remaining))),
                      fill=CORAL, font=('Segoe UI', 30, 'bold'))
        c.create_text(cx, cy + 16, text='seconds left',
                      fill=T3, font=('Segoe UI', 8))

        # ── Progress bar ──────────────────────────────────────────────────────
        self._cal_prog_track.update_idletasks()
        pw = self._cal_prog_track.winfo_width()
        if pw > 2:
            self._cal_prog_track.delete('all')
            self._cal_prog_track.create_rectangle(0, 0, pw, 8, fill=BORDER, outline='')
            fw = int(pw * progress)
            if fw > 0:
                self._cal_prog_track.create_rectangle(0, 0, fw, 8, fill=CORAL, outline='')

        # ── Camera preview ────────────────────────────────────────────────────
        with self._lock:
            frame_data = self._last_frame

        face_detected = False
        if frame_data is not None:
            frame, fd = frame_data
            face_detected = bool(fd and fd.get('detected'))
            overlay = frame.copy()
            fh, fw_ = overlay.shape[:2]
            if face_detected:
                bx, by, bw_, bh_ = fd['box_norm']
                cv2.rectangle(overlay,
                               (int(bx * fw_),        int(by * fh)),
                               (int((bx + bw_) * fw_), int((by + bh_) * fh)),
                               (82, 107, 201), 2)   # CORAL in BGR
            small = cv2.resize(overlay, (420, 315))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img   = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._cal_cam_canvas.create_image(0, 0, anchor='nw', image=img)
            self._cal_cam_canvas._img = img   # Prevent garbage collection

        # ── Face status dot & label ───────────────────────────────────────────
        face_color = BRONZE if face_detected else BORDER
        self._cal_face_dot.delete('dot')
        self._cal_face_dot.create_oval(1, 1, 11, 11, fill=face_color, outline='', tags='dot')
        self._cal_face_lbl.configure(
            text='Detected ✓' if face_detected else 'Looking for face…',
            fg=BRONZE if face_detected else T3,
        )

        # ── Voice status dot & label ──────────────────────────────────────────
        voice_samples = len(self._calibration_vocal['pitch_stability'])
        voice_ok      = voice_samples > 3
        voice_color   = GRAPE if voice_ok else BORDER
        self._cal_voice_dot.delete('dot')
        self._cal_voice_dot.create_oval(1, 1, 11, 11, fill=voice_color, outline='', tags='dot')
        self._cal_voice_lbl.configure(
            text=f'Captured ({voice_samples} samples) ✓' if voice_ok else 'Speak to calibrate…',
            fg=GRAPE if voice_ok else T3,
        )

        self.root.after(100, self._update_cal)   # Reschedule for the next 100 ms tick

    # ── Render helpers ─────────────────────────────────────────────────────────

    def _render_camera(self, frame_data):
        if frame_data is None: return
        frame, fd = frame_data
        overlay = frame.copy()
        h, w    = overlay.shape[:2]

        if fd and fd.get('detected'):
            bx, by, bw, bh = fd['box_norm']
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            # Face bounding box in coral
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (111, 107, 229), 2)   # BGR of CORAL #E56B6F
            cv2.putText(overlay, fd['dominant'], (x1 + 4, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (111, 107, 229), 1, cv2.LINE_AA)
            if fd.get('eye_norm'):
                for pts_norm, ear in [(fd['eye_norm']['l'], fd.get('l_ear', 0.3)),
                                       (fd['eye_norm']['r'], fd.get('r_ear', 0.3))]:
                    pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in pts_norm], dtype=np.int32)
                    # Bronze when open, mauve when squinting/closed
                    color = (139, 172, 234) if ear > 0.21 else (118, 89, 178)  # BGR BRONZE / MAUVE
                    cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)

        display = cv2.resize(overlay, (self.CAM_W, self.CAM_H))
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.cam_canvas.create_image(0, 0, anchor='nw', image=img)
        self.cam_canvas._img = img

    def _render_gauge(self, scores):
        c = self.gauge_canvas
        c.delete('all')
        s     = scores['total']
        color = MAUVE if self._calibrating else _trust_color(s)

        cx, cy, r, lw = 100, 100, 80, 14
        x1, y1, x2, y2 = cx-r, cy-r, cx+r, cy+r

        # Dark track arc
        c.create_arc(x1, y1, x2, y2, start=0, extent=180,
                     style='arc', outline=BORDER, width=lw)
        if s > 0:
            c.create_arc(x1, y1, x2, y2, start=180, extent=-(180 * s / 100),
                         style='arc', outline=color, width=lw)
        c.create_text(cx, cy - 14, text=str(s), fill=color,
                      font=('Segoe UI', 32, 'bold'))
        c.create_text(cx, cy + 4, text='TRUST LEVEL', fill=T3,
                      font=('Segoe UI', 7))
        if not self._calibrating:
            label = TrustEngine.trust_label(s)
            self.trust_label_var.set(label['text'])
            self.trust_badge.configure(fg=color)

    def _render_bars(self, scores):
        for key, (track, color) in self._bars.items():
            track.update_idletasks()
            w = track.winfo_width()
            if w < 2: continue
            track.delete('all')
            track.create_rectangle(0, 0, w, 8, fill=BORDER, outline='')
            fill_w = int(w * scores[key] / 100)
            if fill_w > 0:
                track.create_rectangle(0, 0, fill_w, 8, fill=color, outline='')
            self._bar_nums[key].configure(text=str(scores[key]))

    def _render_face_metrics(self, fd):
        if fd and fd.get('detected'):
            self._face_labels['expr'].configure(text=fd['dominant'])
            self._face_labels['ear'].configure(text=f"{fd['eye_ar'] * 100:.0f}%")
            self._face_labels['blink'].configure(text=f"{fd['blink_rate']:.0f}/min")
            self._face_labels['gaze_dev'].configure(text=f"{fd['gaze_deviation']*100:.0f}%")
        else:
            for lbl in self._face_labels.values(): lbl.configure(text='—')

    def _render_vocal_metrics(self, vd):
        if not vd: return
        self._vocal_labels['pitch'].configure(text=f"{vd['pitch_stability'] * 100:.0f}%")
        self._vocal_labels['energy'].configure(text=f"{vd['energy_level'] * 100:.0f}%")
        self._vocal_labels['tremor'].configure(text=f"{vd['tremor_index'] * 100:.0f}%")
        hz = vd.get('dominant_hz', 0)
        self._vocal_labels['hz'].configure(text=f"{hz:.0f} Hz" if hz else '—')
        speaking = vd.get('is_speaking', False)
        self._vocal_labels['speaking'].configure(
            text='Yes' if speaking else 'No',
            fg=BRONZE if speaking else T2
        )

    def _render_waveform(self, buf, vd):
        c = self.wave_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 2 or H < 2: return
        c.delete('all')
        c.create_rectangle(0, 0, W, H, fill=WAVE_BG, outline='')
        color  = GRAPE if (vd and vd.get('is_speaking')) else BORDER
        step   = max(1, len(buf) // W)
        points = []
        for i in range(W):
            sample = buf[min(i * step, len(buf) - 1)]
            y      = H / 2 + sample * H * 0.42
            points.extend([i, y])
        if len(points) >= 4:
            c.create_line(points, fill=color, width=1.5, smooth=True)

    def _render_spectrum(self, buf):
        c = self.spec_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 2 or H < 2: return
        c.delete('all')
        c.create_rectangle(0, 0, W, H, fill=WAVE_BG, outline='')
        window = np.hanning(len(buf))
        fft    = np.abs(np.fft.rfft(buf * window))
        fft    = fft[:len(fft) // 4]
        fft    = fft / (fft.max() + 1e-10)
        bars   = min(W // 4, 64)
        bw     = W / bars
        step   = max(1, len(fft) // bars)
        for i in range(bars):
            val  = float(np.mean(fft[i * step: i * step + step]))
            bh   = val * H
            # Gradient from GRAPE → CORAL based on intensity
            r1, g1, b1 = 0x6D, 0x59, 0x7A   # GRAPE
            r2, g2, b2 = 0xE5, 0x6B, 0x6F   # CORAL
            r = int(r1 + (r2 - r1) * val)
            g = int(g1 + (g2 - g1) * val)
            b = int(b1 + (b2 - b1) * val)
            color = f'#{r:02x}{g:02x}{b:02x}'
            c.create_rectangle(i * bw, H - bh, (i + 1) * bw - 1, H,
                                fill=color, outline='')

    def _render_status_icons(self):
        """Redraws the camera and mic header icons with live status colours."""
        cam_color = BRONZE if self._cam_ok else T3   # Bronze = camera delivering frames; grey = no signal yet
        mic_color = GRAPE  if self._mic_ok else T3   # Grape  = microphone active; grey = not yet detected
        self._draw_camera_icon(cam_color)
        self._draw_mic_icon(mic_color)

    def _render_workload_strip(self, wl: dict):
        if not wl:
            return

        high      = wl.get('is_high_workload', False)
        progress  = wl.get('spike_progress', 0.0)
        elapsed_s = int(progress * 60)

        # Pick semantic color based on load level
        if progress >= 1.0 or (high and progress > 0.66):
            arc_color = '#f87171'   # red
        elif high or progress > 0.33:
            arc_color = '#fbbf24'   # amber
        else:
            arc_color = '#34d399'   # green

        state_color = '#f87171' if high else '#34d399'
        self._wl_state_lbl.configure(text='HIGH' if high else 'LOW', fg=state_color)

        # ── Draw donut ─────────────────────────────────────────────────────────
        c = self._wl_canvas
        c.delete('all')
        # Background ring (full circle)
        c.create_arc(6, 6, 74, 74, start=90, extent=359.9,
                     style='arc', outline=BORDER, width=12)
        # Progress arc (clockwise from top)
        if progress > 0:
            c.create_arc(6, 6, 74, 74, start=90, extent=-(progress * 359.9),
                         style='arc', outline=arc_color, width=12)
        # Center percentage text
        c.create_text(40, 38, text=f'{int(progress * 100)}%',
                      fill=arc_color, font=('Segoe UI', 11, 'bold'))
        c.create_text(40, 54, text='load', fill=T3, font=('Segoe UI', 7))

        # ── Status labels ──────────────────────────────────────────────────────
        if progress >= 1.0:
            self._wl_bar_label.configure(text='Sustained high load', fg='#f87171')
            self._spike_lbl.configure(text=f'{elapsed_s}s / 60s', fg=T3)
            self._wl_status_lbl.configure(text='Assessment due →', fg=CORAL)
        elif high and progress > 0:
            self._wl_bar_label.configure(text='Building high load…', fg=T2)
            self._spike_lbl.configure(text=f'{elapsed_s}s / 60s', fg=T3)
            self._wl_status_lbl.configure(text='High load detected', fg=BRONZE)
        else:
            self._wl_bar_label.configure(text='No sustained load', fg=T3)
            self._spike_lbl.configure(text='')
            self._wl_status_lbl.configure(text='Monitoring', fg=T3)

    def _on_workload_spike(self):
        """Called by WorkloadEngine when a 60s spike ends at a low-WL moment."""
        if self._tlx_open:
            return
        # Schedule on the next Tk tick so we're not nested inside _update
        self.root.after(0, self._show_tlx)

    def _show_tlx(self):
        if self._tlx_open:
            return
        self._tlx_open = True
        import time as _time
        NasaTLX(self.root, on_complete=self._on_tlx_complete,
                trigger_ts=_time.time())

    def _on_tlx_complete(self, result: dict | None):
        self._tlx_open = False
        if result is None:
            return   # user dismissed
        weighted = result.get('weighted_tlx', '?')
        raw      = result.get('raw_tlx', '?')
        print(f"[NASA TLX] Weighted={weighted}  Raw={raw}  "
              f"Ratings={result.get('ratings', {})}")

    # ── End session & summary screen ──────────────────────────────────────────

    def _end_session(self):
        """Stop recording and show the session summary overlay."""
        if self._calibrating:
            messagebox.showinfo("Not started",
                                "Calibration hasn't finished yet.\n"
                                "Please complete calibration before ending the session.")
            return
        if not self._session_rows:
            messagebox.showinfo("No data",
                                "No session data has been recorded yet.\n"
                                "Wait at least a few seconds after calibration.")
            return
        self._session_ended = True
        self._build_session_summary()

    def _compute_session_stats(self) -> dict:
        rows = self._session_rows
        if not rows:
            return {}
        n = len(rows)

        def avg(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / len(vals), 1) if vals else 0

        def pct_yes(key):
            return round(sum(1 for r in rows if r.get(key) == "Yes") / n * 100, 1)

        duration_s = rows[-1].get("elapsed_s", 0)
        mins, secs = divmod(int(duration_s), 60)
        hrs,  mins = divmod(mins, 60)
        duration_str = (f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs
                        else f"{mins:02d}:{secs:02d}")

        return {
            "duration_str":       duration_str,
            "n_samples":          n,
            "trust_total":        avg("trust_total"),
            "trust_facial":       avg("trust_facial"),
            "trust_vocal":        avg("trust_vocal"),
            "trust_gaze":         avg("trust_gaze"),
            "trust_hrv":          avg("trust_hrv"),
            "pct_speaking":       pct_yes("speaking"),
            "avg_pitch_stability":avg("pitch_stability_pct"),
            "avg_voice_energy":   avg("voice_energy_pct"),
            "avg_tremor":         avg("tremor_index_pct"),
            "pct_face_detected":  pct_yes("face_detected"),
            "avg_blink_rate":     avg("blink_rate_bpm"),
            "avg_gaze_deviation": avg("gaze_deviation_pct"),
            "pct_high_workload":  pct_yes("workload_high"),
            "trust_history":      [r["trust_total"] for r in rows
                                   if isinstance(r.get("trust_total"), (int, float))],
        }

    def _build_session_summary(self):
        """Full-window summary overlay shown after End Session."""
        stats = self._compute_session_stats()
        total = stats.get("trust_total", 50)
        tlabel = TrustEngine.trust_label(int(total))

        # ── Outer overlay (covers entire window) ──────────────────────────
        ov = tk.Frame(self.root, bg=BG)
        ov.place(relx=0, rely=0, relwidth=1, relheight=1)
        ov.lift()
        self._summary_overlay = ov

        # ── Top bar ───────────────────────────────────────────────────────
        topbar = tk.Frame(ov, bg=HEADER_BG)
        topbar.pack(fill='x')
        tk.Label(topbar, text='Session Complete',
                 bg=HEADER_BG, fg='#e2e8f0',
                 font=('Segoe UI', 15, 'bold'),
                 padx=24, pady=14).pack(side='left')
        tk.Label(topbar,
                 text=f'Duration  {stats["duration_str"]}  ·  {stats["n_samples"]} samples recorded',
                 bg=HEADER_BG, fg=T3,
                 font=('Segoe UI', 9), padx=24).pack(side='right')

        # ── Three cards side by side (pure pack, no grid) ─────────────────
        cards_row = tk.Frame(ov, bg=BG)
        cards_row.pack(fill='x', padx=20, pady=(16, 10))

        def make_card(parent, title):
            """Return a white card frame with a labelled header."""
            f = tk.Frame(parent, bg=SURFACE)
            f.pack(side='left', fill='both', expand=True, padx=6)
            tk.Label(f, text=title, bg=SURFACE, fg=T3,
                     font=('Segoe UI', 8, 'bold'),
                     padx=18, pady=(12, 6)).pack(anchor='w')
            tk.Frame(f, bg=BORDER, height=1).pack(fill='x', padx=18, pady=(0, 8))
            return f

        # Card 1 — Overall score
        c1 = make_card(cards_row, 'OVERALL TRUST')
        tk.Label(c1, text=str(int(total)),
                 bg=SURFACE, fg=tlabel["color"],
                 font=('Segoe UI', 54, 'bold')).pack(pady=(8, 0))
        tk.Label(c1, text=tlabel["text"],
                 bg=SURFACE, fg=tlabel["color"],
                 font=('Segoe UI', 12, 'bold')).pack()
        tk.Label(c1, text='session average',
                 bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8)).pack(pady=(4, 18))

        # Card 2 — Channel breakdown with Canvas bars
        c2 = make_card(cards_row, 'CHANNEL BREAKDOWN')
        channels = [
            ('Facial', stats['trust_facial'], CORAL),
            ('Vocal',  stats['trust_vocal'],  GRAPE),
            ('Gaze',   stats['trust_gaze'],   MAUVE),
            ('HRV',    stats['trust_hrv'],    '#818cf8'),
        ]

        def _draw_bar(canvas, pct, color):
            canvas.delete('all')
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 2:
                return
            canvas.create_rectangle(0, 0, w, h, fill=BORDER, outline='')
            fill_w = max(4, int(w * pct / 100))
            canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline='')

        for ch_name, ch_val, ch_color in channels:
            row_f = tk.Frame(c2, bg=SURFACE)
            row_f.pack(fill='x', padx=18, pady=5)
            tk.Label(row_f, text=ch_name, bg=SURFACE, fg=T2,
                     font=('Segoe UI', 9), width=7, anchor='w').pack(side='left')
            bar_cv = tk.Canvas(row_f, height=10, bg=BORDER,
                               highlightthickness=0, bd=0)
            bar_cv.pack(side='left', fill='x', expand=True, padx=(6, 8))
            tk.Label(row_f, text=str(int(ch_val)), bg=SURFACE, fg=T1,
                     font=('Segoe UI', 9, 'bold'), width=3, anchor='e').pack(side='left')
            # Draw after widget has a real width
            bar_cv.bind('<Configure>',
                        lambda e, c=bar_cv, v=ch_val, clr=ch_color:
                        _draw_bar(c, v, clr))
        tk.Frame(c2, bg=SURFACE, height=10).pack()

        # Card 3 — Key metrics
        c3 = make_card(cards_row, 'KEY METRICS')
        metrics = [
            ('Face detected',   f"{stats['pct_face_detected']:.0f}%"),
            ('Time speaking',   f"{stats['pct_speaking']:.0f}% of session"),
            ('Pitch stability', f"{stats['avg_pitch_stability']:.0f}%"),
            ('Voice tremor',    f"{stats['avg_tremor']:.0f}%"),
            ('Avg blink rate',  f"{stats['avg_blink_rate']:.0f} /min"),
            ('Gaze deviation',  f"{stats['avg_gaze_deviation']:.0f}%"),
            ('High workload',   f"{stats['pct_high_workload']:.0f}% of session"),
        ]
        for m_label, m_val in metrics:
            mrow = tk.Frame(c3, bg=SURFACE)
            mrow.pack(fill='x', padx=18, pady=4)
            tk.Label(mrow, text=m_label, bg=SURFACE, fg=T3,
                     font=('Segoe UI', 9), anchor='w').pack(side='left')
            tk.Label(mrow, text=m_val, bg=SURFACE, fg=T1,
                     font=('Segoe UI', 9, 'bold')).pack(side='right')
        tk.Frame(c3, bg=SURFACE, height=10).pack()

        # ── Trust history chart ───────────────────────────────────────────
        chart_card = tk.Frame(ov, bg=SURFACE)
        chart_card.pack(fill='x', padx=28, pady=(0, 10))
        tk.Label(chart_card, text='TRUST SCORE HISTORY',
                 bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold'),
                 padx=14, pady=(10, 2)).pack(anchor='w')

        hist = stats.get("trust_history", [])
        if len(hist) >= 2:
            fig_s, ax_s = plt.subplots(figsize=(11, 2.0))
            fig_s.patch.set_facecolor(SURFACE)
            ax_s.set_facecolor(SURFACE)
            ax_s.set_ylim(0, 100)
            ax_s.set_xlim(0, max(len(hist) - 1, 1))
            ax_s.tick_params(colors=T3, labelsize=8)
            for sp in ax_s.spines.values():
                sp.set_color(BORDER)
            ax_s.grid(color=BORDER, linewidth=0.5, alpha=0.6)
            ax_s.xaxis.set_visible(False)
            xs = list(range(len(hist)))
            ax_s.fill_between(xs, hist, alpha=0.10, color=CORAL)
            ax_s.plot(xs, hist, color=CORAL, lw=2.5, label='Trust')
            ax_s.axhline(total, color=CORAL, lw=1.2, ls='--', alpha=0.5,
                         label=f'Avg {int(total)}')
            ax_s.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER,
                        labelcolor=T2, loc='upper left')
            fig_s.tight_layout(pad=0.8)
            cv_s = FigureCanvasTkAgg(fig_s, master=chart_card)
            cv_s.draw()
            cv_s.get_tk_widget().pack(fill='x', padx=12, pady=(0, 12))
        else:
            tk.Label(chart_card,
                     text='Not enough data to plot history.',
                     bg=SURFACE, fg=T3,
                     font=('Segoe UI', 9), pady=18).pack()

        # ── Footer ────────────────────────────────────────────────────────
        footer = tk.Frame(ov, bg=BG)
        footer.pack(fill='x', padx=28, pady=(4, 22))

        new_btn = tk.Button(footer, text='↺  New Session',
                            command=self._restart_session,
                            bg=SURFACE, fg=T2,
                            font=('Segoe UI', 10, 'bold'),
                            relief='flat', bd=1, padx=20, pady=10,
                            cursor='hand2')
        new_btn.pack(side='left')
        new_btn.bind('<Enter>', lambda e: new_btn.configure(bg=BORDER))
        new_btn.bind('<Leave>', lambda e: new_btn.configure(bg=SURFACE))

        exp_btn = tk.Button(footer, text='⬇  Export Session Data',
                            command=self._export_excel,
                            bg=BRONZE, fg=SURFACE,
                            font=('Segoe UI', 10, 'bold'),
                            relief='flat', bd=0, padx=24, pady=10,
                            cursor='hand2',
                            activebackground='#3a7bd5',
                            activeforeground=SURFACE)
        exp_btn.pack(side='right')
        exp_btn.bind('<Enter>', lambda e: exp_btn.configure(bg='#3a7bd5'))
        exp_btn.bind('<Leave>', lambda e: exp_btn.configure(bg=BRONZE))

        # Force Tkinter to lay out and paint all overlay widgets immediately.
        ov.update_idletasks()

    def _restart_session(self):
        """Tear down the summary overlay and restart calibration."""
        if hasattr(self, '_summary_overlay') and self._summary_overlay.winfo_exists():
            self._summary_overlay.destroy()
        # Reset all session state
        self._session_rows       = []
        self._session_start      = 0.0
        self._last_record_time   = 0.0
        self._session_ended      = False
        self._history            = {k: [] for k in ('total', 'facial', 'vocal', 'gaze', 'hrv')}
        self._calibration_pupil  = []
        self._calibration_face   = {"eye_ar": [], "blink_rate": [], "gaze_deviation": []}
        self._calibration_vocal  = {"pitch_stability": [], "energy_level": [], "tremor_index": []}
        self._calibration_baseline = {}
        self.trust               = TrustEngine()
        self.workload            = WorkloadEngine()
        self.workload.set_tlx_callback(self._on_workload_spike)
        self._calibrating        = False
        self._calibration_started_at = None
        self._build_calibration_overlay()

    # ── Session recording & export ─────────────────────────────────────────────

    # Column order used for both CSV and Excel
    _CSV_COLUMNS = [
        "timestamp", "elapsed_s",
        "trust_total", "trust_facial", "trust_vocal", "trust_gaze", "trust_hrv",
        "face_detected", "face_expression", "eye_ar_pct", "blink_rate_bpm",
        "gaze_deviation_pct", "pupil_norm", "duchenne",
        "speaking", "pitch_stability_pct", "voice_energy_pct",
        "tremor_index_pct", "vocal_hz",
        "workload_high", "pcps", "wiv", "spike_progress_pct",
    ]

    def _record_row(self, scores: dict, fd: dict | None, vd: dict | None,
                    wl: dict) -> None:
        """Append one row of sensor readings to the in-memory session log."""
        now = time.time()
        row: dict = {
            "timestamp":          datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":          round(now - self._session_start, 1),
            # Trust scores
            "trust_total":        scores.get("total",  50),
            "trust_facial":       scores.get("facial", 50),
            "trust_vocal":        scores.get("vocal",  50),
            "trust_gaze":         scores.get("gaze",   50),
            "trust_hrv":          scores.get("hrv",    65),
            # Face / gaze
            "face_detected":      "Yes" if (fd and fd.get("detected")) else "No",
            "face_expression":    fd.get("dominant", "—")       if fd else "—",
            "eye_ar_pct":         round(fd.get("eye_ar", 0) * 100, 1)         if fd else "",
            "blink_rate_bpm":     round(fd.get("blink_rate", 0), 1)           if fd else "",
            "gaze_deviation_pct": round(fd.get("gaze_deviation", 0) * 100, 1) if fd else "",
            "pupil_norm":         round(fd.get("pupil_norm", 0), 4)            if fd and fd.get("pupil_norm") is not None else "",
            "duchenne":           round(fd.get("duchenne", 0), 3)              if fd else "",
            # Voice
            "speaking":           "Yes" if (vd and vd.get("is_speaking")) else "No",
            "pitch_stability_pct":round(vd.get("pitch_stability", 0) * 100, 1) if vd else "",
            "voice_energy_pct":   round(vd.get("energy_level",    0) * 100, 1) if vd else "",
            "tremor_index_pct":   round(vd.get("tremor_index",    0) * 100, 1) if vd else "",
            "vocal_hz":           round(vd.get("dominant_hz",     0), 1)       if vd else "",
            # Workload
            "workload_high":      "Yes" if wl.get("is_high_workload") else "No",
            "pcps":               wl.get("pcps",           ""),
            "wiv":                wl.get("wiv",            ""),
            "spike_progress_pct": round(wl.get("spike_progress", 0) * 100, 1) if wl else "",
        }
        self._session_rows.append(row)

    def _export_excel(self) -> None:
        """Write session data to an Excel file chosen by the user."""
        if not self._session_rows:
            messagebox.showinfo("No data", "No session data recorded yet.\n"
                                   "Complete calibration and wait at least 1 second.")
            return

        default = f"trust_session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        path = filedialog.asksaveasfilename(
            title="Save session data",
            initialfile=default,
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("CSV file", "*.csv")],
        )
        if not path:
            return

        if path.endswith(".csv"):
            self._write_csv(path)
        else:
            self._write_xlsx(path)

    def _write_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._session_rows)
        messagebox.showinfo("Exported", f"CSV saved to:\n{path}")

    def _write_xlsx(self, path: str) -> None:
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            csv_path = path.replace(".xlsx", ".csv")
            self._write_csv(csv_path)
            messagebox.showwarning(
                "openpyxl missing",
                f"openpyxl is not installed — saved as CSV instead:\n{csv_path}\n\n"
                "Install it with:  pip install openpyxl")
            return

        # ── Shared style constants ──────────────────────────────────────────
        _HEADERS = {
            "timestamp":          "Timestamp",
            "elapsed_s":          "Elapsed (s)",
            "trust_total":        "Trust Total",
            "trust_facial":       "Facial Trust",
            "trust_vocal":        "Vocal Trust",
            "trust_gaze":         "Gaze Trust",
            "trust_hrv":          "HRV Trust",
            "face_detected":      "Face Detected",
            "face_expression":    "Expression",
            "eye_ar_pct":         "Eye Openness %",
            "blink_rate_bpm":     "Blink Rate /min",
            "gaze_deviation_pct": "Gaze Deviation %",
            "pupil_norm":         "Pupil (norm.)",
            "duchenne":           "Duchenne Smile",
            "speaking":           "Speaking",
            "pitch_stability_pct":"Pitch Stability %",
            "voice_energy_pct":   "Voice Energy %",
            "tremor_index_pct":   "Tremor Index %",
            "vocal_hz":           "Vocal Hz",
            "workload_high":      "High Workload",
            "pcps":               "PCPS",
            "wiv":                "WIV",
            "spike_progress_pct": "Spike Progress %",
        }
        _WIDTHS = {
            "timestamp": 20, "elapsed_s": 10,
            "trust_total": 13, "trust_facial": 12, "trust_vocal": 12,
            "trust_gaze": 12, "trust_hrv": 12,
            "face_detected": 13, "face_expression": 14,
            "eye_ar_pct": 14, "blink_rate_bpm": 15,
            "gaze_deviation_pct": 17, "pupil_norm": 14, "duchenne": 15,
            "speaking": 10, "pitch_stability_pct": 17,
            "voice_energy_pct": 15, "tremor_index_pct": 15, "vocal_hz": 10,
            "workload_high": 14, "pcps": 10, "wiv": 10, "spike_progress_pct": 16,
        }
        _SCORE_COLS = {"trust_total", "trust_facial", "trust_vocal",
                       "trust_gaze", "trust_hrv"}

        thin       = Side(style="thin", color="E8D5C4")
        cell_bdr   = Border(left=thin, right=thin, top=thin, bottom=thin)
        alt_fill   = PatternFill("solid", fgColor="FDF0E6")
        norm_font  = Font(size=9)
        num_align  = Alignment(horizontal="right")
        hdr_fill   = PatternFill("solid", fgColor="C94D52")
        hdr_font   = Font(bold=True, color="FFFFFF", size=10)
        hdr_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)

        def score_fill(score):
            try:
                s = int(score)
            except (TypeError, ValueError):
                return None
            if s >= 64:
                r = int(0x4a + (0xb8 - 0x4a) * (100 - s) / 36)
                g = int(0xde + (0x73 - 0xde) * (100 - s) / 36)
                b = int(0x80 + (0x40 - 0x80) * (100 - s) / 36)
            else:
                r = int(0xb8 + (0xC9 - 0xb8) * (64 - s) / 64)
                g = int(0x73 + (0x4d - 0x73) * (64 - s) / 64)
                b = int(0x40 + (0x52 - 0x40) * (64 - s) / 64)
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            return PatternFill("solid", fgColor=f"{r:02X}{g:02X}{b:02X}")

        def populate_sheet(ws, columns):
            """Write header + all session rows for the given column list."""
            # Header
            for col_idx, key in enumerate(columns, start=1):
                cell = ws.cell(row=1, column=col_idx, value=_HEADERS[key])
                cell.font      = hdr_font
                cell.fill      = hdr_fill
                cell.alignment = hdr_align
                cell.border    = cell_bdr
            ws.row_dimensions[1].height = 36

            # Data rows
            for row_idx, row_data in enumerate(self._session_rows, start=2):
                bg = alt_fill if row_idx % 2 == 0 else None
                for col_idx, key in enumerate(columns, start=1):
                    val  = row_data.get(key, "")
                    cell = ws.cell(row=row_idx, column=col_idx, value=val)
                    cell.font   = norm_font
                    cell.border = cell_bdr
                    if key in _SCORE_COLS:
                        sf = score_fill(val)
                        if sf:
                            cell.fill = sf
                            cell.font = Font(size=9, bold=True, color="2A1A24")
                    elif bg:
                        cell.fill = bg
                    if isinstance(val, (int, float)):
                        cell.alignment = num_align

            # Column widths + freeze
            for col_idx, key in enumerate(columns, start=1):
                ws.column_dimensions[get_column_letter(col_idx)].width = _WIDTHS.get(key, 12)
            ws.freeze_panes = "A2"

        # ── Sheet definitions ───────────────────────────────────────────────
        # Each entry: (sheet title, [column keys])
        SHEETS = [
            ("All Data", self._CSV_COLUMNS),
            ("Face & Gaze", [
                "timestamp", "elapsed_s",
                "trust_facial", "trust_gaze",
                "face_detected", "face_expression",
                "eye_ar_pct", "blink_rate_bpm", "gaze_deviation_pct",
                "pupil_norm", "duchenne",
            ]),
            ("Voice", [
                "timestamp", "elapsed_s",
                "trust_vocal",
                "speaking", "pitch_stability_pct",
                "voice_energy_pct", "tremor_index_pct", "vocal_hz",
            ]),
            ("Workload", [
                "timestamp", "elapsed_s",
                "workload_high", "pcps", "wiv", "spike_progress_pct",
            ]),
            ("HRV", [
                "timestamp", "elapsed_s",
                "trust_hrv",
            ]),
        ]

        # ── Build workbook ──────────────────────────────────────────────────
        wb = openpyxl.Workbook()
        wb.remove(wb.active)   # remove the default blank sheet

        for title, cols in SHEETS:
            ws = wb.create_sheet(title=title)
            populate_sheet(ws, cols)

        wb.save(path)
        messagebox.showinfo("Exported",
                            f"Excel file saved to:\n{path}\n\n"
                            f"Sheets: {', '.join(t for t, _ in SHEETS)}")

    def _render_chart(self):
        h = self._history
        if len(h['total']) < 2: return
        x = list(range(len(h['total'])))
        for key, line in self._chart_lines.items():
            line.set_data(x, h[key])
        for key, line in self._chart_lines2.items():
            line.set_data(x, h[key])
        xlim = (0, max(len(h['total']) - 1, 1))
        self.ax1.set_xlim(*xlim)
        self.ax2.set_xlim(*xlim)
        try:
            self._fig_canvas.draw_idle()
        except Exception:
            pass

    # ── Shutdown ────────────────────────────────────────────────────────────────

    def run(self):
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, '_audio_stream'):
            self._audio_stream.stop()
            self._audio_stream.close()
        self.root.destroy()


if __name__ == '__main__':
    app = TrustDashboard()
    app.run()
