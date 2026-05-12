import os
import logging
os.environ["TQDM_DISABLE"] = "1"   # Suppress py-feat's per-frame tqdm progress bars
logging.getLogger("root").setLevel(logging.ERROR)  # Suppress py-feat's "NO FACE detected" warnings

# Compatibility shims for py-feat with newer library versions:
# 3. PyTorch 2.x forbids .numpy() on grad-tracked tensors; py-feat calls it without .detach()
import torch as _torch
_orig_np = _torch.Tensor.numpy
def _safe_numpy(self, *a, **kw):
    return self.detach().numpy(*a, **kw)
_torch.Tensor.numpy = _safe_numpy
# 1. torchvision 0.21+ removed read_video; py-feat imports it but only uses it for video files (unused here)
import torchvision.io as _tvio
if not hasattr(_tvio, "read_video"):
    _tvio.read_video = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("read_video unavailable"))
# 2. scipy 1.14+ renamed simps → simpson; py-feat still imports the old name
import scipy.integrate as _sci
if not hasattr(_sci, "simps"):
    _sci.simps = _sci.simpson

import tkinter as tk                    # Python's built-in GUI toolkit — creates the desktop window and all widgets
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

from face_analyzer import FaceAnalyzer  # Runs MediaPipe face landmark detection and maps blendshapes to emotion scores
from vocal_analyzer import VocalAnalyzer  # Analyses pitch stability, energy, and tremor from raw audio samples
from trust_engine import TrustEngine    # Combines face/vocal/gaze scores into a smoothed overall trust percentage

# ── Colour palette (matches the web version) ─────────────────────────────────
BG      = '#080812'   # Near-black page background
SURFACE = '#111122'   # Slightly lighter card/panel background
BORDER  = '#1e2040'   # Subtle border colour used for dividers and chart grid lines
CYAN    = '#00d4ff'   # Primary accent — used for headings, gauge arc, and camera overlay
GREEN   = '#4ade80'   # Positive / high trust colour
YELLOW  = '#fbbf24'   # Warning / medium trust colour
RED     = '#f87171'   # Danger / low trust colour
BLUE    = '#60a5fa'   # Vocal channel accent colour
T1      = '#e2e8f0'   # Primary text (near white)
T2      = '#94a3b8'   # Secondary text (light grey)
T3      = '#475569'   # Muted text (dark grey) used for labels and axis ticks


class TrustDashboard:
    CAM_W, CAM_H = 320, 240   # Width and height (pixels) at which the camera feed is displayed in the panel

    def __init__(self):
        self.root = tk.Tk()                         # Creates the main application window
        self.root.title('Trust Level Dashboard')    # Sets the title bar text
        self.root.configure(bg=BG)                  # Applies the dark background colour to the window
        self.root.minsize(1100, 680)                # Prevents the window from being resized smaller than this

        self.face  = FaceAnalyzer()                 # Initialises MediaPipe and downloads the face landmarker model on first run
        self.vocal = VocalAnalyzer()                # Initialises the vocal analyser (no setup needed — pure NumPy)
        self.trust = TrustEngine()                  # Initialises the trust engine with neutral starting scores

        self._lock         = threading.Lock()       # Mutex that protects shared data written by background threads and read by the UI thread
        self._last_frame   = None                   # Holds the most recent (frame_bgr, face_data) tuple from the camera thread
        self._last_vocal   = None                   # Holds the most recent vocal_data dict from the audio callback
        self._audio_buffer = np.zeros(4096)         # Ring buffer that stores the latest audio samples for the waveform display
        self._sample_rate  = 44100                  # Default sample rate; overwritten with the actual device rate at startup
        self._running      = True                   # Flag that tells background threads to keep looping; set to False on window close
        self._history      = {k: [] for k in ('total', 'facial', 'vocal', 'gaze')}  # Lists of past scores used to draw the history chart

        self._build_ui()      # Constructs all widgets and lays out the window
        self._start_camera()  # Launches the camera capture and face analysis thread
        self._start_audio()   # Starts the microphone input stream
        self._update()        # Kicks off the recurring UI refresh loop

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()         # Top bar with title and status indicator dots
        self._build_main_grid()      # Three-column panel row: camera | score | voice
        self._build_chart_panel()    # Full-width history chart at the bottom

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=SURFACE)                # Header bar frame with the surface background colour
        hdr.pack(fill='x', padx=12, pady=(12, 6))            # Stretches across the full window width with padding

        title_f = tk.Frame(hdr, bg=SURFACE)                  # Inner frame that groups the two title lines on the left
        title_f.pack(side='left', padx=14, pady=8)           # Aligned to the left side of the header
        tk.Label(title_f, text='Trust Level Dashboard',
                 bg=SURFACE, fg=CYAN,
                 font=('Segoe UI', 16, 'bold')).pack(anchor='w')   # Large cyan title text
        tk.Label(title_f, text='Real-time facial · vocal · gaze analysis',
                 bg=SURFACE, fg=T3,
                 font=('Segoe UI', 9)).pack(anchor='w')             # Smaller subtitle beneath the title

        dots_f = tk.Frame(hdr, bg=SURFACE)                   # Right-aligned frame that holds the three status indicators
        dots_f.pack(side='right', padx=14)
        self._status_dots = {}                               # Dict that maps channel names to their Canvas widgets for later updates
        for label, key in [('Face', 'face'), ('Gaze', 'gaze'), ('Voice', 'voice')]:
            f = tk.Frame(dots_f, bg=SURFACE)                 # One small frame per indicator (dot + text)
            f.pack(side='left', padx=10)
            dot = tk.Canvas(f, width=10, height=10, bg=SURFACE, highlightthickness=0)  # Tiny canvas that draws the coloured circle
            dot.pack(side='left')
            dot.create_oval(1, 1, 9, 9, fill=T3, outline='', tags='dot')  # Starts grey; coloured later when signal is active
            tk.Label(f, text=label, bg=SURFACE, fg=T2,
                     font=('Segoe UI', 8)).pack(side='left', padx=(3, 0))  # Text label next to the dot
            self._status_dots[key] = dot                     # Saves the canvas so _render_dots can update its fill colour

    def _build_main_grid(self):
        grid = tk.Frame(self.root, bg=BG)                    # Container frame for the three-column layout
        grid.pack(fill='both', expand=True, padx=12, pady=(0, 6))
        grid.columnconfigure(0, weight=2)                    # Camera column gets slightly more space
        grid.columnconfigure(1, weight=2)                    # Score column
        grid.columnconfigure(2, weight=2)                    # Voice column
        grid.rowconfigure(0, weight=1)                       # Single row expands vertically to fill available space

        self._build_camera_panel(grid)   # Left panel: live camera feed + face metrics
        self._build_score_panel(grid)    # Centre panel: gauge, trust label, bar breakdown
        self._build_voice_panel(grid)    # Right panel: waveform, spectrum, vocal metrics

    def _card(self, parent, col, title):
        f = tk.Frame(parent, bg=SURFACE)                     # Creates a card-style frame with the surface colour
        f.grid(row=0, column=col, sticky='nsew',
               padx=(0 if col == 0 else 5, 0 if col == 2 else 5), pady=4)  # Adds horizontal gutters between cards
        tk.Label(f, text=title, bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=14, pady=(10, 4))  # Section title label in muted text
        return f                                             # Returns the frame so the caller can add content inside it

    def _build_camera_panel(self, grid):
        card = self._card(grid, 0, 'CAMERA FEED')

        self.cam_canvas = tk.Canvas(card, width=self.CAM_W, height=self.CAM_H,
                                    bg='#000', highlightthickness=0)   # Black canvas that shows each camera frame as an image
        self.cam_canvas.pack(padx=12, pady=(0, 8))

        metrics = tk.Frame(card, bg=SURFACE)                           # 2×2 grid of small metric boxes below the camera
        metrics.pack(fill='x', padx=12, pady=(0, 10))
        for r in range(2): metrics.rowconfigure(r, weight=1)          # Two rows in the metrics grid
        for c in range(2): metrics.columnconfigure(c, weight=1)       # Two columns in the metrics grid

        self._face_labels = {}                                         # Dict mapping metric keys to their Label widgets for later updates
        for i, (label, key) in enumerate([('Expression', 'expr'), ('Eye Openness', 'ear'),
                                           ('Blink Rate', 'blink'), ('Gaze Deviation', 'gaze_dev')]):
            box = tk.Frame(metrics, bg=BG)                            # Dark box background for each metric cell
            box.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky='nsew')
            tk.Label(box, text=label, bg=BG, fg=T3,
                     font=('Segoe UI', 7)).pack(anchor='w', padx=8, pady=(5, 0))   # Small muted label at the top of the cell
            lbl = tk.Label(box, text='—', bg=BG, fg=CYAN,
                           font=('Segoe UI', 10, 'bold'))             # Cyan value that gets updated each frame
            lbl.pack(anchor='w', padx=8, pady=(0, 5))
            self._face_labels[key] = lbl                              # Saves the label widget for update calls

    def _build_score_panel(self, grid):
        card = self._card(grid, 1, 'TRUST SCORE')

        self.gauge_canvas = tk.Canvas(card, width=200, height=115,
                                      bg=SURFACE, highlightthickness=0)   # Canvas where the arc gauge is drawn each frame
        self.gauge_canvas.pack(pady=(0, 4))

        self.trust_label_var = tk.StringVar(value='Calibrating…')         # StringVar lets us update the badge text without recreating the widget
        self.trust_badge = tk.Label(card, textvariable=self.trust_label_var,
                                    bg=BG, fg=CYAN,
                                    font=('Segoe UI', 11, 'bold'),
                                    padx=18, pady=6)                       # Pill-shaped label that shows the current trust category text
        self.trust_badge.pack(pady=(0, 10))

        bars_frame = tk.Frame(card, bg=SURFACE)                            # Container for the three channel score bars
        bars_frame.pack(fill='x', padx=16, pady=(0, 12))

        self._bars = {}      # Maps channel key → (Canvas widget, fill colour)
        self._bar_nums = {}  # Maps channel key → Label widget showing the numeric score
        for key, color in [('facial', CYAN), ('vocal', BLUE), ('gaze', YELLOW)]:
            row = tk.Frame(bars_frame, bg=SURFACE)                         # One row per channel
            row.pack(fill='x', pady=4)
            tk.Label(row, text=key.capitalize(), bg=SURFACE, fg=T2,
                     font=('Segoe UI', 9), width=7, anchor='w').pack(side='left')  # Channel name label
            track = tk.Canvas(row, height=8, bg=BG, highlightthickness=0) # Grey track that the coloured fill bar sits inside
            track.pack(side='left', fill='x', expand=True, padx=(4, 0))
            num = tk.Label(row, text='50', bg=SURFACE, fg=T1,
                           font=('Segoe UI', 9, 'bold'), width=4, anchor='e')  # Numeric score label on the right
            num.pack(side='left', padx=(4, 0))
            self._bars[key] = (track, color)                               # Saves track canvas and colour for update calls
            self._bar_nums[key] = num                                      # Saves the number label for update calls

    def _build_voice_panel(self, grid):
        card = self._card(grid, 2, 'VOICE ANALYSIS')

        self.wave_canvas = tk.Canvas(card, height=70, bg='#0d0d1c', highlightthickness=0)  # Dark canvas for the time-domain waveform
        self.wave_canvas.pack(fill='x', padx=12, pady=(0, 6))

        self.spec_canvas = tk.Canvas(card, height=50, bg='#0d0d1c', highlightthickness=0)  # Dark canvas for the frequency spectrum bars
        self.spec_canvas.pack(fill='x', padx=12, pady=(0, 10))

        metrics = tk.Frame(card, bg=SURFACE)                               # Grid of vocal metric boxes
        metrics.pack(fill='x', padx=12, pady=(0, 10))
        for r in range(3): metrics.rowconfigure(r, weight=1)
        for c in range(2): metrics.columnconfigure(c, weight=1)

        self._vocal_labels = {}                                            # Dict mapping metric keys to Label widgets
        items = [('Pitch Stability', 'pitch'), ('Voice Energy', 'energy'),
                 ('Tremor Index', 'tremor'), ('Dominant Hz', 'hz'), ('Speaking', 'speaking')]
        for i, (label, key) in enumerate(items):
            box = tk.Frame(metrics, bg=BG)
            if i == 4:                                                     # "Speaking" spans both columns on the last row
                box.grid(row=2, column=0, columnspan=2, padx=3, pady=3, sticky='nsew')
            else:
                box.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky='nsew')
            tk.Label(box, text=label, bg=BG, fg=T3,
                     font=('Segoe UI', 7)).pack(anchor='w', padx=8, pady=(5, 0))
            lbl = tk.Label(box, text='—', bg=BG, fg=CYAN,
                           font=('Segoe UI', 10, 'bold'))
            lbl.pack(anchor='w', padx=8, pady=(0, 5))
            self._vocal_labels[key] = lbl

    def _build_chart_panel(self):
        chart_frame = tk.Frame(self.root, bg=SURFACE)                      # Full-width card at the bottom of the window
        chart_frame.pack(fill='x', padx=12, pady=(0, 12))
        tk.Label(chart_frame, text='TRUST HISTORY', bg=SURFACE, fg=T3,
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=14, pady=(10, 2))

        fig, self.ax = plt.subplots(figsize=(11, 1.8))                     # Creates a wide, short matplotlib figure
        fig.patch.set_facecolor(SURFACE)                                   # Matches the figure background to the card colour
        self.ax.set_facecolor(SURFACE)                                     # Matches the axes background to the card colour
        self.ax.set_ylim(0, 100)                                           # Y axis always shows the full 0–100 score range
        self.ax.tick_params(colors=T3, labelsize=8)                        # Muted tick label colour
        for spine in self.ax.spines.values(): spine.set_color(BORDER)     # Subtle border around the chart area
        self.ax.grid(color=BORDER, linewidth=0.5, alpha=0.5)              # Faint horizontal grid lines for readability
        self.ax.xaxis.set_visible(False)                                   # Hides x-axis labels (time ticks) to save space

        self._chart_lines = {                                              # Creates one line artist per channel; updated each tick
            'total':  self.ax.plot([], [], color=CYAN,    lw=2.5, label='Trust')[0],
            'facial': self.ax.plot([], [], color=GREEN,   lw=1.2, ls='--', label='Facial')[0],
            'vocal':  self.ax.plot([], [], color='#818cf8', lw=1.2, ls='--', label='Vocal')[0],
            'gaze':   self.ax.plot([], [], color=YELLOW,  lw=1.2, ls='--', label='Gaze')[0],
        }
        self.ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER,
                       labelcolor=T2, loc='upper left')                    # Legend inside the chart area, dark-themed
        fig.tight_layout(pad=0.8)                                          # Removes excess whitespace around the figure

        widget = FigureCanvasTkAgg(fig, master=chart_frame)               # Wraps the matplotlib figure as a Tkinter widget
        widget.get_tk_widget().pack(fill='x', padx=12, pady=(0, 10))     # Places the chart widget inside the card frame
        self._fig_canvas = widget                                         # Saves a reference so we can call draw_idle() each tick

    # ── Camera thread ──────────────────────────────────────────────────────────

    def _pick_camera(self) -> int:
        import sys, io
        available = []
        for i in range(6):                                                 # Checks indices 0–5 for connected cameras
            old_err = os.dup(2)                                            # Suppress OpenCV "out of bound" stderr spam while probing
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

        if len(available) <= 1:                                            # If only one camera found, use it automatically
            return available[0] if available else 0

        # More than one camera — show a small picker dialog
        chosen = tk.IntVar(value=available[0])
        dialog = tk.Toplevel(self.root)
        dialog.title('Select Camera')
        dialog.configure(bg=SURFACE)
        dialog.resizable(False, False)
        dialog.grab_set()                                                  # Blocks the main window until the user picks

        tk.Label(dialog, text='Multiple cameras detected.\nChoose the one to use:',
                 bg=SURFACE, fg=T1, font=('Segoe UI', 10),
                 pady=10, padx=20).pack()

        for idx in available:
            # Try to read one frame to show a tiny preview label
            cap = cv2.VideoCapture(idx)
            ok, frame = cap.read()
            cap.release()
            label = f'Camera {idx}'
            if ok and frame is not None:
                h, w = frame.shape[:2]
                label += f'  ({w}×{h})'
            tk.Radiobutton(dialog, text=label, variable=chosen, value=idx,
                           bg=SURFACE, fg=T1, selectcolor=BG,
                           activebackground=SURFACE, activeforeground=CYAN,
                           font=('Segoe UI', 10)).pack(anchor='w', padx=24, pady=2)

        tk.Button(dialog, text='Use this camera', command=dialog.destroy,
                  bg=CYAN, fg=BG, font=('Segoe UI', 10, 'bold'),
                  relief='flat', padx=14, pady=6).pack(pady=12)

        self.root.wait_window(dialog)                                      # Waits here until the dialog is closed
        return chosen.get()

    def _start_camera(self):
        idx = self._pick_camera()                                          # Shows picker if multiple cameras are detected
        self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)            # AVFoundation backend required on macOS for reliable camera access
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(0.5)                                                    # Give AVFoundation time to initialise before reading
        self._pending_frame = None                                         # Frame queued for face analysis
        threading.Thread(target=self._camera_loop,  daemon=True).start()  # Reads frames at display rate
        threading.Thread(target=self._analysis_loop, daemon=True).start() # Runs py-feat at its own (slower) pace

    def _camera_loop(self):
        while self._running:                                               # Keeps running until the window is closed
            ok, frame = self.cap.read()                                    # Reads one frame from the webcam
            if ok and frame is not None and frame.mean() > 1.0:           # Skip black/empty frames that AVFoundation emits during warm-up
                frame = cv2.flip(frame, 1)                                 # Mirrors the frame horizontally so it behaves like a mirror
                with self._lock:
                    self._pending_frame = frame                            # Always show the latest frame immediately
                    if self._last_frame is None:
                        self._last_frame = (frame, {"detected": False})
                    else:
                        self._last_frame = (frame, self._last_frame[1])   # Keep last known face_data until analysis updates it
            time.sleep(0.033)                                              # ~30 fps display rate

    def _analysis_loop(self):
        print("[analysis] thread started", flush=True)
        while self._running:                                               # Runs face analysis independently so it never blocks the display
            with self._lock:
                frame = self._pending_frame
            if frame is not None:
                print("[analysis] got frame, starting detect…", flush=True)
                small = cv2.resize(frame, (640, 360))                      # Downscale to 640×360 — keeps 16:9 aspect ratio so face proportions stay correct
                face_data = self.face.analyze(small)
                print(f"[analysis] detected={face_data.get('detected')} dominant={face_data.get('dominant','—')}", flush=True)
                with self._lock:
                    if self._last_frame is not None:
                        self._last_frame = (self._last_frame[0], face_data)
            time.sleep(0.1)                                                # Analyse at ~10 fps max; py-feat typically takes longer than this anyway

    # ── Audio thread ────────────────────────────────────────────────────────────

    def _start_audio(self):
        try:
            self._sample_rate = int(sd.query_devices(kind='input')['default_samplerate'])  # Queries the microphone's native sample rate
        except Exception:
            self._sample_rate = 44100                                      # Falls back to 44.1 kHz if the query fails

        def callback(indata, frames, time_info, status):                   # Called by sounddevice every time a new block of audio arrives
            samples = indata[:, 0].copy()                                  # Takes the first (mono) channel as a 1-D float32 array
            result  = self.vocal.analyze(samples, self._sample_rate)       # Runs pitch/energy/tremor analysis on the new samples
            n = min(len(samples), len(self._audio_buffer))                 # Number of samples to roll into the waveform buffer
            with self._lock:
                self._last_vocal   = result                                # Stores the analysis result for the UI thread to read
                self._audio_buffer = np.roll(self._audio_buffer, -n)      # Shifts the ring buffer left by n positions
                self._audio_buffer[-n:] = samples[:n]                     # Writes the newest samples into the end of the ring buffer

        try:
            self._audio_stream = sd.InputStream(channels=1, blocksize=4096, callback=callback)  # Opens a mono input stream with 4096-sample blocks
            self._audio_stream.start()                                     # Starts the stream; callback fires automatically from here
        except Exception as e:
            print(f'Microphone unavailable: {e}')                         # Prints a warning but lets the app continue without audio

    # ── Main UI update loop ────────────────────────────────────────────────────

    def _update(self):
        with self._lock:                                                   # Acquires the lock to safely read shared data
            frame_data  = self._last_frame                                 # Gets the latest camera frame and face result
            vocal_data  = self._last_vocal                                 # Gets the latest vocal analysis result
            audio_buf   = self._audio_buffer.copy()                        # Copies the audio ring buffer for waveform drawing

        face_data = frame_data[1] if frame_data else None                  # Unpacks face_data from the tuple (or None if no frame yet)
        scores    = self.trust.update(face_data, vocal_data)               # Recalculates smoothed trust scores from the latest inputs

        for k, v in scores.items():                                        # Appends each score to its history list
            self._history[k].append(v)
            if len(self._history[k]) > 120: self._history[k].pop(0)       # Caps the history at 120 points (4 seconds at 30 fps)

        self._render_camera(frame_data)                                    # Draws the camera frame and face overlay on the camera canvas
        self._render_gauge(scores)                                         # Redraws the arc gauge with the latest total score
        self._render_bars(scores)                                          # Updates the three channel bar widths and numbers
        self._render_face_metrics(face_data)                               # Updates the four face metric labels
        self._render_vocal_metrics(vocal_data)                             # Updates the five vocal metric labels
        self._render_waveform(audio_buf, vocal_data)                       # Draws the audio waveform line
        self._render_spectrum(audio_buf)                                   # Draws the frequency spectrum bars
        self._render_dots(face_data, vocal_data)                           # Colours the three status dots in the header
        self._render_chart()                                               # Pushes new data points to the matplotlib history chart

        if self._running:
            self.root.after(33, self._update)                              # Schedules the next UI update in 33 ms (~30 fps)

    # ── Render helpers ─────────────────────────────────────────────────────────

    def _render_camera(self, frame_data):
        if frame_data is None: return                                      # Nothing to draw before the first frame arrives
        frame, fd = frame_data
        overlay = frame.copy()                                             # Works on a copy so the original frame isn't modified
        h, w    = overlay.shape[:2]

        if fd and fd.get('detected'):
            bx, by, bw, bh = fd['box_norm']                               # Normalised bounding box [0,1]
            x1, y1 = int(bx * w), int(by * h)                            # Converts to pixel coordinates
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 212, 255), 2) # Cyan bounding box around the face
            cv2.putText(overlay, fd['dominant'], (x1 + 4, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1, cv2.LINE_AA)  # Expression label above the box
            if fd.get('eye_norm'):
                for pts_norm, ear in [(fd['eye_norm']['l'], fd.get('l_ear', 0.3)),
                                       (fd['eye_norm']['r'], fd.get('r_ear', 0.3))]:
                    pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in pts_norm], dtype=np.int32)
                    color = (74, 222, 128) if ear > 0.21 else (251, 191, 36)  # Green when eye is open, yellow when squinting/closed
                    cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)  # Draws the 6-point eye outline polygon

        display = cv2.resize(overlay, (self.CAM_W, self.CAM_H))           # Scales the frame down to fit the camera panel
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)                # Converts from OpenCV BGR to RGB for PIL
        img     = ImageTk.PhotoImage(Image.fromarray(rgb))                # Converts the NumPy array to a Tkinter-compatible image
        self.cam_canvas.create_image(0, 0, anchor='nw', image=img)        # Draws the image on the canvas starting at the top-left corner
        self.cam_canvas._img = img                                        # Keeps a reference to prevent garbage collection

    def _render_gauge(self, scores):
        c = self.gauge_canvas
        c.delete('all')                                                    # Clears the previous frame's arc so we can redraw fresh
        s     = scores['total']
        label = TrustEngine.trust_label(s)                                 # Gets the colour and text for this score value
        color = label['color']

        cx, cy, r, lw = 100, 100, 80, 14                                  # Gauge centre, radius, and stroke width in canvas pixels
        x1, y1, x2, y2 = cx-r, cy-r, cx+r, cy+r                         # Bounding box of the full circle (arc is drawn inside this)

        c.create_arc(x1, y1, x2, y2, start=0, extent=180,
                     style='arc', outline=BORDER, width=lw)               # Grey background arc spanning the full top semicircle
        if s > 0:
            c.create_arc(x1, y1, x2, y2, start=180, extent=-(180 * s / 100),
                         style='arc', outline=color, width=lw)            # Coloured fill arc: start=left (180°), grows clockwise as score rises
        c.create_text(cx, cy - 14, text=str(s), fill=color,
                      font=('Segoe UI', 32, 'bold'))                      # Large score number in the centre of the gauge
        c.create_text(cx, cy + 4, text='TRUST LEVEL', fill=T3,
                      font=('Segoe UI', 7))                               # Small label beneath the score number
        self.trust_label_var.set(label['text'])                           # Updates the badge text below the gauge
        self.trust_badge.configure(fg=color)                              # Recolours the badge to match the score category

    def _render_bars(self, scores):
        for key, (track, color) in self._bars.items():
            track.update_idletasks()                                       # Forces Tkinter to calculate the widget's current pixel width
            w = track.winfo_width()
            if w < 2: continue                                             # Skips drawing before the widget has been laid out
            track.delete('all')
            track.create_rectangle(0, 0, w, 8, fill=BG, outline='')      # Redraws the dark track background
            fill_w = int(w * scores[key] / 100)                           # Calculates fill width proportional to the score
            if fill_w > 0:
                track.create_rectangle(0, 0, fill_w, 8, fill=color, outline='')  # Draws the coloured fill portion
            self._bar_nums[key].configure(text=str(scores[key]))          # Updates the numeric score label on the right

    def _render_face_metrics(self, fd):
        if fd and fd.get('detected'):
            self._face_labels['expr'].configure(text=fd['dominant'])                       # Dominant emotion name
            self._face_labels['ear'].configure(text=f"{fd['eye_ar'] * 100:.0f}%")         # Eye openness as a percentage
            self._face_labels['blink'].configure(text=f"{fd['blink_rate']:.0f}/min")      # Blink rate in blinks per minute
            self._face_labels['gaze_dev'].configure(text=f"{fd['gaze_deviation']*100:.0f}%")  # Gaze deviation as a percentage
        else:
            for lbl in self._face_labels.values(): lbl.configure(text='—')                # Shows dashes when no face is detected

    def _render_vocal_metrics(self, vd):
        if not vd: return
        self._vocal_labels['pitch'].configure(text=f"{vd['pitch_stability'] * 100:.0f}%")   # Pitch stability percentage
        self._vocal_labels['energy'].configure(text=f"{vd['energy_level'] * 100:.0f}%")     # Voice energy percentage
        self._vocal_labels['tremor'].configure(text=f"{vd['tremor_index'] * 100:.0f}%")     # Tremor index percentage
        hz = vd.get('dominant_hz', 0)
        self._vocal_labels['hz'].configure(text=f"{hz:.0f} Hz" if hz else '—')             # Dominant pitch frequency in Hz
        speaking = vd.get('is_speaking', False)
        self._vocal_labels['speaking'].configure(                                            # Shows "Yes" in green or "No" in grey
            text='Yes' if speaking else 'No',
            fg=GREEN if speaking else T2
        )

    def _render_waveform(self, buf, vd):
        c = self.wave_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 2 or H < 2: return
        c.delete('all')
        c.create_rectangle(0, 0, W, H, fill='#0d0d1c', outline='')       # Dark background
        color   = BLUE if (vd and vd.get('is_speaking')) else BORDER      # Blue when speaking, grey when silent
        step    = max(1, len(buf) // W)                                    # How many audio samples map to one pixel column
        points  = []
        for i in range(W):
            sample = buf[min(i * step, len(buf) - 1)]                     # Gets the audio sample for this pixel column
            y      = H / 2 + sample * H * 0.42                           # Maps the sample amplitude to a y pixel position
            points.extend([i, y])
        if len(points) >= 4:
            c.create_line(points, fill=color, width=1.5, smooth=True)     # Draws the waveform as a smooth polyline

    def _render_spectrum(self, buf):
        c = self.spec_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 2 or H < 2: return
        c.delete('all')
        c.create_rectangle(0, 0, W, H, fill='#0d0d1c', outline='')       # Dark background
        window  = np.hanning(len(buf))                                    # Hanning window reduces spectral leakage at the edges of the buffer
        fft     = np.abs(np.fft.rfft(buf * window))                       # Real FFT of the windowed signal; gives magnitude spectrum
        fft     = fft[:len(fft) // 4]                                     # Keeps only the lower quarter (speech-range frequencies)
        fft     = fft / (fft.max() + 1e-10)                               # Normalises so the tallest bar always reaches near the top
        bars    = min(W // 4, 64)                                          # Number of bars; never more than 64 or the canvas allows
        bw      = W / bars                                                 # Width of each bar in pixels
        step    = max(1, len(fft) // bars)                                 # Number of FFT bins averaged into each bar
        for i in range(bars):
            val  = float(np.mean(fft[i * step: i * step + step]))        # Averages the FFT bins that map to this bar
            bh   = val * H                                                 # Converts the 0–1 magnitude to a pixel height
            hue  = int(185 + val * 55)                                    # Shifts from teal (185°) toward cyan/blue (240°) with intensity
            sat  = 80                                                      # Fixed saturation percentage for vibrant colours
            lit  = int(40 + val * 20)                                     # Darker bars when quiet, brighter when loud
            r, g, b = self._hsl_to_rgb(hue, sat / 100, lit / 100)        # Converts HSL to an RGB tuple for Tkinter
            color = f'#{r:02x}{g:02x}{b:02x}'                            # Formats as a hex colour string
            c.create_rectangle(i * bw, H - bh, (i + 1) * bw - 1, H,
                                fill=color, outline='')                   # Draws one vertical bar from the bottom

    @staticmethod
    def _hsl_to_rgb(h, s, l):
        h /= 360                                                           # Normalises hue to 0–1
        if s == 0:
            v = int(l * 255)
            return v, v, v                                                 # Grey when saturation is zero
        q = l * (1 + s) if l < 0.5 else l + s - l * s                    # Chroma helper value
        p = 2 * l - q                                                      # Second chroma helper
        def f(t):                                                          # Converts a hue component to an 8-bit channel value
            t = t % 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        return int(f(h + 1/3) * 255), int(f(h) * 255), int(f(h - 1/3) * 255)  # Returns R, G, B as 0–255 integers

    def _render_dots(self, fd, vd):
        for key, color in [
            ('face',  GREEN if (fd and fd.get('detected')) else T3),              # Green when face is detected, grey otherwise
            ('gaze',  GREEN if (fd and fd.get('detected')) else T3),              # Green when gaze is being tracked, grey otherwise
            ('voice', GREEN if (vd and vd.get('is_speaking')) else                # Green when speaking, blue when mic is active but silent
                      (BLUE if vd else T3)),
        ]:
            dot = self._status_dots[key]
            dot.delete('dot')
            dot.create_oval(1, 1, 9, 9, fill=color, outline='', tags='dot')      # Redraws the dot with the updated colour

    def _render_chart(self):
        h = self._history
        if len(h['total']) < 2: return                                     # Needs at least two points to draw a line
        x = list(range(len(h['total'])))                                   # Simple integer x-axis: each tick is one UI update
        for key, line in self._chart_lines.items():
            line.set_data(x, h[key])                                       # Pushes new data into the line artist
        self.ax.set_xlim(0, max(len(h['total']) - 1, 1))                  # Expands the x-axis to fit the new data
        try:
            self._fig_canvas.draw_idle()                                   # Queues a non-blocking redraw of the matplotlib figure
        except Exception:
            pass                                                           # Suppresses any drawing errors during window resize

    # ── Shutdown ────────────────────────────────────────────────────────────────

    def run(self):
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)             # Intercepts the close button so we can clean up threads
        self.root.mainloop()                                               # Enters the Tkinter event loop (blocks until the window is closed)

    def _on_close(self):
        self._running = False                                              # Signals background threads to stop their loops
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()                                             # Releases the webcam so other applications can use it
        if hasattr(self, '_audio_stream'):
            self._audio_stream.stop()                                      # Stops the microphone input stream
            self._audio_stream.close()                                     # Frees the audio device
        self.root.destroy()                                                # Closes the Tkinter window and exits mainloop


if __name__ == '__main__':
    app = TrustDashboard()   # Creates the dashboard instance (builds the window, starts camera and audio)
    app.run()                # Opens the window and enters the event loop — this line blocks until the window is closed
