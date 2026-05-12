"""
face_analyzer.py — Hybrid MediaPipe + OpenFace face analysis
─────────────────────────────────────────────────────────────
MediaPipe runs synchronously on every frame (~10 ms) and provides:
  • Face bounding box and eye landmark overlays
  • Eye Aspect Ratio (EAR) and blink detection
  • Iris-based gaze deviation
  • Head-pose estimate

OpenFace runs asynchronously in a background thread (~600 ms) and provides:
  • FACS Action Unit intensities (AU_r) and binary presence flags (AU_c)
  • Presence-gated emotion scores (happy, sad, angry, surprised, fearful, disgusted)
  • Duchenne smile score

The two results are merged: every call to analyze() returns real-time tracking
data from MediaPipe annotated with the most recent emotion snapshot from OpenFace.
This means the bounding box and eye outlines update at 30 fps while emotion labels
update at roughly 1–2 fps — without any subprocess lag blocking the display.
"""

import os                      # Path checks and temp file deletion
import time                    # Blink-rate timing window
import tempfile                # Temp JPEG and output directory for OpenFace
import subprocess              # Spawns the FeatureExtraction binary
import shutil                  # Cleans up the temp output directory
import csv                     # Parses the CSV that OpenFace writes
import threading               # Background thread that drives OpenFace calls
import urllib.request          # Downloads the MediaPipe model if missing
import numpy as np             # Vector maths for EAR and landmark geometry
import cv2                     # Writes temp JPEGs and converts colour spaces

import mediapipe as mp                                                   # MediaPipe runtime
from mediapipe.tasks import python as mp_python                          # BaseOptions
from mediapipe.tasks.python import vision as mp_vision                   # FaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode  # Task config

# ── Paths ──────────────────────────────────────────────────────────────────────
OPENFACE_BIN = os.path.expanduser("~/OpenFace/build/bin/FeatureExtraction")   # Compiled OpenFace binary
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "face_landmarker.task")  # Cached alongside this file

# ── MediaPipe eye landmark indices (478-point mesh) ───────────────────────────
# Upper/lower lid pairs used for Eye Aspect Ratio
L_EYE_MP = {"h": (33, 133), "v1": (159, 145), "v2": (158, 153)}   # Left eye horizontal span + two vertical spans
R_EYE_MP = {"h": (362, 263), "v1": (386, 374), "v2": (385, 380)}  # Right eye same layout

# ── OpenFace AU → emotion weights ─────────────────────────────────────────────
# Keys are short AU names ("AU06", not "AU06_r").
# Scoring formula: score = Σ( AU_c × AU_r/5 × weight )
#   AU_c is 0 or 1 (SVM presence classifier) — gates out AU_r noise on neutral faces
#   AU_r is 0–5 (regression intensity) — provides magnitude when AU is genuinely active
AU_EMOTION_WEIGHTS = {
    "happy":     [("AU06", 0.40), ("AU12", 0.60)],                              # Cheek raiser + lip corner puller
    "sad":       [("AU01", 0.30), ("AU04", 0.25), ("AU15", 0.30), ("AU17", 0.15)],  # Inner brow raise, brow lower, lip corner depress, chin raise
    "angry":     [("AU04", 0.25), ("AU05", 0.15), ("AU07", 0.35), ("AU23", 0.25)],  # Brow lower, upper lid raise, lid tighten, lip tighten
    "surprised": [("AU01", 0.25), ("AU02", 0.25), ("AU05", 0.20), ("AU26", 0.30)],  # Both brow raises, upper lid raise, jaw drop
    "fearful":   [("AU01", 0.20), ("AU02", 0.15), ("AU04", 0.15),                   # Brows raised and pulled together,
                  ("AU05", 0.15), ("AU07", 0.10), ("AU20", 0.15), ("AU26", 0.10)],  #   lids wide, mouth stretched
    "disgusted": [("AU09", 0.35), ("AU15", 0.30), ("AU25", 0.35)],              # Nose wrinkle, lip corner depress, lips part
}

# Neutral emotion startup defaults (used until OpenFace returns its first result)
_NEUTRAL_EMOTIONS = {
    "expressions": {"happy": 0.0, "sad": 0.0, "angry": 0.0,
                    "surprised": 0.0, "fearful": 0.0, "disgusted": 0.0, "neutral": 1.0},
    "aus":         {},
    "duchenne":    0.0,
    "dominant":    "neutral",
    "au_emotions": {"happy": 0.0, "sad": 0.0, "angry": 0.0,
                    "surprised": 0.0, "fearful": 0.0, "disgusted": 0.0, "neutral": 1.0},
}


class FaceAnalyzer:
    def __init__(self):
        # ── MediaPipe setup ───────────────────────────────────────────────────
        print("Loading MediaPipe FaceLandmarker…")
        if not os.path.exists(MODEL_PATH):                          # Download the ~3 MB model on first run
            print("  Downloading face_landmarker.task…")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,                         # Single-image mode (called per frame)
            num_faces=1,                                            # Track one face at a time
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=False,                          # Not needed — OpenFace handles emotions
            output_facial_transformation_matrixes=True,             # Used for head-pose estimate
        )
        self.mp_detector = mp_vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe ready.")

        # ── OpenFace setup ────────────────────────────────────────────────────
        if not os.path.exists(OPENFACE_BIN):
            raise FileNotFoundError(
                f"OpenFace binary not found at {OPENFACE_BIN}\n"
                "Build OpenFace: https://github.com/TadasBaltrusaitis/OpenFace"
            )
        print(f"  OpenFace binary: {OPENFACE_BIN}")
        self._warmup_openface()                                     # Pre-load OpenFace models to avoid first-call latency
        print("OpenFace ready.")

        # ── Blink tracking state ──────────────────────────────────────────────
        self.blink_count        = 0
        self.blink_window_start = time.time()
        self.blink_rate         = 15.0      # blinks/min; 15 is a normal resting baseline
        self.is_blinking        = False
        self.EAR_THRESH         = 0.21      # EAR below this → eye considered closed

        # ── Gaze smoothing (30-frame rolling average) ─────────────────────────
        self.gaze_hist: list[float] = []

        # ── Emotion cache shared between analyze() and the OpenFace thread ───
        # analyze() always returns real-time MediaPipe tracking data merged with
        # this cache, so callers always get a complete dict even before OpenFace
        # has finished its first analysis.
        self._emotion_cache: dict = dict(_NEUTRAL_EMOTIONS)   # Starts as neutral
        self._of_lock        = threading.Lock()               # Guards _emotion_cache and _of_pending
        self._of_pending     = None                           # Latest frame queued for OpenFace analysis
        self._of_running     = True                           # Set to False to stop the background thread
        self._of_thread = threading.Thread(target=self._openface_loop, daemon=True)
        self._of_thread.start()                               # Start the background OpenFace thread

        self.last_result: dict = {"detected": False}

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        """
        Fast path (~10 ms): MediaPipe tracks the face in real-time.
        Also queues the frame for OpenFace so emotions update asynchronously.
        Returns a merged dict: current tracking + most-recent emotion snapshot.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_result = {"detected": False}
            return self.last_result

        h, w = frame_bgr.shape[:2]
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)           # MediaPipe requires RGB input
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.mp_detector.detect(mp_image)
        except Exception as e:
            print(f"[face] MediaPipe error: {e}", flush=True)
            self.last_result = {"detected": False}
            return self.last_result

        if not result.face_landmarks:                                    # No face in frame
            self.last_result = {"detected": False}
            return self.last_result

        try:
            lms = result.face_landmarks[0]                               # 478 NormalizedLandmark objects

            # Queue this frame for OpenFace (non-blocking; old pending frame is replaced)
            with self._of_lock:
                self._of_pending = frame_bgr.copy()                      # OpenFace thread picks this up next cycle

            # ── EAR from MediaPipe landmarks ──────────────────────────────
            def pt(idx):
                return np.array([lms[idx].x * w, lms[idx].y * h])       # Converts normalised → pixel coords

            def ear(e):
                horizontal = np.linalg.norm(pt(e["h"][0]) - pt(e["h"][1]))
                if horizontal < 1e-6:
                    return 0.3
                v = (np.linalg.norm(pt(e["v1"][0]) - pt(e["v1"][1])) +
                     np.linalg.norm(pt(e["v2"][0]) - pt(e["v2"][1]))) / 2
                return v / horizontal

            l_ear  = ear(L_EYE_MP)
            r_ear  = ear(R_EYE_MP)
            eye_ar = (l_ear + r_ear) / 2
            self._track_blink(eye_ar)                                    # Updates blink_count and blink_rate

            # ── Iris gaze deviation ───────────────────────────────────────
            def iris_dev(iris_idx, eye_h):
                iris         = pt(iris_idx)
                left_corner  = pt(eye_h[0])
                right_corner = pt(eye_h[1])
                eye_w = np.linalg.norm(right_corner - left_corner)
                if eye_w < 1e-6:
                    return 0.0
                centre = (left_corner + right_corner) / 2
                return float(np.linalg.norm(iris - centre) / eye_w)

            gaze_dev = (iris_dev(468, (33, 133)) + iris_dev(473, (362, 263))) / 2   # Average of left/right iris offset
            self.gaze_hist.append(gaze_dev)
            if len(self.gaze_hist) > 30:
                self.gaze_hist.pop(0)
            avg_gaze = float(sum(self.gaze_hist) / len(self.gaze_hist))

            # ── Head pose from transformation matrix ──────────────────────
            try:
                mat   = result.facial_transformation_matrixes[0]
                yaw   = abs(float(np.degrees(np.arctan2(mat[1][0], mat[0][0]))))
                pitch = abs(float(np.degrees(np.arctan2(
                                -mat[2][0], np.sqrt(mat[2][1]**2 + mat[2][2]**2)))))
                pose_dev = min(1.0, (yaw + pitch * 0.5) / 40.0)
            except Exception:
                pose_dev = avg_gaze                                      # Fall back to iris deviation if matrix missing

            # ── Iris/pupil diameter estimate ───────────────────────────────
            def iris_radius(center_idx, edge_idxs):
                c = pt(center_idx)
                return float(np.mean([np.linalg.norm(pt(e) - c) for e in edge_idxs]))

            try:
                l_iris_r   = iris_radius(468, [469, 470, 471, 472])
                r_iris_r   = iris_radius(473, [474, 475, 476, 477])
                avg_iris   = (l_iris_r + r_iris_r) / 2.0
                inter_ocu  = float(np.linalg.norm(pt(33) - pt(263)))
                pupil_norm = avg_iris / (inter_ocu + 1e-9)
            except Exception:
                pupil_norm = None

            # ── Bounding box and eye outlines (normalised) ────────────────
            xs   = [lm.x for lm in lms]
            ys   = [lm.y for lm in lms]
            bx   = min(xs);  by  = min(ys)
            bw_  = max(xs) - bx; bh_ = max(ys) - by

            l_pts = [[lms[i].x, lms[i].y] for i in [33, 246, 161, 160, 159, 158,
                                                       157, 173, 133, 155, 154, 153,
                                                       145, 144, 163,   7]]          # 16-point left eye outline
            r_pts = [[lms[i].x, lms[i].y] for i in [362, 398, 384, 385, 386, 387,
                                                       388, 466, 263, 249, 390, 373,
                                                       374, 380, 381, 382]]          # 16-point right eye outline

            # ── Merge with latest OpenFace emotion snapshot ───────────────
            with self._of_lock:
                emotions = dict(self._emotion_cache)                     # Snapshot; never blocks for long

            self.last_result = {
                "detected":       True,
                # Tracking from MediaPipe (real-time)
                "eye_ar":         float(eye_ar),
                "l_ear":          float(l_ear),
                "r_ear":          float(r_ear),
                "blink_rate":     float(self.blink_rate),
                "gaze_deviation": float(pose_dev),
                "pupil_norm":     pupil_norm,    # iris radius / inter-ocular dist
                "box_norm":       [bx, by, bw_, bh_],
                "eye_norm":       {"l": l_pts, "r": r_pts},
                # Emotions from OpenFace (updated ~1 fps in background thread)
                "expressions":    emotions["expressions"],
                "au_emotions":    emotions["au_emotions"],
                "aus":            emotions["aus"],
                "duchenne":       emotions["duchenne"],
                "dominant":       emotions["dominant"],
            }

        except Exception as e:
            print(f"[face] analysis error: {e}", flush=True)
            self.last_result = {"detected": False}

        return self.last_result

    # ── OpenFace background thread ─────────────────────────────────────────────

    def _openface_loop(self):
        """
        Runs continuously in a daemon thread.
        Picks up the latest queued frame, runs FeatureExtraction (~600 ms),
        parses the CSV, and writes AU-based emotions into _emotion_cache.
        Because this is fire-and-forget, the main thread never blocks on it.
        """
        while self._of_running:
            with self._of_lock:
                frame = self._of_pending
                self._of_pending = None                                  # Consume the pending frame

            if frame is not None:
                result = self._openface_analyze(frame)
                if result is not None:
                    with self._of_lock:
                        self._emotion_cache = result                     # Atomically replace the emotion snapshot
            else:
                time.sleep(0.05)                                         # Nothing queued — poll every 50 ms

    def _openface_analyze(self, frame_bgr: np.ndarray) -> dict | None:
        """Runs OpenFace on a single frame and returns a parsed emotion dict."""
        fd, jpg_path = tempfile.mkstemp(suffix=".jpg", prefix="of_frame_")
        os.close(fd)
        out_dir = tempfile.mkdtemp(prefix="of_out_")
        try:
            cv2.imwrite(jpg_path, frame_bgr)
            row = self._run_openface(jpg_path, out_dir)
            if row is None or int(float(row.get("success", 0))) != 1:
                return None
            return self._parse_emotions(row)
        except Exception as e:
            print(f"[face/of] error: {e}", flush=True)
            return None
        finally:
            try:
                os.unlink(jpg_path)
            except OSError:
                pass
            shutil.rmtree(out_dir, ignore_errors=True)

    def _warmup_openface(self):
        """Runs a blank image through FeatureExtraction to pre-load all models."""
        fd, jpg_path = tempfile.mkstemp(suffix=".jpg", prefix="of_warm_")
        os.close(fd)
        out_dir = tempfile.mkdtemp(prefix="of_warm_")
        try:
            blank = np.ones((360, 640, 3), dtype=np.uint8) * 128        # Plain grey; no face will be found, but models load
            cv2.imwrite(jpg_path, blank)
            self._run_openface(jpg_path, out_dir)
        except Exception:
            pass
        finally:
            try:
                os.unlink(jpg_path)
            except OSError:
                pass
            shutil.rmtree(out_dir, ignore_errors=True)

    def _run_openface(self, jpg_path: str, out_dir: str) -> dict | None:
        """Calls FeatureExtraction and returns the first CSV row as a stripped dict."""
        cmd = [
            OPENFACE_BIN,
            "-f",       jpg_path,
            "-out_dir", out_dir,
            "-q",                    # Suppress progress output
            "-aus",                  # Output AU intensities (_r) and presence flags (_c)
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=8.0)        # 8-second hard timeout
        except subprocess.TimeoutExpired:
            print("[face/of] timed out", flush=True)
            return None

        csv_path = None
        for fname in os.listdir(out_dir):
            if fname.endswith(".csv"):
                csv_path = os.path.join(out_dir, fname)
                break
        if csv_path is None:
            return None

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                return {k.strip(): v.strip() for k, v in row.items()}   # Strip whitespace from OpenFace column names
        return None

    def _parse_emotions(self, row: dict) -> dict:
        """
        Converts a raw OpenFace CSV row into the emotion dict stored in _emotion_cache.
        Uses AU_c presence flags to gate AU_r intensities so that low-level noise in
        neutral faces does not bleed into emotion scores.
        """
        def f(key, default=0.0):
            try:
                return float(row.get(key, default))
            except (ValueError, TypeError):
                return default

        # AU intensities (0–5 normalised to 0–1)
        aus_r = {au: f(f"{au}_r") / 5.0
                 for au in ["AU01","AU02","AU04","AU05","AU06","AU07",
                             "AU09","AU10","AU12","AU14","AU15","AU17",
                             "AU20","AU23","AU25","AU26","AU45"]}

        # AU presence flags (0 or 1) — the critical noise gate
        aus_c = {au: int(f(f"{au}_c"))
                 for au in ["AU01","AU02","AU04","AU05","AU06","AU07",
                             "AU09","AU10","AU12","AU14","AU15","AU17",
                             "AU20","AU23","AU25","AU26","AU28"]}

        # Presence-gated emotion scores: AU_c=0 zeroes out the entire term
        expressions: dict[str, float] = {}
        for emotion, components in AU_EMOTION_WEIGHTS.items():
            score = sum(aus_c.get(au, 0) * aus_r.get(au, 0.0) * wt     # only scores when AU is genuinely present
                        for au, wt in components)
            expressions[emotion] = min(1.0, score)
        expressions["neutral"] = max(0.0, 1.0 - sum(expressions.values()))
        dominant = max(expressions, key=expressions.get)

        # Duchenne smile: AU6 AND AU12 both present and active
        duchenne = float(
            (aus_c.get("AU06", 0) * aus_r.get("AU06", 0.0) +
             aus_c.get("AU12", 0) * aus_r.get("AU12", 0.0)) / 2.0
        )

        return {
            "expressions": expressions,
            "au_emotions": expressions,                                  # trust_engine reads this key too
            "aus":         {au: v for au, v in aus_r.items()},          # Short keys for trust_engine
            "duchenne":    duchenne,
            "dominant":    dominant,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _track_blink(self, eye_ar: float):
        """Detects blink events and recalculates blink_rate every 10 s."""
        if eye_ar < self.EAR_THRESH and not self.is_blinking:
            self.is_blinking = True                                      # Eye just closed
        elif eye_ar >= self.EAR_THRESH and self.is_blinking:
            self.is_blinking = False                                     # Eye reopened → one complete blink
            self.blink_count += 1
        elapsed = time.time() - self.blink_window_start
        if elapsed >= 10:
            self.blink_rate        = (self.blink_count / elapsed) * 60  # Convert to blinks per minute
            self.blink_count       = 0
            self.blink_window_start = time.time()
