"""
face_analyzer.py — Hybrid MediaPipe (real-time) + OpenFace (post-hoc) face analysis
──────────────────────────────────────────────────────────────────────────────────────
During a live session MediaPipe handles everything synchronously on every frame:
  • 478 face landmark points
  • Eye Aspect Ratio (EAR) → blink detection and blink rate
  • Iris position → gaze deviation
  • Head rotation matrix → pose deviation
  • Iris radius / inter-ocular distance → pupil size proxy
  • 52 blendshape scores → real-time emotion scoring and approximate Action Units

After a session ends, OpenFace is run on the saved .mp4 recording (post-hoc):
  • Accurate FACS Action Unit intensities (AU_r) and presence flags (AU_c)
  • Precise emotion scores from AU combinations
  • Accurate Duchenne smile detection
  These results are written into the Excel export by main.py.

Why this split?
  MediaPipe runs in ~10 ms per frame and keeps the display at 30 fps.
  OpenFace takes ~600 ms per frame — too slow for real-time — but processes
  a full video in a single efficient pass, making it ideal for post-session analysis.
  See FACE_ANALYSIS_ARCHITECTURE.md for a full explanation.
"""

import os                      # File existence checks and temp-file deletion
import time                    # Blink-rate timing window
import tempfile                # Temp files and directories for OpenFace
import subprocess              # Launches the OpenFace FeatureExtraction binary
import shutil                  # Deletes the temp output directory after reading
import csv                     # Parses the CSV that OpenFace writes
import urllib.request          # Downloads the MediaPipe model file on first run
import numpy as np             # Vector and geometry maths
import cv2                     # Colour-space conversion and temp JPEG writing

import mediapipe as mp                                                   # MediaPipe runtime
from mediapipe.tasks import python as mp_python                          # BaseOptions
from mediapipe.tasks.python import vision as mp_vision                   # FaceLandmarker task
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode  # Task config

# ── File paths ─────────────────────────────────────────────────────────────────
OPENFACE_BIN = os.path.expanduser("~/OpenFace/build/bin/FeatureExtraction")   # Compiled OpenFace binary
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "face_landmarker.task")  # Cached alongside this file

# ── MediaPipe eye landmark indices (478-point mesh) ───────────────────────────
# "h" = horizontal span (left corner → right corner)
# "v1", "v2" = two vertical spans (upper lid → lower lid at two positions)
L_EYE_MP = {"h": (33, 133), "v1": (159, 145), "v2": (158, 153)}   # Left eye
R_EYE_MP = {"h": (362, 263), "v1": (386, 374), "v2": (385, 380)}  # Right eye

# ── Blendshape → emotion weights ───────────────────────────────────────────────
# MediaPipe outputs 52 blendshape scores (0–1) on every frame.
# Each score represents a specific facial deformation (e.g. mouthSmileLeft).
# These weights define how to combine blendshapes into a 0–1 emotion score.
# Emotion definitions follow the Ekman AU classification table:
#   Anger:    4+5+7+23   Contempt: R12A+R14A   Disgust:  9+15+16
#   Fear:     1+2+4+5+7+20+26      Happiness: 6+12
#   Sadness:  1+4+15               Surprise:  1+2+5B+26
BLENDSHAPE_EMOTION_WEIGHTS = {
    "happy": [
        ("mouthSmileLeft",   0.25), ("mouthSmileRight",  0.25),  # Mirrors AU12 (lip corner puller)
        ("cheekSquintLeft",  0.25), ("cheekSquintRight", 0.25),  # Mirrors AU06 (cheek raiser)
    ],
    "sad": [
        ("browInnerUp",      0.35),                              # Mirrors AU01 (inner brow raise)
        ("browDownLeft",     0.15), ("browDownRight",    0.15),  # Mirrors AU04 (brow lowerer)
        ("mouthFrownLeft",   0.175), ("mouthFrownRight", 0.175), # Mirrors AU15 (lip corner depressor)
    ],
    "angry": [
        ("browDownLeft",     0.25), ("browDownRight",    0.25),  # Mirrors AU04 (brow lowerer)
        ("eyeWideLeft",      0.10), ("eyeWideRight",     0.10),  # Mirrors AU05 (upper lid raiser)
        ("eyeSquintLeft",    0.10), ("eyeSquintRight",   0.10),  # Mirrors AU07 (lid tightener)
        # AU23 (lip tightener) has no direct blendshape — mouthPressLeft/Right is the closest
        ("mouthPressLeft",   0.05), ("mouthPressRight",  0.05),
    ],
    "surprised": [
        ("browOuterUpLeft",  0.125), ("browOuterUpRight", 0.125), # Mirrors AU02 (outer brow raise)
        ("browInnerUp",      0.25),                               # Mirrors AU01 (inner brow raise)
        ("eyeWideLeft",      0.125), ("eyeWideRight",    0.125),  # Mirrors AU05B (upper lid raiser)
        ("jawOpen",          0.25),                               # Mirrors AU26 (jaw drop)
    ],
    "fearful": [
        ("browInnerUp",      0.12),                              # Mirrors AU01
        ("browOuterUpLeft",  0.06), ("browOuterUpRight", 0.06),  # Mirrors AU02
        ("browDownLeft",     0.08), ("browDownRight",    0.08),  # Mirrors AU04 (brows raised AND pulled together)
        ("eyeWideLeft",      0.08), ("eyeWideRight",     0.08),  # Mirrors AU05
        ("eyeSquintLeft",    0.06), ("eyeSquintRight",   0.06),  # Mirrors AU07
        ("mouthStretchLeft", 0.08), ("mouthStretchRight", 0.08), # Mirrors AU20 (lip stretcher)
        ("jawOpen",          0.16),                              # Mirrors AU26
    ],
    "disgusted": [
        ("noseSneerLeft",    0.30), ("noseSneerRight",   0.30),  # Mirrors AU09 (nose wrinkler)
        ("mouthFrownLeft",   0.175), ("mouthFrownRight", 0.175), # Mirrors AU15 (lip corner depressor)
        # AU16 (lower lip depressor) not available; mouthLowerDown is the closest proxy
        ("mouthLowerDownLeft", 0.025), ("mouthLowerDownRight", 0.025),
    ],
    # Contempt is computed via asymmetry in _blendshapes_to_emotions(), not from this table.
    # The table is kept here for documentation only — it is not used in the weighted sum.
    # Formula: max(0, mouthSmileRight − mouthSmileLeft − threshold) + mouthDimpleRight bonus.
    # A bilateral smile gives ~equal left/right → difference ≈ 0 → contempt ≈ 0.
    # A unilateral smirk gives high right, low left → large difference → contempt rises.
    "contempt": [],
}

# ── Blendshape → approximate Action Unit mapping ───────────────────────────────
# Provides approximate AU intensities from blendshapes during the live session.
# These are used by trust_engine for real-time scoring of AU04, AU07, AU20 etc.
# Post-hoc OpenFace analysis replaces these with accurate values in the Excel export.
# Format: AU_code → list of (blendshape_name, weight) pairs
BLENDSHAPE_AU_MAP = {
    "AU01": [("browInnerUp",      1.00)],                               # Inner brow raise
    "AU02": [("browOuterUpLeft",  0.50), ("browOuterUpRight",  0.50)],  # Outer brow raise
    "AU04": [("browDownLeft",     0.50), ("browDownRight",     0.50)],  # Brow furrow
    "AU05": [("eyeWideLeft",      0.50), ("eyeWideRight",      0.50)],  # Upper lid raise
    "AU06": [("cheekSquintLeft",  0.50), ("cheekSquintRight",  0.50)],  # Cheek raiser
    "AU07": [("eyeSquintLeft",    0.50), ("eyeSquintRight",    0.50)],  # Lid tightener
    "AU09": [("noseSneerLeft",    0.50), ("noseSneerRight",    0.50)],  # Nose wrinkler
    "AU12": [("mouthSmileLeft",   0.50), ("mouthSmileRight",   0.50)],  # Lip corner puller
    "AU14": [("mouthDimpleLeft",  0.50), ("mouthDimpleRight",  0.50)],  # Dimpler
    "AU15": [("mouthFrownLeft",   0.50), ("mouthFrownRight",   0.50)],  # Lip corner depressor
    "AU20": [("mouthStretchLeft", 0.50), ("mouthStretchRight", 0.50)],  # Lip stretcher
    "AU25": [("mouthClose",       1.00)],                               # Lips part (inverted)
    "AU26": [("jawOpen",          1.00)],                               # Jaw drop
    "AU45": [("eyeBlinkLeft",     0.50), ("eyeBlinkRight",     0.50)],  # Blink / eye closure
}

# ── OpenFace AU → emotion weights (used only in post-hoc analysis) ─────────────
# Follows the Ekman AU classification table exactly:
#   Anger:    4+5+7+23     Contempt: R12A+R14A   Disgust:  9+15+16
#   Fear:     1+2+4+5+7+20+26       Happiness: 6+12
#   Sadness:  1+4+15                Surprise:  1+2+5B+26
#
# Scoring formula: score = Σ( AU_c × AU_r/5 × weight )
#   AU_c = presence flag (0 or 1) — gates out noise on neutral faces
#   AU_r = intensity (0–5, divided by 5 to give 0–1)
#
# Notes on edge cases:
#   AU16 (lower lip depressor) is not tracked by OpenFace.
#     → Substituted with AU10 (upper lip raiser), which co-occurs in disgust.
#   R12A / R14A (right-side unilateral AUs) are not separated by OpenFace.
#     → Approximated with bilateral AU12 + AU14; asymmetry is not captured.
#   AU5B in surprise is the wide-eyed variant of AU05 — mapped directly to AU05.
AU_EMOTION_WEIGHTS = {
    "happy":    [("AU06", 0.50), ("AU12", 0.50)],                           # 6+12: equal weight
    "sad":      [("AU01", 0.35), ("AU04", 0.30), ("AU15", 0.35)],           # 1+4+15
    "angry":    [("AU04", 0.25), ("AU05", 0.25), ("AU07", 0.25), ("AU23", 0.25)],  # 4+5+7+23: equal
    "surprised":[("AU01", 0.25), ("AU02", 0.25), ("AU05", 0.25), ("AU26", 0.25)],  # 1+2+5B+26: equal
    "fearful":  [("AU01", 0.12), ("AU02", 0.12), ("AU04", 0.16),            # 1+2+4+5+7+20+26
                 ("AU05", 0.16), ("AU07", 0.12), ("AU20", 0.16), ("AU26", 0.16)],
    "disgusted":[("AU09", 0.40), ("AU15", 0.35), ("AU10", 0.25)],           # 9+15+16(≈AU10)
    "contempt": [("AU12", 0.50), ("AU14", 0.50)],                           # R12A+R14A (bilateral approx.)
}


class FaceAnalyzer:
    def __init__(self):
        # ── MediaPipe setup ───────────────────────────────────────────────────
        print("Loading MediaPipe FaceLandmarker…")

        # Download the ~3 MB model file on first run if it is not already cached.
        if not os.path.exists(MODEL_PATH):
            print("  Downloading face_landmarker.task…")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,                         # Process one image at a time (called manually each frame)
            num_faces=1,                                            # Track one face at a time
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=True,                           # Enable 52 blendshape scores for real-time emotion analysis
            output_facial_transformation_matrixes=True,             # Enable 3D rotation matrix for head-pose estimation
        )
        self.mp_detector = mp_vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe ready.")

        # ── OpenFace setup (optional — used for post-hoc analysis only) ───────
        # If the OpenFace binary is not found we skip the warmup and set a flag.
        # The app still runs fully; post-hoc AU analysis is simply skipped.
        self.openface_available = os.path.exists(OPENFACE_BIN)
        if self.openface_available:
            print(f"  OpenFace binary: {OPENFACE_BIN}")
            self._warmup_openface()   # Pre-load models so the first post-hoc run is fast
            print("OpenFace ready (post-hoc mode).")
        else:
            print("  OpenFace not found — post-hoc AU analysis disabled.")

        # ── Blink tracking state ──────────────────────────────────────────────
        self.blink_count        = 0
        self.blink_window_start = time.time()
        self.blink_rate         = 15.0      # blinks/min; 15 is a normal resting baseline
        self.is_blinking        = False
        self.EAR_THRESH         = 0.21      # EAR below this value = eye is considered closed

        # ── Gaze smoothing (30-frame rolling average) ─────────────────────────
        # Iris position jumps slightly frame-to-frame due to measurement noise.
        # Averaging the last 30 readings gives a stable gaze estimate.
        self.gaze_hist: list[float] = []

        self.last_result: dict = {"detected": False}

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one camera frame (~10 ms) and return a complete measurement dict.

        MediaPipe tracks the face, computes geometric measurements, and extracts
        blendshape scores which are immediately mapped to emotion intensities and
        approximate Action Unit values. Everything updates at ~30 fps.

        Returns a dict with:
          detected, eye_ar, blink_rate, gaze_deviation, pupil_norm,
          expressions (from blendshapes), aus (approximate, from blendshapes),
          duchenne (approximate), dominant, box_norm, eye_norm
        """
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_result = {"detected": False}
            return self.last_result

        h, w = frame_bgr.shape[:2]

        # MediaPipe requires RGB colour order; cameras give BGR.
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.mp_detector.detect(mp_image)
        except Exception as e:
            print(f"[face] MediaPipe error: {e}", flush=True)
            self.last_result = {"detected": False}
            return self.last_result

        if not result.face_landmarks:
            self.last_result = {"detected": False}
            return self.last_result

        try:
            lms = result.face_landmarks[0]   # List of 478 NormalizedLandmark objects

            # ── Eye Aspect Ratio (EAR) ────────────────────────────────────────
            # EAR = average vertical eye height ÷ horizontal eye width.
            # Fully open eye ≈ 0.27–0.35. Closed eye (mid-blink) ≈ 0.

            def pt(idx):
                # Convert normalised (0–1) landmark coordinates to pixel coordinates.
                return np.array([lms[idx].x * w, lms[idx].y * h])

            def ear(e):
                horizontal = np.linalg.norm(pt(e["h"][0]) - pt(e["h"][1]))   # Eye width in pixels
                if horizontal < 1e-6:
                    return 0.3
                # Average of two vertical measurements for robustness
                v = (np.linalg.norm(pt(e["v1"][0]) - pt(e["v1"][1])) +
                     np.linalg.norm(pt(e["v2"][0]) - pt(e["v2"][1]))) / 2
                return v / horizontal

            l_ear  = ear(L_EYE_MP)
            r_ear  = ear(R_EYE_MP)
            eye_ar = (l_ear + r_ear) / 2
            self._track_blink(eye_ar)   # Update blink count and blink rate

            # ── Iris gaze deviation ───────────────────────────────────────────
            # Measures how far the iris centre has drifted from the midpoint
            # of the eye's horizontal span. 0 = looking straight ahead.

            def iris_dev(iris_idx, eye_h):
                iris         = pt(iris_idx)
                left_corner  = pt(eye_h[0])
                right_corner = pt(eye_h[1])
                eye_w = np.linalg.norm(right_corner - left_corner)
                if eye_w < 1e-6:
                    return 0.0
                centre = (left_corner + right_corner) / 2
                return float(np.linalg.norm(iris - centre) / eye_w)

            gaze_dev = (iris_dev(468, (33, 133)) + iris_dev(473, (362, 263))) / 2
            self.gaze_hist.append(gaze_dev)
            if len(self.gaze_hist) > 30:
                self.gaze_hist.pop(0)
            avg_gaze = float(sum(self.gaze_hist) / len(self.gaze_hist))

            # ── Head pose from 3D transformation matrix ───────────────────────
            # Extracts yaw (left-right) and pitch (up-down) rotation angles.
            # Combined into a 0–1 deviation where 1.0 = 40 degrees off-centre.
            try:
                mat      = result.facial_transformation_matrixes[0]
                yaw      = abs(float(np.degrees(np.arctan2(mat[1][0], mat[0][0]))))
                pitch    = abs(float(np.degrees(np.arctan2(
                                    -mat[2][0], np.sqrt(mat[2][1]**2 + mat[2][2]**2)))))
                pose_dev = min(1.0, (yaw + pitch * 0.5) / 40.0)
            except Exception:
                pose_dev = avg_gaze   # Fall back to iris deviation if matrix is missing

            # ── Iris/pupil size estimate ──────────────────────────────────────
            # iris radius / inter-ocular distance = size independent of camera distance

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

            # ── Blendshape-based emotions and AUs ─────────────────────────────
            # MediaPipe outputs 52 blendshape scores every frame.
            # We map them to emotion intensities and approximate AU values.
            # These replace the OpenFace background-thread results from the old design.
            if result.face_blendshapes:
                # Build a name → score lookup from the blendshape list.
                bs_dict = {bs.category_name: bs.score
                           for bs in result.face_blendshapes[0]}
                expressions = self._blendshapes_to_emotions(bs_dict)
                aus_approx  = self._blendshapes_to_aus(bs_dict)
                # Approximate Duchenne smile: both cheek raiser and lip corner puller active.
                # This is a rough equivalent to the OpenFace AU06×AU12 check.
                duchenne = float(
                    (bs_dict.get("cheekSquintLeft",  0) +
                     bs_dict.get("cheekSquintRight", 0) +
                     bs_dict.get("mouthSmileLeft",   0) +
                     bs_dict.get("mouthSmileRight",  0)) / 4.0
                )
                dominant = max(expressions, key=expressions.get)
            else:
                # Blendshapes unavailable for this frame — return neutral defaults.
                expressions = {"happy": 0.0, "sad": 0.0, "angry": 0.0,
                               "surprised": 0.0, "fearful": 0.0,
                               "disgusted": 0.0, "neutral": 1.0}
                aus_approx  = {}
                duchenne    = 0.0
                dominant    = "neutral"

            # ── Bounding box and eye outlines ─────────────────────────────────
            xs   = [lm.x for lm in lms]
            ys   = [lm.y for lm in lms]
            bx   = min(xs);  by  = min(ys)
            bw_  = max(xs) - bx; bh_ = max(ys) - by

            l_pts = [[lms[i].x, lms[i].y] for i in [33, 246, 161, 160, 159, 158,
                                                       157, 173, 133, 155, 154, 153,
                                                       145, 144, 163,   7]]   # 16-point left eye outline
            r_pts = [[lms[i].x, lms[i].y] for i in [362, 398, 384, 385, 386, 387,
                                                       388, 466, 263, 249, 390, 373,
                                                       374, 380, 381, 382]]   # 16-point right eye outline

            self.last_result = {
                "detected":       True,
                # Tracking from MediaPipe landmarks (updates every frame at ~30 fps)
                "eye_ar":         float(eye_ar),
                "l_ear":          float(l_ear),
                "r_ear":          float(r_ear),
                "blink_rate":     float(self.blink_rate),
                "gaze_deviation": float(pose_dev),
                "pupil_norm":     pupil_norm,
                "box_norm":       [bx, by, bw_, bh_],
                "eye_norm":       {"l": l_pts, "r": r_pts},
                # Full 478-point landmark list (normalised 0–1) used to draw the
                # face mesh overlay on recorded video frames.
                "landmarks_norm": [[lm.x, lm.y] for lm in lms],
                # Emotions from blendshapes (updates every frame at ~30 fps)
                "expressions":    expressions,
                "au_emotions":    expressions,   # Alias kept for trust_engine compatibility
                "aus":            aus_approx,    # Approximate AUs; replaced by accurate values in post-hoc Excel
                "duchenne":       duchenne,
                "dominant":       dominant,
            }

        except Exception as e:
            print(f"[face] analysis error: {e}", flush=True)
            self.last_result = {"detected": False}

        return self.last_result

    # ── Blendshape mapping helpers ─────────────────────────────────────────────

    def _blendshapes_to_emotions(self, bs: dict) -> dict:
        """
        Convert a blendshape name→score dictionary into a 0–1 score for each emotion.
        Uses the BLENDSHAPE_EMOTION_WEIGHTS table for all emotions except contempt,
        which is handled separately via asymmetry detection.
        """
        expressions = {}

        for emotion, components in BLENDSHAPE_EMOTION_WEIGHTS.items():
            if emotion == "contempt":
                # Contempt cannot be detected by summing absolute blendshape values because
                # the same blendshapes (mouthSmileRight, mouthDimpleRight) also activate
                # during a genuine bilateral smile, causing false positives.
                #
                # Instead we measure LEFT-RIGHT ASYMMETRY of the lip corner:
                #   r − l > 0.15 means the right side is meaningfully higher than the left.
                #   A genuine smile raises both sides equally → difference ≈ 0 → contempt ≈ 0.
                #   A unilateral smirk raises only the right → large difference → contempt rises.
                #
                # A 0.15 dead-zone filters out minor natural asymmetry in a normal smile.
                r_smile  = bs.get("mouthSmileRight",  0.0)
                l_smile  = bs.get("mouthSmileLeft",   0.0)
                r_dimple = bs.get("mouthDimpleRight", 0.0)
                asymmetry = max(0.0, r_smile - l_smile - 0.15)   # positive only when right >> left
                expressions["contempt"] = min(1.0, asymmetry * 0.70 + r_dimple * 0.30)
            else:
                # All other emotions: weighted sum of their blendshapes, capped at 1.0.
                score = sum(bs.get(name, 0.0) * wt for name, wt in components)
                expressions[emotion] = min(1.0, score)

        # Neutral is whatever is left after all other emotions are accounted for.
        expressions["neutral"] = max(0.0, 1.0 - sum(expressions.values()))
        return expressions

    def _blendshapes_to_aus(self, bs: dict) -> dict:
        """
        Map blendshape scores to approximate Action Unit intensities (0–1 scale).
        These are anatomically motivated but less precise than OpenFace AU_r values.
        They give trust_engine something to work with in real time; the post-hoc
        OpenFace pass replaces them with accurate values in the Excel export.
        """
        aus = {}
        for au, components in BLENDSHAPE_AU_MAP.items():
            aus[au] = min(1.0, sum(bs.get(name, 0.0) * wt for name, wt in components))
        return aus

    # ── Post-hoc OpenFace analysis ─────────────────────────────────────────────

    @classmethod
    def analyze_video(cls, video_path: str) -> list[dict]:
        """
        Run OpenFace on a session recording and return a list of per-frame AU dicts.

        This is called automatically by main.py after a session ends, in a background
        thread. It does not affect the live dashboard display.

        Each returned dict has keys:
            frame_idx, timestamp_s, success,
            AU01..AU45 (intensity 0–1), AU01_c..AU45_c (presence 0/1),
            expressions (emotion scores), duchenne

        Returns an empty list if OpenFace is unavailable or fails.
        """
        if not os.path.exists(OPENFACE_BIN):
            print("[post-hoc] OpenFace binary not found — skipping AU analysis.")
            return []

        out_dir = tempfile.mkdtemp(prefix="of_video_")
        try:
            cmd = [
                OPENFACE_BIN,
                "-f",       video_path,   # Process the full video in one pass
                "-out_dir", out_dir,
                "-q",                     # Suppress progress output
                "-aus",                   # Output AU intensities and presence flags
            ]
            print(f"[post-hoc] Running OpenFace on {video_path} …", flush=True)
            try:
                # Allow up to 30 minutes for a long session recording.
                subprocess.run(cmd, capture_output=True, timeout=1800)
            except subprocess.TimeoutExpired:
                print("[post-hoc] OpenFace timed out.", flush=True)
                return []

            # Find the CSV that OpenFace wrote.
            csv_path = None
            for fname in os.listdir(out_dir):
                if fname.endswith(".csv"):
                    csv_path = os.path.join(out_dir, fname)
                    break
            if csv_path is None:
                print("[post-hoc] OpenFace wrote no CSV.", flush=True)
                return []

            rows = []
            AU_NAMES = ["AU01","AU02","AU04","AU05","AU06","AU07",
                        "AU09","AU10","AU12","AU14","AU15","AU17",
                        "AU20","AU23","AU25","AU26","AU45"]

            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for raw in reader:
                    # Strip whitespace from all column names (OpenFace pads them).
                    row = {k.strip(): v.strip() for k, v in raw.items()}

                    def f(key, default=0.0):
                        try: return float(row.get(key, default))
                        except (ValueError, TypeError): return default

                    # Success flag: 1 if OpenFace found a face in this frame, 0 if not.
                    success = int(f("success"))

                    # AU intensities normalised from the 0–5 OpenFace scale to 0–1.
                    aus_r = {au: f(f"{au}_r") / 5.0 for au in AU_NAMES}

                    # AU presence flags (0 or 1) — the noise gate.
                    aus_c = {au: int(f(f"{au}_c")) for au in AU_NAMES}

                    # Compute emotion scores using the same AU weighting as before.
                    expressions = {}
                    for emotion, components in AU_EMOTION_WEIGHTS.items():
                        score = sum(aus_c.get(au, 0) * aus_r.get(au, 0.0) * wt
                                    for au, wt in components)
                        expressions[emotion] = min(1.0, score)
                    expressions["neutral"] = max(0.0, 1.0 - sum(expressions.values()))

                    # Accurate Duchenne smile: both AU06 and AU12 present and active.
                    duchenne = float(
                        (aus_c.get("AU06", 0) * aus_r.get("AU06", 0.0) +
                         aus_c.get("AU12", 0) * aus_r.get("AU12", 0.0)) / 2.0
                    )

                    rows.append({
                        "frame_idx":   int(f("frame")),
                        "timestamp_s": round(f("timestamp"), 3),   # Seconds from video start
                        "success":     success,
                        "aus":         aus_r,                       # Intensity 0–1 per AU
                        "aus_c":       aus_c,                       # Presence flag per AU
                        "expressions": expressions,
                        "duchenne":    duchenne,
                    })

            print(f"[post-hoc] OpenFace parsed {len(rows)} frames.", flush=True)
            return rows

        except Exception as e:
            print(f"[post-hoc] Error during OpenFace video analysis: {e}", flush=True)
            return []
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)   # Always clean up the temp directory

    # ── OpenFace utilities (used by analyze_video) ─────────────────────────────

    def _warmup_openface(self):
        """
        Sends a blank image through OpenFace to pre-load its models into memory.
        Called once at startup so the first post-hoc analysis doesn't have extra delay.
        """
        fd, jpg_path = tempfile.mkstemp(suffix=".jpg", prefix="of_warm_")
        os.close(fd)
        out_dir = tempfile.mkdtemp(prefix="of_warm_")
        try:
            blank = np.ones((360, 640, 3), dtype=np.uint8) * 128   # Plain grey — no face, but models still load
            cv2.imwrite(jpg_path, blank)
            cmd = [OPENFACE_BIN, "-f", jpg_path, "-out_dir", out_dir, "-q", "-aus"]
            subprocess.run(cmd, capture_output=True, timeout=15.0)
        except Exception:
            pass   # Warmup failure is non-critical
        finally:
            try: os.unlink(jpg_path)
            except OSError: pass
            shutil.rmtree(out_dir, ignore_errors=True)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _track_blink(self, eye_ar: float):
        """
        Detects blink events from EAR and recalculates blink_rate every 10 seconds.

        A complete blink = EAR drops below threshold (eye closes)
        then rises back above it (eye reopens).
        """
        if eye_ar < self.EAR_THRESH and not self.is_blinking:
            self.is_blinking = True       # Eye just closed — start of a blink

        elif eye_ar >= self.EAR_THRESH and self.is_blinking:
            self.is_blinking = False      # Eye just reopened — one complete blink
            self.blink_count += 1

        elapsed = time.time() - self.blink_window_start
        if elapsed >= 10:                 # Recalculate blinks per minute every 10 seconds
            self.blink_rate         = (self.blink_count / elapsed) * 60
            self.blink_count        = 0
            self.blink_window_start = time.time()
