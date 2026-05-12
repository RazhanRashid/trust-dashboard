import time                             # Used to measure elapsed time for the blink-rate calculation window
import numpy as np                      # Used for vector maths when computing Eye Aspect Ratio and landmark coordinates
import cv2                              # Converts BGR frames to RGB before passing them to py-feat

from feat import Detector               # py-feat: Python Facial Expression Analysis Toolbox — OpenFace-style AU detection in pure Python

# ── Action Unit → emotion mapping (FACS-based) ────────────────────────────────
# Each emotion is scored by summing the relevant AUs weighted by their FACS importance.
# AU names follow the Facial Action Coding System (Ekman & Friesen, 1978).
AU_EMOTION_MAP = {
    "happy":     [("AU06", 0.5), ("AU12", 0.5)],                          # AU06=Cheek Raiser + AU12=Lip Corner Puller = Duchenne smile (genuine happiness)
    "angry":     [("AU04", 0.4), ("AU05", 0.2), ("AU07", 0.2),           # AU04=Brow Lowerer, AU05=Upper Lid Raiser, AU07=Lid Tightener
                  ("AU23", 0.1), ("AU24", 0.1)],                          # AU23=Lip Tightener, AU24=Lip Pressor (pressed lips)
    "surprised": [("AU01", 0.2), ("AU02", 0.2), ("AU05", 0.3),           # AU01=Inner Brow Raise, AU02=Outer Brow Raise, AU05=Upper Lid Raiser
                  ("AU26", 0.3)],                                          # AU26=Jaw Drop
    "sad":       [("AU01", 0.3), ("AU04", 0.3), ("AU15", 0.3),           # AU01=Inner Brow Raise, AU04=Brow Lowerer, AU15=Lip Corner Depressor
                  ("AU17", 0.1)],                                          # AU17=Chin Raiser
    "fearful":   [("AU01", 0.2), ("AU02", 0.1), ("AU04", 0.1),           # Inner brow raise + brow lowerer combination signals fear
                  ("AU05", 0.2), ("AU20", 0.2), ("AU26", 0.2)],          # AU20=Lip Stretcher, AU26=Jaw Drop
    "disgusted": [("AU09", 0.4), ("AU15", 0.2), ("AU16", 0.2),           # AU09=Nose Wrinkler (hallmark of disgust), AU15=Lip Corner Depressor
                  ("AU25", 0.1), ("AU26", 0.1)],                          # AU16=Lower Lip Depressor, AU25=Lips Part
}

# Eye landmark indices for the 68-point dlib/OpenFace scheme (same as original OpenFace)
L_EYE_IDX = slice(36, 42)   # Left eye:  landmark indices 36–41
R_EYE_IDX = slice(42, 48)   # Right eye: landmark indices 42–47


class FaceAnalyzer:
    def __init__(self):
        print("Loading OpenFace-style models via py-feat…")
        self.detector = Detector(
            face_model="retinaface",    # RetinaFace: accurate multi-scale face detector
            landmark_model="mobilenet", # MobileNet-based 68-point facial landmark detector
            au_model="xgb",             # XGBoost AU intensity predictor (fast, accurate)
            emotion_model="resmasknet", # ResNet + attention mask emotion classifier
            facepose_model="img2pose",  # Head pose estimation (pitch, roll, yaw)
        )
        print("Models ready.")

        self.blink_count        = 0             # Running blink count within the current 10-second measurement window
        self.blink_window_start = time.time()   # Timestamp for the start of the current measurement window
        self.blink_rate         = 0.0           # Smoothed blink rate in blinks per minute
        self.is_blinking        = False         # True while the eye is in the closed phase of a blink
        self.EAR_THRESH         = 0.21          # Eye Aspect Ratio below this value is classified as a blink
        self.gaze_hist: list[float] = []        # Rolling buffer of gaze deviation values for temporal smoothing
        self.last_result: dict  = {"detected": False}   # Most recent analysis result, read by the WebSocket handler or UI loop

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:    # Guards against empty or corrupted frames
            self.last_result = {"detected": False}
            return self.last_result

        h, w = frame_bgr.shape[:2]                      # Gets frame dimensions for normalising coordinates to [0,1]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # py-feat expects RGB; OpenCV gives BGR

        try:
            result = self.detector.detect_image(rgb)    # Runs the full OpenFace-style pipeline: detect → landmarks → AUs → emotions → pose
        except Exception as e:
            print(f"Detection error: {e}")
            self.last_result = {"detected": False}
            return self.last_result

        # Check whether py-feat found a face in this frame
        if result is None or len(result) == 0:
            self.last_result = {"detected": False}
            return self.last_result
        try:
            fb = result.facebox.iloc[0]                 # Bounding box of the first detected face
            if any(fb[c] != fb[c] for c in fb.index):  # NaN check — py-feat returns NaN columns when no face is found
                self.last_result = {"detected": False}
                return self.last_result
        except Exception:
            self.last_result = {"detected": False}
            return self.last_result

        try:
            # ── Emotions ──────────────────────────────────────────────────────
            em = result.emotions.iloc[0]                # Row of emotion probabilities for the first face
            expressions = {
                "happy":     max(0.0, float(em.get("happiness", 0))),   # py-feat uses "happiness" not "happy"
                "sad":       max(0.0, float(em.get("sadness",   0))),
                "angry":     max(0.0, float(em.get("anger",     0))),
                "fearful":   max(0.0, float(em.get("fear",      0))),
                "disgusted": max(0.0, float(em.get("disgust",   0))),
                "surprised": max(0.0, float(em.get("surprise",  0))),
                "neutral":   max(0.0, float(em.get("neutral",   0))),
            }
            dominant = max(expressions, key=expressions.get)            # Emotion with the highest probability

            # ── Action Units (OpenFace FACS output) ───────────────────────────
            au_row = result.aus.iloc[0]                                 # Row of AU intensity values (0–1 continuous)
            aus    = {col: max(0.0, float(au_row[col]))
                      for col in result.aus.columns}                    # Converts to a plain dict for easy lookup

            # Duchenne marker: AU06 (Cheek Raiser) + AU12 (Lip Corner Puller) = genuine smile
            duchenne = (aus.get("AU06", 0) + aus.get("AU12", 0)) / 2   # 0 = no smile, 1 = full genuine smile

            # AU-derived emotion scores (used alongside the deep-learning emotion model)
            au_emotions = self._aus_to_emotions(aus)                    # Maps FACS AUs → emotion scores via FACS rules

            # ── Landmarks → Eye Aspect Ratio ──────────────────────────────────
            x_cols = [f"x_{i}" for i in range(68)]                     # py-feat stores 68 x-coordinates as columns "x_0"…"x_67"
            y_cols = [f"y_{i}" for i in range(68)]                     # and 68 y-coordinates as columns "y_0"…"y_67"
            lm_row = result.landmarks.iloc[0]
            xs  = lm_row[x_cols].values.astype(float)                  # Extract x pixel coordinates
            ys  = lm_row[y_cols].values.astype(float)                  # Extract y pixel coordinates
            lms = np.stack([xs, ys], axis=1)                           # Shape (68, 2) — one (x, y) pair per landmark

            l_ear  = self._ear(lms[L_EYE_IDX])                        # Eye Aspect Ratio for the left eye
            r_ear  = self._ear(lms[R_EYE_IDX])                        # Eye Aspect Ratio for the right eye
            eye_ar = (l_ear + r_ear) / 2                               # Average both eyes for robustness
            self._track_blink(eye_ar)                                  # Updates blink counter and blink rate

            # Eye outline points normalised to [0,1] for the frontend overlay
            l_pts = [[lms[i,0]/w, lms[i,1]/h] for i in range(36, 42)]
            r_pts = [[lms[i,0]/w, lms[i,1]/h] for i in range(42, 48)]

            # ── Head pose as gaze proxy ────────────────────────────────────────
            # py-feat gives head yaw (left-right rotation) and pitch (up-down tilt).
            # Large yaw indicates the person is looking away from the camera.
            try:
                pose    = result.poses.iloc[0]
                yaw     = abs(float(pose.get("Yaw",   0)))             # Horizontal head rotation in degrees
                pitch   = abs(float(pose.get("Pitch", 0)))             # Vertical head tilt in degrees
                gaze_dev = min(1.0, (yaw + pitch * 0.5) / 40.0)       # Normalises to 0–1 (40° combined = max deviation)
            except Exception:
                gaze_dev = 0.0                                         # Falls back to zero if pose estimation failed
            self.gaze_hist.append(gaze_dev)
            if len(self.gaze_hist) > 30: self.gaze_hist.pop(0)
            avg_gaze = sum(self.gaze_hist) / len(self.gaze_hist)       # Smoothed average over the last 30 frames

            # ── Face bounding box (normalised) ────────────────────────────────
            bx = float(fb["FaceRectX"])      / w                       # Left edge normalised to [0,1]
            by = float(fb["FaceRectY"])      / h                       # Top edge normalised to [0,1]
            bw = float(fb["FaceRectWidth"])  / w                       # Width normalised to [0,1]
            bh = float(fb["FaceRectHeight"]) / h                       # Height normalised to [0,1]

            self.last_result = {
                "detected":       True,
                "expressions":    expressions,                          # Deep-learning emotion probabilities (7 classes)
                "au_emotions":    au_emotions,                          # FACS rule-based emotion scores from AUs
                "aus":            aus,                                  # Raw AU intensities (AU01, AU04, AU06, AU12, …)
                "duchenne":       float(duchenne),                      # Genuine smile indicator (AU06+AU12 combined)
                "dominant":       dominant,                             # Name of the highest-probability emotion
                "eye_ar":         float(eye_ar),                       # Average Eye Aspect Ratio (0=closed, ~0.3=open)
                "l_ear":          float(l_ear),                        # Left eye EAR
                "r_ear":          float(r_ear),                        # Right eye EAR
                "blink_rate":     float(self.blink_rate),              # Blinks per minute (10-second rolling window)
                "gaze_deviation": float(avg_gaze),                     # Smoothed head-pose-derived gaze deviation (0–1)
                "box_norm":       [bx, by, bw, bh],                    # Face bounding box as normalised [x, y, w, h]
                "eye_norm":       {"l": l_pts, "r": r_pts},            # Eye outline points in normalised coords for the overlay
            }

        except Exception as e:
            print(f"Analysis error: {e}")
            self.last_result = {"detected": False}

        return self.last_result

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _aus_to_emotions(aus: dict) -> dict:
        scores = {}
        for emotion, components in AU_EMOTION_MAP.items():
            scores[emotion] = min(1.0, sum(aus.get(au, 0) * w for au, w in components))  # Weighted AU sum, capped at 1.0
        total = sum(scores.values()) or 1                               # Avoids division by zero
        return {k: round(v / total, 4) for k, v in scores.items()}     # Normalises so all AU emotion scores sum to 1.0

    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        def d(a, b): return float(np.linalg.norm(a - b))               # Euclidean distance between two 2-D points
        h = d(eye_pts[0], eye_pts[3])                                   # Horizontal distance: outer corner to inner corner
        return (d(eye_pts[1], eye_pts[5]) + d(eye_pts[2], eye_pts[4])) / (2 * h) if h > 0 else 0.0  # EAR formula

    def _track_blink(self, ear: float):
        if ear < self.EAR_THRESH and not self.is_blinking:             # EAR drops below threshold → eye entering closed phase
            self.is_blinking = True
        elif ear >= self.EAR_THRESH and self.is_blinking:              # EAR rises back above threshold → blink completed
            self.is_blinking = False
            self.blink_count += 1                                      # Counts this as one complete blink
        elapsed = time.time() - self.blink_window_start
        if elapsed >= 10:                                              # Recalculates blink rate every 10 seconds
            self.blink_rate = (self.blink_count / elapsed) * 60       # Converts to blinks per minute
            self.blink_count = 0
            self.blink_window_start = time.time()
