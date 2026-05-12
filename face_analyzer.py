import os                               # Used to check whether the model file already exists before downloading it
import time                             # Used to measure elapsed time for the blink-rate calculation window
import urllib.request                   # Downloads the MediaPipe face landmarker model file from Google's servers

import cv2                              # Converts BGR frames (OpenCV default) to RGB (required by MediaPipe)
import numpy as np                      # Used for vector maths when computing eye aspect ratio and gaze deviation
import mediapipe as mp                  # Core MediaPipe library; provides the Image class and ImageFormat enum
from mediapipe.tasks import python as mp_python          # Provides BaseOptions for specifying the model file path
from mediapipe.tasks.python import vision as mp_vision  # Provides FaceLandmarkerOptions and FaceLandmarker task class

MODEL_URL = (                           # URL to download the pre-trained face landmarker model from Google's model repository
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"     # Local filename where the model will be saved (in the project root directory)

# Defines how MediaPipe blendshape coefficients map to the 6 emotional states the dashboard displays.
# Each entry is a list of (blendshape_name, weight) tuples; higher weights give that blendshape more influence.
EMOTION_WEIGHTS = {
    "happy":     [("mouthSmileLeft", 0.5), ("mouthSmileRight", 0.5),    # Corner lip raises indicate a smile
                  ("cheekSquintLeft", 0.25), ("cheekSquintRight", 0.25)],  # Cheek squinting is a marker of genuine smiling
    "angry":     [("browDownLeft", 0.55), ("browDownRight", 0.55),       # Brow lowering is the primary anger indicator
                  ("noseSneerLeft", 0.3), ("noseSneerRight", 0.3)],       # Nose wrinkling co-occurs with anger
    "surprised": [("browOuterUpLeft", 0.3), ("browOuterUpRight", 0.3),   # Outer brow raise signals surprise
                  ("eyeWideLeft", 0.3), ("eyeWideRight", 0.3),            # Wide eyes are a key surprise marker
                  ("jawOpen", 0.2)],                                       # Open jaw often accompanies surprise
    "sad":       [("mouthFrownLeft", 0.5), ("mouthFrownRight", 0.5),     # Downturned lip corners indicate sadness
                  ("browInnerUp", 0.2)],                                   # Inner brow raise adds a worried/sad quality
    "fearful":   [("browInnerUp", 0.4),                                   # Inner brow raise is the strongest fear signal
                  ("eyeWideLeft", 0.25), ("eyeWideRight", 0.25),          # Wide eyes appear during fear
                  ("jawOpen", 0.1)],                                       # Slight jaw drop can accompany fear
    "disgusted": [("noseSneerLeft", 0.55), ("noseSneerRight", 0.55),      # Nose wrinkling is the hallmark of disgust
                  ("mouthShrugUpper", 0.2)],                               # Upper lip raise adds to the disgust expression
}

# MediaPipe 478-point mesh indices for the six eye-outline points used in the Eye Aspect Ratio (EAR) calculation.
# EAR = (vertical_distances) / (2 × horizontal_distance) — low EAR means the eye is closing (blink).
L_EYE  = (362, 385, 387, 263, 373, 380)  # Left eye: outer-corner, upper-1, upper-2, inner-corner, lower-2, lower-1
R_EYE  = (33,  160, 158, 133, 153, 144)  # Right eye: outer-corner, upper-1, upper-2, inner-corner, lower-2, lower-1
L_IRIS = 468                              # Index of the left iris centre landmark (available when refine_landmarks=True)
R_IRIS = 473                              # Index of the right iris centre landmark


class FaceAnalyzer:
    def __init__(self):
        self._ensure_model()            # Downloads the model file if it isn't already present on disk
        options = mp_vision.FaceLandmarkerOptions(       # Configures the FaceLandmarker task
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),  # Points to the local .task model file
            output_face_blendshapes=True,                # Enables the 52 blendshape coefficients needed for emotion mapping
            output_facial_transformation_matrixes=False, # Disables 3-D head pose matrices (not needed here)
            num_faces=1,                                 # Only track a single face to keep processing fast
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)  # Builds the ready-to-use detector from the options

        self.blink_count        = 0             # Running count of blinks detected within the current 10-second window
        self.blink_window_start = time.time()   # Timestamp marking the start of the current blink-rate measurement window
        self.blink_rate         = 0.0           # Most recently calculated blink rate in blinks-per-minute
        self.is_blinking        = False         # Flag to track whether the eye is currently in the closed phase of a blink
        self.EAR_THRESH         = 0.21          # EAR values below this threshold are classified as a blink (eye closed)
        self.gaze_hist: list[float] = []        # Rolling buffer of the last 30 gaze-deviation values for smoothing
        self.last_result: dict = {"detected": False}  # Stores the most recent analysis result so the WebSocket handler can read it

    # ── Public ─────────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:    # Guards against empty or corrupted frames from the browser
            self.last_result = {"detected": False}       # Returns a "no face" result rather than crashing
            return self.last_result

        h, w = frame_bgr.shape[:2]                      # Extracts the pixel height and width of the frame
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Converts from OpenCV's BGR order to the RGB order MediaPipe expects
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # Wraps the NumPy array in a MediaPipe Image object
        res = self.detector.detect(mp_img)              # Runs face landmark detection; returns landmarks and blendshapes

        if not res.face_landmarks or not res.face_blendshapes:  # Checks whether a face was actually found in the frame
            self.last_result = {"detected": False}              # No face detected; return early with a "not detected" result
            return self.last_result

        lm = res.face_landmarks[0]                      # Gets the list of 478 landmark points for the first (and only) detected face
        bs = {b.category_name: b.score for b in res.face_blendshapes[0]}  # Converts the blendshape list to a name→score dict for easy lookup

        expressions = self._blendshapes_to_emotions(bs)  # Maps the 52 raw blendshape values to 7 normalised emotion scores
        dominant    = max(expressions, key=expressions.get)  # Finds the emotion with the highest score to display as the label

        def pt(i): return np.array([lm[i].x * w, lm[i].y * h])  # Helper: converts a normalised landmark to pixel coordinates

        l_ear  = self._ear(*[pt(i) for i in L_EYE])    # Computes Eye Aspect Ratio for the left eye using its 6 outline landmarks
        r_ear  = self._ear(*[pt(i) for i in R_EYE])    # Computes Eye Aspect Ratio for the right eye using its 6 outline landmarks
        eye_ar = (l_ear + r_ear) / 2                    # Averages both eyes for a more robust overall eye-openness measure
        self._track_blink(eye_ar)                       # Updates the blink counter and recalculates blink rate if the window has elapsed

        # Iris-based gaze deviation: measures how far the iris centre is from the eye centre, normalised by eye width.
        has_iris = len(lm) > L_IRIS                     # Checks that the refined 478-point mesh (with iris) was returned
        l_iris = pt(L_IRIS) if has_iris else pt(L_EYE[0])  # Uses the iris landmark if available, otherwise falls back to the eye corner
        r_iris = pt(R_IRIS) if has_iris else pt(R_EYE[0])  # Same fallback for the right iris
        l_dev = abs(l_iris[0] - (pt(L_EYE[0])[0] + pt(L_EYE[3])[0]) / 2) / max(abs(pt(L_EYE[3])[0] - pt(L_EYE[0])[0]), 1)  # Left iris offset ÷ eye width
        r_dev = abs(r_iris[0] - (pt(R_EYE[0])[0] + pt(R_EYE[3])[0]) / 2) / max(abs(pt(R_EYE[3])[0] - pt(R_EYE[0])[0]), 1)  # Right iris offset ÷ eye width
        gaze_dev = (l_dev + r_dev) / 2                  # Averages the two eyes for a single gaze-deviation score
        self.gaze_hist.append(gaze_dev)                 # Adds the current deviation to the rolling history buffer
        if len(self.gaze_hist) > 30:                    # Keeps the buffer at a maximum of 30 samples (≈2 seconds at 15 fps)
            self.gaze_hist.pop(0)                       # Removes the oldest sample when the buffer is full
        avg_gaze = sum(self.gaze_hist) / len(self.gaze_hist)  # Computes the smoothed average gaze deviation

        xs = [p.x for p in lm]; ys = [p.y for p in lm]  # Extracts all x and y coordinates from the 478 landmarks
        bx, by = min(xs), min(ys)                          # Finds the top-left corner of the face bounding box (normalised 0–1)
        bw, bh = max(xs) - bx, max(ys) - by               # Computes the width and height of the bounding box

        l_pts = [[lm[i].x, lm[i].y] for i in L_EYE]   # Packages the 6 left-eye landmark positions as normalised [x, y] pairs for the frontend
        r_pts = [[lm[i].x, lm[i].y] for i in R_EYE]   # Packages the 6 right-eye landmark positions the same way

        self.last_result = {                             # Builds the final result dict that the WebSocket handler will send to the browser
            "detected":       True,                      # Tells the frontend a face was found this frame
            "expressions":    expressions,               # Dict of 7 normalised emotion scores (sum ≈ 1.0)
            "dominant":       dominant,                  # Name of the highest-scoring emotion (e.g. "happy")
            "eye_ar":         float(eye_ar),             # Average Eye Aspect Ratio (0.0 = fully closed, ~0.3 = fully open)
            "l_ear":          float(l_ear),              # Left eye EAR (used for per-eye colour coding in the overlay)
            "r_ear":          float(r_ear),              # Right eye EAR (used for per-eye colour coding in the overlay)
            "blink_rate":     float(self.blink_rate),    # Blinks per minute calculated over the last 10-second window
            "gaze_deviation": float(avg_gaze),           # Smoothed iris offset ratio (0 = looking straight ahead, 1 = far to one side)
            "box_norm":       [bx, by, bw, bh],          # Face bounding box as normalised [x, y, width, height] values
            "eye_norm":       {"l": l_pts, "r": r_pts},  # Eye outline points as normalised coords so the frontend can draw them at any size
        }
        return self.last_result                          # Returns the result and also stores it in self.last_result for the WebSocket handler

    # ── Private ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_model():
        if not os.path.exists(MODEL_PATH):               # Skips the download if the model file is already on disk
            print("Downloading MediaPipe face landmarker model (~3 MB)…")  # Informs the user so they know why startup is slow the first time
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)  # Downloads the file and saves it to MODEL_PATH
            print("Model ready.")                        # Confirms the download finished successfully

    @staticmethod
    def _blendshapes_to_emotions(bs: dict) -> dict:
        scores = {}                                      # Will hold the raw (un-normalised) score for each emotion
        for emotion, components in EMOTION_WEIGHTS.items():  # Iterates over each of the 6 mapped emotions
            scores[emotion] = min(1.0, sum(bs.get(n, 0) * w for n, w in components))  # Weighted sum of relevant blendshapes, capped at 1.0
        raw_total = sum(scores.values()) or 1            # Sums all emotion scores; "or 1" prevents division by zero
        neutral   = max(0.0, 1.0 - raw_total)           # Neutral fills whatever emotional space the other 6 emotions don't occupy
        scores["neutral"] = neutral                      # Adds neutral as the 7th emotion
        total = sum(scores.values()) or 1                # Recalculates the total including neutral
        return {k: round(v / total, 4) for k, v in scores.items()}  # Normalises all scores so they sum to 1.0

    @staticmethod
    def _ear(p1, p2, p3, p4, p5, p6) -> float:
        def d(a, b): return float(np.linalg.norm(a - b))  # Computes the Euclidean distance between two 2-D points
        h = d(p1, p4)                                    # Horizontal distance between the two eye corner landmarks
        return (d(p2, p6) + d(p3, p5)) / (2 * h) if h > 0 else 0.0  # EAR formula: sum of vertical distances ÷ (2 × horizontal distance)

    def _track_blink(self, ear: float):
        if ear < self.EAR_THRESH and not self.is_blinking:    # Detects the start of a blink: EAR drops below threshold for the first time
            self.is_blinking = True                            # Marks that the eye has entered the closed phase
        elif ear >= self.EAR_THRESH and self.is_blinking:     # Detects the end of a blink: EAR rises back above threshold
            self.is_blinking = False                           # Resets the flag so the next closing will count as a new blink
            self.blink_count += 1                              # Increments the blink counter for the current measurement window
        elapsed = time.time() - self.blink_window_start       # Calculates how many seconds have passed in the current window
        if elapsed >= 10:                                      # Recalculates blink rate every 10 seconds for a stable estimate
            self.blink_rate = (self.blink_count / elapsed) * 60  # Converts blinks-per-second to blinks-per-minute
            self.blink_count = 0                               # Resets the counter for the next 10-second window
            self.blink_window_start = time.time()              # Resets the window start time
