import time
import urllib.request
import os
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode

# ── Blendshape → emotion mapping ──────────────────────────────────────────────
# MediaPipe returns 52 blendshape coefficients (0–1). We combine them into the
# same 7-emotion scores that trust_engine.py expects.
BLENDSHAPE_EMOTION = {
    "happy":     [("mouthSmileLeft", 0.5), ("mouthSmileRight", 0.5)],
    "sad":       [("mouthFrownLeft", 0.4), ("mouthFrownRight", 0.4), ("browInnerUp", 0.2)],
    "angry":     [("browDownLeft", 0.4), ("browDownRight", 0.4), ("noseSneerLeft", 0.1), ("noseSneerRight", 0.1)],
    "surprised": [("jawOpen", 0.4), ("eyeWideLeft", 0.3), ("eyeWideRight", 0.3)],
    "fearful":   [("browInnerUp", 0.4), ("jawOpen", 0.2), ("eyeWideLeft", 0.2), ("eyeWideRight", 0.2)],
    "disgusted": [("noseSneerLeft", 0.5), ("noseSneerRight", 0.5)],
    "neutral":   [("browDownLeft", 0.0)],   # computed as residual below
}

# Eye landmark indices in MediaPipe's 478-point mesh (same layout as dlib 68-pt for EAR)
# Upper/lower lid points for each eye for Eye Aspect Ratio calculation
L_EYE = {"h": (33, 133), "v1": (159, 145), "v2": (158, 153)}
R_EYE = {"h": (362, 263), "v1": (386, 374), "v2": (385, 380)}

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


class FaceAnalyzer:
    def __init__(self):
        print("Loading MediaPipe FaceLandmarker…")
        if not os.path.exists(MODEL_PATH):
            print("  Downloading face_landmarker.task (~3 MB)…")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)
        print("Models ready.")

        self.blink_count        = 0
        self.blink_window_start = time.time()
        self.blink_rate         = 15.0
        self.is_blinking        = False
        self.EAR_THRESH         = 0.21
        self.gaze_hist: list[float] = []
        self.last_result: dict  = {"detected": False}

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_result = {"detected": False}
            return self.last_result

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"[face] detection error: {e}", flush=True)
            self.last_result = {"detected": False}
            return self.last_result

        if not result.face_landmarks:
            self.last_result = {"detected": False}
            return self.last_result

        try:
            lms = result.face_landmarks[0]          # 478 NormalizedLandmark objects

            # ── Blendshapes → emotions ─────────────────────────────────────
            bs = {}
            if result.face_blendshapes:
                for cat in result.face_blendshapes[0]:
                    bs[cat.category_name] = float(cat.score)

            expressions = {}
            for emotion, components in BLENDSHAPE_EMOTION.items():
                if emotion == "neutral":
                    continue
                expressions[emotion] = min(1.0, sum(bs.get(k, 0) * w_ for k, w_ in components))

            # Neutral is whatever is left after all other emotions
            expressions["neutral"] = max(0.0, 1.0 - sum(expressions.values()))
            dominant = max(expressions, key=expressions.get)

            # ── Duchenne smile (genuine): cheek raise + lip corner pull ────
            duchenne = (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0) +
                        bs.get("mouthSmileLeft", 0)  + bs.get("mouthSmileRight", 0)) / 4

            # ── AU proxies from blendshapes ────────────────────────────────
            aus = {
                "AU04": (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2,
                "AU06": (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0)) / 2,
                "AU07": (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2,
                "AU12": (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2,
                "AU14": (bs.get("mouthDimpleLeft", 0) + bs.get("mouthDimpleRight", 0)) / 2,
                "AU20": (bs.get("mouthStretchLeft", 0) + bs.get("mouthStretchRight", 0)) / 2,
            }

            # ── Eye Aspect Ratio ───────────────────────────────────────────
            def pt(idx):
                return np.array([lms[idx].x * w, lms[idx].y * h])

            def ear(e):
                horizontal = np.linalg.norm(pt(e["h"][0]) - pt(e["h"][1]))
                if horizontal < 1e-6:
                    return 0.3
                v = (np.linalg.norm(pt(e["v1"][0]) - pt(e["v1"][1])) +
                     np.linalg.norm(pt(e["v2"][0]) - pt(e["v2"][1]))) / 2
                return v / horizontal

            l_ear  = ear(L_EYE)
            r_ear  = ear(R_EYE)
            eye_ar = (l_ear + r_ear) / 2
            self._track_blink(eye_ar)

            # ── Iris gaze deviation ────────────────────────────────────────
            # MediaPipe 478-point mesh includes iris landmarks: L 468-472, R 473-477
            def iris_deviation(iris_center_idx, eye_h):
                iris = pt(iris_center_idx)
                left_corner  = pt(eye_h[0])
                right_corner = pt(eye_h[1])
                eye_w = np.linalg.norm(right_corner - left_corner)
                if eye_w < 1e-6:
                    return 0.0
                center = (left_corner + right_corner) / 2
                return float(np.linalg.norm(iris - center) / eye_w)

            gaze_dev = (iris_deviation(468, (33, 133)) + iris_deviation(473, (362, 263))) / 2
            self.gaze_hist.append(gaze_dev)
            if len(self.gaze_hist) > 30:
                self.gaze_hist.pop(0)
            avg_gaze = sum(self.gaze_hist) / len(self.gaze_hist)

            # ── Head pose (yaw/pitch from transformation matrix) ───────────
            try:
                mat  = result.facial_transformation_matrixes[0]
                yaw  = abs(float(np.degrees(np.arctan2(mat[1][0], mat[0][0]))))
                pitch = abs(float(np.degrees(np.arctan2(-mat[2][0],
                             np.sqrt(mat[2][1]**2 + mat[2][2]**2)))))
                pose_dev = min(1.0, (yaw + pitch * 0.5) / 40.0)
            except Exception:
                pose_dev = avg_gaze

            # ── Iris/pupil diameter estimate ───────────────────────────────
            # Left iris:  center=468, edges=469-472
            # Right iris: center=473, edges=474-477
            # Normalise by inter-ocular distance so head distance cancels out.
            def iris_radius(center_idx, edge_idxs):
                c = pt(center_idx)
                return float(np.mean([np.linalg.norm(pt(e) - c) for e in edge_idxs]))

            try:
                l_iris_r  = iris_radius(468, [469, 470, 471, 472])
                r_iris_r  = iris_radius(473, [474, 475, 476, 477])
                avg_iris  = (l_iris_r + r_iris_r) / 2.0
                inter_ocu = float(np.linalg.norm(pt(33) - pt(263)))
                pupil_norm = avg_iris / (inter_ocu + 1e-9)
            except Exception:
                pupil_norm = None

            # ── Face bounding box from landmark extents ────────────────────
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            bx, by = min(xs), min(ys)
            bw_, bh_ = max(xs) - bx, max(ys) - by

            # Eye outline points (normalised) for the camera overlay
            l_pts = [[lms[i].x, lms[i].y] for i in [33, 246, 161, 160, 159, 158,
                                                       157, 173, 133, 155, 154, 153,
                                                       145, 144, 163,  7]]
            r_pts = [[lms[i].x, lms[i].y] for i in [362, 398, 384, 385, 386, 387,
                                                       388, 466, 263, 249, 390, 373,
                                                       374, 380, 381, 382]]

            self.last_result = {
                "detected":       True,
                "expressions":    expressions,
                "au_emotions":    expressions,   # same source; trust_engine uses both
                "aus":            aus,
                "duchenne":       float(duchenne),
                "dominant":       dominant,
                "eye_ar":         float(eye_ar),
                "l_ear":          float(l_ear),
                "r_ear":          float(r_ear),
                "blink_rate":     float(self.blink_rate),
                "gaze_deviation": float(pose_dev),
                "pupil_norm":     pupil_norm,    # iris radius / inter-ocular dist
                "box_norm":       [bx, by, bw_, bh_],
                "eye_norm":       {"l": l_pts, "r": r_pts},
            }

        except Exception as e:
            print(f"[face] analysis error: {e}", flush=True)
            self.last_result = {"detected": False}

        return self.last_result

    # ── Private ───────────────────────────────────────────────────────────────

    def _track_blink(self, ear: float):
        if ear < self.EAR_THRESH and not self.is_blinking:
            self.is_blinking = True
        elif ear >= self.EAR_THRESH and self.is_blinking:
            self.is_blinking = False
            self.blink_count += 1
        elapsed = time.time() - self.blink_window_start
        if elapsed >= 10:
            self.blink_rate = (self.blink_count / elapsed) * 60
            self.blink_count = 0
            self.blink_window_start = time.time()
