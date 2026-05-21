"""
face_analyzer.py — Hybrid MediaPipe + OpenFace face analysis
─────────────────────────────────────────────────────────────
This file does two things at once using two separate face-analysis tools:

MediaPipe (fast, runs every frame, ~10 ms per frame):
  • Finds the face in the camera image and marks 478 landmark points on it
  • Measures eye openness (Eye Aspect Ratio) to detect blinks
  • Tracks iris position to estimate where the person is looking
  • Estimates head rotation (are they looking straight at the camera?)
  • Estimates pupil size (a proxy for mental workload)

OpenFace (slower, runs in the background, ~600 ms per frame):
  • Identifies which specific facial muscles are moving (Action Units / AUs)
  • From those muscle movements, scores each emotion (happy, sad, angry, etc.)
  • Detects whether a smile is genuine (Duchenne) or forced

Because MediaPipe is fast and OpenFace is slow, they run on different schedules:
  - The face bounding box, eye outlines, and blink rate update at ~30 fps (MediaPipe).
  - The emotion labels update at ~1–2 fps (OpenFace running in a background thread).
  - The result you get from analyze() always has both, merged together.
"""

import os                      # Used to check if files exist and delete temporary files
import time                    # Used to measure elapsed time for blink-rate calculation
import tempfile                # Used to create temporary image files for OpenFace to read
import subprocess              # Used to launch the external OpenFace program
import shutil                  # Used to delete the temporary folder OpenFace writes results into
import csv                     # Used to read the results table that OpenFace writes out
import threading               # Used to run OpenFace in the background without freezing the UI
import urllib.request          # Used to download the MediaPipe model file on first run
import numpy as np             # Used for all vector and geometry calculations
import cv2                     # Used to write temporary JPEG images and convert colour formats

import mediapipe as mp                                                   # The MediaPipe face-tracking library
from mediapipe.tasks import python as mp_python                          # Base configuration options
from mediapipe.tasks.python import vision as mp_vision                   # The face landmark detection task
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode  # Task settings classes

# ── File paths ─────────────────────────────────────────────────────────────────
OPENFACE_BIN = os.path.expanduser("~/OpenFace/build/bin/FeatureExtraction")   # Where the compiled OpenFace program lives
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "face_landmarker.task")  # Where the downloaded model is cached

# ── MediaPipe landmark index groups ───────────────────────────────────────────
# MediaPipe places 478 numbered dot-markers on the face.
# These dictionaries name the specific dot numbers we need for each eye.
# "h" = horizontal span (left corner to right corner of the eye)
# "v1" and "v2" = two vertical spans (top lid to bottom lid at two positions)
# Having two vertical measurements makes the Eye Aspect Ratio more reliable.
L_EYE_MP = {"h": (33, 133), "v1": (159, 145), "v2": (158, 153)}   # Left eye landmark indices
R_EYE_MP = {"h": (362, 263), "v1": (386, 374), "v2": (385, 380)}  # Right eye landmark indices

# ── How Action Units map to emotions ──────────────────────────────────────────
# OpenFace detects "Action Units" (AUs) — individual facial muscle contractions.
# For example, AU12 is the zygomatic muscle that pulls the lip corners up into a smile.
# This dictionary defines which muscles to combine, and how much weight each gets,
# to produce a score for each of six emotions.
#
# Each entry is: emotion name → list of (AU_code, weight) pairs
# Weight 0.60 means "AU12 accounts for 60% of the happy score."
#
# The formula also uses two numbers per AU from OpenFace:
#   AU_c = 0 or 1: is this muscle actually firing? (presence flag — cuts out noise on neutral faces)
#   AU_r = 0–5: how intensely is it firing? (divided by 5 to get a 0–1 range)
# Score = sum of (AU_c × AU_r/5 × weight) across all AUs for that emotion.
AU_EMOTION_WEIGHTS = {
    "happy":     [("AU06", 0.40), ("AU12", 0.60)],                              # Cheek raiser + lip corner puller
    "sad":       [("AU01", 0.30), ("AU04", 0.25), ("AU15", 0.30), ("AU17", 0.15)],  # Inner brow raise + brow lower + lip corner depress + chin raise
    "angry":     [("AU04", 0.25), ("AU05", 0.15), ("AU07", 0.35), ("AU23", 0.25)],  # Brow lower + upper lid raise + lid tighten + lip tighten
    "surprised": [("AU01", 0.25), ("AU02", 0.25), ("AU05", 0.20), ("AU26", 0.30)],  # Both brow raises + upper lid raise + jaw drop
    "fearful":   [("AU01", 0.20), ("AU02", 0.15), ("AU04", 0.15),                   # Brows raised and pulled together
                  ("AU05", 0.15), ("AU07", 0.10), ("AU20", 0.15), ("AU26", 0.10)],  #   + lids wide + mouth stretched
    "disgusted": [("AU09", 0.35), ("AU15", 0.30), ("AU25", 0.35)],              # Nose wrinkle + lip corner depress + lips part
}

# Default emotion values used until OpenFace produces its first result (takes ~600 ms).
# Starting as "neutral 1.0" means the dashboard shows a calm baseline while it warms up.
_NEUTRAL_EMOTIONS = {
    "expressions": {"happy": 0.0, "sad": 0.0, "angry": 0.0,
                    "surprised": 0.0, "fearful": 0.0, "disgusted": 0.0, "neutral": 1.0},
    "aus":         {},        # No Action Units detected yet
    "duchenne":    0.0,       # No genuine smile detected yet
    "dominant":    "neutral", # The strongest emotion right now
    "au_emotions": {"happy": 0.0, "sad": 0.0, "angry": 0.0,
                    "surprised": 0.0, "fearful": 0.0, "disgusted": 0.0, "neutral": 1.0},
}


class FaceAnalyzer:
    def __init__(self):
        # ── Set up MediaPipe ──────────────────────────────────────────────────
        print("Loading MediaPipe FaceLandmarker…")

        # Download the ~3 MB model file if it has never been downloaded before.
        if not os.path.exists(MODEL_PATH):
            print("  Downloading face_landmarker.task…")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        # Configure MediaPipe with the settings we need.
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,                         # Process one image at a time (we call it manually per frame)
            num_faces=1,                                            # Only track one face at a time
            min_face_detection_confidence=0.4,                     # Accept a detection if MediaPipe is at least 40% confident
            min_face_presence_confidence=0.4,                      # Same threshold for confirming the face is still there
            min_tracking_confidence=0.4,                           # Same for tracking the face frame-to-frame
            output_face_blendshapes=False,                         # We don't need MediaPipe blendshapes — OpenFace handles emotions
            output_facial_transformation_matrixes=True,            # We do need the 3D rotation matrix for head-pose estimation
        )
        self.mp_detector = mp_vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe ready.")

        # ── Set up OpenFace ───────────────────────────────────────────────────
        # OpenFace must be compiled and placed at OPENFACE_BIN before the app can run.
        if not os.path.exists(OPENFACE_BIN):
            raise FileNotFoundError(
                f"OpenFace binary not found at {OPENFACE_BIN}\n"
                "Build OpenFace: https://github.com/TadasBaltrusaitis/OpenFace"
            )
        print(f"  OpenFace binary: {OPENFACE_BIN}")
        self._warmup_openface()   # Run a blank image through OpenFace so its models are loaded into memory
        print("OpenFace ready.")

        # ── Blink tracking ────────────────────────────────────────────────────
        self.blink_count        = 0            # Number of complete blinks since the last rate calculation
        self.blink_window_start = time.time()  # When the current 10-second counting window started
        self.blink_rate         = 15.0         # Current estimate: blinks per minute (15 is a normal resting rate)
        self.is_blinking        = False        # True while the eye is currently closed mid-blink
        self.EAR_THRESH         = 0.21         # Eye Aspect Ratio below this value = eye is considered closed

        # ── Gaze smoothing ────────────────────────────────────────────────────
        # Iris position jumps slightly frame-to-frame due to measurement noise.
        # We keep the last 30 readings and average them to get a stable gaze estimate.
        self.gaze_hist: list[float] = []

        # ── Emotion cache shared between the main thread and the OpenFace thread ──
        # analyze() always returns the latest MediaPipe tracking data merged with
        # the most recent OpenFace emotion result, so callers always get a complete
        # dictionary — even before OpenFace finishes its first analysis.
        self._emotion_cache: dict = dict(_NEUTRAL_EMOTIONS)   # Start with neutral emotions
        self._of_lock        = threading.Lock()               # Lock that protects _emotion_cache and _of_pending
        self._of_pending     = None                           # The latest camera frame waiting to be processed by OpenFace
        self._of_running     = True                           # Set to False when the app closes, to stop the background thread
        self._of_thread = threading.Thread(target=self._openface_loop, daemon=True)
        self._of_thread.start()   # Launch the background OpenFace processing thread

        self.last_result: dict = {"detected": False}   # The result from the most recent analyze() call

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one camera frame and return a dictionary with all face measurements.

        This function is called roughly 30 times per second.
        MediaPipe analysis (~10 ms) happens here synchronously.
        OpenFace analysis (~600 ms) is handed off to a background thread.

        The returned dictionary always contains:
          - detected: True if a face was found, False if not
          - eye_ar: Eye Aspect Ratio (how open the eyes are)
          - blink_rate: blinks per minute
          - gaze_deviation: how far the person is looking away from the camera
          - pupil_norm: normalised iris size (proxy for mental workload)
          - expressions: emotion intensities from OpenFace (updated ~1–2 fps)
          - aus: Action Unit intensities from OpenFace
          - duchenne: genuine smile score
        """
        # If the frame is empty (camera not ready yet), return a "no face" result.
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_result = {"detected": False}
            return self.last_result

        h, w = frame_bgr.shape[:2]   # Height and width of the frame in pixels

        # MediaPipe expects RGB colour order; cameras give BGR, so we convert.
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.mp_detector.detect(mp_image)   # Run MediaPipe face detection
        except Exception as e:
            print(f"[face] MediaPipe error: {e}", flush=True)
            self.last_result = {"detected": False}
            return self.last_result

        # If MediaPipe found no face in this frame, return immediately.
        if not result.face_landmarks:
            self.last_result = {"detected": False}
            return self.last_result

        try:
            lms = result.face_landmarks[0]   # The list of 478 landmark points for the first (only) face

            # Queue this frame for OpenFace analysis.
            # The background thread will pick it up whenever it finishes its current frame.
            # If it hasn't finished yet, the old pending frame is simply replaced — we
            # only ever need the most recent frame for emotion analysis.
            with self._of_lock:
                self._of_pending = frame_bgr.copy()

            # ── Eye Aspect Ratio (EAR) ────────────────────────────────────────
            # The EAR formula measures how open each eye is:
            #   EAR = (average vertical eye height) / (horizontal eye width)
            # A fully open eye has EAR ~0.27–0.35. When the eye closes during a blink,
            # EAR drops toward 0. Below our threshold (0.21) we call the eye "closed".

            def pt(idx):
                # Convert a landmark's normalised (0–1) coordinates to actual pixel coordinates.
                return np.array([lms[idx].x * w, lms[idx].y * h])

            def ear(e):
                # Compute EAR for one eye using its landmark index groups.
                horizontal = np.linalg.norm(pt(e["h"][0]) - pt(e["h"][1]))   # Width of the eye in pixels
                if horizontal < 1e-6:   # Guard against division by zero if the eye is too tiny
                    return 0.3
                # Average of two vertical measurements (top lid to bottom lid at two horizontal positions)
                v = (np.linalg.norm(pt(e["v1"][0]) - pt(e["v1"][1])) +
                     np.linalg.norm(pt(e["v2"][0]) - pt(e["v2"][1]))) / 2
                return v / horizontal   # EAR = vertical / horizontal

            l_ear  = ear(L_EYE_MP)              # EAR for the left eye
            r_ear  = ear(R_EYE_MP)              # EAR for the right eye
            eye_ar = (l_ear + r_ear) / 2        # Average of both eyes for a single openness value
            self._track_blink(eye_ar)           # Update blink count and blink rate with this new EAR value

            # ── Iris gaze deviation ───────────────────────────────────────────
            # MediaPipe places a landmark at the centre of each iris (dot 468 = left, 473 = right).
            # We measure how far the iris centre is from the midpoint of the eye's horizontal span.
            # If the iris sits perfectly centred, deviation = 0. Looking to the side increases it.

            def iris_dev(iris_idx, eye_h):
                iris         = pt(iris_idx)            # Pixel position of the iris centre
                left_corner  = pt(eye_h[0])            # Pixel position of the left corner of the eye
                right_corner = pt(eye_h[1])            # Pixel position of the right corner of the eye
                eye_w = np.linalg.norm(right_corner - left_corner)   # Eye width in pixels
                if eye_w < 1e-6:
                    return 0.0
                centre = (left_corner + right_corner) / 2             # The midpoint of the eye
                return float(np.linalg.norm(iris - centre) / eye_w)   # Deviation normalised by eye width

            # Average the left and right iris deviations for a single gaze measure.
            gaze_dev = (iris_dev(468, (33, 133)) + iris_dev(473, (362, 263))) / 2

            # Add to the 30-frame history and use the average to smooth out jitter.
            self.gaze_hist.append(gaze_dev)
            if len(self.gaze_hist) > 30:
                self.gaze_hist.pop(0)
            avg_gaze = float(sum(self.gaze_hist) / len(self.gaze_hist))

            # ── Head pose estimation ──────────────────────────────────────────
            # MediaPipe outputs a 4×4 rotation/translation matrix describing how the head
            # is rotated in 3D space. We extract yaw (left-right) and pitch (up-down) angles.
            # Combined into a single 0–1 "pose deviation" value where 0 = looking straight ahead.
            try:
                mat   = result.facial_transformation_matrixes[0]   # The 4×4 transformation matrix

                # arctan2 extracts the rotation angle from the matrix columns.
                # The result is in radians; np.degrees converts to degrees for easier reading.
                yaw   = abs(float(np.degrees(np.arctan2(mat[1][0], mat[0][0]))))
                pitch = abs(float(np.degrees(np.arctan2(
                                -mat[2][0], np.sqrt(mat[2][1]**2 + mat[2][2]**2)))))

                # Combine yaw and half-pitch (up-down head tilt matters less than left-right turning).
                # Divide by 40 so that a 40-degree combined rotation maps to a deviation of 1.0.
                pose_dev = min(1.0, (yaw + pitch * 0.5) / 40.0)
            except Exception:
                pose_dev = avg_gaze   # Fall back to iris deviation if the matrix is missing

            # ── Iris/pupil size estimate ──────────────────────────────────────
            # MediaPipe places 5 points on each iris: one at the centre and four on the edge.
            # We measure the average distance from the centre to the four edge points —
            # this is the iris radius in pixels.
            # Dividing by the inter-ocular distance (gap between the two eye centres)
            # makes the measurement independent of how close the face is to the camera.

            def iris_radius(center_idx, edge_idxs):
                c = pt(center_idx)   # Iris centre point
                return float(np.mean([np.linalg.norm(pt(e) - c) for e in edge_idxs]))   # Mean distance to edge points

            try:
                l_iris_r   = iris_radius(468, [469, 470, 471, 472])   # Left iris radius in pixels
                r_iris_r   = iris_radius(473, [474, 475, 476, 477])   # Right iris radius in pixels
                avg_iris   = (l_iris_r + r_iris_r) / 2.0              # Average of both irises
                inter_ocu  = float(np.linalg.norm(pt(33) - pt(263)))  # Distance between left and right eye outer corners
                pupil_norm = avg_iris / (inter_ocu + 1e-9)            # Normalised iris size (1e-9 prevents division by zero)
            except Exception:
                pupil_norm = None   # If any landmark is missing, skip the pupil measurement this frame

            # ── Bounding box and eye outline points ───────────────────────────
            # Collect the x and y coordinates of all 478 landmarks to find the face bounding box.
            xs   = [lm.x for lm in lms]   # All normalised x coordinates
            ys   = [lm.y for lm in lms]   # All normalised y coordinates
            bx   = min(xs);  by  = min(ys)                # Top-left corner of the face bounding box
            bw_  = max(xs) - bx; bh_ = max(ys) - by      # Width and height of the bounding box

            # 16-point outlines of each eye (used to draw the glowing eye overlays in the UI).
            l_pts = [[lms[i].x, lms[i].y] for i in [33, 246, 161, 160, 159, 158,
                                                       157, 173, 133, 155, 154, 153,
                                                       145, 144, 163,   7]]
            r_pts = [[lms[i].x, lms[i].y] for i in [362, 398, 384, 385, 386, 387,
                                                       388, 466, 263, 249, 390, 373,
                                                       374, 380, 381, 382]]

            # ── Merge MediaPipe tracking with the latest OpenFace emotions ────
            # Read the emotion cache under the lock to get a consistent snapshot.
            # This is a very fast operation — the lock is held for microseconds.
            with self._of_lock:
                emotions = dict(self._emotion_cache)

            # Assemble the final result dictionary that the dashboard reads every frame.
            self.last_result = {
                "detected":       True,
                # From MediaPipe (updates at ~30 fps):
                "eye_ar":         float(eye_ar),           # Eye openness ratio
                "l_ear":          float(l_ear),            # Left eye EAR
                "r_ear":          float(r_ear),            # Right eye EAR
                "blink_rate":     float(self.blink_rate),  # Blinks per minute
                "gaze_deviation": float(pose_dev),         # Head/gaze angle away from camera (0 = straight, 1 = far away)
                "pupil_norm":     pupil_norm,              # Normalised iris size (used by workload engine)
                "box_norm":       [bx, by, bw_, bh_],     # Face bounding box in normalised (0–1) coordinates
                "eye_norm":       {"l": l_pts, "r": r_pts},  # Eye outline points in normalised coordinates
                # From OpenFace (updates at ~1–2 fps via background thread):
                "expressions":    emotions["expressions"], # Emotion intensities: happy, sad, angry, etc.
                "au_emotions":    emotions["au_emotions"], # Same as expressions (kept for backwards compatibility)
                "aus":            emotions["aus"],         # Individual Action Unit intensities
                "duchenne":       emotions["duchenne"],    # Genuine smile score (0 = forced, 1 = real)
                "dominant":       emotions["dominant"],    # The name of the strongest emotion right now
            }

        except Exception as e:
            print(f"[face] analysis error: {e}", flush=True)
            self.last_result = {"detected": False}

        return self.last_result

    # ── OpenFace background thread ─────────────────────────────────────────────

    def _openface_loop(self):
        """
        This function runs in a separate background thread for the entire lifetime
        of the app. It continuously checks whether a new frame has been queued
        for OpenFace analysis, processes it if so, and updates the emotion cache.

        Because it runs in the background, the main display thread never has to
        wait 600 ms for OpenFace — it just reads whatever emotion result is already cached.
        """
        while self._of_running:
            # Take the pending frame (if any) and clear the slot so the next frame can be queued.
            with self._of_lock:
                frame = self._of_pending
                self._of_pending = None

            if frame is not None:
                # A frame is waiting — run OpenFace on it.
                result = self._openface_analyze(frame)
                if result is not None:
                    # Replace the cached emotions with the new result, atomically.
                    with self._of_lock:
                        self._emotion_cache = result
            else:
                # No frame is queued right now — sleep briefly before checking again.
                # 50 ms sleep means we poll at most 20 times per second, avoiding busy-waiting.
                time.sleep(0.05)

    def _openface_analyze(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Runs OpenFace on a single camera frame and returns a parsed emotion dictionary.
        Steps:
          1. Save the frame as a temporary JPEG file.
          2. Run the FeatureExtraction binary on that file.
          3. Parse the CSV it writes out.
          4. Clean up all temporary files.
        Returns None if OpenFace fails or doesn't detect a face.
        """
        # Create a temporary file to hold the JPEG image.
        fd, jpg_path = tempfile.mkstemp(suffix=".jpg", prefix="of_frame_")
        os.close(fd)   # Close the OS-level file descriptor (cv2.imwrite will reopen it)

        # Create a temporary directory for OpenFace to write its output CSV into.
        out_dir = tempfile.mkdtemp(prefix="of_out_")

        try:
            cv2.imwrite(jpg_path, frame_bgr)         # Save the camera frame as a JPEG
            row = self._run_openface(jpg_path, out_dir)   # Run OpenFace and get back one CSV row
            if row is None or int(float(row.get("success", 0))) != 1:
                return None   # OpenFace reported failure or found no face
            return self._parse_emotions(row)          # Convert the raw CSV row into an emotion dictionary
        except Exception as e:
            print(f"[face/of] error: {e}", flush=True)
            return None
        finally:
            # Always clean up temporary files, even if an error occurred above.
            try:
                os.unlink(jpg_path)   # Delete the temporary JPEG
            except OSError:
                pass
            shutil.rmtree(out_dir, ignore_errors=True)   # Delete the entire OpenFace output directory

    def _warmup_openface(self):
        """
        Sends a blank grey image through OpenFace before the app starts.
        OpenFace loads several large model files into memory on its first run.
        Running a warmup image means those models are already loaded when the
        first real camera frame arrives, so there is no 1–2 second freeze.
        """
        fd, jpg_path = tempfile.mkstemp(suffix=".jpg", prefix="of_warm_")
        os.close(fd)
        out_dir = tempfile.mkdtemp(prefix="of_warm_")
        try:
            # A uniform grey image — no face will be found, but the models still load.
            blank = np.ones((360, 640, 3), dtype=np.uint8) * 128
            cv2.imwrite(jpg_path, blank)
            self._run_openface(jpg_path, out_dir)
        except Exception:
            pass   # If warmup fails it's not critical — the first real frame will just be slow
        finally:
            try:
                os.unlink(jpg_path)
            except OSError:
                pass
            shutil.rmtree(out_dir, ignore_errors=True)

    def _run_openface(self, jpg_path: str, out_dir: str) -> dict | None:
        """
        Launches the OpenFace FeatureExtraction program as a subprocess and
        returns the first row of the CSV it writes, as a dictionary.
        Returns None if OpenFace times out or writes no CSV.
        """
        cmd = [
            OPENFACE_BIN,       # Path to the compiled OpenFace binary
            "-f",       jpg_path,   # Input image file
            "-out_dir", out_dir,    # Where to write the output CSV
            "-q",                   # Quiet mode: suppress progress messages printed to the terminal
            "-aus",                 # Tell OpenFace to output AU intensities (_r) and presence flags (_c)
        ]
        try:
            # Run the command and wait up to 8 seconds for it to finish.
            # If it takes longer than 8 seconds, kill it and return None.
            subprocess.run(cmd, capture_output=True, timeout=8.0)
        except subprocess.TimeoutExpired:
            print("[face/of] timed out", flush=True)
            return None

        # Find the CSV file that OpenFace wrote into out_dir.
        csv_path = None
        for fname in os.listdir(out_dir):
            if fname.endswith(".csv"):
                csv_path = os.path.join(out_dir, fname)
                break
        if csv_path is None:
            return None   # OpenFace wrote no CSV — likely no face was detected

        # Read the first data row of the CSV and return it as a dictionary.
        # The keys are the column names (e.g. "AU06_r", "AU12_c").
        # We strip whitespace from the keys because OpenFace sometimes pads them with spaces.
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                return {k.strip(): v.strip() for k, v in row.items()}
        return None   # CSV was empty (should not happen, but guard anyway)

    def _parse_emotions(self, row: dict) -> dict:
        """
        Converts a raw OpenFace CSV row into the emotion dictionary stored in _emotion_cache.

        OpenFace outputs two numbers per Action Unit:
          AU_r (e.g. "AU06_r"): intensity, 0–5 scale — how strongly is this muscle contracting?
          AU_c (e.g. "AU06_c"): presence, 0 or 1 — is this muscle genuinely active at all?

        Using AU_c as a gate (multiplying by it) zeroes out any AU_r value when the
        muscle is not classified as truly active. This prevents faint resting-face noise
        from leaking into emotion scores on neutral expressions.
        """
        def f(key, default=0.0):
            # Safely read a float value from the CSV row; return default if missing or invalid.
            try:
                return float(row.get(key, default))
            except (ValueError, TypeError):
                return default

        # Read all AU intensities and normalise from the 0–5 OpenFace scale down to 0–1.
        aus_r = {au: f(f"{au}_r") / 5.0
                 for au in ["AU01","AU02","AU04","AU05","AU06","AU07",
                             "AU09","AU10","AU12","AU14","AU15","AU17",
                             "AU20","AU23","AU25","AU26","AU45"]}

        # Read all AU presence flags (0 = not active, 1 = active).
        # These act as noise gates: if AU_c is 0, that Action Unit is completely excluded
        # from the emotion scores no matter how large AU_r might be.
        aus_c = {au: int(f(f"{au}_c"))
                 for au in ["AU01","AU02","AU04","AU05","AU06","AU07",
                             "AU09","AU10","AU12","AU14","AU15","AU17",
                             "AU20","AU23","AU25","AU26","AU28"]}

        # Compute a 0–1 score for each emotion using the weighted AU combination defined above.
        # AU_c=0 means that entire AU term becomes zero (noise gate in action).
        expressions: dict[str, float] = {}
        for emotion, components in AU_EMOTION_WEIGHTS.items():
            score = sum(
                aus_c.get(au, 0) * aus_r.get(au, 0.0) * wt
                for au, wt in components
            )
            expressions[emotion] = min(1.0, score)   # Cap at 1.0 (can't be more than fully expressed)

        # Neutral = whatever is left after all other emotions are accounted for.
        # If the person is 30% happy and 20% sad, neutral = max(0, 1 − 0.5) = 0.5.
        expressions["neutral"] = max(0.0, 1.0 - sum(expressions.values()))

        # The dominant emotion is simply whichever has the highest score.
        dominant = max(expressions, key=expressions.get)

        # Duchenne smile: requires BOTH the cheek raiser (AU06) AND the lip corner puller (AU12)
        # to be present and active. A forced smile typically shows AU12 but not AU06.
        # The Duchenne score is the average of the two, gated by their presence flags.
        duchenne = float(
            (aus_c.get("AU06", 0) * aus_r.get("AU06", 0.0) +
             aus_c.get("AU12", 0) * aus_r.get("AU12", 0.0)) / 2.0
        )

        return {
            "expressions": expressions,                              # Emotion intensities (happy, sad, angry, etc.)
            "au_emotions": expressions,                              # Duplicate key kept because trust_engine also reads "au_emotions"
            "aus":         {au: v for au, v in aus_r.items()},      # Individual AU intensities (short keys like "AU06")
            "duchenne":    duchenne,                                  # Genuine smile score
            "dominant":    dominant,                                  # Name of the strongest emotion
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _track_blink(self, eye_ar: float):
        """
        Detects blink events from the Eye Aspect Ratio and recalculates the blink rate
        every 10 seconds.

        A blink is detected as: EAR drops below the threshold (eye closes)
        and then rises back above it (eye reopens). That full cycle = one blink.
        """
        if eye_ar < self.EAR_THRESH and not self.is_blinking:
            self.is_blinking = True   # The eye just closed — start of a blink

        elif eye_ar >= self.EAR_THRESH and self.is_blinking:
            self.is_blinking = False  # The eye just reopened — end of a blink
            self.blink_count += 1     # Count this as one complete blink

        elapsed = time.time() - self.blink_window_start
        if elapsed >= 10:   # Every 10 seconds, recalculate the blinks-per-minute rate
            self.blink_rate        = (self.blink_count / elapsed) * 60   # Convert count/10s to count/60s
            self.blink_count       = 0                                    # Reset the counter for the next window
            self.blink_window_start = time.time()                         # Start a new 10-second window
