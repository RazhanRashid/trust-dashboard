# Face Analysis Architecture

## The Problem

The dashboard needs to do two things that pull in opposite directions:

1. **Update the screen 30 times per second** — the face box, eye outlines, blink count, gaze direction, and emotion bars must always feel live.
2. **Accurately measure facial Action Units** — which requires OpenFace, a heavyweight tool that takes ~600 ms per frame and is far too slow for real-time display.

The solution is a **two-phase split**: MediaPipe handles everything that must be instant during a live session, and OpenFace runs a single efficient batch pass on the saved video once the session is over.

---

## The Two Tools

### MediaPipe (Google) — live session
- **Speed:** ~10 ms per frame
- **Runs:** Every camera frame during the session (~30 fps)
- **Outputs:** 478 face landmark coordinates, 52 blendshape scores, 3D head rotation matrix, iris landmarks

### OpenFace (Carnegie Mellon University) — post-session
- **Speed:** ~600 ms per frame, but processes a full video in one efficient batch
- **Runs:** Once, after the session ends, on JPEG frames extracted from the `.mp4` recording
- **Outputs:** Accurate FACS Action Unit intensities (AU_r, 0–5 scale) and presence flags (AU_c, 0 or 1) for every frame

---

## Live Session Pipeline (MediaPipe only)

During a session, `FaceAnalyzer.analyze()` is called on every camera frame. Everything comes from MediaPipe — there is no second thread and no OpenFace involvement.

```
Camera frame arrives (~30 fps)
        │
        ▼
MediaPipe FaceLandmarker (~10 ms)
        │
        ├── 478 landmarks ──► Eye Aspect Ratio (EAR) ──► blink detection & rate
        │                 ──► 3D rotation matrix       ──► gaze_deviation (primary)
        │                 ──► Iris position             ──► gaze_deviation fallback
        │                 ──►   (30-frame rolling avg)      if matrix unavailable
        │                 ──► Iris radius / IOD        ──► pupil size proxy
        │                 ──► Bounding box + eye outlines + landmarks_norm
        │
        └── 52 blendshapes ──► emotion scores (weighted sums)
                           ──► approximate AU intensities (13 AUs; see below)
                           ──► Duchenne proxy: avg(cheekSquintL/R, mouthSmileL/R)
        │
        ▼
Complete result dict returned to main.py
(used for live display, trust score, session row recording)
```

### How Blendshapes Become Emotions

MediaPipe outputs 52 blendshape scores on every frame — each one is a 0–1 value for a specific facial deformation (e.g. `mouthSmileLeft`, `noseSneerRight`, `browInnerUp`). The code maps these to eight emotions via weighted sums following the Ekman AU classification table:

| Emotion | Key blendshapes |
|---|---|
| Happy | mouthSmileLeft/Right (×0.25 each), cheekSquintLeft/Right (×0.25 each) |
| Sad | browInnerUp (×0.30), browDownLeft/Right (×0.15 each), mouthFrownLeft/Right (×0.15 each), mouthRollLower (×0.075), mouthShrugLower (×0.025) |
| Angry | browDownLeft/Right (×0.22 each), eyeWideLeft/Right (×0.10 each), eyeSquintLeft/Right (×0.10 each), mouthPressLeft/Right (×0.04 each), mouthRollUpper (×0.08) |
| Surprised | browOuterUpLeft/Right (×0.10 each), browInnerUp (×0.20), eyeWideLeft/Right (×0.10 each), jawOpen (×0.225), mouthFunnel (×0.125), mouthPucker (×0.05) |
| Fearful | browInnerUp (×0.12), browOuterUpLeft/Right (×0.06 each), browDownLeft/Right (×0.08 each), eyeWideLeft/Right (×0.08 each), eyeSquintLeft/Right (×0.06 each), mouthStretchLeft/Right (×0.08 each), jawOpen (×0.16) |
| Disgusted | noseSneerLeft/Right (×0.25 each), mouthFrownLeft/Right (×0.15 each), mouthShrugUpper (×0.10), mouthUpperUpLeft/Right (×0.05 each) |
| **Contempt** | left–right asymmetry only — see formula below |
| Neutral | max(0, 1 − sum of all other emotion scores) |

**Contempt formula:**
```
asymmetry   = max(0, mouthSmileRight − mouthSmileLeft − 0.15)
contempt    = min(1, asymmetry × 0.70 + mouthDimpleRight × 0.30)
```
A 0.15 dead-zone filters out natural asymmetry in a normal smile. A genuine bilateral smile raises both sides equally (difference ≈ 0), while a unilateral smirk raises only the right side, producing a large asymmetry value. The `mouthDimpleRight` term adds a small contribution from the right cheek dimple that often accompanies a true contemptuous smirk. Contempt uses asymmetry rather than a weighted sum to avoid false positives — `mouthSmileRight` also activates during a full bilateral smile, which would incorrectly trigger contempt if treated as an absolute value.

**Sources**

- **MediaPipe blendshape definitions** — the 52 blendshape names and what each one measures are defined by Google's Face Landmarker model, which adopts the blendshape coefficient set from Apple ARKit:
  - MediaPipe Face Landmarker guide: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
  - Apple ARKit blendshape locations reference: https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation

- **Ekman AU → emotion mapping** — the assignment of Action Units to each basic emotion follows the canonical Ekman & Friesen FACS classification table:
  - Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Consulting Psychologists Press.
  - Ekman, P. (1993). Facial expression and emotion. *American Psychologist, 48*(4), 384–392. https://doi.org/10.1037/0003-066X.48.4.384

- **Contempt asymmetry** — unilateral lip corner raise as the anatomical marker of contempt is described in:
  - Ekman, P., & Friesen, W. V. (1986). A new pan-cultural facial expression of emotion. *Motivation and Emotion, 10*(2), 159–168. https://doi.org/10.1007/BF00992253

### Approximate AUs During the Session

The same 52 blendshape scores are mapped to approximate Action Unit intensities via `BLENDSHAPE_AU_MAP`. These are anatomically motivated equivalents — for example, `browInnerUp` maps to AU01, `eyeSquintLeft + eyeSquintRight` maps to AU07.

**13 AUs have a blendshape equivalent:**

| AU | Name | Blendshape(s) |
|---|---|---|
| AU01 | Inner Brow Raise | browInnerUp |
| AU02 | Outer Brow Raise | browOuterUpLeft/Right |
| AU04 | Brow Lowerer | browDownLeft/Right |
| AU05 | Upper Lid Raiser | eyeWideLeft/Right |
| AU06 | Cheek Raiser | cheekSquintLeft/Right |
| AU07 | Lid Tightener | eyeSquintLeft/Right |
| AU09 | Nose Wrinkler | noseSneerLeft/Right |
| AU12 | Lip Corner Puller | mouthSmileLeft/Right |
| AU14 | Dimpler | mouthDimpleLeft/Right |
| AU15 | Lip Corner Depressor | mouthFrownLeft/Right |
| AU20 | Lip Stretcher | mouthStretchLeft/Right |
| AU26 | Jaw Drop | jawOpen |
| AU45 | Blink / Eye Closure | eyeBlinkLeft/Right |

**4 AUs have no blendshape equivalent** and are only available from OpenFace post-hoc:

| AU | Name | Reason |
|---|---|---|
| AU10 | Upper Lip Raiser | No ARKit blendshape for this muscle |
| AU17 | Chin Raiser | Mentalis muscle not in the 52 blendshapes |
| AU23 | Lip Tightener | Orbicularis oris tension not captured |
| AU25 | Lips Part | `mouthClose` is the inverse; no direct equivalent exists |

These are called *approximate* because blendshapes are a different measurement abstraction than true FACS AUs — they describe how close the face looks to a set of pre-defined 3D mesh poses, not which muscles contracted and by how much. OpenFace's geometric regression model is more accurate.

---

## Post-Session Pipeline (OpenFace)

When the session ends:

```
Session ends
     │
     ├──► Base Excel saved immediately (MediaPipe metrics only, no OpenFace data yet)
     │
     ├──► Waiting screen shown to the user
     │
     └──► Background thread: FaceAnalyzer.analyze_video(recording.mp4)
               │
               ▼
          ffprobe reads video FPS from the source file
          (OpenFace sets timestamp=0 for all frames in -fdir mode;
           timestamps are reconstructed as (frame_idx − 1) / fps)
               │
               ▼
          ffmpeg extracts every frame as a JPEG into a temp directory
          (OpenFace's bundled OpenCV lacks video-decoding support;
           image directory mode bypasses this limitation)
               │
               ▼
          OpenFace FeatureExtraction binary (-fdir mode)
          (single batch pass over the image directory)
               │
               ▼
          CSV parsed → per-frame dicts:
            frame_idx, timestamp_s (reconstructed), success,
            AU01–AU45 intensities (AU_r ÷ 5 → 0–1),
            AU presence flags (AU_c),
            emotion scores from AU combinations,
            accurate Duchenne smile (AU06 × AU12)
               │
               ▼
          Excel re-saved with OpenFace AU data (OpenFace Raw + AU Timeline sheets)
          mp4v recording transcoded to H.264 with ffmpeg (-movflags +faststart)
               │
               ▼
     Signal emitted → waiting screen dismissed → summary screen shown
```

OpenFace is not available frame-by-frame during the session — it only sees the final recording. This is intentional: processing the full video in one pass is far more efficient than calling it per-frame, and the results are written into the Excel export rather than the live display.

---

## What Each Source Contributes

### `FaceAnalyzer.analyze()` return dict (live, ~30 fps)

| Field | Source | Notes |
|---|---|---|
| `detected` | MediaPipe | Whether a face was found in this frame |
| `eye_ar` | MediaPipe landmarks | Average Eye Aspect Ratio (both eyes); fully open ≈ 0.27–0.35, closed ≈ 0 |
| `l_ear`, `r_ear` | MediaPipe landmarks | Individual left and right EAR values |
| `blink_rate` | MediaPipe EAR | Rolling blinks/min, recalculated every 10 s; EAR threshold = 0.21 |
| `gaze_deviation` | 3D rotation matrix (primary) | `(yaw + 0.5×pitch) / 40°`, clamped 0–1; falls back to 30-frame rolling average of iris deviation if matrix unavailable |
| `pupil_norm` | MediaPipe iris landmarks | Iris radius ÷ inter-ocular distance; `None` if iris tracking fails |
| `expressions` | MediaPipe blendshapes | 0–1 score per emotion: happy, sad, angry, surprised, fearful, disgusted, contempt, neutral |
| `au_emotions` | Alias of `expressions` | Same dict as `expressions`; kept for `trust_engine` compatibility |
| `aus` | MediaPipe blendshapes | Approximate AU intensities (0–1) for 13 mapped AUs; unmapped AUs (AU10, AU17, AU23, AU25) absent |
| `duchenne` | MediaPipe blendshapes | Average of cheekSquintLeft/Right and mouthSmileLeft/Right — proxy for genuine smile |
| `dominant` | Derived from `expressions` | Highest-scoring emotion label |
| `blendshapes` | MediaPipe | All 52 raw blendshape scores (name → 0–1); written to Excel and used for AU Timeline |
| `landmarks_norm` | MediaPipe | All 478 landmark (x, y) coordinates (normalised 0–1); used to draw the face mesh overlay on recordings |
| `box_norm`, `eye_norm` | MediaPipe landmarks | Face bounding box and 16-point eye outlines for live display |

### `FaceAnalyzer.analyze_video()` return list (post-hoc)

| Field | Source | Notes |
|---|---|---|
| `frame_idx` | OpenFace CSV | Frame number in the video |
| `timestamp_s` | Reconstructed | `(frame_idx − 1) / video_fps` — OpenFace CSV timestamps are always 0 in `-fdir` mode |
| `success` | OpenFace CSV | 1 if face was tracked in this frame |
| `aus` | OpenFace AU_r ÷ 5 | Accurate AU intensities for all 17 tracked AUs, 0–1 per frame |
| `aus_c` | OpenFace AU_c | Presence flags (noise gate) per frame |
| `expressions` | AU_EMOTION_WEIGHTS | Emotion scores from AU combinations |
| `duchenne` | AU06 × AU12 | Accurate Duchenne smile score |

---

## Excel Output

| Sheet | Time resolution | Contents |
|---|---|---|
| Facial Analysis | 1 fps (session rows) | MediaPipe live metrics only — expression, eye openness, blink rate, gaze deviation, pupil, Duchenne smile. No AU columns. |
| OpenFace Raw | ~30 fps (every frame) | OpenFace AU_r values for all 17 AUs — only present when post-hoc analysis completed |
| AU Timeline | 1 fps (aligned) | Per-AU line charts: MediaPipe blendshape value vs nearest OpenFace frame value over time. AUs with no blendshape mapping (AU10, AU17, AU23, AU25) show OpenFace only. |

The Facial Analysis sheet always has data. The OpenFace Raw and AU Timeline sheets are only added if OpenFace completed post-hoc analysis before the Excel was last saved.

---

## Why This Split?

| Approach | Problem |
|---|---|
| MediaPipe only | No accurate FACS AUs, approximate emotions only |
| OpenFace real-time per frame | ~1–2 fps display, blinks and gaze would lag visibly |
| Run both per-frame sequentially | Still ~610 ms per frame → 1–2 fps |
| Run both per-frame on separate threads (old design) | OpenFace still misses most frames; temp-file I/O per frame; complex state |
| **MediaPipe live + OpenFace post-hoc batch (current)** | 30 fps display with accurate AU data available after the session ✓ |

The post-hoc approach also produces more accurate results than per-frame OpenFace would have: the batch pass lets OpenFace use temporal context across the whole video rather than treating each frame in isolation.
