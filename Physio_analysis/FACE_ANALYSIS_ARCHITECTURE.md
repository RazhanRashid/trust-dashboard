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
- **Runs:** Once, on the saved `.mp4` recording, after the session ends
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
        │                 ──► Iris position            ──► gaze deviation
        │                 ──► 3D rotation matrix       ──► head pose deviation
        │                 ──► Iris radius / IOD        ──► pupil size proxy
        │                 ──► Bounding box + eye outlines
        │
        └── 52 blendshapes ──► emotion scores (weighted sums)
                           ──► approximate AU intensities
                           ──► Duchenne smile proxy
        │
        ▼
Complete result dict returned to main.py
(used for live display, trust score, session row recording)
```

### How Blendshapes Become Emotions

MediaPipe outputs 52 blendshape scores on every frame — each one is a 0–1 value for a specific facial deformation (e.g. `mouthSmileLeft`, `noseSneerRight`, `browInnerUp`). The code maps these to eight emotions via weighted sums following the Ekman AU classification table:

| Emotion | Key blendshapes |
|---|---|
| Happy | mouthSmileLeft/Right, cheekSquintLeft/Right |
| Sad | browInnerUp, browDownLeft/Right, mouthFrownLeft/Right |
| Angry | browDownLeft/Right, eyeSquintLeft/Right, mouthPressLeft/Right |
| Surprised | browInnerUp, browOuterUpLeft/Right, eyeWideLeft/Right, jawOpen |
| Fearful | browInnerUp, browOuterUp, browDown, eyeWide, eyeSquint, mouthStretch, jawOpen |
| Disgusted | noseSneerLeft/Right, mouthFrownLeft/Right, mouthLowerDownLeft/Right |
| **Contempt** | left–right asymmetry of mouthSmileRight − mouthSmileLeft (not a weighted sum) |
| Neutral | 1 − sum of all other emotion scores |

Contempt uses asymmetry rather than absolute values to avoid false positives from bilateral smiling — a genuine smile raises both sides equally (difference ≈ 0), while a unilateral smirk raises only one side (large difference).

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

The same 52 blendshape scores are also mapped to approximate Action Unit intensities via `BLENDSHAPE_AU_MAP`. These are anatomically motivated equivalents — for example, `browInnerUp` maps to AU01, `noseSneerLeft + noseSneerRight` maps to AU09. They are called *approximate* because blendshapes are a different measurement than true FACS AUs; OpenFace's regression model is more accurate.

---

## Post-Session Pipeline (OpenFace)

When the session ends:

```
Session ends
     │
     ├──► Base Excel saved immediately (blendshape AUs, no OpenFace data yet)
     │
     ├──► Waiting screen shown to the user
     │
     └──► Background thread: FaceAnalyzer.analyze_video(recording.mp4)
               │
               ▼
          OpenFace FeatureExtraction binary
          (single batch pass over the full .mp4)
               │
               ▼
          CSV parsed → per-frame dicts:
            frame_idx, timestamp_s, success,
            AU01–AU45 intensities (AU_r ÷ 5 → 0–1),
            AU presence flags (AU_c),
            emotion scores from AU combinations,
            accurate Duchenne smile (AU06 × AU12)
               │
               ▼
          Excel re-saved with accurate AU data
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
| `eye_ar` | MediaPipe landmarks | Average Eye Aspect Ratio (both eyes) |
| `blink_rate` | MediaPipe EAR | Rolling blinks/min, recalculated every 10 s |
| `gaze_deviation` | MediaPipe iris + rotation matrix | Combined head pose deviation, 0–1 |
| `pupil_norm` | MediaPipe iris landmarks | Iris radius ÷ inter-ocular distance |
| `expressions` | MediaPipe blendshapes | 0–1 score per emotion (8 emotions + neutral) |
| `aus` | MediaPipe blendshapes | Approximate AU intensities, 0–1 |
| `duchenne` | MediaPipe blendshapes | Approximate Duchenne score (AU06+AU12 proxy) |
| `dominant` | Derived from `expressions` | Highest-scoring emotion label |
| `box_norm`, `eye_norm` | MediaPipe landmarks | Face bounding box and eye outlines for display |

### `FaceAnalyzer.analyze_video()` return list (post-hoc)

| Field | Source | Notes |
|---|---|---|
| `frame_idx` | OpenFace CSV | Frame number in the video |
| `timestamp_s` | OpenFace CSV | Seconds from video start |
| `success` | OpenFace CSV | 1 if face was tracked in this frame |
| `aus` | OpenFace AU_r ÷ 5 | Accurate AU intensities, 0–1, per frame |
| `aus_c` | OpenFace AU_c | Presence flags (noise gate) per frame |
| `expressions` | AU_EMOTION_WEIGHTS | Emotion scores from AU combinations |
| `duchenne` | AU06 × AU12 | Accurate Duchenne smile score |

---

## Excel Output

| Sheet | Time resolution | AU source |
|---|---|---|
| Facial Analysis | 1 fps (session rows) | Blendshape AUs (teal columns), always present |
| OpenFace Raw | ~30 fps (every frame) | OpenFace AU_r values — only when post-hoc ran |

The Facial Analysis sheet always has data. The OpenFace Raw sheet is only added if OpenFace is installed and completed post-hoc analysis before the Excel was last saved.

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
