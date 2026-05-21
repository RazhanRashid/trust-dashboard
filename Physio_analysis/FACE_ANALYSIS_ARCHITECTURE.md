# How Face Analysis Works: Combining MediaPipe and OpenFace

## The Problem

The dashboard needs to do two things at the same time that pull in opposite directions:

1. **Update the screen 30 times per second** — so the face box, eye outlines, blink count, and gaze direction always feel live and responsive.
2. **Accurately detect emotions** — which requires a slow, heavyweight analysis tool that takes around 600 milliseconds per frame.

600 ms per frame means roughly 1–2 frames per second. That is far too slow for a live dashboard. But the slower tool is also far more accurate at reading emotions than anything fast enough to run in real time.

The solution is to use **both tools simultaneously**, each doing the job it is best suited for.

---

## The Two Tools

### MediaPipe (made by Google)
- **Speed:** ~10 ms per frame — fast enough for 30 fps
- **What it does:** Places 478 landmark dots on the face and uses their positions to compute geometric measurements
- **Used for:** Eye openness, blink detection, gaze direction, head rotation, pupil size estimate, face bounding box, eye outlines

### OpenFace (from Carnegie Mellon University)
- **Speed:** ~600 ms per frame — too slow for real-time display
- **What it does:** Identifies which individual facial muscles are contracting (called Action Units) and uses them to score emotions
- **Used for:** Happy, sad, angry, surprised, fearful, disgusted scores; genuine vs forced smile detection; individual muscle intensities

These are completely independent tools made by different organisations. Neither knows the other exists. The code in `face_analyzer.py` is the glue that runs both and combines their outputs.

---

## How They Run Together

The key idea is that MediaPipe and OpenFace run **on different threads** — meaning they work in parallel rather than one waiting for the other to finish.

```
Main thread (30 fps)
─────────────────────────────────────────────────────────
Frame arrives → MediaPipe runs (~10 ms) → result ready
                                         ↓
                              Queue frame for OpenFace
                                         ↓
                     Merge with latest cached emotion result
                                         ↓
                              Return complete result dict

Background thread (1–2 fps)
─────────────────────────────────────────────────────────
Wait for queued frame → OpenFace runs (~600 ms) → update emotion cache
Wait for queued frame → OpenFace runs (~600 ms) → update emotion cache
...
```

The background thread runs continuously and independently. Whenever MediaPipe finishes a frame, it drops a copy into a shared slot. The background thread picks it up whenever it is free, processes it, and writes the emotion result into a shared cache. The main thread always reads from that cache — so it always has *some* emotion result, just not necessarily from the exact current frame.

---

## What Each Tool Contributes to the Final Result

Every call to `analyze()` in `face_analyzer.py` returns a single dictionary. Here is where each field comes from:

| Field | Source | Update rate |
|---|---|---|
| Face bounding box | MediaPipe | ~30 fps |
| Eye outlines | MediaPipe | ~30 fps |
| Eye openness (EAR) | MediaPipe | ~30 fps |
| Blink rate | MediaPipe | ~30 fps |
| Gaze direction | MediaPipe | ~30 fps |
| Head rotation | MediaPipe | ~30 fps |
| Pupil size estimate | MediaPipe | ~30 fps |
| Emotion scores (happy, sad, etc.) | OpenFace | ~1–2 fps |
| Action Unit intensities | OpenFace | ~1–2 fps |
| Genuine smile score | OpenFace | ~1–2 fps |

The emotion fields update more slowly, but that is acceptable — emotions change gradually, and a 600 ms lag on an emotion label is not noticeable to a person watching the dashboard.

---

## Why Not Just Use One Tool?

| Approach | Problem |
|---|---|
| MediaPipe only | No Action Unit detection, no accurate emotion scoring |
| OpenFace only | ~1–2 fps display rate, blinks would be missed, gaze tracking would lag visibly |
| Run them sequentially per frame | Every frame takes 610 ms → still ~1–2 fps |
| Run them in parallel on separate threads | Both jobs complete at their own pace → 30 fps display + accurate emotions ✓ |

---

## Why the Emotion Cache Matters

When the app first starts, OpenFace has not finished its first analysis yet. If the code tried to wait for it before showing anything, the dashboard would freeze for the first 600 ms.

Instead, the emotion cache is pre-filled with neutral values (`happy: 0, sad: 0, neutral: 1, ...`). The dashboard starts immediately showing a neutral emotion state, and the cache is updated with real results as soon as OpenFace finishes its first frame. The user sees a smooth launch rather than a blank screen.

---

## Summary

MediaPipe and OpenFace are combined because no single tool can simultaneously be fast enough for real-time display *and* accurate enough for reliable emotion scoring. MediaPipe handles everything that must update instantly (eye tracking, blinks, face position), while OpenFace handles everything that requires deep analysis (emotions, muscle movements). Running them in parallel on separate threads means the dashboard gets both without either one slowing the other down.
