# Trust Level Dashboard

A real-time desktop application that measures behavioural trust indicators from a webcam and microphone. Four analysis channels — facial expressions, gaze, voice, and heart rate variability — are combined into a single 0–100 trust score that updates live during a conversation. All processing runs locally; no data leaves the machine.

---

## What it measures

| Channel | Features extracted | How |
|---|---|---|
| **Facial** | 8 emotion scores, genuine vs forced smile (Duchenne), eye openness, blink rate, Action Units | MediaPipe blendshapes (live, every frame) + OpenFace FACS (post-hoc on the recording) |
| **Gaze** | Iris deviation, head rotation (yaw/pitch), pupil size proxy | MediaPipe 3D landmarks |
| **Voice** | Pitch (F0), loudness, jitter, shimmer, HNR, spectral flux, alpha ratio, Hammarberg index, MFCCs 1–4, formants F1–F3, glottal source features | OpenSMILE / eGeMAPSv02 (falls back to NumPy if unavailable) |
| **HRV** | Heart rate variability score | Placeholder — see [Adding a real HRV sensor](#adding-a-real-hrv-sensor) |

The four channels feed into a weighted trust score:

```
Trust = 25% Facial + 25% Vocal + 25% Gaze + 25% HRV
```

Exponential smoothing (α = 0.2) is applied so the score drifts gradually rather than jumping frame to frame.

---

## How a session works

1. **Calibration (30 seconds)** — the app records the subject's baseline facial metrics, pupil size, and voice characteristics while they are relaxed and unguarded. This baseline is used throughout the session to compute relative changes rather than absolute values.

2. **Live session** — the dashboard runs in real time. Trust score, sub-scores, and waveforms update every second. A face mesh overlay and emotion bars are drawn onto the live video feed.

3. **End of session** — the app saves an Excel workbook with per-second data across all channels and a per-second trust score. It also saves an `.mp4` recording and a `.jpg` thumbnail. OpenFace then runs a post-hoc analysis on the recording to add accurate FACS Action Unit data to the workbook.

---

## Project structure

```
trust-dashboard/
├── main.py                          # App entry point, UI, session management
├── panels.py                        # Dashboard panel components (camera, voice, score)
├── overlays.py                      # Face mesh and emotion overlay rendering
├── widgets.py                       # Custom UI widgets (gauges, bars, charts)
├── theme.py                         # Colours and visual styling
├── requirements.txt                 # Python dependencies
├── Physio_analysis/
│   ├── face_analyzer.py             # MediaPipe + OpenFace dual-path face analysis
│   ├── vocal_analyzer.py            # eGeMAPSv02 voice analysis via OpenSMILE
│   ├── trust_engine.py              # Combines all channels into the trust score
│   ├── hrv_analyzer.py              # HRV (placeholder — see below)
│   ├── workload_engine.py           # Pupil-dilation cognitive load detection
│   ├── nasa_tlx.py                  # NASA Task Load Index questionnaire
│   ├── FACE_ANALYSIS_ARCHITECTURE.md   # How MediaPipe and OpenFace work together
│   └── VOCAL_ANALYZER_EXPLAINED.md    # Plain-English walkthrough of the voice pipeline
└── static/                          # Icons and fonts
```

Session output is written to `~/Desktop/trust-dashboard/`:
```
~/Desktop/trust-dashboard/
├── session-data/
│   ├── sessions.json                # Session history index
│   └── trust-session-<timestamp>.xlsx
└── recordings/
    ├── <session-id>.mp4
    └── <session-id>.jpg             # Thumbnail
```

---

## How it works

### Face analysis

Two tools run in parallel on different timescales:

**MediaPipe** runs live on every webcam frame (~30 fps). It locates 478 landmarks on the face, computes Eye Aspect Ratio for blink detection, measures iris position for gaze deviation, extracts a 3D head rotation matrix for pose, estimates pupil size from iris radius, and scores 8 emotions from 52 blendshape coefficients.

**OpenFace** runs once after the session ends on the saved recording. It produces accurate FACS Action Unit intensities at full frame rate. The results are appended to the Excel workbook as a separate sheet and do not affect the live display.

See [`Physio_analysis/FACE_ANALYSIS_ARCHITECTURE.md`](Physio_analysis/FACE_ANALYSIS_ARCHITECTURE.md) for a full breakdown.

### Voice analysis

The microphone delivers audio at the device sample rate (typically 48 kHz) in 4096-sample chunks (~85 ms each). Each chunk is:

1. Checked against an RMS silence threshold — silent chunks are skipped
2. Resampled from 48 kHz to 16 kHz via `scipy.signal.resample_poly`
3. Passed to OpenSMILE, which runs eGeMAPSv02 and returns ~8 rows of features (one per 10 ms frame)
4. Aggregated — voiced-only features (jitter, shimmer, HNR, F0, formants) are averaged over voiced frames only; spectral and cepstral features are averaged over all frames

A 60-entry rolling history buffer tracks F0 over time to compute pitch stability as the coefficient of variation of recent pitch.

The tremor index is a clinical composite: 40% jitter + 40% shimmer + 20% inverted HNR.

If OpenSMILE is not installed, the analyzer falls back to NumPy autocorrelation pitch detection and RMS energy. All eGeMAPS columns in the Excel will be 0 in that case.

See [`Physio_analysis/VOCAL_ANALYZER_EXPLAINED.md`](Physio_analysis/VOCAL_ANALYZER_EXPLAINED.md) for a full walkthrough.

### Cognitive load

Pupil size (estimated from iris radius relative to inter-ocular distance) is tracked continuously. The workload engine computes a PCPS score `(pupil − baseline) / baseline + 1000` and maintains a 60-second rolling average as the WIV threshold. A sustained spike above that threshold for 60 seconds triggers a NASA TLX questionnaire when it resolves.

### Trust score

All four channels are weighted equally and combined:

```
Trust = 25% Facial + 25% Vocal + 25% Gaze + 25% HRV
```

Exponential smoothing (α = 0.2) is applied per channel — each new value gets 20% weight, the running history gets 80%. All channels start at 50 (neutral). The output is a 0–100 score plus a per-channel breakdown.

**Facial (starts at 50)**
Ekman emotion intensities (0–1) are multiplied by fixed point weights and summed:

| Emotion | Points |
|---|---|
| Happy | +30 |
| Neutral | +10 |
| Surprised | +4 |
| Sad | −18 |
| Disgusted | −30 |
| Fearful | −30 |
| Angry | −35 |
| Contempt | −40 |

A genuine (Duchenne) smile adds up to +20 on top of the happy weight. Three OpenFace Action Units apply additional deductions: AU04 brow furrow (−12), AU20 lip stretch (−10), AU14 dimpler (−8). An interaction penalty of −15 fires only when AU07 (lid tightener) and AU04 are both active simultaneously — the combination reads as a hostile stare.

**Vocal (starts at 55)**
Active speech is already a small positive signal, so the baseline is 55 rather than 50. Silent frames drift slowly back toward 50 rather than dropping immediately.

- Pitch stability (0–1): ±19 pts centred at 0.5, scaled by 38
- Energy level: +8 pts for comfortable volume, −18 for very quiet, −6 for very loud
- Tremor index (0–1): up to −32 pts at maximum tremor
- Alpha ratio (eGeMAPS): ±4 pts, centred at −10 dB; skipped if OpenSMILE unavailable
- Spectral flux (eGeMAPS): up to −5 pts above a 0.005 baseline; skipped if OpenSMILE unavailable

**Gaze (starts at 62)**
Sustained eye contact is a positive signal, so the baseline is 62.

- Eye Aspect Ratio: +10 for wide-open eyes, −12 for narrowed, −28 for nearly shut
- Blink rate: +8 for normal range (10–20/min), −10 for elevated (>23/min), −22 for rapid (>32/min)
- Gaze deviation: up to −18 for maximum look-away

**HRV (fixed placeholder: 65)**
Returns a fixed score of 65 until a real sensor is connected.

---

## Excel export

Each session produces an `.xlsx` workbook:

| Sheet | Contents | Rate |
|---|---|---|
| Trust Session Summary | Total score and all sub-scores | 1 fps |
| Facial Analysis | Expression, eye openness, blink rate, gaze, pupil, Duchenne smile | 1 fps |
| Vocal Analysis | Pitch, loudness, tremor, jitter, shimmer, HNR, spectral flux, alpha ratio, Hammarberg index, F1–F2, MFCCs 1–4 | 1 fps |
| Gaze Analysis | Gaze deviation, pupil normalised | 1 fps |
| Cognitive Load | PCPS workload score | 1 fps |
| HRV | Heart rate variability | 1 fps |
| AU Timeline | Per-AU line charts: MediaPipe blendshape vs OpenFace values over session time | Post-hoc |

> eGeMAPS columns in Vocal Analysis will be 0 for any session recorded before OpenSMILE was installed, or if `import opensmile` fails for the Python interpreter running the app.

---

## Requirements

- Python 3.10
- Webcam
- Microphone (optional — voice analysis is skipped if unavailable)
- `ffmpeg` and `ffprobe` on your PATH
- OpenFace built and placed at `~/OpenFace/build/bin/FeatureExtraction`
  - Build guide: https://github.com/TadasBaltrusaitis/OpenFace

> **Important — OpenSMILE and the correct Python:** `opensmile` and `scipy` are listed in `requirements.txt` and installed by `pip install -r requirements.txt`, but they must be installed under the exact Python interpreter used to launch the app. If the eGeMAPS columns in the Excel are all zero, verify with:
> ```bash
> python3 -c "import opensmile; print(opensmile.__version__)"
> ```
> using the same `python3` you run `main.py` with.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/RazhanRashid/trust-dashboard.git
cd trust-dashboard

# 2. Install dependencies
pip install -r requirements.txt
```

The MediaPipe face landmarker model is downloaded automatically on first run.

---

## Running

```bash
python3 main.py
```

The desktop window opens immediately.

**In Thonny:** File → Open → `main.py` → press Run (F5)

**In VS Code:** Open the folder → F5 → select *Run Trust Dashboard*

---

## Adding a real HRV sensor

`hrv_analyzer.py` returns a fixed score of 65. Two upgrade paths are documented inside the file:

- **Option A — Bluetooth monitor** (Polar H10, Garmin, etc.) via the `bleak` library
- **Option B — Webcam rPPG** — extract the pulse signal from the green channel of a forehead ROI, no extra hardware needed

---

## Notes

- Scores reflect behavioural indicators associated with comfort and openness. This is not a lie detector.
- All processing is local. No video, audio, or scores are sent anywhere.
