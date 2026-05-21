# Trust Level Dashboard

A real-time desktop application that measures behavioural trust indicators using your webcam and microphone. All analysis runs locally in Python — no data leaves your machine.

---

## What it measures

| Channel | What is detected | Tools used |
|---|---|---|
| **Facial** | Emotion scores (happy, sad, angry, fearful, disgusted, surprised, neutral), individual facial muscle activity (Action Units), genuine vs forced smile | MediaPipe + OpenFace |
| **Gaze** | Eye openness (Eye Aspect Ratio), blink rate, iris deviation, head rotation | MediaPipe |
| **Vocal** | Pitch stability, voice energy level, tremor index | NumPy |
| **HRV** | Heart rate variability (placeholder — see [Adding a real HRV sensor](#adding-a-real-hrv-sensor)) | — |

The four channels are combined into a single **Trust Score** (0–100) using weighted averaging (facial 35%, vocal 25%, gaze 25%, HRV 15%) with exponential smoothing so the display updates fluidly without jumping erratically.

---

## Requirements

- Python 3.10 or later
- A webcam
- A microphone (optional — vocal analysis is skipped if unavailable)
- OpenFace compiled and placed at `~/OpenFace/build/bin/FeatureExtraction`
  - Build instructions: https://github.com/TadasBaltrusaitis/OpenFace

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/RazhanRashid/trust-dashboard.git
cd trust-dashboard

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

On first run, the MediaPipe face landmarker model (~3 MB) is downloaded automatically.

---

## Running

```bash
python3 main.py
```

A desktop window opens immediately — no browser needed.

---

## Running in Thonny

1. Open Thonny and go to **File → Open** → select `main.py`
2. Press the green **Run** button (or F5)
3. The dashboard window opens automatically

---

## Running in VS Code

1. Open the `trust-dashboard` folder in VS Code (`File → Open Folder`)
2. Install the **Python** extension if prompted
3. Press `F5` or go to **Run → Start Debugging**
4. Select **Run Trust Dashboard** from the dropdown

---

## Project structure

```
trust-dashboard/
├── main.py                    # App entry point and main window
├── overlays.py                # On-screen face and eye overlay rendering
├── panels.py                  # Dashboard panel components
├── widgets.py                 # Custom UI widgets (gauges, bars, charts)
├── theme.py                   # Colours and visual styling
├── requirements.txt           # Python dependencies
├── sessions.json              # Session history log
├── recordings/                # Session recordings (.mp4 + .jpg thumbnail per session)
├── static/                    # Static assets (icons, fonts)
├── Physio_analysis/
│   ├── face_analyzer.py       # MediaPipe + OpenFace face analysis (see below)
│   ├── vocal_analyzer.py      # Voice pitch, energy, and tremor analysis
│   ├── trust_engine.py        # Combines all channels into a single trust score
│   ├── hrv_analyzer.py        # Heart rate variability (stub — see below)
│   ├── workload_engine.py     # Pupil-dilation workload detection (PCPS / WIV)
│   ├── nasa_tlx.py            # NASA Task Load Index questionnaire dialog
│   └── FACE_ANALYSIS_ARCHITECTURE.md  # Detailed explanation of the dual-tool face analysis
└── .vscode/
    └── launch.json            # VS Code debug configuration
```

---

## How it works

```
Webcam frame
    │
    ▼
face_analyzer.py
    ├─ MediaPipe (~10 ms, runs every frame)
    │    • Locates 478 landmark points on the face
    │    • Computes Eye Aspect Ratio → blink detection + blink rate
    │    • Measures iris position → gaze deviation
    │    • Extracts head rotation matrix → pose deviation
    │    • Estimates iris radius / inter-ocular distance → pupil size proxy
    │
    └─ OpenFace (~600 ms, runs in background thread)
         • Detects Action Unit intensities and presence flags
         • Scores emotions from muscle combinations
         • Detects genuine (Duchenne) vs forced smile

Microphone audio
    │
    ▼
vocal_analyzer.py
    • Autocorrelation pitch detection (80–450 Hz range)
    • Pitch stability (coefficient of variation)
    • Energy level (RMS normalised)
    • Tremor index (frame-to-frame energy variation)

Pupil size (from face_analyzer)
    │
    ▼
workload_engine.py
    • Computes PCPS: (pupil − baseline) / baseline + 1000
    • Maintains 60-second rolling average as the WIV threshold
    • Declares a workload spike after 60 s of continuous high PCPS
    • Triggers NASA TLX questionnaire when the spike ends

All channels
    │
    ▼
trust_engine.py
    • facial  35%  (expressions + Action Units + Duchenne smile)
    • vocal   25%  (pitch stability + energy + tremor)
    • gaze    25%  (eye openness + blink rate + gaze deviation)
    • hrv     15%  (heart rate variability — placeholder until sensor is connected)
    • Exponential smoothing (α = 0.2) applied to all channels
    • Outputs 0–100 trust score + per-channel breakdown
```

For a detailed explanation of why MediaPipe and OpenFace are used together, see [`Physio_analysis/FACE_ANALYSIS_ARCHITECTURE.md`](Physio_analysis/FACE_ANALYSIS_ARCHITECTURE.md).

---

## Session recording and export

Each recorded session is saved to `~/Desktop/trust-dashboard/` and includes:
- An `.mp4` video of the session
- A `.jpg` thumbnail
- An Excel workbook (`.xlsx`) with six sheets covering raw scores, emotion timelines, Action Units, vocal metrics, workload, and session summary

---

## Adding a real HRV sensor

`hrv_analyzer.py` currently returns a fixed placeholder score of 65. Two upgrade paths are documented inside the file:

- **Option A — Bluetooth heart-rate monitor** (Polar H10, Garmin, etc.) using the `bleak` library
- **Option B — Webcam rPPG** (no extra hardware) by extracting the pulse signal from the green channel of a forehead region of interest

---

## Notes

- Trust scores reflect behavioural indicators correlated with comfort and openness — they are not a lie detector and should not be used as one.
- All processing happens on your local machine. No video, audio, or scores are sent to any external server.
