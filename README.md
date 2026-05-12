# Trust Level Dashboard

A real-time dashboard that measures behavioural trust indicators using your webcam and microphone. All analysis runs locally in Python — no data leaves your machine.

## What it measures

| Channel | Signal | Library |
|---|---|---|
| **Facial** | Expression probabilities (happy, neutral, angry, fearful, sad, disgusted, surprised) derived from 52 MediaPipe face blendshapes | MediaPipe |
| **Gaze** | Eye Aspect Ratio (blink detection), blink rate, iris deviation from eye centre | MediaPipe |
| **Vocal** | Pitch stability, voice energy, tremor index via autocorrelation | NumPy |

The three channels are combined (40% facial / 30% vocal / 30% gaze) into a single **Trust Score** (0–100) with exponential smoothing applied so the display doesn't jump erratically.

---

## Requirements

- Python 3.10 or later
- A webcam
- A microphone (optional — vocal analysis is skipped if unavailable)

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

The server starts on **http://localhost:8000** and your browser opens automatically.

---

## Running in VS Code

1. Open the `trust-dashboard` folder in VS Code (`File → Open Folder`)
2. Install the **Python** extension if prompted
3. Press `F5` or go to **Run → Start Debugging**
4. Select **Run Trust Dashboard** from the dropdown

The integrated terminal shows server logs and the browser opens automatically.

---

## Project structure

```
trust-dashboard/
├── main.py              # FastAPI server + WebSocket handler
├── face_analyzer.py     # MediaPipe face landmark detection, emotion mapping, EAR, iris gaze
├── vocal_analyzer.py    # Autocorrelation pitch detection, energy, tremor (NumPy)
├── trust_engine.py      # Weighted score combination with exponential smoothing
├── requirements.txt     # Python dependencies
├── face_landmarker.task # MediaPipe model (downloaded automatically on first run)
├── static/
│   ├── index.html       # Dashboard UI
│   ├── styles.css       # Dark-theme styles
│   └── client.js        # Browser capture layer (sends frames + audio to Python via WebSocket)
└── .vscode/
    └── launch.json      # VS Code debug configuration
```

---

## How it works

```
Browser                          Python (FastAPI)
───────                          ────────────────
Camera frame (JPEG, base64) ──►  face_analyzer.py
                                   └─ MediaPipe FaceLandmarker
                                      • 478 landmarks
                                      • 52 blendshapes → emotions
                                      • EAR → blink rate
                                      • Iris offset → gaze deviation

Audio chunk (Float32 PCM)   ──►  vocal_analyzer.py
                                   └─ NumPy autocorrelation
                                      • Dominant pitch (Hz)
                                      • Pitch stability (CV)
                                      • Energy level (RMS)
                                      • Tremor index

                                 trust_engine.py
                                   └─ Weighted combination
                                      • 40% facial
                                      • 30% vocal
                                      • 30% gaze
                                      • Exponential smoothing (α=0.2)

JSON scores + metrics       ◄──  WebSocket response

Renders gauge, bars, chart
```

---

## Notes

- The dashboard shows a **demo mode** with simulated signals when the Python server is not running (e.g. in a static preview environment).
- Trust scores reflect behavioural indicators correlated with comfort and openness — they are not a lie detector.
- All processing happens in your browser and on your local machine. No video, audio, or scores are sent to any external server.
