# Trust Level Dashboard

A real-time desktop application that measures behavioural trust indicators from a webcam, a microphone, and a Bluetooth heart-rate strap. Four analysis channels — facial expressions, gaze, voice, and heart rate variability — are combined into a single 0–100 trust score that updates live during a conversation. All processing runs locally; no data leaves the machine.

---

## What it measures

| Channel | Features extracted | How |
|---|---|---|
| **Facial** | 8 emotion scores, genuine vs forced smile (Duchenne), eye openness, blink rate, Action Units | MediaPipe blendshapes (live, every frame) |
| **Gaze** | Iris deviation, head rotation (yaw/pitch), pupil size proxy | MediaPipe 3D landmarks |
| **Voice** | Pitch (F0), loudness, jitter, shimmer, HNR, spectral flux, alpha ratio, Hammarberg index, MFCCs 1–4, formants F1–F3, glottal source features | OpenSMILE / eGeMAPSv02 (falls back to NumPy if unavailable) |
| **HRV** | RMSSD over a rolling 60-second R-R window, heart rate | Polar H10 or any BLE Heart Rate Service strap, via `bleak` |

The four channels are weighted equally:

```
Trust = 25% Facial + 25% Vocal + 25% Gaze + 25% HRV
```

Exponential smoothing (α = 0.2) is applied so the score drifts gradually rather than jumping frame to frame.

---

## Everything is measured against your own baseline

Nothing in the scoring is a fixed threshold applied to everybody. Resting eye shape, blink rate, vocal loudness, and RMSSD vary several-fold between healthy people, and plenty of relaxed faces register a permanent trace of sadness or a habitually furrowed brow. Scoring those against a population average penalises people for their face rather than for their state.

So the 30-second calibration window measures **this participant at rest**, and every threshold moves with them:

| Signal | What calibration measures | How it is then used |
|---|---|---|
| Eye Aspect Ratio | Resting eye openness | The shut / narrowed / wide cut-offs are scaled by it |
| Blink rate | Resting blinks per minute | The rapid / elevated / normal bands are scaled by it |
| Gaze deviation | Where they naturally rest their gaze | Deviation is measured from there, not from dead centre |
| Expressions, Action Units, Duchenne | What their resting face registers as | Each emotion and AU is scored as the difference from resting |
| Pitch stability | Their resting speech steadiness | Becomes the centre point instead of a fixed 0.5 |
| Energy level | Their resting speaking loudness | The quiet / shouting thresholds are scaled by it |
| Tremor index | Any permanent waver in their voice | Only the increase on top of it counts |
| Alpha ratio, spectral flux | Their resting voice spectrum | Become the centre points |
| RMSSD | Their resting HRV, in ms | Scored as a ratio: resting → 50, double → 75, half → 25 |

On top of that, each channel's resting **score** is subtracted, so a calibrated participant sitting at rest reads exactly **50** on every channel and on the total. That 50 is the reference point every later movement is read against — the gauge marks it, and a marker sitting anywhere else means something in the calibration window did not take.

Two things follow from this:

- **A calibration window only personalises what it actually captured.** No face detected, nobody spoke, no strap worn — each of those falls back to a population default *for that signal alone*, and everything else still personalises.
- **Skipping calibration is legitimate.** The defaults are chosen so that an uncalibrated session scores exactly as the app did before per-user baselines existed: EAR cut-offs at 0.14 / 0.20 / 0.28, blink bands at 10–20 / 23 / 32 per minute, pitch centred at 0.5, RMSSD on a linear 0–80 ms → 20–90 band. It just means "assume an average person at rest".

A deliberately wild calibration window — a yawn, a half-covered camera, someone leaning out of frame — cannot rescale a whole channel: the stretch factor is clamped to between 0.5× and 2× the population reference.

The measured baseline is shown in a review popup right after calibration (see [How a session works](#how-a-session-works)), and written into the **Score Config** sheet plus a baseline banner on every other sheet of the export, so a session's numbers can be reproduced and compared against another participant's.

---

## How a session works

1. **Participant details** — a short dialog collects sex, age, and cultural background. These are recorded in the export; nothing is inferred from them during scoring.

2. **Camera** — the app scans for every camera attached to the machine and asks which one to use. See [Choosing a camera](#choosing-a-camera).

3. **Calibration (30 seconds)** — the app records the participant's resting facial metrics, pupil size, voice characteristics, and RMSSD while they are relaxed and unguarded. The camera, microphone, and BLE scan all start before this screen so the preview is already live. A coverage indicator reports what fraction of frames actually found a face. Calibration can be skipped, at the cost described above.

   Right after calibration ends, a **baseline review popup** (`baseline_dialog.py`) shows every sensor's resting value before any session data is recorded — each row marked *measured* or *default* (fell back to the population reference because that signal was never captured), with a warning if face-detection coverage was low. A bad calibration window (participant out of frame, nobody speaking, strap unpaired) is cheap to redo here and expensive to discover after the session.

4. **Live session** — the dashboard runs in real time. Trust score, sub-scores, and waveforms update every second. A face mesh overlay and emotion bars are drawn onto the live video feed.

   - **Phases** — the researcher marks Trust Establishment → Trust Violation → Trust Recovery by hand, with the top-strip button or hotkeys `1`/`2`/`3`. The system never infers a phase change from the score. Phases appear as bands on the history chart and as a breakdown in the export.
   - **Behavioural flags** — rapid blink rate, sustained gaze aversion, the hostile-gaze AU04+AU07 combination, elevated voice tremor, and a sharp trust drop (more than 10 points in a second) are surfaced live in the sidebar. Each flag type is debounced for 8 seconds.
   - **Event markers** — `B` breach, `C` control, `M` generic marker. Each is stamped with a nanosecond master clock, flashed on screen, and written to the Events sheet for aligning against external recordings.

5. **End of session** — the app saves an Excel workbook with per-second data across all channels, a per-second trust score, and the raw per-frame logs. It also saves an `.mp4` recording and a `.jpg` thumbnail; the recording is transcoded to H.264 in the background so it plays back cleanly in QuickTime and other players.

Past sessions are listed on the overview screen and can be clicked to reopen their full summary.

---

## Project structure

```
trust-dashboard/
├── main.py                          # App entry point, UI, session management, Excel export
├── panels.py                        # Dashboard panel components (camera, voice, score, workload)
├── overlays.py                      # Calibration overlay, face mesh and emotion overlay rendering
├── widgets.py                       # Custom UI widgets (gauges, channel bars, charts)
├── demographics_dialog.py           # Participant details collected before calibration
├── baseline_dialog.py               # Post-calibration popup reviewing every sensor's measured baseline
├── bpm_monitor.py                   # TEMPORARY — always-on-top BPM readout for checking the strap against a watch (press H)
├── camera_scanner.py                # Finds cameras and identifies how each is connected
├── camera_dialog.py                 # Camera picker with live preview
├── theme.py                         # Colours, fonts, and screen-relative sizing
├── trust_dashboard.spec             # PyInstaller spec for building a Windows .exe
├── requirements.txt                 # Python dependencies
├── Physio_analysis/
│   ├── face_analyzer.py             # MediaPipe live face analysis
│   ├── vocal_analyzer.py            # eGeMAPSv02 voice analysis via OpenSMILE
│   ├── hrv_analyzer.py              # Polar H10 / BLE heart-rate client, RMSSD → score
│   ├── trust_engine.py              # Combines all channels into the trust score
│   ├── workload_engine.py           # Pupil-dilation cognitive load detection
│   ├── FACE_ANALYSIS_ARCHITECTURE.md   # How the MediaPipe face pipeline works
│   └── VOCAL_ANALYZER_EXPLAINED.md     # Plain-English walkthrough of the voice pipeline
└── static/                          # Unused: leftover web client from a previous version
```

Session output is written to `~/Desktop/trust-dashboard/`:
```
~/Desktop/trust-dashboard/
├── session-data/
│   ├── sessions.json                # Session history index
│   └── trust-session-<timestamp>.xlsx
└── recordings/
    ├── <session-id>.mp4
    ├── <session-id>.jpg             # Thumbnail
    └── <session-id>_start.json      # Start timestamp + score config, for external sync
```

---

## Choosing a camera

Before calibration the app scans the machine and lists every camera it finds, each labelled with how it is connected:

| Badge | Meaning |
|---|---|
| **Built-in camera** | The laptop's own camera |
| **USB — wired** | An external webcam plugged in over USB |
| **Bluetooth** | A camera paired and connected over Bluetooth |
| **Wireless** | Continuity Camera — an iPhone or iPad acting as a webcam |
| **Virtual camera** | A software device (OBS, Snap Camera, screen capture) |

Selecting one opens a live preview, which is the only reliable way to tell two plugged-in webcams apart. **Rescan** picks up a camera attached after launch. The choice is remembered and preselected next time, and the camera panel's ⇄ button reopens the picker mid-session.

Two questions have to be answered separately, because no single API answers both. *Which devices deliver frames* is only knowable by opening each one and reading a frame — OpenCV has no device list. *What each device is* comes from the platform:

| | Device names + order | Connection type |
|---|---|---|
| **macOS** | AVFoundation device list (PyObjC), or `ffmpeg -f avfoundation` if PyObjC is missing | AVFoundation device type, refined against `system_profiler` USB and Bluetooth trees |
| **Windows** | `ffmpeg -f dshow -list_devices`, which enumerates in the same order OpenCV's `CAP_DSHOW` indices follow | `Get-PnpDevice` instance IDs: `USB\`, `BTHENUM\` for Bluetooth, `ROOT\`/`SWD\` for virtual |
| **Linux** | sysfs (`/sys/class/video4linux`) | the `device` symlink target |

The two lists are unioned rather than intersected. A camera the OS lists but that will not open is still offered, marked *listed, but would not open* — the usual cause is another app holding it or camera permission not yet granted, both of which the researcher can fix. A working index the OS never named is offered too, as `Camera N`.

Two caveats worth knowing:

- **A laptop's built-in camera sits on an internal USB bus**, so on Windows and Linux the transport alone cannot distinguish it from a plugged-in webcam. The device name is used instead (`Integrated`, `Built-in`, `FaceTime`, …). macOS has no such ambiguity — AVFoundation reports the device type directly.
- **On Windows, ffmpeg is what maps a name to an index.** Without it the PnP order is used instead, which usually matches but is not guaranteed; the picker logs that it has fallen back.

If a badge or a name is wrong, the preview still shows which physical camera an entry is — that is what it is for.

The chosen camera is recorded in the export (Summary sheet). Facial and gaze readings are not comparable across a laptop camera and an external one at a different height and field of view, so the device travels with the session.

To see what the scanner detects without launching the app:

```bash
python camera_scanner.py
```

---

## How it works

### Face analysis

**MediaPipe** runs live on every webcam frame (~30 fps). It locates 478 landmarks on the face, computes Eye Aspect Ratio for blink detection, measures iris position for gaze deviation, extracts a 3D head rotation matrix for pose, estimates pupil size from iris radius, scores 8 emotions from 52 blendshape coefficients, and maps blendshapes to approximate Action Unit intensities.

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

### Heart rate variability

A background thread runs its own asyncio loop and, via `bleak`:

1. Scans for a device advertising the standard Heart Rate Service (`0x180D`), preferring one named "Polar H10"
2. Subscribes to the Heart Rate Measurement characteristic (`0x2A37`), which the strap pushes roughly once per heartbeat
3. Keeps a rolling 60-second window of R-R intervals and computes RMSSD from it
4. Maps RMSSD to a 0–100 score, relative to the calibrated resting RMSSD

The strap is optional. Without one, the channel reports a constant, and that constant becomes its own baseline — so a missing strap leaves the total neutral rather than quietly lifting it. The HRV bar in the sidebar greys out whenever the sensor is not streaming, and re-colours when it reconnects.

To test a strap without launching the dashboard:

```bash
python -m Physio_analysis.hrv_analyzer
```

It scans, reports every device it sees with signal strength, connects to a heart-rate strap if it finds one, and prints live beats. On Windows it also reports the Bluetooth adapter's power state, which separates "the adapter is off" from "nothing is advertising".

### Cognitive load

Pupil size (estimated from iris radius relative to inter-ocular distance) is tracked continuously. The workload engine computes a PCPS score `(pupil − baseline) / baseline + 1000` against the pupil baseline measured during calibration, and maintains a 60-second rolling average as the WIV threshold. A sustained spike above that threshold for 60 seconds is surfaced to the UI as `spike_progress`/`is_high_workload` (workload panel progress bar + Excel export columns).

### Trust score

All four channels are weighted equally and combined:

```
Trust = 25% Facial + 25% Vocal + 25% Gaze + 25% HRV
```

Exponential smoothing (α = 0.2) is applied per channel — each new value gets 20% weight, the running history gets 80%. Each channel is anchored at a neutral 50 and, after calibration, shifted so the participant's own resting state lands there. The output is a 0–100 score plus a per-channel breakdown, with the top two contributing signals per channel exposed for tooltips.

**Facial**
Ekman emotion intensities are measured as the difference from the participant's resting face, multiplied by fixed point weights and summed:

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

A genuine (Duchenne) smile beyond resting adds up to +20 on top of the happy weight. Three Action Units (approximated from MediaPipe blendshapes) apply additional deductions: AU04 brow furrow (−12), AU20 lip stretch (−10), AU14 dimpler (−8). An interaction penalty of −15 fires only when AU07 (lid tightener) and AU04 are both active above resting at the same time — the combination reads as a hostile stare.

**Vocal**
Silent frames drift slowly back toward the participant's baseline rather than dropping immediately.

- Pitch stability: ±19 pts, centred on their resting stability, scaled by 38
- Energy level: +8 pts for comfortable volume, −18 for very quiet, −6 for very loud, all relative to their resting loudness
- Tremor index: up to −32 pts, counting only tremor above their resting level
- Alpha ratio (eGeMAPS): ±4 pts from their resting ratio; skipped if OpenSMILE unavailable
- Spectral flux (eGeMAPS): up to −5 pts above their resting flux; skipped if OpenSMILE unavailable

**Gaze**

- Eye Aspect Ratio: +10 for wide-open eyes, −12 for narrowed, −28 for nearly shut — all scaled to their resting eye openness
- Blink rate: +8 for their normal range, −10 for elevated, −22 for rapid — all scaled to their resting rate
- Gaze deviation: up to −18 for maximum look-away, measured from where they naturally rest their gaze

**HRV**
RMSSD on a log ratio against resting: resting → 50, twice resting → 75, half resting → 25. Uncalibrated, a linear 0–80 ms → 20–90 population band.

---

## Excel export

Each session produces an `.xlsx` workbook. Most sheets carry a legend below the data explaining what each column means.

| Sheet | Contents | Rate |
|---|---|---|
| Summary | Participant details, camera used, duration, averages, peak/low trust, flag count, phase breakdown | once |
| Trust Session | Total score, all four sub-scores, and the headline metric from each channel | 1 fps |
| Facial Analysis | Expression, eye openness, blink rate, gaze deviation, pupil, Duchenne smile | 1 fps |
| Vocal Analysis | Pitch, loudness, tremor, jitter, shimmer, HNR, spectral flux, alpha ratio, Hammarberg index, F1–F2, MFCCs 1–4 | 1 fps |
| Gaze Analysis | Gaze deviation, pupil normalised | 1 fps |
| Cognitive Load | PCPS, WIV, high-workload state, spike progress | 1 fps |
| HRV | Score, sensor connected, heart rate, RMSSD | 1 fps |
| Raw Facial | Unthrottled per-frame facial log | ~30 fps |
| Raw Vocal | Unthrottled per-chunk vocal log | ~11 fps |
| Score Config | Every scoring constant, plus the baseline measured for this participant | once |
| Events | Timestamped session events, with nanosecond master clock for external sync | per event |
| Flags | Behavioural triggers, with the trust band in effect when each fired | per flag |

Every per-second sheet carries the marked phase alongside the timestamp, so rows can be grouped by protocol stage.

> eGeMAPS columns in Vocal Analysis will be 0 for any session recorded before OpenSMILE was installed, or if `import opensmile` fails for the Python interpreter running the app.

---

## Requirements

- Python 3.14 (3.10+ should work, but 3.14 is what the app is tested on)
- A camera — the built-in one, a USB webcam, or a Bluetooth camera (you pick at session start)
- Microphone (optional — voice analysis is skipped if unavailable)
- Polar H10 or another BLE Heart Rate Service strap (optional)
- `ffmpeg` and `ffprobe` on your PATH (used for thumbnail extraction and H.264 transcoding)

> **Important — OpenSMILE and the correct Python:** `opensmile` and `scipy` are listed in `requirements.txt` and installed by `pip install -r requirements.txt`, but they must be installed under the exact Python interpreter used to launch the app. If the eGeMAPS columns in the Excel are all zero, verify with:
> ```bash
> python3 -c "import opensmile; print(opensmile.__version__)"
> ```
> using the same `python3` you run `main.py` with. The same applies to `bleak` and the heart-rate strap.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/RazhanRashid/trust-dashboard.git
cd trust-dashboard

# 2. Create a virtual environment (3.14 recommended)
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`opencv-python` and `opencv-contrib-python` must stay on the same major line — they ship the same `cv2` module, and left unpinned pip resolves them to different majors and whichever installs last wins. `requirements.txt` pins both.

The MediaPipe face landmarker model (`face_landmarker.task`) is committed to the repo; if it is ever missing it is downloaded automatically on first run.

---

## Running

```bash
python3 main.py
```

The desktop window opens immediately.

**In VS Code:** open the folder → F5 → select *Run Trust Dashboard*

> **macOS + Bluetooth:** run the app from **VS Code's integrated terminal**, not Terminal.app. macOS aborts any process that touches CoreBluetooth without an `NSBluetoothAlwaysUsageDescription` key in the bundle that macOS holds responsible — Terminal.app has no such key, VS Code does. Wrapping the interpreter in a custom `.app` does not help: framework Python builds re-exec into their own bundled binary regardless. Without this the whole app aborts the moment the HRV thread starts scanning. The strap also needs to be worn and its electrodes moistened before it advertises.

If the HRV panel stays greyed out, check the console for the `[hrv]` lines — they distinguish "bleak not importable", "scanning", "no device found", and "connected". If a device is found but never connects, check whether the strap is already paired to another app (Polar Flow, Polar Beat, a phone), which blocks a second BLE connection.

The app also broadcasts a passive live tick stream on `ws://127.0.0.1:8765` for external subscribers.

---

## Windows

The app runs natively on Windows; two platform differences are handled in code:

- **Camera backend** — DirectShow (`cv2.CAP_DSHOW`) on Windows, AVFoundation on macOS. Device identification follows: PnP instance IDs on Windows, AVFoundation device types on macOS. Continuity Camera exists only on macOS and is not looked for elsewhere.
- **Camera scanning speed** — a failed DirectShow open blocks for a second or more, so the scan probes only the range the OS reported (plus two spares) rather than a fixed 0–7. Blind-probing every index is what makes a Windows camera scan crawl.
- **Console windows** — every subprocess the scanner runs (`powershell`, `ffmpeg`) is launched with `CREATE_NO_WINDOW`. Without it Windows flashes a black console box for each one, and in a packaged build they linger.
- **No standard streams** — `trust_dashboard.spec` builds with `console=False`, so a packaged app starts with `sys.stdout` and `sys.stderr` set to `None` and any `print()` raises. Since the camera scan, HRV, and recording all log from background threads, that would not print a stray error — it would kill the thread, and the camera scan would report no cameras on a machine with a working webcam. `main.py` substitutes null streams at import when they are missing.
- **BLE scanning** — the scan is deliberately unfiltered. bleak's WinRT backend cannot use the native service-UUID filter and falls back to filtering client-side on advertised UUIDs, which silently drops any strap that lists `0x180D` only in its scan response. The code scans unfiltered and matches afterwards on service UUID or name. **Do not "fix" this by re-adding a `service_uuids` filter** — the same code then finds the strap on macOS and nothing on Windows.

### Building a standalone .exe

```bash
pyinstaller trust_dashboard.spec
```

Produces `dist/TrustDashboard/`. It is a onedir build, not onefile — onefile re-extracts MediaPipe's large native libraries on every launch, which is slow and trips some antivirus scanners.

PyInstaller does not cross-compile: the spec has to be run on an actual Windows machine or Windows CI runner. It cannot be built or verified from macOS.

---

## Notes

- Scores reflect behavioural indicators associated with comfort and openness. This is not a lie detector.
- All processing is local. No video, audio, or scores are sent anywhere.
