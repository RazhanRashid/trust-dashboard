# vocal_analyzer.py — Plain-English Walkthrough

This document explains the vocal analysis code in three sections matching the
structure of the file: setup (lines 1–41), the public entry point and opensmile
call (lines 41–107), and the data processing that follows (lines 107–end).

---

## Section 1 — Setup (lines 1–41)

### Imports and fallback flags

The file starts by importing three external libraries, each wrapped in a
`try/except` so the app can still run if one is missing.

**numpy** (line 1) is always imported unconditionally. It provides the fast
array maths used throughout — squaring thousands of samples at once, taking
means, computing standard deviations. Without it the file would not load at all.

**opensmile** (lines 8–12) is the acoustic feature extraction library. It
contains a C++ engine that runs eGeMAPSv02, the standard 25-feature set used in
speech emotion and stress research. If the package is not installed, the flag
`_OPENSMILE_OK` is set to `False` and the app falls back to a simpler
hand-written analysis path.

**scipy** (lines 19–24) provides `resample_poly`, a high-quality audio
resampler. The microphone delivers audio at 48 000 samples per second, but
opensmile expects 16 000. `resample_poly` converts between the two using a
polyphase FIR filter, which is the cleanest way to do this because it removes
high-frequency content before discarding samples rather than just dropping every
third one. If scipy is not installed, the code falls back to numpy's simpler
linear interpolation, which is slightly lower quality but still functional.

### Class constants (lines 28–29)

```
_TARGET_SR  = 16000
_F0_BASE_HZ = 27.5
```

`_TARGET_SR` is the sample rate opensmile requires. Every audio chunk is
resampled to this before being handed over.

`_F0_BASE_HZ` is used to convert pitch out of the units eGeMAPS uses internally.
eGeMAPS stores fundamental frequency (F0) as semitones above 27.5 Hz (the note
A0, the lowest A on a piano) rather than in Hz directly. The conversion is:

```
Hz = 27.5 × 2^(semitones / 12)
```

For example, 24 semitones → 27.5 × 2^(24/12) = 27.5 × 4 = 110 Hz.

### `__init__` — initialisation (lines 31–77)

**`SPEAK_THRESH = 0.018`** is a silence gate. Before running any analysis, the
code computes RMS (root mean square) energy of the incoming audio chunk — a
single number that represents how loud it is. Audio samples are floating point
numbers between −1.0 and +1.0, so RMS also falls in that range. 0.018 is
approximately 1.8% of full scale, which sits in the gap between typical
background noise (RMS 0.002–0.010) and quiet speech (RMS 0.020+). Chunks below
this threshold are treated as silence and skipped. The value was carried over
from an earlier version of the analyzer and was chosen empirically rather than
derived from a formula — if someone speaks very quietly it may need lowering.

**`pitch_hist` and `energy_hist`** (lines 33–34) are rolling history buffers,
each capped at 60 entries. They work like conveyor belts: new values are added
to the right, and once the list exceeds 60 entries the oldest value is dropped
from the left. This means they always hold at most the last ~5 seconds of data
(60 chunks × ~85 ms per chunk). The pitch buffer is used to compute pitch
stability — how much the person's pitch has varied recently. The energy buffer
is used by the legacy tremor calculation. Neither buffer is part of opensmile;
they are plain Python lists managed entirely by our own code.

**The `Smile` object** (lines 39–44) is the opensmile engine. It is created
once at startup rather than on every audio callback, because loading the
eGeMAPSv02 configuration file takes roughly 100 ms. Since a new audio chunk
arrives every ~85 ms, creating a fresh engine on each call would be slower than
the chunk itself. Creating it once in `__init__` pays that cost a single time
and then reuses the same object throughout the session.

The `feature_level=LowLevelDescriptors` setting tells opensmile to return one
row of features per 10 ms frame rather than a single pre-averaged summary. An
85 ms chunk therefore produces roughly 8 rows. This matters because we want to
average only the frames where the voice is actually voiced (pitched), not
silence frames — something opensmile's built-in averaging would not do for us.

**`last_result`** (lines 48–77) is a dictionary of all 22 output fields
pre-filled with zeros (or 0.5 for pitch stability, which is the neutral
mid-point). It serves two purposes: it is returned immediately if the first
audio chunk arrives before any analysis has run, and it acts as the zero
template for the legacy fallback path where eGeMAPS features are unavailable.

---

## Section 2 — Entry point and opensmile call (lines 41–107)

### `analyze()` — the public method (lines 81–97)

This is the only method called from outside the class. The audio thread in
`main.py` calls it roughly 11 times per second with a fresh 4096-sample chunk.

The first thing it does is compute RMS energy and compare it to `SPEAK_THRESH`.
This is the silence gate described above. If the chunk is silent, the code skips
opensmile entirely and calls the legacy path instead, which returns the cached
pitch stability and energy values with all eGeMAPS fields as zero.

If the chunk is above the threshold and opensmile is available, it calls
`_analyze_opensmile`. The result is cached in `self.last_result` before being
returned, so that if the main UI thread reads the result slightly late it still
gets the most recent frame rather than nothing.

### `_analyze_opensmile()` — the opensmile path (lines 101–179)

The whole body is wrapped in a `try/except`. If anything goes wrong — a chunk
too short for opensmile to process, a C++ crash inside opensmile, a column name
that has changed between versions — the exception is silently caught and the
code falls back to the legacy path. This ensures the dashboard never freezes
mid-session because of a vocal analysis failure.

**Resampling (line 106):**
```python
sig16k = self._to_16k(samples, sr)
```
The raw microphone chunk (at 48 000 Hz) is resampled down to 16 000 Hz. See
Section 1 for why this is necessary. The resampled array is what gets handed
to opensmile.

**The opensmile call (line 107):**
```python
df = self._smile.process_signal(sig16k, self._TARGET_SR)
```
This is where opensmile actually runs. It takes the 16 kHz audio array, runs
the eGeMAPSv02 pipeline across it in 10 ms frames, and returns a pandas
DataFrame — a table where each row is one 10 ms frame and each column is one of
the 25 acoustic features. For an 85 ms chunk this table will have roughly 8
rows and 25 columns.

If the chunk was too short to produce even one frame, the DataFrame is empty
and the code falls back to the legacy path.

---

## Section 3 — Data processing (lines 107–end)

### Helper functions `am()` and `vm()` (lines 112–123)

These are small functions defined inside `_analyze_opensmile` that read values
out of the DataFrame.

**`am(col)`** — "all-frame mean". Returns the mean of a column across every row
in the DataFrame, including unvoiced (silent) frames. Used for features that are
meaningful even during silence, such as loudness, spectral flux, and MFCCs.

**`vm(col)`** — "voiced-frame mean". Returns the mean of a column but only for
rows where the value is non-zero. OpenSMILE writes 0 into voiced-only columns
(those ending in `_sma3nz`) whenever a frame is unvoiced. If `vm()` averaged
those zeroes in alongside the real values, features like jitter, shimmer, and
HNR would be artificially dragged toward zero. By filtering them out first, we
get the true average for the frames where the voice was actually producing
pitched sound.

### Pitch (F0) conversion (lines 126–129)

```python
f0s    = df["F0semitoneFrom27.5Hz_sma3nz"]
voiced = f0s[f0s > 0]
dominant_hz = 27.5 × 2^(voiced.median() / 12)
```

The F0 column is extracted from the DataFrame. Unvoiced frames (value = 0) are
discarded. The median of the remaining voiced frames is taken and converted from
semitones back to Hz using the formula described in Section 1. The median is
used rather than the mean because it is more resistant to outlier frames where
F0 tracking briefly goes wrong.

### Voice quality features (lines 132–134)

These three are extracted using `vm()` because they are only meaningful in
voiced frames:

- **Jitter** — how much the pitch wobbles from cycle to cycle. A steady voice
  has very low jitter. Stress and tremor raise it.
- **Shimmer** — how much the amplitude wobbles from cycle to cycle, measured in
  decibels. Fatigue and vocal strain raise it.
- **HNR (Harmonics-to-Noise Ratio)** — the ratio of clean harmonic energy to
  background noise in the voice, in decibels. A clear, relaxed voice has HNR
  above 20 dB. A tense or noisy voice drops below 10 dB.

### Energy (line 137)

```python
energy_level = clip(am("Loudness_sma3") / 0.6, 0.0, 1.0)
```

Rather than using raw RMS, the eGeMAPS path uses opensmile's perceptual loudness
feature, which is calibrated to how the human ear actually perceives volume. It
is divided by 0.6 (roughly the loudness of comfortable conversational speech)
to normalise it to a 0–1 scale.

### History buffer updates (lines 140–146)

The pitch and energy history buffers (from Section 1) are updated here with the
values just computed. Only frames with a detected voiced F0 are added to the
pitch buffer, so silent intervals do not dilute the stability calculation. The
60-entry cap is enforced by popping the oldest entry whenever the list exceeds
that length.

### Return dictionary (lines 148–176)

The function returns a dictionary of all 22 features. These fall into groups:

| Group | Features | Aggregation |
|---|---|---|
| Core | is_speaking, pitch_stability, energy_level, tremor_index, dominant_hz | Derived from buffers and the calculations above |
| Voice quality | jitter, shimmer_db, hnr_db | `vm()` — voiced frames only |
| Spectral | spectral_flux, alpha_ratio, hammarberg_idx, slope_low, slope_mid | `am()` — all frames |
| Cepstral | mfcc1–mfcc4 | `am()` — all frames |
| Glottal source | log_h1h2, log_h1a3 | `vm()` — voiced frames only |
| Formants | f1_hz, f2_hz, f3_hz | `vm()` — voiced frames only |

### Tremor calculation — `_tremor_from_features()` (lines 237–244)

Tremor is computed from the three voice quality features rather than from raw
energy variance (which the legacy path uses). Each component is normalised
against a clinical threshold from the speech pathology literature:

- Jitter contributes 40% (threshold: 0.05, beyond which perturbation is severe)
- Shimmer contributes 40% (threshold: 3 dB)
- HNR contributes 20%, inverted (0 dB = all noise = maximum tremor score)

The result is a 0–1 score where 0 means no tremor and 1 means severe
instability across all three markers.

### Legacy fallback path — `_analyze_legacy()` (lines 186–211)

Used whenever opensmile is unavailable or a chunk is silent. Computes pitch via
autocorrelation (finding the repeating period of the waveform directly) and
energy as a simple RMS ratio. All 17 eGeMAPS-specific fields are returned as
0.0. The five core fields (is_speaking, pitch_stability, energy_level,
tremor_index, dominant_hz) are returned with real values so the dashboard
remains functional without opensmile.

### Resampling helper — `_to_16k()` (lines 215–234)

Converts the raw audio chunk from the device sample rate to 16 000 Hz. Uses
`resample_poly` from scipy if available (higher quality, uses a polyphase FIR
filter), otherwise falls back to numpy's `interp` (linear interpolation, faster
but introduces mild aliasing). If the audio is already at 16 000 Hz the function
returns immediately with no work done.

### Pitch stability helper — `_stability()` (lines 261–269)

Takes the pitch history buffer and returns a 0–1 score. Computes the coefficient
of variation (CV = standard deviation / mean) — a scale-independent measure of
how spread out the values are. CV is then inverted so that a tight, stable pitch
(low CV) maps to a score near 1.0, and a wildly varying pitch (high CV) maps
near 0.0. Returns 0.5 (neutral) if there are fewer than 4 entries in the buffer,
to avoid misleading scores at the very start of a session.

### Legacy tremor helper — `_tremor()` (lines 271–275)

Used only when opensmile is unavailable. Sums the absolute frame-to-frame
differences in the energy history buffer (total variation), divides by buffer
length and a calibrated constant, and clamps to 0–1. A high number means energy
is jumping around rapidly, which was the original proxy for vocal tremor before
the jitter/shimmer/HNR composite replaced it.
