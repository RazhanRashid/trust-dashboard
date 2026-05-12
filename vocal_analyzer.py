import numpy as np                      # Provides fast array maths for all signal-processing calculations


class VocalAnalyzer:
    def __init__(self):
        self.pitch_hist:  list[float] = []   # Rolling buffer of dominant pitch values (Hz) recorded while the user is speaking
        self.energy_hist: list[float] = []   # Rolling buffer of RMS energy values recorded while the user is speaking
        self.SPEAK_THRESH = 0.018            # RMS energy below this value is treated as silence (no speech detected)
        self.last_result: dict = {           # Default result returned before any audio has been analysed
            "is_speaking":     False,        # Whether the user is currently speaking
            "pitch_stability": 0.5,          # Coefficient of variation inverted to 0–1 (1 = perfectly stable pitch)
            "energy_level":    0.0,          # Normalised volume level (0 = silent, 1 = very loud)
            "tremor_index":    0.0,          # Measure of high-frequency energy fluctuation (0 = steady, 1 = very tremory)
            "dominant_hz":     0.0,          # Most prominent pitch frequency in Hz (0 when not speaking)
        }

    def analyze(self, samples: np.ndarray, sample_rate: int = 44100) -> dict:
        if samples is None or len(samples) == 0:  # Guards against empty audio chunks sent by the browser
            return self.last_result               # Returns the previous result unchanged rather than crashing

        samples = samples.astype(np.float32)      # Ensures the array is float32 (consistent type for all maths below)

        rms         = float(np.sqrt(np.mean(samples ** 2)))  # Root Mean Square: standard measure of audio signal energy/volume
        energy      = min(1.0, rms / 0.09)                   # Normalises RMS to a 0–1 scale (0.09 RMS ≈ comfortable speaking volume)
        is_speaking = rms > self.SPEAK_THRESH                 # Classifies the chunk as speech if energy exceeds the silence threshold

        dominant_hz = self._pitch_autocorr(samples, sample_rate) if is_speaking else 0.0  # Only runs pitch detection when speech is present

        if is_speaking:                            # Only updates the history buffers when the user is actively speaking
            self.pitch_hist.append(dominant_hz)   # Adds the current pitch to the rolling pitch history
            self.energy_hist.append(rms)          # Adds the current RMS to the rolling energy history
            if len(self.pitch_hist)  > 60: self.pitch_hist.pop(0)   # Keeps the pitch buffer at 60 samples max (≈ last few seconds)
            if len(self.energy_hist) > 60: self.energy_hist.pop(0)  # Keeps the energy buffer at 60 samples max

        self.last_result = {                       # Builds and stores the result dict for the WebSocket handler to read
            "is_speaking":     is_speaking,        # True if the current chunk was classified as speech
            "pitch_stability": self._stability(self.pitch_hist),  # 0–1 score; lower variance in pitch = higher stability = higher trust
            "energy_level":    energy,             # 0–1 normalised volume
            "tremor_index":    self._tremor(),     # 0–1 score; high rapid energy fluctuations = higher tremor = lower trust
            "dominant_hz":     dominant_hz,        # Pitch in Hz shown in the voice metrics panel
        }
        return self.last_result                    # Returns the result and stores it in self.last_result

    # ── Signal analysis ────────────────────────────────────────────────────────

    def _pitch_autocorr(self, samples: np.ndarray, sr: int) -> float:
        lo = max(1, int(sr / 450))                 # Minimum lag index corresponding to the upper pitch limit (450 Hz)
        hi = min(len(samples) - 1, int(sr / 80))  # Maximum lag index corresponding to the lower pitch limit (80 Hz)
        if lo >= hi:                               # If the audio chunk is too short to contain a full speech cycle, bail out
            return 0.0
        corr = np.correlate(samples, samples, mode="full")  # Full autocorrelation: measures similarity of the signal with delayed copies of itself
        corr = corr[len(corr) // 2:]              # Keeps only the positive-lag half (the mirror-image negative half is redundant)
        if len(corr) <= hi:                        # Guard: corr must be long enough to search in the [lo, hi] range
            return 0.0
        d = np.diff(corr[lo:hi])                   # First difference of the autocorrelation in the speech-frequency range
        peaks = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]  # Finds local minima→maxima transitions which mark the autocorrelation peaks
        if len(peaks) == 0:                        # No periodicity found in the speech range means pitch is indeterminate
            return 0.0
        return float(sr / (peaks[0] + lo))         # Converts the lag (in samples) of the first peak back to Hz (frequency = sample_rate / lag)

    def _stability(self, arr: list[float]) -> float:
        if len(arr) < 4:                           # Not enough history to compute a meaningful variance yet
            return 0.5                             # Returns neutral 0.5 so the score doesn't bias toward low or high
        a    = np.array(arr, dtype=np.float32)     # Converts the list to a NumPy array for vectorised maths
        mean = np.mean(a)                          # Calculates the average pitch over the history window
        if mean == 0:                              # Avoids division by zero if all pitch values are somehow zero
            return 0.5
        cv = np.std(a) / mean                      # Coefficient of Variation: standard deviation relative to the mean (lower = more stable)
        return float(np.clip(1.0 - cv, 0.0, 1.0)) # Inverts CV so that stable (low CV) maps to a high score, clamped to [0, 1]

    def _tremor(self) -> float:
        if len(self.energy_hist) < 8:              # Needs at least 8 samples to detect meaningful short-term fluctuations
            return 0.0                             # Returns zero tremor when there isn't enough data
        delta = np.sum(np.abs(np.diff(self.energy_hist)))  # Sums the absolute frame-to-frame energy changes (total variation)
        return float(min(1.0, delta / len(self.energy_hist) / 0.008))  # Normalises to 0–1 (0.008 per-frame change ≈ full tremor)
