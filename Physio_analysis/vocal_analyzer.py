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
        if samples is None or len(samples) == 0:  # Guards against empty audio chunks passed in before the microphone warms up
            return self.last_result               # Returns the previous result unchanged rather than crashing or returning partial data

        samples = samples.astype(np.float32)      # Ensures the array is float32 so all downstream maths uses a consistent numeric type

        rms         = float(np.sqrt(np.mean(samples ** 2)))  # Root Mean Square: squares each sample, averages them, then square-roots — standard measure of signal energy/loudness
        energy      = min(1.0, rms / 0.09)                   # Normalises RMS to a 0–1 scale where 0.09 RMS corresponds roughly to comfortable speaking volume
        is_speaking = rms > self.SPEAK_THRESH                 # Classifies the chunk as speech if the energy exceeds the silence threshold

        dominant_hz = self._pitch_autocorr(samples, sample_rate) if is_speaking else 0.0  # Only runs the relatively expensive pitch detection when the user is actually speaking

        if is_speaking:                            # Only updates the history buffers when the user is actively speaking so silence doesn't skew the averages
            self.pitch_hist.append(dominant_hz)   # Adds the current pitch estimate to the rolling history window
            self.energy_hist.append(rms)          # Adds the current RMS energy to the rolling history window
            if len(self.pitch_hist)  > 60: self.pitch_hist.pop(0)   # Caps the pitch history at 60 entries to bound memory usage and keep the window recent
            if len(self.energy_hist) > 60: self.energy_hist.pop(0)  # Caps the energy history at 60 entries for the same reason

        self.last_result = {                       # Assembles the result dict that main.py reads on every UI tick
            "is_speaking":     is_speaking,        # True if the current audio chunk was classified as speech rather than silence
            "pitch_stability": self._stability(self.pitch_hist),  # 0–1 score derived from pitch variance: low variance = high stability = higher trust signal
            "energy_level":    energy,             # 0–1 normalised loudness shown in the Voice Energy metric box
            "tremor_index":    self._tremor(),     # 0–1 score: high rapid energy fluctuations suggest vocal tremor and lower trust
            "dominant_hz":     dominant_hz,        # Most prominent pitch frequency in Hz shown in the Dominant Hz metric box
        }
        return self.last_result                    # Returns the freshly built dict and also stores it in self.last_result for polling callers

    # ── Signal analysis ────────────────────────────────────────────────────────

    def _pitch_autocorr(self, samples: np.ndarray, sr: int) -> float:
        lo = max(1, int(sr / 450))                 # Converts the 450 Hz upper pitch limit to a lag index (short lag = high freq); max(1,…) prevents a zero lag
        hi = min(len(samples) - 1, int(sr / 80))  # Converts the 80 Hz lower pitch limit to a lag index; capped at the last valid sample index
        if lo >= hi:                               # If the chunk is too short to contain even one full 80 Hz cycle, autocorrelation is meaningless
            return 0.0                             # Returns 0 to signal that pitch could not be determined for this chunk
        corr = np.correlate(samples, samples, mode="full")  # Computes the full autocorrelation: each position measures how similar the signal is to a lagged copy of itself
        corr = corr[len(corr) // 2:]              # Discards the negative-lag mirror half; keeps only lags 0, 1, 2, … which is all we need
        if len(corr) <= hi:                        # Ensures the autocorrelation array is long enough to index up to the hi lag
            return 0.0                             # Returns 0 if the array is too short (can happen with very small audio blocks)
        d = np.diff(corr[lo:hi])                   # Takes the first difference of the autocorrelation in the speech-frequency lag range to find where it rises
        peaks = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]  # Locates lag indices where the autocorrelation transitions from falling to rising — these are the peaks corresponding to periodicity
        if len(peaks) == 0:                        # No peaks means no detectable periodicity in the speech range; the sound is aperiodic or noisy
            return 0.0                             # Returns 0 to indicate indeterminate pitch
        return float(sr / (peaks[0] + lo))         # Converts the lag of the first (strongest) peak to Hz: frequency = sample_rate / lag_in_samples

    def _stability(self, arr: list[float]) -> float:
        if len(arr) < 4:                           # Requires at least 4 samples to compute a variance that isn't dominated by noise
            return 0.5                             # Returns a neutral mid-point score rather than biasing the dashboard before enough data has accumulated
        a    = np.array(arr, dtype=np.float32)     # Converts the Python list to a NumPy array for efficient vectorised statistics
        mean = np.mean(a)                          # Computes the mean pitch over the history window
        if mean == 0:                              # Guards against division by zero if all pitch readings happened to be zero
            return 0.5                             # Returns neutral 0.5 in the degenerate all-zero case
        cv = np.std(a) / mean                      # Coefficient of Variation = std / mean: a scale-independent measure of pitch variability
        return float(np.clip(1.0 - cv, 0.0, 1.0)) # Inverts CV so that stable (low variance) maps to a score near 1.0, then clamps to the valid [0, 1] range

    def _tremor(self) -> float:
        if len(self.energy_hist) < 8:              # Requires at least 8 energy samples to distinguish genuine tremor from normal breath variation
            return 0.0                             # Returns zero tremor rather than a spurious high reading while the buffer is still filling
        delta = np.sum(np.abs(np.diff(self.energy_hist)))  # Sums the absolute frame-to-frame energy differences — the total variation, a measure of rapid fluctuation
        return float(min(1.0, delta / len(self.energy_hist) / 0.008))  # Normalises by buffer length and a calibrated constant (0.008 per-frame change ≈ maximum expected tremor) then clamps to [0, 1]
