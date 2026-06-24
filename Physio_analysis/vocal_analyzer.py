import numpy as np                      # Provides fast array maths for all signal-processing calculations

# opensmile is the C++ acoustic feature extraction library wrapped for Python.
# eGeMAPSv02 is the Extended Geneva Minimalistic Acoustic Parameter Set — the standard
# feature set for speech emotion and stress research, covering 25 low-level descriptors
# per audio frame: F0, loudness, jitter, shimmer, HNR, MFCCs, formants, and spectral balance.
# If the package is not installed the analyzer falls back to the original hand-rolled numpy path.
try:
    import opensmile as _opensmile
    _OPENSMILE_OK = True
except ImportError:
    _OPENSMILE_OK = False

# scipy.signal.resample_poly is the highest-quality resampler available in the Python stack.
# It uses a polyphase FIR filter, which avoids the aliasing artefacts that simple decimation
# would introduce when downsampling from 44.1 kHz to 16 kHz before handing audio to opensmile.
# If scipy is not installed the fallback is numpy linear interpolation, which is lower quality
# but good enough for real-time feedback.
try:
    from scipy.signal import resample_poly as _resample_poly
    from math import gcd as _gcd
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


class VocalAnalyzer:
    _TARGET_SR  = 16000   # eGeMAPSv02 is designed for 16 kHz audio; all chunks are resampled before analysis
    _F0_BASE_HZ = 27.5    # eGeMAPSv02 stores F0 as semitones above 27.5 Hz (A0); convert back with 27.5 × 2^(s/12)

    def __init__(self):
        self.SPEAK_THRESH = 0.008            # RMS energy below this is treated as silence; lowered to catch quiet speech
        self.pitch_hist:  list[float] = []   # Rolling 60-entry buffer of per-chunk median F0 values used to compute pitch stability
        self.energy_hist: list[float] = []   # Rolling 60-entry buffer of per-chunk RMS values used by the legacy tremor calculation

        # Instantiate one shared opensmile Smile object at startup rather than per-call,
        # because loading the eGeMAPSv02 config file takes ~100 ms on first use.
        # LowLevelDescriptors gives one row per 10 ms frame so we can aggregate ourselves.
        self._smile = None
        if _OPENSMILE_OK:
            self._smile = _opensmile.Smile(
                feature_set=_opensmile.FeatureSet.eGeMAPSv02,
                feature_level=_opensmile.FeatureLevel.LowLevelDescriptors,  # per-frame output, not pre-averaged functionals
            )

        # Default result returned before any audio has been analysed, and also used as the
        # zero-value template for the legacy fallback path where eGeMAPS features are unavailable.
        self.last_result: dict = {
            # ── Core features — keys unchanged from original so nothing in main.py or trust_engine needs updating ──
            "is_speaking":     False,   # True when the current chunk's RMS exceeds the silence threshold
            "pitch_stability": 0.5,     # 0–1: inverse coefficient of variation of F0 history; 1 = perfectly stable pitch
            "energy_level":    0.0,     # 0–1: perceptual loudness normalised to conversational volume
            "tremor_index":    0.0,     # 0–1: composite vocal instability (see _tremor_from_features)
            "dominant_hz":     0.0,     # Most prominent F0 in Hz; 0 when not speaking or no voiced frames detected
            # ── eGeMAPS voice-quality features — added to expose acoustic stress markers from the literature ──
            "jitter":          0.0,     # Local jitter: cycle-to-cycle F0 perturbation; normal < 0.01, elevated > 0.02
            "shimmer_db":      0.0,     # Local shimmer in dB: amplitude perturbation; normal < 1 dB, elevated > 2 dB
            "hnr_db":          0.0,     # Harmonics-to-Noise Ratio in dB; clean voice > 20 dB, noisy/tense < 10 dB
            # ── eGeMAPS spectral features — added to capture breathiness, vocal effort, and spectral instability ──
            "spectral_flux":   0.0,     # Mean frame-to-frame spectral change; higher values indicate vocal agitation
            "alpha_ratio":     0.0,     # log(energy 1–5 kHz / 50 Hz–1 kHz); more negative = relaxed, less negative = strained
            "hammarberg_idx":  0.0,     # Strongest peak 2–5 kHz vs energy below 2 kHz; higher = greater vocal effort
            "slope_low":       0.0,     # Spectral slope in the 0–500 Hz band; reflects low-frequency tilt
            "slope_mid":       0.0,     # Spectral slope in the 500–1500 Hz band; reflects mid-frequency tilt
            # ── eGeMAPS cepstral/MFCC features — added for ML-based emotion/stress classification downstream ──
            "mfcc1":           0.0,     # 1st MFCC: encodes overall spectral shape and vocal tract length
            "mfcc2":           0.0,     # 2nd MFCC: encodes spectral tilt and low-frequency energy distribution
            "mfcc3":           0.0,     # 3rd MFCC: encodes mid-frequency spectral shape
            "mfcc4":           0.0,     # 4th MFCC: encodes high-frequency spectral detail
            # ── eGeMAPS glottal source features — added to detect vocal tension and open-quotient changes ──
            "log_h1h2":        0.0,     # Log ratio of 1st to 2nd harmonic amplitude; proxy for open quotient / breathiness
            "log_h1a3":        0.0,     # Log ratio of 1st harmonic to 3rd formant amplitude; reflects vocal tract coupling
            # ── eGeMAPS formants — added to track vowel articulation quality and vocal tract configuration ──
            "f1_hz":           0.0,     # 1st formant frequency in Hz; reflects jaw opening / vowel height (typical: 300–900 Hz)
            "f2_hz":           0.0,     # 2nd formant frequency in Hz; reflects tongue front/back position (typical: 800–2500 Hz)
            "f3_hz":           0.0,     # 3rd formant frequency in Hz; contributes to voice quality and speaker identity
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, samples: np.ndarray, sample_rate: int = 44100) -> dict:
        if samples is None or len(samples) == 0:  # Guards against empty chunks before the microphone warms up
            return self.last_result               # Returns the previous result unchanged rather than crashing

        samples     = samples.astype(np.float32)                      # Ensures consistent float32 arithmetic downstream
        rms         = float(np.sqrt(np.mean(samples ** 2)))           # Root Mean Square: standard measure of signal energy
        is_speaking = rms > self.SPEAK_THRESH                         # Classify as speech if energy exceeds silence threshold

        # Route to the eGeMAPS path when opensmile is loaded and the person is speaking;
        # fall back to the original numpy autocorrelation path when silent or opensmile unavailable.
        if is_speaking and self._smile is not None:
            result = self._analyze_opensmile(samples, sample_rate, rms, is_speaking)
        else:
            result = self._analyze_legacy(samples, sample_rate, rms, is_speaking)

        self.last_result = result   # Cache so callers that poll rather than wait get the most recent frame
        return result

    # ── eGeMAPS path ─────────────────────────────────────────────────────────

    def _analyze_opensmile(self, samples: np.ndarray, sr: int,
                           rms: float, is_speaking: bool) -> dict:
        # Any exception (short chunk, opensmile C++ crash, column mismatch) silently falls back
        # to the legacy path so the dashboard never freezes mid-session.
        try:
            sig16k = self._to_16k(samples, sr)                        # Downsample to 16 kHz before passing to opensmile
            df     = self._smile.process_signal(sig16k, self._TARGET_SR)  # Run eGeMAPSv02 — returns a DataFrame of per-frame LLDs

            if df is None or df.empty:                                # Chunk too short to produce any frames
                return self._analyze_legacy(samples, sr, rms, is_speaking)

            def am(col: str) -> float:
                # Mean over all frames (all-frame features: loudness, spectral flux, MFCCs, slopes)
                return float(df[col].mean()) if col in df.columns else 0.0

            def vm(col: str) -> float:
                # Mean over voiced frames only — columns with the _sma3nz suffix store 0 for unvoiced frames,
                # so averaging those zeroes in would artificially lower jitter, shimmer, F0, and HNR.
                if col not in df.columns:
                    return 0.0
                v  = df[col]
                nz = v[v != 0].dropna()               # Keep only the frames where the feature was actually computed
                return float(nz.mean()) if len(nz) else 0.0

            # ── F0: eGeMAPS stores pitch as semitones above 27.5 Hz; convert the per-frame median back to Hz ──
            f0s         = df["F0semitoneFrom27.5Hz_sma3nz"]          # Per-frame F0 in semitones; 0 = unvoiced frame
            voiced      = f0s[f0s > 0]                               # Discard unvoiced frames before taking the median
            dominant_hz = float(self._F0_BASE_HZ * 2 ** (voiced.median() / 12.0)) \
                          if len(voiced) >= 2 else 0.0               # Need ≥ 2 voiced frames for a stable median; otherwise report 0

            # ── Voice-quality features from voiced frames — fed into tremor_index and logged to Excel ──
            jitter     = vm("jitterLocal_sma3nz")        # Cycle-to-cycle F0 perturbation; rises under vocal tremor and stress
            shimmer_db = vm("shimmerLocaldB_sma3nz")     # Cycle-to-cycle amplitude perturbation in dB; rises under fatigue and strain
            hnr_db     = vm("HNRdBACF_sma3nz")           # Harmonics-to-Noise Ratio; falls when the voice becomes tense or noisy

            # ── Energy: use eGeMAPS perceptual loudness rather than raw RMS for a more natural 0–1 scale ──
            energy_level = float(np.clip(am("Loudness_sma3") / 0.6, 0.0, 1.0))  # 0.6 Nepers ≈ comfortable conversational volume

            # ── Update rolling history buffers — same 60-entry window as original ──
            if dominant_hz > 0:
                self.pitch_hist.append(dominant_hz)          # Only append when a voiced F0 was detected; silence doesn't dilute history
                if len(self.pitch_hist) > 60:
                    self.pitch_hist.pop(0)                   # Drop oldest entry once the buffer is full
            self.energy_hist.append(rms)                     # Energy history uses raw RMS for consistency with the legacy tremor helper
            if len(self.energy_hist) > 60:
                self.energy_hist.pop(0)

            return {
                "is_speaking":     is_speaking,
                "pitch_stability": self._stability(self.pitch_hist),                       # CV of eGeMAPS F0 history → more accurate than autocorrelation
                "energy_level":    energy_level,
                "tremor_index":    self._tremor_from_features(jitter, shimmer_db, hnr_db), # Replaces energy-variance tremor with jitter+shimmer+HNR composite
                "dominant_hz":     dominant_hz,
                # voice quality
                "jitter":          jitter,
                "shimmer_db":      shimmer_db,
                "hnr_db":          hnr_db,
                # spectral — am() averages over all frames including unvoiced
                "spectral_flux":   am("spectralFlux_sma3"),       # Frame-to-frame spectral change; used in trust_engine spectral flux term
                "alpha_ratio":     am("alphaRatio_sma3"),         # Breathiness/strain indicator; used in trust_engine alpha ratio term
                "hammarberg_idx":  am("hammarbergIndex_sma3"),    # Vocal effort and brightness; logged to Excel for offline analysis
                "slope_low":       am("slope0-500_sma3"),         # Low-frequency spectral slope; logged to Excel
                "slope_mid":       am("slope500-1500_sma3"),      # Mid-frequency spectral slope; logged to Excel
                # cepstral — logged to Excel for use in ML-based stress classification
                "mfcc1":           am("mfcc1_sma3"),
                "mfcc2":           am("mfcc2_sma3"),
                "mfcc3":           am("mfcc3_sma3"),
                "mfcc4":           am("mfcc4_sma3"),
                # glottal source — vm() because these are only valid on voiced frames
                "log_h1h2":        vm("logRelF0-H1-H2_sma3nz"),  # Open quotient proxy; detects breathiness vs pressed phonation
                "log_h1a3":        vm("logRelF0-H1-A3_sma3nz"),  # Vocal tract coupling; complementary glottal source feature
                # formants — vm() because formant tracking only runs on voiced frames
                "f1_hz":           vm("F1frequency_sma3nz"),      # 1st formant; tracks vowel openness and jaw position
                "f2_hz":           vm("F2frequency_sma3nz"),      # 2nd formant; tracks front/back tongue position
                "f3_hz":           vm("F3frequency_sma3nz"),      # 3rd formant; contributes to voice quality and timbre
            }

        except Exception:
            return self._analyze_legacy(samples, sr, rms, is_speaking)   # Silent fallback keeps the dashboard running

    # ── Legacy fallback path ─────────────────────────────────────────────────
    # Used when opensmile is not installed, when the chunk is silent, or when
    # the eGeMAPS path throws an exception.  Produces the same five core keys
    # as the original implementation; all eGeMAPS-specific keys are returned as 0.0.

    def _analyze_legacy(self, samples: np.ndarray, sr: int,
                        rms: float, is_speaking: bool) -> dict:
        energy      = min(1.0, rms / 0.09)                                                   # Normalise RMS to 0–1; 0.09 ≈ comfortable speaking volume
        dominant_hz = self._pitch_autocorr(samples, sr) if is_speaking else 0.0              # Run autocorrelation only during speech to save CPU

        if is_speaking:
            self.pitch_hist.append(dominant_hz)                   # Accumulate pitch history only while speaking so silence doesn't skew it
            self.energy_hist.append(rms)                          # Accumulate energy history for the legacy energy-variance tremor calculation
            if len(self.pitch_hist)  > 60: self.pitch_hist.pop(0)
            if len(self.energy_hist) > 60: self.energy_hist.pop(0)

        return {
            "is_speaking":     is_speaking,
            "pitch_stability": self._stability(self.pitch_hist),  # Same CV calculation as the eGeMAPS path; only the F0 source differs
            "energy_level":    energy,
            "tremor_index":    self._tremor(),                    # Original energy-variance tremor; used when opensmile is unavailable
            "dominant_hz":     dominant_hz,
            # All eGeMAPS-specific keys returned as 0.0 so downstream consumers can treat
            # 0.0 as "not available" without special-casing the opensmile vs legacy paths.
            "jitter":          0.0, "shimmer_db":      0.0, "hnr_db":         0.0,
            "spectral_flux":   0.0, "alpha_ratio":     0.0, "hammarberg_idx": 0.0,
            "slope_low":       0.0, "slope_mid":       0.0,
            "mfcc1":           0.0, "mfcc2":           0.0, "mfcc3":          0.0, "mfcc4": 0.0,
            "log_h1h2":        0.0, "log_h1a3":        0.0,
            "f1_hz":           0.0, "f2_hz":           0.0, "f3_hz":          0.0,
        }

    # ── Signal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_16k(samples: np.ndarray, src_sr: int) -> np.ndarray:
        # eGeMAPSv02 expects 16 kHz input.  Passing 44.1 kHz audio directly would shift all
        # frequency features by a factor of 2.75× and produce nonsense formant values.
        if src_sr == VocalAnalyzer._TARGET_SR:
            return samples                                        # Already at target rate; no work to do
        if _SCIPY_OK:
            g = _gcd(src_sr, VocalAnalyzer._TARGET_SR)           # Greatest common divisor so the up/down ratio is in lowest terms
            return _resample_poly(
                samples,
                VocalAnalyzer._TARGET_SR // g,                   # Upsample factor (16000 / gcd)
                src_sr // g,                                      # Downsample factor (src_sr / gcd)
            ).astype(np.float32)
        # numpy fallback: linear interpolation is fast but introduces some aliasing
        n_out = int(len(samples) * VocalAnalyzer._TARGET_SR / src_sr)   # Number of output samples at 16 kHz
        return np.interp(
            np.linspace(0, len(samples) - 1, n_out),             # Target sample positions in the original index space
            np.arange(len(samples)),                              # Source sample positions
            samples,
        ).astype(np.float32)

    @staticmethod
    def _tremor_from_features(jitter: float, shimmer_db: float, hnr_db: float) -> float:
        # Replaces the original energy-variance tremor with a clinically-grounded composite
        # drawn from eGeMAPS voiced-frame features.  Each component is normalised to [0, 1]
        # against a "severe" threshold taken from the speech pathology literature.
        j_score   = min(1.0, jitter / 0.05)              # 0.05 local jitter ≈ severe perturbation threshold
        s_score   = min(1.0, shimmer_db / 3.0)           # 3 dB shimmer ≈ severe amplitude perturbation threshold
        hnr_score = float(np.clip((20.0 - hnr_db) / 20.0, 0.0, 1.0))  # Inverted: 0 dB HNR = all noise = score of 1.0
        return float(0.4 * j_score + 0.4 * s_score + 0.2 * hnr_score) # Weighted sum: jitter and shimmer carry equal weight; HNR is secondary

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
