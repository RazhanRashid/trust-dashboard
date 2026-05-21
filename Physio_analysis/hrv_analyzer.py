"""
HRV trust channel — stub implementation.

Returns a fixed placeholder score of 65 until a real sensor is wired in.

To replace with a live implementation pick one of:

  Option A — BLE heart-rate monitor (Polar H10, Garmin, etc.)
      pip install bleak
      Read R-R intervals from the HR service characteristic (0x2A37).
      RMSSD = sqrt(mean(diff(rr_intervals_ms) ** 2))
      Map RMSSD → trust score (see _rmssd_to_score below).

  Option B — Webcam rPPG (no extra hardware)
      Extract the mean green-channel value from a forehead ROI each frame.
      Band-pass filter (0.7–3 Hz) to isolate the cardiac pulse.
      Detect peaks, compute R-R intervals, then RMSSD as above.
      Accuracy degrades under poor lighting; requires stable head position.

RMSSD → trust score heuristic (literature-derived):
    RMSSD > 50 ms  →  ~80   (relaxed, parasympathetic dominance, high trust)
    RMSSD 20–50 ms →  50–80 (moderate arousal)
    RMSSD < 20 ms  →  ~30   (stressed, sympathetic dominance, low trust)
"""


class HRVAnalyzer:
    STUB_SCORE = 65   # placeholder until a real sensor is connected

    def get_score(self) -> int:
        """Return a 0–100 trust score derived from HRV."""
        return self.STUB_SCORE

    def get_display(self) -> dict:
        return {
            "rmssd_ms":   None,           # float (ms) when real sensor is connected
            "heart_rate": None,           # int (bpm) when real sensor is connected
            "score":      self.STUB_SCORE,
            "status":     "stub",
        }

    # ── Helper for future real implementations ─────────────────────────────────

    @staticmethod
    def _rmssd_to_score(rmssd_ms: float) -> int:
        """Linear mapping: RMSSD 0–80 ms → trust score 20–90."""
        score = 20.0 + (rmssd_ms / 80.0) * 70.0
        return int(max(0, min(100, round(score))))
