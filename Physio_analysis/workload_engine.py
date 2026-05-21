"""
WIV-based workload monitor — Katidioti et al. (2016).

Algorithm:
    PCPS = (pupil - baseline) / baseline + 1000          # eq. 1
    WIV  = live_average_60s × threshold_adapter          # eq. 2

A 60-second sustained high-workload period (PCPS > WIV) is a "spike".
The first low-workload moment after a spike (PCPS < WIV for 200 ms
consecutively) triggers the NASA TLX callback.
"""

import time
import threading
from collections import deque


class WorkloadEngine:
    THRESHOLD_ADAPTER_INIT = 0.997
    THRESHOLD_ADAPT_STEP   = 0.001
    SPIKE_DURATION_S       = 60.0   # seconds of sustained high workload → spike
    LOW_WINDOW_S           = 0.200  # seconds below WIV to confirm low-workload moment

    def __init__(self):
        self.baseline_pupil    = None   # set after calibration via set_baseline()
        self._pcps_history: deque = deque()   # (timestamp, pcps) rolling 60 s
        self.threshold_adapter = self.THRESHOLD_ADAPTER_INIT
        self._lock             = threading.Lock()

        self._high_start       = None   # wall-clock start of current high-WL period
        self._spike_occurred   = False  # True once a 60 s spike has been confirmed
        self._low_start        = None   # wall-clock start of current low-WL period
        self._tlx_callback     = None   # callable — set via set_tlx_callback()

        # Exposed read-only state for the UI (written under _lock, read without)
        self.pcps             = 1000.0
        self.live_average     = 1000.0
        self.wiv              = 1000.0
        self.is_high_workload = False
        self.spike_progress   = 0.0    # 0 → 1 over the 60 s build-up

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_baseline(self, baseline_pupil: float):
        """Call once after calibration with the mean normalised pupil size."""
        self.baseline_pupil = baseline_pupil

    def set_tlx_callback(self, cb):
        """Register the function to call when a spike ends at a low-WL moment."""
        self._tlx_callback = cb

    def update(self, pupil_norm: float | None) -> dict:
        """
        Feed one pupil measurement (normalised by inter-ocular distance).
        Returns current engine state as a dict for the UI.
        """
        now = time.time()

        if pupil_norm is None or self.baseline_pupil is None or self.baseline_pupil < 1e-9:
            return self._snapshot()

        pcps = (pupil_norm - self.baseline_pupil) / self.baseline_pupil + 1000.0

        fire_tlx = False
        with self._lock:
            # Maintain 60 s rolling history
            self._pcps_history.append((now, pcps))
            cutoff = now - 60.0
            while self._pcps_history and self._pcps_history[0][0] < cutoff:
                self._pcps_history.popleft()

            # Live average and WIV
            vals = [v for _, v in self._pcps_history]
            self.live_average = sum(vals) / len(vals) if vals else 1000.0
            self.wiv          = self.live_average * self.threshold_adapter
            self.pcps         = pcps

            high = pcps > self.wiv
            self.is_high_workload = high

            if high:
                self._low_start = None
                if self._high_start is None:
                    self._high_start = now
                elapsed_high = now - self._high_start
                self.spike_progress = min(1.0, elapsed_high / self.SPIKE_DURATION_S)
                if elapsed_high >= self.SPIKE_DURATION_S:
                    self._spike_occurred = True
            else:
                self._high_start    = None
                self.spike_progress = 0.0

                if self._spike_occurred:
                    if self._low_start is None:
                        self._low_start = now
                    elif now - self._low_start >= self.LOW_WINDOW_S:
                        self._spike_occurred = False
                        self._low_start      = None
                        fire_tlx             = True
                else:
                    self._low_start = None

        if fire_tlx and self._tlx_callback:
            self._tlx_callback()

        return self._snapshot()

    def _snapshot(self) -> dict:
        return {
            "pcps":              round(self.pcps, 2),
            "live_average":      round(self.live_average, 2),
            "wiv":               round(self.wiv, 2),
            "is_high_workload":  self.is_high_workload,
            "spike_progress":    self.spike_progress,
            "threshold_adapter": round(self.threshold_adapter, 4),
        }
