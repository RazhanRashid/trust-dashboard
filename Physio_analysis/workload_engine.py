"""
WorkloadEngine monitors mental workload by tracking how much the person's
pupils dilate compared to their own personal baseline.

How it works in plain English:
  1. When the app starts, it measures the person's pupil size at rest (the baseline).
  2. Every frame it computes a PCPS score — how much the current pupil size
     has changed relative to that baseline. Bigger pupils = more mental effort.
  3. It also tracks a rolling 60-second average of PCPS (called the WIV threshold).
  4. If PCPS stays above the threshold continuously for 60 seconds,
     that counts as a "workload spike" — the person has been under sustained strain.
     `spike_progress` (0.0 → 1.0) tracks how close the current high-workload
     period is to that 60-second mark, for display in the UI.

The PCPS formula comes from Katidioti et al. (2016):
    PCPS = (pupil_now − pupil_baseline) / pupil_baseline + 1000
    Adding 1000 keeps the number above zero even when pupils shrink below baseline.

WIV (Working Index Value) = rolling_60s_average × threshold_adapter
    threshold_adapter starts at 0.997 and is reserved for future self-calibration.
"""

import time        # Used to get the current wall-clock time for measuring durations
import threading   # Used to make shared data safe when the UI reads it at the same time the engine writes it
from collections import deque   # A fast queue that lets us efficiently keep only the last 60 seconds of data


class WorkloadEngine:
    # ── Class-level constants ─────────────────────────────────────────────────────
    THRESHOLD_ADAPTER_INIT = 0.997   # Starting multiplier that converts the live average into the WIV threshold
    THRESHOLD_ADAPT_STEP   = 0.001   # Reserved for future auto-calibration; not yet used
    SPIKE_DURATION_S       = 60.0    # Seconds of continuous high workload needed before a "spike" is declared

    def __init__(self):
        # The person's pupil size when they are calm and not doing anything demanding.
        # Set by calling set_baseline() after a short eyes-open rest period.
        # Until it is set, the engine cannot compute PCPS and returns placeholder data.
        self.baseline_pupil    = None

        # A rolling queue of (timestamp, pcps) pairs covering the last 60 seconds.
        # Old entries are automatically removed as time moves forward.
        self._pcps_history: deque = deque()

        # Multiplier applied to the live average to produce the WIV decision threshold.
        # Values below 1.0 make the threshold slightly lower than the average,
        # so small sustained increases still get flagged.
        self.threshold_adapter = self.THRESHOLD_ADAPTER_INIT

        # A lock prevents the UI thread and the analysis thread from reading/writing
        # the same variables at exactly the same moment, which could corrupt the data.
        self._lock             = threading.Lock()

        # Wall-clock time when the current high-workload period started.
        # Reset to None whenever workload drops back below the threshold.
        self._high_start       = None

        # ── Read-only state for the UI ────────────────────────────────────────────
        # These are updated inside the lock but read by the UI without locking
        # (a slightly stale value is acceptable for display purposes).
        self.pcps             = 1000.0   # Most recent PCPS value (1000 = exactly at baseline)
        self.live_average     = 1000.0   # Rolling 60-second average of PCPS
        self.wiv              = 1000.0   # Current WIV decision threshold (live_average × adapter)
        self.is_high_workload = False    # True when the current PCPS is above the WIV
        self.spike_progress   = 0.0     # Progress toward the 60-second spike threshold (0.0 → 1.0)

    # ── Public API ─────────────────────────────────────────────────────────────────

    def set_baseline(self, baseline_pupil: float):
        """
        Call this once after a calibration rest period.
        baseline_pupil should be the mean normalised pupil size
        (iris radius divided by inter-ocular distance) while the person is relaxed.
        """
        self.baseline_pupil = baseline_pupil

    def update(self, pupil_norm: float | None) -> dict:
        """
        Feed the engine one pupil measurement from the current camera frame.
        pupil_norm is the iris radius divided by the inter-ocular distance,
        which makes it independent of how close the person is to the camera.
        Returns a snapshot dict with the current engine state for the UI.
        """
        now = time.time()   # The exact moment this update is being processed

        # If we don't have a pupil reading this frame, or if baseline hasn't been set yet,
        # return the last known state without changing anything.
        if pupil_norm is None or self.baseline_pupil is None or self.baseline_pupil < 1e-9:
            return self._snapshot()

        # Compute PCPS: how much has the pupil changed from the personal baseline?
        # +1000 keeps the number positive even when pupils are smaller than baseline.
        # A value of 1000 means exactly at baseline. Above 1000 = dilated (more effort).
        pcps = (pupil_norm - self.baseline_pupil) / self.baseline_pupil + 1000.0

        with self._lock:   # Lock so the UI thread can't read half-updated values while we write
            # Add the new reading to the history queue with its timestamp.
            self._pcps_history.append((now, pcps))

            # Remove any entries older than 60 seconds from the front of the queue.
            cutoff = now - 60.0
            while self._pcps_history and self._pcps_history[0][0] < cutoff:
                self._pcps_history.popleft()

            # Recompute the rolling 60-second average from all entries still in the queue.
            vals = [v for _, v in self._pcps_history]
            self.live_average = sum(vals) / len(vals) if vals else 1000.0

            # The WIV decision threshold is the average multiplied by the adapter.
            # Because the adapter is 0.997 (just below 1.0), the threshold sits
            # slightly below the average, so the engine is sensitive to sustained increases.
            self.wiv          = self.live_average * self.threshold_adapter
            self.pcps         = pcps

            # Decide whether the current frame is a "high workload" moment.
            high = pcps > self.wiv
            self.is_high_workload = high

            if high:
                # Record when this high-workload period started (only on the first high frame).
                if self._high_start is None:
                    self._high_start = now

                # How long has workload been continuously elevated?
                elapsed_high = now - self._high_start

                # Express progress as a 0→1 fraction of the 60-second spike window.
                # The UI uses this to show a growing progress bar.
                self.spike_progress = min(1.0, elapsed_high / self.SPIKE_DURATION_S)

            else:
                # Workload dropped back below the threshold this frame.
                self._high_start    = None    # Reset the high-workload timer
                self.spike_progress = 0.0     # Reset the progress bar

        return self._snapshot()   # Return the current engine state to the caller

    def _snapshot(self) -> dict:
        # Packages the engine's current state into a plain dictionary for the UI to display.
        # Values are rounded to avoid showing excessive decimal places on screen.
        return {
            "pcps":              round(self.pcps, 2),              # Current pupil change score
            "live_average":      round(self.live_average, 2),      # 60-second rolling average
            "wiv":               round(self.wiv, 2),               # Current decision threshold
            "is_high_workload":  self.is_high_workload,            # True when above the threshold this frame
            "spike_progress":    self.spike_progress,              # 0.0–1.0 progress toward a 60-second spike
            "threshold_adapter": round(self.threshold_adapter, 4), # The adapter multiplier (for debugging)
        }
