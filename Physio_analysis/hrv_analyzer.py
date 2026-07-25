"""
HRV trust channel — live Polar H10 (or any BLE Heart Rate Service device).

Runs a background thread with its own asyncio event loop that:
  1. Scans for a BLE device advertising the standard Heart Rate Service
     (0x180D), preferring one named "Polar H10" if multiple are found.
  2. Subscribes to the Heart Rate Measurement characteristic (0x2A37),
     which the H10 pushes roughly once per heartbeat, each notification
     carrying the instantaneous HR plus any R-R intervals since the last
     notification.
  3. Maintains a rolling window of R-R intervals and computes RMSSD
     (root mean square of successive differences) — the standard
     short-window time-domain HRV metric.
  4. Maps RMSSD to a 0-100 trust sub-score using literature-derived
     bands (see _rmssd_to_score).

If no sensor is connected (bleak missing, Bluetooth off, device out of
range), falls back to STUB_SCORE so the rest of the dashboard keeps working.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

try:
    from bleak import BleakClient, BleakScanner
    _BLEAK_AVAILABLE = True
except ImportError:
    _BLEAK_AVAILABLE = False

# Standard Bluetooth SIG UUIDs — same on every BLE heart-rate strap/chest belt.
HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# R-R intervals older than this fall out of the RMSSD window.
RR_WINDOW_SECONDS = 60.0
# Need at least this many intervals before RMSSD is considered meaningful.
MIN_RR_FOR_SCORE = 4
# Re-scan/reconnect backoff after a dropped connection.
RECONNECT_DELAY_S = 3.0


def _parse_hr_measurement(data: bytes) -> tuple[int | None, list[int]]:
    """Parse the Heart Rate Measurement characteristic payload (GATT spec).

    Returns (heart_rate_bpm, rr_intervals_ms). RR intervals in the BLE
    payload are in units of 1/1024 s; converted to ms here.
    """
    flags = data[0]
    hr_16bit = flags & 0x01
    rr_present = flags & 0x10

    offset = 1
    if hr_16bit:
        heart_rate = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2
    else:
        heart_rate = data[offset]
        offset += 1

    # Energy expenditure field, if present, sits between HR and RR data.
    if flags & 0x08:
        offset += 2

    rr_intervals_ms: list[int] = []
    if rr_present:
        while offset + 1 < len(data):
            rr_1024 = int.from_bytes(data[offset:offset + 2], "little")
            rr_intervals_ms.append(round(rr_1024 * 1000 / 1024))
            offset += 2

    return heart_rate, rr_intervals_ms


class HRVAnalyzer:
    STUB_SCORE = 65   # used until a sensor connects, and as a safe fallback

    def __init__(self, device_name_hint: str = "Polar H10"):
        self._device_name_hint = device_name_hint

        self._lock = threading.Lock()
        self._rr_buffer: deque[tuple[float, int]] = deque()  # (received_at, rr_ms)
        self._latest_hr: int | None = None
        self._latest_rmssd: float | None = None
        self._latest_score: int = self.STUB_SCORE
        self._status: str = "disabled" if not _BLEAK_AVAILABLE else "disconnected"

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()

    # ── Public control ──────────────────────────────────────────────────────

    def start(self):
        """Begin scanning/connecting in a background thread. Safe to call
        even if bleak isn't installed or no sensor is available — the
        dashboard just keeps using the stub score."""
        if not _BLEAK_AVAILABLE or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Disconnect and stop the background thread. Safe to call multiple
        times or if start() was never called."""
        if self._thread is None:
            return
        self._stop_event.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)  # wake the loop
            except RuntimeError:
                pass  # loop already stopped/closed
        self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        with self._lock:
            self._status = "disconnected"

    # ── Public read API (called from the Qt main thread every tick) ────────

    def get_score(self) -> int:
        """Return a 0-100 trust score derived from HRV."""
        with self._lock:
            return self._latest_score

    def get_display(self) -> dict:
        with self._lock:
            return {
                "rmssd_ms":   self._latest_rmssd,
                "heart_rate": self._latest_hr,
                "score":      self._latest_score,
                "status":     self._status,
            }

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._status == "connected"

    # ── Background thread: asyncio/bleak lifecycle ──────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_stream())
        finally:
            self._loop.close()

    async def _connect_and_stream(self):
        while not self._stop_event.is_set():
            try:
                self._set_status("scanning")
                print("[hrv] scanning for a BLE heart-rate device...", flush=True)
                device = await self._find_device()
                if device is None:
                    print("[hrv] no heart-rate device found (is the H10 worn/moistened and awake?)", flush=True)
                    self._set_status("disconnected")
                    await self._sleep_or_stop(RECONNECT_DELAY_S)
                    continue

                print(f"[hrv] found {device.name!r} ({device.address}) — connecting...", flush=True)
                self._set_status("connecting")
                async with BleakClient(device) as client:
                    self._set_status("connected")
                    print("[hrv] connected, subscribing to heart-rate notifications", flush=True)

                    def _on_notify(_char, data: bytes):
                        self._handle_notification(bytes(data))

                    await client.start_notify(HEART_RATE_MEASUREMENT_UUID, _on_notify)
                    try:
                        while not self._stop_event.is_set() and client.is_connected:
                            await asyncio.sleep(0.5)
                    finally:
                        try:
                            await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
                        except Exception:
                            pass
                print("[hrv] disconnected", flush=True)
            except Exception as exc:
                # Connection dropped, device unreachable, adapter error, etc.
                # Fall back to stub score and retry rather than crashing the thread.
                print(f"[hrv] error: {exc!r}", flush=True)
                with self._lock:
                    self._latest_score = self.STUB_SCORE
                self._set_status("error")

            if not self._stop_event.is_set():
                await self._sleep_or_stop(RECONNECT_DELAY_S)

    async def _find_device(self):
        devices = await BleakScanner.discover(timeout=5.0, service_uuids=[HEART_RATE_SERVICE_UUID])
        print(f"[hrv] devices advertising heart-rate service: {[(d.name, d.address) for d in devices]}", flush=True)
        if not devices:
            return None
        for d in devices:
            if d.name and self._device_name_hint.lower() in d.name.lower():
                return d
        return devices[0]

    async def _sleep_or_stop(self, seconds: float):
        try:
            await asyncio.wait_for(self._stop_wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _stop_wait(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)

    def _set_status(self, status: str):
        with self._lock:
            self._status = status

    # ── Notification handling / RMSSD ───────────────────────────────────────

    def _handle_notification(self, data: bytes):
        heart_rate, rr_intervals_ms = _parse_hr_measurement(data)
        now = time.time()
        with self._lock:
            if heart_rate is not None:
                self._latest_hr = heart_rate
            for rr in rr_intervals_ms:
                self._rr_buffer.append((now, rr))
            cutoff = now - RR_WINDOW_SECONDS
            while self._rr_buffer and self._rr_buffer[0][0] < cutoff:
                self._rr_buffer.popleft()

            rmssd = self._compute_rmssd([rr for _, rr in self._rr_buffer])
            self._latest_rmssd = rmssd
            self._latest_score = (
                self._rmssd_to_score(rmssd) if rmssd is not None else self.STUB_SCORE
            )

    @staticmethod
    def _compute_rmssd(rr_intervals_ms: list[int]) -> float | None:
        if len(rr_intervals_ms) < MIN_RR_FOR_SCORE:
            return None
        diffs = [b - a for a, b in zip(rr_intervals_ms, rr_intervals_ms[1:])]
        mean_sq = sum(d * d for d in diffs) / len(diffs)
        return mean_sq ** 0.5

    # ── Helper for RMSSD → trust score ──────────────────────────────────────

    @staticmethod
    def _rmssd_to_score(rmssd_ms: float) -> int:
        """Linear mapping: RMSSD 0-80 ms → trust score 20-90."""
        score = 20.0 + (rmssd_ms / 80.0) * 70.0
        return int(max(0, min(100, round(score))))
