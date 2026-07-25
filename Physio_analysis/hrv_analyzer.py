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

# Catch Exception, not just ImportError. On Windows bleak pulls in the native
# winrt-* extension modules, and a bad/missing one can surface as OSError or a
# DLL-load error rather than ImportError — which would take the whole app down
# on import instead of degrading to the stub score. The reason is kept so the
# app can say *why* HRV is unavailable rather than failing mute.
try:
    from bleak import BleakClient, BleakScanner
    _BLEAK_AVAILABLE = True
    _BLEAK_IMPORT_ERROR: str | None = None
except Exception as _exc:                      # noqa: BLE001 - deliberately broad
    _BLEAK_AVAILABLE = False
    _BLEAK_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    print(f"[hrv] bleak unavailable — HRV disabled ({_BLEAK_IMPORT_ERROR})", flush=True)

# Standard Bluetooth SIG UUIDs — same on every BLE heart-rate strap/chest belt.
HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
# Some backends report 16-bit SIG UUIDs in short form rather than expanded.
HEART_RATE_SERVICE_SHORT = "180d"

# How long each scan pass runs. Windows tends to need longer than macOS before
# a strap's advertisement is picked up, so this is deliberately not tight.
SCAN_TIMEOUT_S = 10.0

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
        dashboard just keeps using the stub score.

        Every early return logs its reason. A silent no-op here is
        indistinguishable from "scanned and found nothing", which makes the
        difference between a missing dependency and an absent strap
        impossible to tell apart from the dashboard alone."""
        if not _BLEAK_AVAILABLE:
            print(f"[hrv] not starting — bleak is not importable "
                  f"({_BLEAK_IMPORT_ERROR}). Install it into the *same* Python "
                  f"you launch main.py with: pip install -r requirements.txt", flush=True)
            return
        if self._thread is not None:
            print("[hrv] not starting — already running", flush=True)
            return
        print(f"[hrv] starting BLE heart-rate thread (looking for "
              f"{self._device_name_hint!r})", flush=True)
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
        """Scan for a heart-rate strap and return the best match, or None.

        Deliberately scans *unfiltered* and matches afterwards, rather than
        passing service_uuids= to discover(). The two backends treat that
        kwarg very differently:

          - macOS hands it to CoreBluetooth's scanForPeripheralsWithServices,
            so the OS itself does the filtering and it works reliably.
          - Windows can't use its native filter (doing so would return either
            the advertisement or the scan response, never both), so bleak
            filters client-side on whatever service UUIDs happened to be in
            the advertisement. A strap that only lists 0x180D in its scan
            response — or whose first packets arrive without it — gets dropped
            before it ever reaches us, which is why a filtered scan can come
            back empty on Windows while finding the strap fine on a Mac.

        Matching on the advertised service UUID *or* the device name recovers
        those devices and behaves the same on both platforms.
        """
        found = await BleakScanner.discover(timeout=SCAN_TIMEOUT_S, return_adv=True)

        if not found:
            print("[hrv] scan found no BLE devices at all — is Bluetooth on?", flush=True)
            return None

        hint = self._device_name_hint.lower()
        by_service: list = []
        by_name: list = []

        print(f"[hrv] scan found {len(found)} BLE device(s):", flush=True)
        for device, adv in found.values():
            uuids = {str(u).lower() for u in (adv.service_uuids or [])}
            name = device.name or adv.local_name or "(unnamed)"
            has_hr = HEART_RATE_SERVICE_UUID in uuids or HEART_RATE_SERVICE_SHORT in uuids
            name_match = hint in name.lower()

            marker = "  <-- heart-rate service" if has_hr else ("  <-- name match" if name_match else "")
            print(f"[hrv]    {name!r} [{device.address}] rssi={adv.rssi}{marker}", flush=True)

            if has_hr:
                by_service.append(device)
            elif name_match:
                by_name.append(device)

        # Prefer a device that actually advertises the heart-rate service, and
        # among those prefer one matching the name hint (e.g. two straps in the room).
        for device in by_service:
            if hint in (device.name or "").lower():
                return device
        if by_service:
            return by_service[0]
        if by_name:
            print(f"[hrv] no device advertised the heart-rate service; falling back to "
                  f"name match {by_name[0].name!r}", flush=True)
            return by_name[0]
        return None

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


# ── Standalone diagnostic ───────────────────────────────────────────────────
# Run with:  python -m Physio_analysis.hrv_analyzer
#
# Scans for BLE devices and, if a heart-rate strap is found, connects and
# prints live beats. Use this to tell "the strap isn't advertising" apart from
# "the app isn't picking it up" without launching the whole dashboard.

async def _report_bluetooth_radio():
    """On Windows, print the Bluetooth adapter's power state.

    Uses winrt-windows-devices-radios, which bleak already depends on, so no
    extra install. This separates "the adapter is off or absent" from "the
    adapter is fine and nothing is advertising" — an unfiltered scan returns
    an empty dict in both cases, so the scan alone cannot tell them apart.
    Best-effort: any failure here is reported, never fatal.
    """
    import sys
    if sys.platform != "win32":
        return
    try:
        from winrt.windows.devices.radios import Radio, RadioKind, RadioState
        radios = await Radio.get_radios_async()
        bt = [r for r in radios if r.kind == RadioKind.BLUETOOTH]
        if not bt:
            print("radio   : no Bluetooth adapter found by Windows")
            return
        for r in bt:
            state = "ON" if r.state == RadioState.ON else str(r.state)
            print(f"radio   : Bluetooth adapter {r.name!r} is {state}")
    except Exception as exc:                   # noqa: BLE001
        print(f"radio   : could not query adapter state ({type(exc).__name__}: {exc})")


async def _diagnostic() -> int:
    import sys

    print(f"python  : {sys.version.split()[0]}")
    print(f"platform: {sys.platform}")
    print(f"executable: {sys.executable}")
    if not _BLEAK_AVAILABLE:
        print(f"\nbleak is NOT importable ({_BLEAK_IMPORT_ERROR})")
        print("Install it into this exact interpreter:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        return 1
    from importlib.metadata import version
    print(f"bleak   : {version('bleak')}")
    await _report_bluetooth_radio()
    print()

    analyzer = HRVAnalyzer()
    device = await analyzer._find_device()
    if device is None:
        print("\nNo heart-rate strap found. Things to check, in order:")
        print("  1. Is the strap worn, with the electrodes moistened? A Polar H10")
        print("     does not advertise at all until it detects skin contact.")
        print("  2. Is it already connected to something else (phone, Polar Flow/Beat,")
        print("     a watch)? BLE straps accept only one connection at a time.")
        print("  3. Is Bluetooth on, and does this terminal have permission to use it?")
        print("  4. If the scan listed other devices but not the strap, the strap is")
        print("     not advertising — that is a device/pairing problem, not the app.")
        print("     If the scan listed NOTHING at all, suspect the adapter or drivers.")
        return 1

    print(f"\nConnecting to {device.name!r} ({device.address})...")
    async with BleakClient(device) as client:
        print("Connected. Listening for 20 s of beats...\n")
        done = asyncio.Event()
        beats = 0

        def _on_notify(_char, data: bytearray):
            nonlocal beats
            hr, rr = _parse_hr_measurement(bytes(data))
            beats += 1
            print(f"  HR {hr} bpm    R-R {rr or '(none reported)'}")
            if beats >= 20:
                done.set()

        await client.start_notify(HEART_RATE_MEASUREMENT_UUID, _on_notify)
        try:
            await asyncio.wait_for(done.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            pass
        await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)

    if beats == 0:
        print("Connected but received no notifications — the strap may not have "
              "good skin contact.")
        return 1
    print(f"\nReceived {beats} notifications. The strap works; if the dashboard "
          f"still shows no HRV, the problem is in the app wiring, not Bluetooth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_diagnostic()))
