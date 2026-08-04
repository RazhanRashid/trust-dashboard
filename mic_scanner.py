"""mic_scanner.py — find every audio input device and say how it is connected.

Unlike cameras (see camera_scanner.py), this is comparatively easy: PortAudio
(via the `sounddevice` package) already gives every device a real name, a
channel count, and a default sample rate cross-platform — no OS-specific
subprocess digging required to even get a usable list.

What PortAudio does *not* give us is transport (built-in vs. USB vs.
Bluetooth). That's inferred from the device name only, same best-effort
approach as camera_scanner's name hints — good enough to badge a device in
the picker, not a guarantee. A paired GoPro (or any Bluetooth mic/headset)
shows up here as an ordinary input device once the OS has connected it; this
module does not do any Bluetooth pairing itself.

Pure Python — no Qt in here, so it can run on a worker thread.
"""

from __future__ import annotations

import sounddevice as sd

# ── Transport kinds ────────────────────────────────────────────────────────
BUILTIN   = "builtin"
USB       = "usb"
BLUETOOTH = "bluetooth"
UNKNOWN   = "unknown"

TRANSPORT_LABEL = {
    BUILTIN:   "Built-in microphone",
    USB:       "USB — wired",
    BLUETOOTH: "Bluetooth",
    UNKNOWN:   "Unknown connection",
}

# Sort order for the picker: the system default first, then built-in, then
# wired, then Bluetooth, with unknowns last.
TRANSPORT_ORDER = {BUILTIN: 0, USB: 1, BLUETOOTH: 2, UNKNOWN: 3}

_BUILTIN_NAME_HINTS = ("macbook", "built-in", "builtin", "internal",
                       "integrated", "microphone array", "realtek")
_BLUETOOTH_NAME_HINTS = ("airpods", "bluetooth", "gopro", "beats", "buds",
                         "headset", "wireless", "hands-free", "hfp")
_USB_NAME_HINTS = ("usb", "webcam", "yeti", "audio interface")


def _name_transport_hint(name: str) -> str:
    lc = name.lower()
    if any(h in lc for h in _BLUETOOTH_NAME_HINTS):
        return BLUETOOTH
    if any(h in lc for h in _BUILTIN_NAME_HINTS):
        return BUILTIN
    if any(h in lc for h in _USB_NAME_HINTS):
        return USB
    return UNKNOWN


def _verified(index: int, samplerate: float) -> bool:
    """Cheap sanity check: does this device accept a mono stream at its own
    reported rate? `check_input_settings` only validates parameters against
    the driver — it never opens the device — so this cannot steal it from
    another app or grab an exclusive lock, unlike actually opening a stream.
    """
    try:
        sd.check_input_settings(device=index, channels=1,
                                samplerate=samplerate or None)
        return True
    except Exception:
        return False


def list_input_devices() -> list[dict]:
    """Every audio input device PortAudio can see, richest info first.

    Each entry: {index, name, channels, samplerate, hostapi, transport,
    label, is_default, verified}.
    """
    try:
        raw = sd.query_devices()
    except Exception as exc:
        print(f"[mic] device enumeration failed ({exc})", flush=True)
        return []

    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    devices = []
    for idx, d in enumerate(raw):
        if int(d.get("max_input_channels", 0) or 0) < 1:
            continue
        name = d.get("name") or f"Device {idx}"
        samplerate = float(d.get("default_samplerate") or 44100.0)
        hostapi_name = ""
        try:
            hostapi_name = hostapis[d["hostapi"]]["name"]
        except Exception:
            pass
        transport = _name_transport_hint(name)
        devices.append({
            "index":       idx,
            "name":        name,
            "channels":    int(d.get("max_input_channels", 1)),
            "samplerate":  samplerate,
            "hostapi":     hostapi_name,
            "transport":   transport,
            "label":       TRANSPORT_LABEL[transport],
            "is_default":  idx == default_in,
            "verified":    _verified(idx, samplerate),
        })

    devices.sort(key=lambda d: (not d["is_default"],
                                TRANSPORT_ORDER.get(d["transport"], 9),
                                d["name"].lower()))
    return devices
