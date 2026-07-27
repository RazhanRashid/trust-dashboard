"""camera_scanner.py — find every attached camera and say how it is connected.

Two separate questions have to be answered, and no single API answers both:

  1. *Which OpenCV indices actually deliver frames?* Only opening them and
     reading a frame proves that. A device can be listed by the OS and still
     fail to open because another app holds it.
  2. *What is each one — the built-in laptop camera, a USB webcam, or something
     paired over Bluetooth?* OpenCV cannot tell you; it has no concept of a
     device name, let alone a transport. That comes from the platform.

So each backend below builds a {index: (name, transport)} map from the OS, the
indices are probed with OpenCV, and the two are merged. Everything is
best-effort: a probe that fails leaves the transport as "unknown" and the
camera is still offered to the user, who can see the device name and decide.

Pure Python — no Qt in here, so it can run on a worker thread.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import cv2

# ── Transport kinds ────────────────────────────────────────────────────────
BUILTIN   = "builtin"
USB       = "usb"
BLUETOOTH = "bluetooth"
WIRELESS  = "wireless"    # Continuity Camera / other wireless-but-not-Bluetooth
VIRTUAL   = "virtual"     # OBS, Snap Camera, screen-capture drivers
UNKNOWN   = "unknown"

TRANSPORT_LABEL = {
    BUILTIN:   "Built-in camera",
    USB:       "USB — wired",
    BLUETOOTH: "Bluetooth",
    WIRELESS:  "Wireless",
    VIRTUAL:   "Virtual camera",
    UNKNOWN:   "Unknown connection",
}

# Sort order for the picker: the built-in laptop camera first, then wired, then
# wireless, with virtual devices last since they are almost never the intent.
TRANSPORT_ORDER = {BUILTIN: 0, USB: 1, BLUETOOTH: 2, WIRELESS: 3,
                   UNKNOWN: 4, VIRTUAL: 5}

# Names that identify a laptop's own camera. Windows and Linux have no reliable
# transport signal for it — an internal webcam sits on an internal USB hub and
# enumerates exactly like an external one — so the name is what is left.
_BUILTIN_NAME_HINTS = ("facetime", "built-in", "builtin", "integrated",
                       "internal", "hd webcam", "hd user facing", "isight")
_VIRTUAL_NAME_HINTS = ("obs", "virtual", "snap camera", "manycam", "droidcam",
                       "screen capture", "ndi")
_PHONE_NAME_HINTS   = ("iphone", "ipad", "continuity", "desk view")

_TIMEOUT_S = 6

# Windows spawns a visible console window for every child process unless told
# not to. Without this the scan flashes several black boxes over the dashboard,
# and in a PyInstaller windowed build they linger. No effect on other platforms.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(cmd: list[str], want_stderr: bool = False) -> str:
    """Run a command and return its output, or "" if anything goes wrong.

    *want_stderr* is for ffmpeg's device listings, which it writes to stderr
    and then exits non-zero — a normal result, not a failure.
    """
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT_S,
                             creationflags=_NO_WINDOW)
        stream = out.stderr if want_stderr else out.stdout
        return (stream or b"").decode(errors="replace")
    except Exception:
        return ""


def _name_transport_hint(name: str) -> str | None:
    """Transport implied by the device name alone, if any."""
    lc = name.lower()
    if any(h in lc for h in _VIRTUAL_NAME_HINTS):
        return VIRTUAL
    if any(h in lc for h in _PHONE_NAME_HINTS):
        return WIRELESS
    if any(h in lc for h in _BUILTIN_NAME_HINTS):
        return BUILTIN
    if "bluetooth" in lc:
        return BLUETOOTH
    return None


# ═══════════════════════════════════════════════════════════════════════════
# macOS
# ═══════════════════════════════════════════════════════════════════════════
def _macos_usb_device_names() -> set[str]:
    """Every device name in the USB tree, flattened."""
    raw = _run(["system_profiler", "SPUSBDataType", "-json"])
    names: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            n = node.get("_name")
            if isinstance(n, str):
                names.add(n.lower())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    try:
        walk(json.loads(raw))
    except Exception:
        pass
    return names


def _macos_bluetooth_device_names() -> set[str]:
    """Names of currently connected Bluetooth devices."""
    raw = _run(["system_profiler", "SPBluetoothDataType", "-json"])
    names: set[str] = set()

    def walk(node, under_connected=False):
        if isinstance(node, dict):
            for key, val in node.items():
                # Connected devices live under "device_connected" as a list of
                # single-key dicts: [{"Razer Kiyo": {...}}, ...]
                if key == "device_connected":
                    walk(val, under_connected=True)
                else:
                    if under_connected and isinstance(key, str):
                        names.add(key.lower())
                    walk(val, under_connected)
        elif isinstance(node, list):
            for v in node:
                walk(v, under_connected)

    try:
        walk(json.loads(raw))
    except Exception:
        pass
    return names


def _macos_devices() -> dict[int, tuple[str, str]]:
    """AVFoundation device list, in the index order OpenCV also uses."""
    devices: dict[int, tuple[str, str]] = {}

    # ── PyObjC: gives the device *type* directly, which is the best signal ──
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        for i, d in enumerate(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)):
            name = str(d.localizedName())
            try:
                dtype = str(d.deviceType()).lower()
            except Exception:
                dtype = ""
            if "continuity" in dtype or "deskview" in dtype:
                transport = WIRELESS
            elif "builtin" in dtype or "built-in" in dtype:
                transport = BUILTIN
            elif "external" in dtype:
                transport = USB      # refined against the USB/BT trees below
            else:
                transport = UNKNOWN
            try:
                if d.isContinuityCamera():
                    transport = WIRELESS
            except Exception:
                pass
            devices[i] = (name, transport)
    except Exception as exc:
        print(f"[camera] PyObjC unavailable ({exc}); falling back to ffmpeg", flush=True)
        devices = _macos_devices_via_ffmpeg()

    if not devices:
        return devices

    # ── Refine: is an "external" camera on the USB bus, or paired over BT? ──
    usb_names = _macos_usb_device_names()
    bt_names  = _macos_bluetooth_device_names()
    for idx, (name, transport) in list(devices.items()):
        hint = _name_transport_hint(name)
        if hint in (VIRTUAL, WIRELESS, BUILTIN):
            devices[idx] = (name, hint)
            continue
        if transport in (USB, UNKNOWN):
            lc = name.lower()
            if any(lc in bt or bt in lc for bt in bt_names):
                devices[idx] = (name, BLUETOOTH)
            elif any(lc in usb or usb in lc for usb in usb_names):
                devices[idx] = (name, USB)
            elif transport == USB:
                # AVFoundation says external and nothing claims it — a wired
                # webcam whose USB product name differs from its camera name is
                # far more likely than an exotic transport.
                devices[idx] = (name, USB)
    return devices


def _macos_devices_via_ffmpeg() -> dict[int, tuple[str, str]]:
    """Fallback when PyObjC is missing: names only, transport inferred later."""
    text = _run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                want_stderr=True)
    devices: dict[int, tuple[str, str]] = {}
    in_video = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if in_video:
            m = re.search(r"\[(\d+)\]\s+(.+)", line)
            if m:
                name = m.group(2).strip()
                devices[int(m.group(1))] = (name, _name_transport_hint(name) or UNKNOWN)
    return devices


# ═══════════════════════════════════════════════════════════════════════════
# Windows
# ═══════════════════════════════════════════════════════════════════════════
def _parse_pnp_cameras(raw: str) -> list[tuple[str, str]]:
    """[(name, transport)] from Get-PnpDevice JSON, in the order reported.

    A device's PnP InstanceId carries its bus: ``USB\\...``, ``BTHENUM\\...``
    for a Bluetooth-paired device, ``ROOT\\`` or ``SWD\\`` for a software one.
    What it does *not* distinguish is a laptop's built-in camera from a plugged
    in webcam — an internal webcam sits on an internal USB hub and enumerates
    exactly like an external one — so that distinction falls back to the name.
    """
    out: list[tuple[str, str]] = []
    try:
        parsed = json.loads(raw) if raw.strip() else []
    except Exception:
        return out
    if isinstance(parsed, dict):        # PowerShell emits a bare object for one result
        parsed = [parsed]
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("FriendlyName") or "").strip()
        inst = (entry.get("InstanceId") or "").upper()
        if not name:
            continue
        if inst.startswith(("BTHENUM", "BTH\\", "BTHLE")):
            transport = BLUETOOTH
        elif inst.startswith(("ROOT", "SWD")):
            transport = VIRTUAL
        elif inst.startswith("USB"):
            transport = _name_transport_hint(name) or USB
        else:
            transport = _name_transport_hint(name) or UNKNOWN
        out.append((name, transport))
    return out


def _parse_dshow_names(text: str) -> list[str]:
    """Video device names from ffmpeg's DirectShow listing, in enumeration order.

    That order is what OpenCV's CAP_DSHOW indices follow — both walk the same
    system device enumerator — which is the only reason a name can be attached
    to an index at all on Windows.
    """
    names: list[str] = []
    in_video = False
    for line in text.splitlines():
        if "DirectShow video devices" in line:
            in_video = True
            continue
        if "DirectShow audio devices" in line:
            break
        if in_video and "Alternative name" not in line:
            m = re.search(r'"([^"]+)"', line)
            if m:
                names.append(m.group(1))
    return names


def _match_transport(name: str, by_name: dict[str, str]) -> str:
    """Transport for a DirectShow name, matched against the PnP list."""
    lc = name.lower()
    if lc in by_name:
        return by_name[lc]
    # PnP and DirectShow names do not always match character for character
    # ("Integrated Camera" vs "Integrated Camera (1bcf:2cd1)") — fall back to a
    # containment match, then to the name itself.
    for pnp_name, transport in by_name.items():
        if pnp_name in lc or lc in pnp_name:
            return transport
    return _name_transport_hint(name) or UNKNOWN


def _windows_devices() -> dict[int, tuple[str, str]]:
    """Name and transport per index, from PnP plus DirectShow ordering.

    Two sources, because each supplies half the answer: PnP knows what every
    camera *is* but not which OpenCV index it will be, and ffmpeg's DirectShow
    listing knows the order but not the bus. Where ffmpeg is unavailable the
    PnP order is used instead — a guess, but a labelled camera the researcher
    can confirm in the preview beats an unnamed one.
    """
    ps = ("Get-PnpDevice -Class Camera,Image -PresentOnly -Status OK | "
          "Select-Object FriendlyName,InstanceId | ConvertTo-Json -Compress")
    pnp = _parse_pnp_cameras(
        _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]))
    by_name = {name.lower(): transport for name, transport in pnp}

    dshow_names = _parse_dshow_names(
        _run(["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
             want_stderr=True))

    if dshow_names:
        return {i: (name, _match_transport(name, by_name))
                for i, name in enumerate(dshow_names)}

    if pnp:
        print("[camera] ffmpeg not available for DirectShow ordering — falling "
              "back to PnP order, so names may not line up with indices "
              "(the preview will show which camera is which)", flush=True)
        return {i: (name, transport) for i, (name, transport) in enumerate(pnp)}

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Linux
# ═══════════════════════════════════════════════════════════════════════════
def _linux_devices() -> dict[int, tuple[str, str]]:
    """Read names from sysfs; the device symlink says whether it is USB."""
    devices: dict[int, tuple[str, str]] = {}
    base = "/sys/class/video4linux"
    try:
        entries = sorted(os.listdir(base), key=lambda s: int(re.sub(r"\D", "", s) or 0))
    except Exception:
        return devices
    for entry in entries:
        m = re.search(r"(\d+)$", entry)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            with open(os.path.join(base, entry, "name")) as fh:
                name = fh.read().strip()
        except Exception:
            name = f"Camera {idx}"
        transport = _name_transport_hint(name) or UNKNOWN
        if transport == UNKNOWN:
            try:
                link = os.readlink(os.path.join(base, entry, "device"))
                if "usb" in link.lower():
                    transport = USB
            except Exception:
                pass
        devices[idx] = (name, transport)
    return devices


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def os_camera_devices() -> dict[int, tuple[str, str]]:
    """{index: (device name, transport)} as reported by the operating system."""
    try:
        if sys.platform == "darwin":
            return _macos_devices()
        if sys.platform == "win32":
            return _windows_devices()
        return _linux_devices()
    except Exception as exc:
        print(f"[camera] device enumeration failed ({exc})", flush=True)
        return {}


def _try_open(index: int, backend: int) -> bool:
    """One attempt at opening *index* and pulling a frame off it."""
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return False
        # The first read after an open regularly fails while the sensor is
        # still spinning up, so give it a few passes before calling it dead.
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.05)
        return False
    except Exception:
        return False
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def probe_indices(backend: int, indices=None, retry=frozenset(),
                  max_index: int = 8) -> list[int]:
    """Indices that open *and* hand back a frame.

    *indices* is what to probe; *retry* is the subset worth a second attempt.
    Both matter for how long this takes. A failed DirectShow open on Windows
    blocks for a second or more, so blind-probing eight indices with retries
    would stall the picker for the best part of a minute — the caller passes a
    range bounded by what the OS actually reported instead.

    An index the OS did list gets two attempts, because a camera released
    moments ago (by the permission warm-up, the previous session, or the
    picker's own preview) is often still busy for a beat, and one failed open
    would report the user's own webcam as broken when it is merely settling.
    Speculative indices get one attempt: nothing is expected there anyway.

    OpenCV prints a wall of driver noise on every failed open, so stderr is
    redirected for the duration — without this a normal scan buries the app's
    own logging.
    """
    if indices is None:
        indices = range(max_index)
    working: list[int] = []
    saved_err = None
    devnull = None
    try:
        saved_err = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except Exception:
        saved_err = None
    try:
        for i in indices:
            attempts = 2 if i in retry else 1
            for attempt in range(attempts):
                if _try_open(i, backend):
                    working.append(i)
                    break
                if attempt + 1 < attempts:
                    time.sleep(0.25)   # let the device settle, then retry
    finally:
        if saved_err is not None:
            try:
                os.dup2(saved_err, 2)
                os.close(saved_err)
            except Exception:
                pass
        if devnull is not None:
            try:
                os.close(devnull)
            except Exception:
                pass
    return working


def scan_cameras(backend: int, max_index: int = 8) -> list[dict]:
    """Every camera the machine has, with a name and a connection type.

    Returns a list of {"index", "name", "transport", "label", "verified"},
    ordered built-in first, then wired, then wireless, then virtual.

    The two sources are unioned rather than intersected, because each catches
    what the other misses:

      - A camera the OS lists but that fails to open is still offered, marked
        unverified. The usual cause is another app holding it (Zoom, Photo
        Booth) or camera permission not yet granted — both fixable by the user,
        and neither a reason to hide the device they are looking for.
      - An index that delivers frames but that the OS never named is offered
        too. Better an unlabelled working camera than a missing one.
    """
    os_devices = os_camera_devices()

    if os_devices:
        # Probe what the OS reported, plus two spares in case its ordering and
        # OpenCV's disagree. Probing the full range regardless is what makes a
        # Windows scan crawl: every empty index costs a blocked DirectShow open.
        ceiling = min(max_index, max(os_devices) + 3)
        candidates = range(ceiling)
    else:
        candidates = range(max_index)
    working = probe_indices(backend, candidates, retry=set(os_devices))

    cameras: list[dict] = []
    for idx in sorted(set(os_devices) | set(working)):
        name, transport = os_devices.get(idx, (f"Camera {idx}", UNKNOWN))
        cameras.append({
            "index":     idx,
            "name":      name,
            "transport": transport,
            "label":     TRANSPORT_LABEL.get(transport, TRANSPORT_LABEL[UNKNOWN]),
            "verified":  idx in working,
        })

    cameras.sort(key=lambda c: (TRANSPORT_ORDER.get(c["transport"], 4), c["index"]))
    for cam in cameras:
        print(f"[camera] index {cam['index']}: {cam['name']!r} — {cam['label']}"
              f"{'' if cam['verified'] else ' (listed but did not open)'}", flush=True)
    if not cameras:
        print("[camera] no camera found at all", flush=True)
    return cameras


if __name__ == "__main__":
    # Standalone diagnostic:  python camera_scanner.py
    _backend = (cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else
                cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    print(f"platform: {sys.platform}")
    print(f"OS device list: {os_camera_devices()}")
    print("\nscanning...")
    for c in scan_cameras(_backend):
        state = "" if c["verified"] else "  ← listed, but would not open"
        print(f"  [{c['index']}] {c['name']}  ({c['label']}){state}")
