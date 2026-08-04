"""mic_dialog.py — pick which microphone the session runs on.

Shown after the participant details and camera are confirmed, and before
calibration, so vocal data is captured through the right input from the very
first second — including a Bluetooth-paired external mic such as a GoPro
used as a wireless lav mic. Reachable again mid-session from the voice
panel's switch button.

Unlike the camera picker, no worker thread is needed to build the list —
PortAudio (via `sounddevice`) already answers instantly. What still needs a
live check is whether the chosen device is actually delivering audio, which
is what the level meter is for: the only reliable way to know a specific
device name refers to the mic that's actually live right now.

Usage:
    dlg = MicDialog(preferred_index=1, parent=win)
    dlg.completed.connect(my_callback)
    dlg.open()   # non-blocking

`completed` emits {"index": int, "name": str, "transport": str,
"samplerate": float} on Use, or None if the researcher cancels.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
                             QPushButton, QRadioButton, QButtonGroup,
                             QScrollArea, QWidget)

from mic_scanner import BLUETOOTH, BUILTIN, USB, list_input_devices
from theme import (ACCENT, BG, C_VOCAL, DANGER, LINE_SOFT, PANEL, PANEL_2,
                   TEXT, TEXT_DIM, TEXT_FAINT, ui_font, panel_qss, sp)
from widgets import BarTrack

# Badge colour per connection type — reuses the same palette role camera_dialog
# uses so the two pickers read as one design.
_BADGE_COLOR = {
    BUILTIN:   "#2872c4",
    USB:       "#2da46a",
    BLUETOOTH: "#6e3fce",
}

# How hard the raw RMS is scaled up to fill the 0-100 meter. This is a "yes,
# audio is arriving" indicator, not a calibrated loudness reading — normal
# speech at a conversational distance should swing it well off zero without
# a shout pinning it at 100.
_LEVEL_GAIN = 350.0


class MicDialog(QDialog):
    """Modal microphone picker: list, choose, live level meter, confirm."""

    completed = Signal(object)   # dict on Use, None on Cancel

    def __init__(self, preferred_index: int | None = None, parent=None):
        super().__init__(parent)
        self._preferred_index = preferred_index
        self._mics: list[dict] = []
        self._selected: dict | None = None
        self._finished = False
        self._preview_stream: sd.InputStream | None = None
        self._level = 0.0
        self._level_lock = threading.Lock()
        self._level_error = False

        self.setWindowTitle("Select Microphone")
        self.setMinimumSize(sp(640), sp(440))
        self.setStyleSheet(f"background: {BG};")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        root = QVBoxLayout(self)
        root.setContentsMargins(sp(32), sp(28), sp(32), sp(24))
        root.setSpacing(sp(16))

        title = QLabel("Which microphone?")
        title.setFont(ui_font(17, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT};")
        sub = QLabel("Every audio input this machine can see — built in, "
                     "plugged in over USB, or paired over Bluetooth (a GoPro "
                     "used as a wireless mic will show up here once paired "
                     "and connected). Pick one and check the level meter "
                     "moves while you speak.")
        sub.setFont(ui_font(10))
        sub.setStyleSheet(f"color: {TEXT_FAINT};")
        sub.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(sp(18))

        # ── Left: the device list ────────────────────────────────────────
        list_card = QFrame()
        list_card.setObjectName("micListCard")
        list_card.setStyleSheet(panel_qss("micListCard"))
        list_v = QVBoxLayout(list_card)
        list_v.setContentsMargins(sp(6), sp(6), sp(6), sp(6))

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("background: transparent;")
        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(sp(8), sp(8), sp(8), sp(8))
        self._list_layout.setSpacing(sp(6))
        self._scroll.setWidget(self._list_host)
        list_v.addWidget(self._scroll)
        body.addWidget(list_card, 3)

        # ── Right: live level meter ────────────────────────────────────────
        lvl_card = QFrame()
        lvl_card.setObjectName("micLvlCard")
        lvl_card.setStyleSheet(panel_qss("micLvlCard"))
        lvl_v = QVBoxLayout(lvl_card)
        lvl_v.setContentsMargins(sp(16), sp(16), sp(16), sp(16))
        lvl_v.setSpacing(sp(10))

        lvl_hdr = QLabel("LEVEL")
        lvl_hdr.setFont(ui_font(9, QFont.Weight.DemiBold))
        lvl_hdr.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 0.5px;")
        lvl_v.addWidget(lvl_hdr)

        self._meter = BarTrack(C_VOCAL)
        lvl_v.addWidget(self._meter)

        self._lvl_msg = QLabel("Select a microphone and speak")
        self._lvl_msg.setFont(ui_font(10))
        self._lvl_msg.setStyleSheet(f"color: {TEXT_FAINT};")
        self._lvl_msg.setWordWrap(True)
        self._lvl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lvl_v.addWidget(self._lvl_msg)
        lvl_v.addStretch()
        body.addWidget(lvl_card, 2)

        root.addLayout(body, 1)

        # ── Footer ───────────────────────────────────────────────────────
        foot = QHBoxLayout()
        cancel = QLabel(f"<a href='#cancel' style='color: {TEXT_FAINT}; "
                        f"text-decoration: underline;'>Cancel</a>")
        cancel.setFont(ui_font(10))
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setOpenExternalLinks(False)
        cancel.linkActivated.connect(lambda _: self._on_cancel())

        self._rescan_btn = QPushButton("Rescan")
        self._rescan_btn.setFont(ui_font(10))
        self._rescan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rescan_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PANEL}; color: {TEXT_DIM};
                border: 1px solid {LINE_SOFT}; border-radius: 6px;
                padding: 9px 18px;
            }}
            QPushButton:hover {{ border-color: {TEXT_FAINT}; color: {TEXT}; }}
        """)
        self._rescan_btn.clicked.connect(self._refresh_list)

        self._use_btn = QPushButton("Use This Microphone")
        self._use_btn.setFont(ui_font(11, QFont.Weight.DemiBold))
        self._use_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._use_btn.clicked.connect(self._on_use)
        self._style_use_btn(False)

        foot.addWidget(cancel)
        foot.addStretch()
        foot.addWidget(self._rescan_btn)
        foot.addSpacing(sp(8))
        foot.addWidget(self._use_btn)
        root.addLayout(foot)

        # Level meter refresh — the callback just stashes a number, this
        # timer is what actually touches Qt widgets, on the GUI thread.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_level)
        self._timer.start(60)

        self._refresh_list()

    # ── Listing ──────────────────────────────────────────────────────────
    def _refresh_list(self):
        self._close_preview()
        self._mics = list_input_devices()
        self._clear_list()

        if not self._mics:
            empty = QLabel("No microphone found.\n\nCheck the OS sees an "
                           "input device, or pair and connect a Bluetooth "
                           "mic, then press Rescan.")
            empty.setFont(ui_font(11))
            empty.setStyleSheet(f"color: {DANGER};")
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._list_layout.addWidget(empty)
            self._list_layout.addStretch()
            return

        old_group = getattr(self, "_group", None)
        if old_group is not None:
            old_group.deleteLater()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for mic in self._mics:
            self._list_layout.addWidget(self._mic_row(mic))
        self._list_layout.addStretch()

        wanted = next((m for m in self._mics
                       if m["index"] == self._preferred_index), None)
        self._select(wanted or self._mics[0])

    def _mic_row(self, mic: dict) -> QWidget:
        row = QFrame()
        row.setObjectName("micRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setStyleSheet(f"""
            QFrame#micRow {{
                background: {PANEL}; border: 1px solid {LINE_SOFT};
                border-radius: 8px;
            }}
            QFrame#micRow:hover {{ border-color: {TEXT_FAINT}; }}
        """)
        h = QHBoxLayout(row)
        h.setContentsMargins(sp(12), sp(10), sp(12), sp(10))
        h.setSpacing(sp(10))

        radio = QRadioButton()
        radio.setCursor(Qt.CursorShape.PointingHandCursor)
        radio.toggled.connect(lambda on, m=mic: self._select(m) if on else None)
        self._group.addButton(radio)
        mic["_radio"] = radio
        h.addWidget(radio)

        text = QVBoxLayout()
        text.setSpacing(sp(2))
        name_bits = [mic["name"]]
        if mic["is_default"]:
            name_bits.append("(system default)")
        name = QLabel("  ".join(name_bits))
        name.setFont(ui_font(11, QFont.Weight.DemiBold))
        name.setStyleSheet(f"color: {TEXT};")
        text.addWidget(name)

        sub_bits = [f"Index {mic['index']}", f"{mic['channels']}ch",
                   f"{mic['samplerate']:.0f} Hz"]
        if not mic["verified"]:
            sub_bits.append("did not pass a settings check — may be in "
                            "exclusive use by another app")
        sub = QLabel("  ·  ".join(sub_bits))
        sub.setFont(ui_font(9))
        sub.setStyleSheet(f"color: {TEXT_FAINT};")
        sub.setWordWrap(True)
        text.addWidget(sub)
        h.addLayout(text, 1)

        badge = QLabel(mic["label"])
        badge.setFont(ui_font(9, QFont.Weight.DemiBold))
        colour = _BADGE_COLOR.get(mic["transport"], TEXT_FAINT)
        badge.setStyleSheet(
            f"color: white; background: {colour}; border-radius: 4px; "
            f"padding: 3px 8px;")
        h.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)

        row.mousePressEvent = lambda _e, r=radio: r.setChecked(True)
        return row

    def _clear_list(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # ── Selection + live level meter ────────────────────────────────────
    def _select(self, mic: dict):
        self._selected = mic
        radio = mic.get("_radio")
        if radio is not None and not radio.isChecked():
            radio.setChecked(True)
        self._style_use_btn(True)
        self._open_preview(mic["index"], mic["samplerate"])

    def _open_preview(self, index: int, samplerate: float):
        self._close_preview()
        self._level_error = False

        def _callback(indata, frames, time_info, status):
            rms = float(np.sqrt(np.mean(np.square(indata[:, 0])))) if frames else 0.0
            with self._level_lock:
                self._level = rms

        try:
            stream = sd.InputStream(device=index, channels=1,
                                    samplerate=samplerate or None,
                                    blocksize=1024, callback=_callback)
            stream.start()
            self._preview_stream = stream
            self._lvl_msg.setText("Speak normally — the bar should move")
        except Exception as exc:                       # noqa: BLE001
            print(f"[mic] preview failed for index {index}: {exc!r}", flush=True)
            self._level_error = True
            self._lvl_msg.setText("Could not open this microphone.\n"
                                  "It may be in use by another app.")

    def _close_preview(self):
        if self._preview_stream is not None:
            try:
                self._preview_stream.stop()
                self._preview_stream.close()
            except Exception:
                pass
            self._preview_stream = None
        with self._level_lock:
            self._level = 0.0
        self._meter.setValue(0)
        if not self._level_error:
            self._lvl_msg.setText("Select a microphone and speak")

    def _tick_level(self):
        if self._preview_stream is None:
            return
        with self._level_lock:
            rms = self._level
        self._meter.setValue(min(100.0, rms * _LEVEL_GAIN))

    # ── Buttons ──────────────────────────────────────────────────────────
    def _style_use_btn(self, enabled: bool):
        self._use_btn.setEnabled(enabled)
        if enabled:
            self._use_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {ACCENT}; color: white;
                    border: 0; border-radius: 6px; padding: 10px 22px;
                }}
                QPushButton:hover {{ background: #1f5fa3; }}
            """)
        else:
            self._use_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {LINE_SOFT}; color: {TEXT_FAINT};
                    border: 0; border-radius: 6px; padding: 10px 22px;
                }}
            """)

    def _on_use(self):
        if self._selected is None:
            return
        self._finished = True
        # Rebuilt rather than passed through: the scan dicts also carry the
        # row's QRadioButton, which must not escape into app state.
        result = {"index":      self._selected["index"],
                  "name":       self._selected["name"],
                  "transport":  self._selected["transport"],
                  "label":      self._selected["label"],
                  "samplerate": self._selected["samplerate"]}
        self._shutdown()
        self.completed.emit(result)
        self.accept()

    def _on_cancel(self):
        self._finished = True
        self._shutdown()
        self.completed.emit(None)
        self.reject()

    def _shutdown(self):
        self._timer.stop()
        self._close_preview()

    def closeEvent(self, event):
        if not self._finished:
            self._finished = True
            self._shutdown()
            self.completed.emit(None)
        super().closeEvent(event)
