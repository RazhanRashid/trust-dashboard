"""camera_dialog.py — pick which camera the session runs on.

Shown after the participant details and before calibration, so the researcher
confirms the right lens is pointed at the participant *before* a 30-second
baseline is recorded through the wrong one. Reachable again mid-session from
the camera panel's switch button.

The scan runs on a worker thread: probing each index means opening a capture
device and waiting for a frame, which takes seconds and would otherwise freeze
the window. While it runs the dialog shows a scanning state.

Selecting a camera opens a live preview of it, which is the only way to be
certain which physical device a name refers to when two are plugged in.

Usage:
    dlg = CameraDialog(backend=cv2.CAP_DSHOW, preferred_index=1, parent=win)
    dlg.completed.connect(my_callback)
    dlg.open()   # non-blocking

`completed` emits {"index": int, "name": str, "transport": str} on Use, or
None if the researcher cancels.
"""

from __future__ import annotations

import cv2
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
                             QPushButton, QRadioButton, QButtonGroup,
                             QScrollArea, QWidget)

from camera_scanner import (BLUETOOTH, BUILTIN, USB, VIRTUAL, WIRELESS,
                            scan_cameras)
from theme import (ACCENT, BG, DANGER, LINE_SOFT, PANEL, PANEL_2, TEXT,
                   TEXT_DIM, TEXT_FAINT, ui_font, panel_qss, sp)

# Badge colour per connection type, reusing the channel hues so the picker
# reads as part of the same app rather than a stock system dialog.
_BADGE_COLOR = {
    BUILTIN:   "#2872c4",
    USB:       "#2da46a",
    BLUETOOTH: "#6e3fce",
    WIRELESS:  "#b88318",
    VIRTUAL:   "#8a91a1",
}

PREVIEW_W, PREVIEW_H = 320, 240


class _ScanWorker(QThread):
    """Runs the camera scan off the UI thread."""
    done = Signal(object)   # list[dict]

    def __init__(self, backend: int, parent=None):
        super().__init__(parent)
        self._backend = backend

    def run(self):
        try:
            self.done.emit(scan_cameras(self._backend))
        except Exception as exc:                       # noqa: BLE001
            print(f"[camera] scan failed: {exc!r}", flush=True)
            self.done.emit([])


class CameraDialog(QDialog):
    """Modal camera picker: scan, choose, preview, confirm."""

    completed = Signal(object)   # dict on Use, None on Cancel

    def __init__(self, backend: int, preferred_index: int | None = None,
                 parent=None):
        super().__init__(parent)
        self._backend = backend
        self._preferred_index = preferred_index
        self._cameras: list[dict] = []
        self._selected: dict | None = None
        self._finished = False
        self._worker: _ScanWorker | None = None
        self._preview_cap = None

        self.setWindowTitle("Select Camera")
        self.setMinimumSize(sp(720), sp(480))
        self.setStyleSheet(f"background: {BG};")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        root = QVBoxLayout(self)
        root.setContentsMargins(sp(32), sp(28), sp(32), sp(24))
        root.setSpacing(sp(16))

        title = QLabel("Which camera?")
        title.setFont(ui_font(17, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT};")
        sub = QLabel("Every camera attached to this machine — built in, plugged "
                     "in over USB, or paired over Bluetooth. Pick the one "
                     "pointed at the participant; the preview confirms it.")
        sub.setFont(ui_font(10))
        sub.setStyleSheet(f"color: {TEXT_FAINT};")
        sub.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(sp(18))

        # ── Left: the device list ────────────────────────────────────────
        list_card = QFrame()
        list_card.setObjectName("camListCard")
        list_card.setStyleSheet(panel_qss("camListCard"))
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

        # ── Right: live preview ──────────────────────────────────────────
        prev_card = QFrame()
        prev_card.setObjectName("camPrevCard")
        prev_card.setStyleSheet(panel_qss("camPrevCard"))
        prev_v = QVBoxLayout(prev_card)
        prev_v.setContentsMargins(sp(12), sp(12), sp(12), sp(12))
        prev_v.setSpacing(sp(8))

        prev_hdr = QLabel("PREVIEW")
        prev_hdr.setFont(ui_font(9, QFont.Weight.DemiBold))
        prev_hdr.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 0.5px;")
        prev_v.addWidget(prev_hdr)

        self._preview = QLabel("Select a camera")
        self._preview.setFixedSize(sp(PREVIEW_W), sp(PREVIEW_H))
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFont(ui_font(10))
        self._preview.setStyleSheet(
            f"background: {PANEL_2}; color: {TEXT_FAINT}; border-radius: 6px;")
        prev_v.addWidget(self._preview)
        prev_v.addStretch()
        body.addWidget(prev_card, 2)

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
            QPushButton:disabled {{ color: {TEXT_FAINT}; }}
        """)
        self._rescan_btn.clicked.connect(self._start_scan)

        self._use_btn = QPushButton("Use This Camera")
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

        # Preview refresh. Deliberately ~15 fps — this is a "which lens is
        # this?" check, not the analysis feed.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_preview)
        self._timer.start(66)

        self._start_scan()

    # ── Scanning ─────────────────────────────────────────────────────────
    def _start_scan(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self._close_preview()
        self._clear_list()
        self._rescan_btn.setEnabled(False)
        self._style_use_btn(False)

        msg = QLabel("Scanning for cameras…")
        msg.setFont(ui_font(11))
        msg.setStyleSheet(f"color: {TEXT_FAINT};")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list_layout.addWidget(msg)
        self._list_layout.addStretch()

        self._worker = _ScanWorker(self._backend, self)
        self._worker.done.connect(self._on_scan_done)
        self._worker.start()

    def _on_scan_done(self, cameras: list):
        self._cameras = cameras or []
        self._rescan_btn.setEnabled(True)
        self._clear_list()

        if not self._cameras:
            empty = QLabel("No camera found.\n\nPlug in a USB webcam, or pair and "
                           "connect a Bluetooth camera, then press Rescan.")
            empty.setFont(ui_font(11))
            empty.setStyleSheet(f"color: {DANGER};")
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._list_layout.addWidget(empty)
            self._list_layout.addStretch()
            return

        # A fresh group each scan; the previous one still references radio
        # buttons that _clear_list has just destroyed.
        old_group = getattr(self, "_group", None)
        if old_group is not None:
            old_group.deleteLater()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for cam in self._cameras:
            self._list_layout.addWidget(self._camera_row(cam))
        self._list_layout.addStretch()

        # Preselect: the remembered camera if it is still here, otherwise the
        # first entry — which the scanner has already sorted built-in first.
        wanted = next((c for c in self._cameras
                       if c["index"] == self._preferred_index), None)
        self._select(wanted or self._cameras[0])

    def _camera_row(self, cam: dict) -> QWidget:
        row = QFrame()
        row.setObjectName("camRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setStyleSheet(f"""
            QFrame#camRow {{
                background: {PANEL}; border: 1px solid {LINE_SOFT};
                border-radius: 8px;
            }}
            QFrame#camRow:hover {{ border-color: {TEXT_FAINT}; }}
        """)
        h = QHBoxLayout(row)
        h.setContentsMargins(sp(12), sp(10), sp(12), sp(10))
        h.setSpacing(sp(10))

        radio = QRadioButton()
        radio.setCursor(Qt.CursorShape.PointingHandCursor)
        radio.toggled.connect(lambda on, c=cam: self._select(c) if on else None)
        self._group.addButton(radio)
        cam["_radio"] = radio
        h.addWidget(radio)

        text = QVBoxLayout()
        text.setSpacing(sp(2))
        name = QLabel(cam["name"])
        name.setFont(ui_font(11, QFont.Weight.DemiBold))
        name.setStyleSheet(f"color: {TEXT};")
        text.addWidget(name)

        sub_bits = [f"Index {cam['index']}"]
        if not cam["verified"]:
            sub_bits.append("listed, but would not open — may be in use by "
                            "another app")
        sub = QLabel("  ·  ".join(sub_bits))
        sub.setFont(ui_font(9))
        sub.setStyleSheet(f"color: {TEXT_FAINT};")
        sub.setWordWrap(True)
        text.addWidget(sub)
        h.addLayout(text, 1)

        badge = QLabel(cam["label"])
        badge.setFont(ui_font(9, QFont.Weight.DemiBold))
        colour = _BADGE_COLOR.get(cam["transport"], TEXT_FAINT)
        badge.setStyleSheet(
            f"color: white; background: {colour}; border-radius: 4px; "
            f"padding: 3px 8px;")
        h.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)

        # Clicking anywhere on the row selects it, not just the small radio.
        row.mousePressEvent = lambda _e, r=radio: r.setChecked(True)
        return row

    def _clear_list(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # ── Selection + preview ──────────────────────────────────────────────
    def _select(self, cam: dict):
        self._selected = cam
        radio = cam.get("_radio")
        if radio is not None and not radio.isChecked():
            radio.setChecked(True)
        self._style_use_btn(True)
        self._open_preview(cam["index"])

    def _open_preview(self, index: int):
        self._close_preview()
        try:
            cap = cv2.VideoCapture(index, self._backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._preview_cap = cap
            else:
                cap.release()
                self._preview.setText("Could not open this camera.\n"
                                      "It may be in use by another app.")
        except Exception as exc:                       # noqa: BLE001
            print(f"[camera] preview failed for index {index}: {exc!r}", flush=True)
            self._preview.setText("Could not open this camera.")

    def _close_preview(self):
        if self._preview_cap is not None:
            try:
                self._preview_cap.release()
            except Exception:
                pass
            self._preview_cap = None
        self._preview.setPixmap(QPixmap())
        self._preview.setText("Select a camera")

    def _tick_preview(self):
        if self._preview_cap is None:
            return
        try:
            ok, frame = self._preview_cap.read()
        except Exception:
            return
        if not ok or frame is None:
            return
        frame = cv2.flip(frame, 1)   # mirror, matching the live dashboard feed
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._preview.setPixmap(pix)

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
        # Deliberately rebuilt rather than passed through: the scan dicts also
        # carry the row's QRadioButton, which must not escape into app state.
        result = {"index":     self._selected["index"],
                  "name":      self._selected["name"],
                  "transport": self._selected["transport"],
                  "label":     self._selected["label"]}
        self._shutdown()
        self.completed.emit(result)
        self.accept()

    def _on_cancel(self):
        self._finished = True
        self._shutdown()
        self.completed.emit(None)
        self.reject()

    def _shutdown(self):
        """Release the preview device and stop the worker.

        Must happen before the caller opens the chosen camera — on Windows a
        DirectShow device cannot be opened twice, so a preview left running
        would make the real capture fail.
        """
        self._timer.stop()
        self._close_preview()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)

    def closeEvent(self, event):
        # Titlebar X or Escape — treat as Cancel so main.py does not proceed.
        if not self._finished:
            self._finished = True
            self._shutdown()
            self.completed.emit(None)
        else:
            self._shutdown()
        super().closeEvent(event)
