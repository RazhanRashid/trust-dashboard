"""demographics_dialog.py — one-time participant info form shown at session start.

Collected once per session, right after the researcher clicks "Start Session"
on the overview screen and before the calibration overlay opens. The values
are attached to the session's stats dict (see `_compute_session_stats` in
main.py) so they travel through to sessions.json and the Excel export
(Summary sheet), letting a researcher later slice results by sex, age, or
cultural background.

Usage:
    dlg = DemographicsDialog(parent=main_window)
    dlg.completed.connect(my_callback)
    dlg.open()   # non-blocking; fires `completed` when the person finishes or cancels

`completed` emits a dict {"sex": str, "age": int, "culture": str} on Continue,
or None if the person cancels (dialog closed via Cancel or the window's X).
"""

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
                              QComboBox, QLineEdit, QSpinBox, QPushButton)

from theme import (BG, PANEL, LINE_SOFT, TEXT, TEXT_DIM, TEXT_FAINT, ACCENT,
                    ui_font, panel_qss, sp)

SEX_OPTIONS = ["Select…", "Female", "Male", "Non-binary", "Prefer not to say"]


class DemographicsDialog(QDialog):
    """Modal form collecting sex, age, and cultural background before calibration."""

    completed = Signal(object)   # emits dict on Continue, None on Cancel

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Participant Information")
        self.setFixedSize(sp(440), sp(420))
        self.setStyleSheet(f"background: {BG};")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._finished = False   # guards against emitting `completed` twice
                                 # (e.g. Cancel button triggers reject(), which
                                 # then triggers closeEvent as well)

        v = QVBoxLayout(self)
        v.setContentsMargins(sp(32), sp(28), sp(32), sp(24))
        v.setSpacing(sp(18))

        title = QLabel("Before we begin")
        title.setFont(ui_font(17, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT};")
        sub = QLabel("A few details about you, recorded once for this session.")
        sub.setFont(ui_font(10))
        sub.setStyleSheet(f"color: {TEXT_FAINT};")
        sub.setWordWrap(True)
        v.addWidget(title)
        v.addWidget(sub)

        card = QFrame()
        card.setObjectName("demoCard")
        card.setStyleSheet(panel_qss("demoCard"))
        cl = QVBoxLayout(card)
        cl.setContentsMargins(sp(20), sp(20), sp(20), sp(20))
        cl.setSpacing(sp(16))

        combo_qss = f"""
            QComboBox {{
                color: {TEXT}; background: {PANEL};
                border: 1px solid {LINE_SOFT}; border-radius: 6px;
                padding: 7px 10px;
            }}
            QComboBox:hover {{ border-color: {TEXT_FAINT}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {PANEL}; color: {TEXT};
                border: 1px solid {LINE_SOFT};
                selection-background-color: {ACCENT};
                outline: none;
            }}
        """
        field_qss = f"""
            QLineEdit, QSpinBox {{
                color: {TEXT}; background: {PANEL};
                border: 1px solid {LINE_SOFT}; border-radius: 6px;
                padding: 7px 10px;
            }}
            QLineEdit:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
        """

        # ── Sex ──────────────────────────────────────────────────────────
        cl.addWidget(self._field_label("Sex"))
        self._sex_combo = QComboBox()
        self._sex_combo.addItems(SEX_OPTIONS)
        self._sex_combo.setFont(ui_font(10))
        self._sex_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sex_combo.setStyleSheet(combo_qss)
        self._sex_combo.currentIndexChanged.connect(self._update_continue_enabled)
        cl.addWidget(self._sex_combo)

        # ── Age ──────────────────────────────────────────────────────────
        cl.addWidget(self._field_label("Age"))
        self._age_spin = QSpinBox()
        self._age_spin.setRange(1, 110)
        self._age_spin.setValue(30)
        self._age_spin.setFont(ui_font(10))
        self._age_spin.setStyleSheet(field_qss)
        cl.addWidget(self._age_spin)

        # ── Culture ──────────────────────────────────────────────────────
        cl.addWidget(self._field_label("Culture / cultural background (optional)"))
        self._culture_edit = QLineEdit()
        self._culture_edit.setPlaceholderText("e.g. British, Han Chinese, Yoruba, Latino…")
        self._culture_edit.setFont(ui_font(10))
        self._culture_edit.setStyleSheet(field_qss)
        cl.addWidget(self._culture_edit)

        v.addWidget(card)
        v.addStretch()

        # ── Footer: Cancel + Continue ───────────────────────────────────
        foot = QHBoxLayout()
        cancel = QLabel("<a href='#cancel' style='color: " + TEXT_FAINT +
                        "; text-decoration: underline;'>Cancel</a>")
        cancel.setFont(ui_font(10))
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setOpenExternalLinks(False)
        cancel.linkActivated.connect(lambda _: self._on_cancel())

        self._continue_btn = QPushButton("Continue to Calibration")
        self._continue_btn.setFont(ui_font(11, QFont.Weight.DemiBold))
        self._continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._continue_btn.clicked.connect(self._on_continue)
        self._style_continue_btn(enabled=False)

        foot.addWidget(cancel)
        foot.addStretch()
        foot.addWidget(self._continue_btn)
        v.addLayout(foot)

        self._update_continue_enabled()

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(ui_font(9, QFont.Weight.DemiBold))
        lbl.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 0.3px;")
        return lbl

    def _style_continue_btn(self, enabled: bool):
        self._continue_btn.setEnabled(enabled)
        if enabled:
            self._continue_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {ACCENT}; color: white;
                    border: 0; border-radius: 6px; padding: 10px 22px;
                }}
                QPushButton:hover {{ background: #1f5fa3; }}
            """)
        else:
            self._continue_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {LINE_SOFT}; color: {TEXT_FAINT};
                    border: 0; border-radius: 6px; padding: 10px 22px;
                }}
            """)

    def _update_continue_enabled(self, *_):
        self._style_continue_btn(enabled=self._sex_combo.currentIndex() > 0)

    def _on_continue(self):
        if self._sex_combo.currentIndex() <= 0:
            return
        self._finished = True
        result = {
            "sex":     self._sex_combo.currentText(),
            "age":     self._age_spin.value(),
            "culture": self._culture_edit.text().strip(),
        }
        self.completed.emit(result)
        self.accept()

    def _on_cancel(self):
        self._finished = True
        self.completed.emit(None)
        self.reject()

    def closeEvent(self, event):
        # Window closed via the titlebar X (or Escape) without submitting —
        # treat exactly like Cancel so main.py doesn't proceed to calibration.
        if not self._finished:
            self._finished = True
            self.completed.emit(None)
        super().closeEvent(event)
