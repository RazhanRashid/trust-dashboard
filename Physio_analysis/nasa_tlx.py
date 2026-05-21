"""
NASA Task Load Index dialog — PySide6 / PyQt6 implementation.

Phase 1 : six subscale sliders (0–100).
Phase 2 : 15 pairwise comparisons to derive per-subscale weights.
Result  : weighted TLX score = Σ(weight_i × rating_i) / 15,  range 0–100.

The dialog is non-blocking; it emits the `completed` signal with a result
dict when the participant finishes, or None if it is dismissed early.
"""

import itertools
import time

try:
    from PyQt6.QtWidgets import (QDialog, QWidget, QFrame, QLabel, QPushButton,
        QSlider, QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy, QSpacerItem)
    from PyQt6.QtCore import Qt, QRectF, pyqtSignal as Signal
    from PyQt6.QtGui import QFont, QPainter, QPen, QColor
except ImportError:
    from PySide6.QtWidgets import (QDialog, QWidget, QFrame, QLabel, QPushButton,
        QSlider, QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy, QSpacerItem)
    from PySide6.QtCore import Qt, QRectF, Signal
    from PySide6.QtGui import QFont, QPainter, QPen, QColor

# ── Palette (cool slate, matches theme.py) ────────────────────────────────────
BG      = '#f7f8fa'
SURFACE = '#ffffff'
BORDER  = '#d9dce3'
CORAL   = '#1a8aa3'   # Teal  — used as primary accent inside the dialog
BRONZE  = '#b88318'   # Gold
MAUVE   = '#6e3fce'   # Violet
GRAPE   = '#6D597A'   # kept for progress bar
BLUE    = '#2872c4'   # Accent blue
T1      = '#2d3340'
T2      = '#5f6675'
T3      = '#8a91a1'

SUBSCALES = [
    ("Mental Demand",
     "How much mental and perceptual activity was required?\n"
     "Thinking, deciding, calculating, remembering, looking, searching.",
     "Low", "High"),
    ("Physical Demand",
     "How much physical activity was required?\n"
     "Pushing, pulling, turning, activating, controlling.",
     "Low", "High"),
    ("Temporal Demand",
     "How much time pressure did you feel?\n"
     "Was the pace slow and leisurely, or rapid and frantic?",
     "Low", "High"),
    ("Performance",
     "How successful were you in accomplishing what you were asked to do?\n"
     "Lower rating = better performance.",
     "Perfect", "Failure"),
    ("Effort",
     "How hard did you have to work (mentally and physically)\n"
     "to accomplish your level of performance?",
     "Low", "High"),
    ("Frustration",
     "How insecure, discouraged, irritated, stressed and annoyed were you?",
     "Low", "High"),
]

PAIRS = list(itertools.combinations(range(len(SUBSCALES)), 2))   # 15 pairs


# ── Minimal progress bar widget (avoids importing from main.py) ────────────────

class _BarWidget(QWidget):
    def __init__(self, color=CORAL, parent=None):
        super().__init__(parent)
        self._value = 0
        self._color = color
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v):
        self._value = max(0, min(100, v))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(BORDER))
        p.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        if self._value > 0:
            p.setBrush(QColor(self._color))
            p.drawRoundedRect(QRectF(0, 0, w * self._value / 100, h), 4, 4)
        p.end()


# ── NASA TLX dialog ────────────────────────────────────────────────────────────

class NasaTLX(QDialog):
    """
    Non-blocking NASA TLX dialog.

    Connect to the `completed` signal to receive the result dict (or None
    if the user dismissed the dialog without finishing).
    """

    completed = Signal(object)   # emits dict or None

    def __init__(self, parent=None, trigger_ts=None):
        super().__init__(parent)
        self.setWindowTitle('NASA Task Load Index')
        self.setMinimumSize(640, 600)
        self.setStyleSheet(f'background-color: {BG};')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self._trigger_ts   = trigger_ts or time.time()
        self._ratings      = [50] * len(SUBSCALES)
        self._pair_choices = {}          # pair_index → chosen subscale index
        self._current_pair = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Phase container — we swap phase1 / phase2 widgets by hiding/showing
        self._phase1_widget = self._build_phase1()
        self._phase2_widget = self._build_phase2()
        root.addWidget(self._phase1_widget)
        root.addWidget(self._phase2_widget)
        self._phase2_widget.hide()

    # ── Phase 1: subscale ratings ──────────────────────────────────────────────

    def _build_phase1(self):
        page = QWidget()
        page.setStyleSheet(f'background-color: {BG};')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        hdr = QFrame()
        hdr.setStyleSheet(f'background-color: {CORAL};')
        h = QHBoxLayout(hdr)
        h.setContentsMargins(20, 12, 20, 12)
        title = QLabel('NASA Task Load Index  —  Step 1 of 2: Rate each dimension')
        title.setStyleSheet('color: white; font: bold 11pt "Segoe UI";')
        sub   = QLabel('Workload assessment triggered by sustained high load')
        sub.setStyleSheet('color: #fcd5c8; font: 8pt "Segoe UI";')
        h.addWidget(title)
        h.addStretch()
        h.addWidget(sub)
        layout.addWidget(hdr)

        # Scrollable slider cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        body_w = QWidget()
        body_w.setStyleSheet(f'background-color: {BG};')
        body   = QVBoxLayout(body_w)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(10)

        self._sliders     = []
        self._slider_lbls = []

        slider_style = f"""
            QSlider::groove:horizontal {{
                height: 6px; background: {BORDER}; border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {CORAL}; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {CORAL}; border: none;
                width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
            }}
        """

        for i, (name, desc, lo, hi) in enumerate(SUBSCALES):
            card = QFrame()
            card.setStyleSheet(f'background-color: {SURFACE}; border-radius: 6px;')
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 12, 16, 12)
            cl.setSpacing(6)

            # Name + live value
            top = QHBoxLayout()
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f'color: {CORAL}; font: bold 10pt "Segoe UI";')
            val_lbl  = QLabel('50')
            val_lbl.setStyleSheet(f'color: {T1}; font: bold 10pt "Segoe UI";')
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setFixedWidth(36)
            top.addWidget(name_lbl)
            top.addWidget(val_lbl)
            cl.addLayout(top)

            # Description
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f'color: {T2}; font: 8pt "Segoe UI";')
            desc_lbl.setWordWrap(True)
            cl.addWidget(desc_lbl)

            # Slider row
            sl_row = QHBoxLayout()
            lo_lbl = QLabel(lo)
            lo_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
            lo_lbl.setFixedWidth(52)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setStyleSheet(slider_style)

            hi_lbl = QLabel(hi)
            hi_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
            hi_lbl.setFixedWidth(52)
            hi_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

            # Capture loop variable correctly
            def _make_cb(idx, lbl):
                def _cb(v):
                    self._ratings[idx] = v
                    lbl.setText(str(v))
                return _cb
            slider.valueChanged.connect(_make_cb(i, val_lbl))

            sl_row.addWidget(lo_lbl)
            sl_row.addWidget(slider)
            sl_row.addWidget(hi_lbl)
            cl.addLayout(sl_row)

            body.addWidget(card)
            self._sliders.append(slider)
            self._slider_lbls.append(val_lbl)

        body.addStretch()
        scroll.setWidget(body_w)
        layout.addWidget(scroll, stretch=1)

        # Footer buttons
        foot = QFrame()
        foot.setStyleSheet(f'background-color: {BG};')
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(20, 10, 20, 14)

        cancel = QPushButton('Cancel')
        cancel.setStyleSheet(f"""
            QPushButton {{ background: {BORDER}; color: {T2}; border: none;
                padding: 8px 16px; font: 9pt "Segoe UI"; border-radius: 4px; }}
            QPushButton:hover {{ background: #d4c0ae; }}
        """)
        cancel.clicked.connect(self._dismiss)

        nxt = QPushButton('Next: Pairwise Comparisons  →')
        nxt.setStyleSheet(f"""
            QPushButton {{ background: {CORAL}; color: white; border: none;
                padding: 10px 22px; font: bold 10pt "Segoe UI"; border-radius: 4px; }}
            QPushButton:hover {{ background: {MAUVE}; }}
        """)
        nxt.clicked.connect(self._go_phase2)

        fl.addWidget(cancel)
        fl.addStretch()
        fl.addWidget(nxt)
        layout.addWidget(foot)
        return page

    # ── Phase 2: pairwise comparisons ─────────────────────────────────────────

    def _build_phase2(self):
        page = QWidget()
        page.setStyleSheet(f'background-color: {BG};')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        hdr = QFrame()
        hdr.setStyleSheet(f'background-color: {GRAPE};')
        h = QHBoxLayout(hdr)
        h.setContentsMargins(20, 12, 20, 12)
        title = QLabel('NASA Task Load Index  —  Step 2 of 2: Pairwise Comparisons')
        title.setStyleSheet('color: white; font: bold 11pt "Segoe UI";')
        self._pair_counter_lbl = QLabel('')
        self._pair_counter_lbl.setStyleSheet('color: #c9b8d8; font: 9pt "Segoe UI";')
        h.addWidget(title)
        h.addStretch()
        h.addWidget(self._pair_counter_lbl)
        layout.addWidget(hdr)

        # Body
        body_w = QWidget()
        body_w.setStyleSheet(f'background-color: {BG};')
        body = QVBoxLayout(body_w)
        body.setContentsMargins(40, 20, 40, 20)
        body.setSpacing(0)

        instr = QLabel('For each pair, click the dimension that contributed MORE to your workload.')
        instr.setStyleSheet(f'color: {T2}; font: 9pt "Segoe UI";')
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(instr)
        body.addSpacing(20)

        # Dynamic pair display area
        self._pair_area = QFrame()
        self._pair_area.setStyleSheet(f'background-color: {BG};')
        self._pair_area_layout = QVBoxLayout(self._pair_area)
        self._pair_area_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(self._pair_area, stretch=1)

        # Progress
        body.addSpacing(16)
        prog_lbl = QLabel('Progress')
        prog_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        body.addWidget(prog_lbl)
        self._phase2_bar = _BarWidget(GRAPE)
        body.addWidget(self._phase2_bar)
        layout.addWidget(body_w, stretch=1)

        # Footer
        foot = QFrame()
        foot.setStyleSheet(f'background-color: {BG};')
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(20, 10, 20, 14)
        back = QPushButton('← Back')
        back.setStyleSheet(f"""
            QPushButton {{ background: {BORDER}; color: {T2}; border: none;
                padding: 8px 16px; font: 9pt "Segoe UI"; border-radius: 4px; }}
            QPushButton:hover {{ background: #d4c0ae; }}
        """)
        back.clicked.connect(self._go_phase1)
        fl.addWidget(back)
        fl.addStretch()
        layout.addWidget(foot)
        return page

    # ── Pair renderer ──────────────────────────────────────────────────────────

    def _show_pair(self, idx):
        """Clear the pair area and render pair `idx`, or finish if past the end."""
        # Remove old widgets
        while self._pair_area_layout.count():
            item = self._pair_area_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if idx >= len(PAIRS):
            self._finish()
            return

        self._current_pair = idx
        i, j = PAIRS[idx]
        already = self._pair_choices.get(idx)

        self._pair_counter_lbl.setText(f'Pair {idx + 1} / {len(PAIRS)}')
        self._phase2_bar.set_value(int(100 * idx / len(PAIRS)))

        prompt = QLabel('Which dimension contributed\nmore to your workload?')
        prompt.setStyleSheet(f'color: {T1}; font: 12pt "Segoe UI";')
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pair_area_layout.addWidget(prompt)
        self._pair_area_layout.addSpacing(28)

        row_w = QWidget()
        row_w.setStyleSheet(f'background-color: {BG};')
        row = QHBoxLayout(row_w)
        row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.setSpacing(20)

        for choice_idx in (i, j):
            name = SUBSCALES[choice_idx][0]
            highlighted = (already == choice_idx)
            bg = CORAL if highlighted else SURFACE
            fg = 'white' if highlighted else T1

            btn = QPushButton(name)
            btn.setStyleSheet(f"""
                QPushButton {{ background: {bg}; color: {fg}; border: none;
                    padding: 18px 32px; font: bold 11pt "Segoe UI";
                    border-radius: 6px; min-width: 190px; }}
                QPushButton:hover {{ background: {MAUVE}; color: white; }}
            """)
            # Capture idx and choice_idx correctly
            def _pick(checked=False, _idx=idx, _c=choice_idx):
                self._pair_choices[_idx] = _c
                self._show_pair(_idx + 1)
            btn.clicked.connect(_pick)
            row.addWidget(btn)

            if choice_idx == i:       # Insert 'vs' between the two buttons
                vs = QLabel('vs')
                vs.setStyleSheet(f'color: {T3}; font: italic 11pt "Segoe UI";')
                vs.setAlignment(Qt.AlignmentFlag.AlignCenter)
                vs.setFixedWidth(36)
                row.addWidget(vs)

        self._pair_area_layout.addWidget(row_w)

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _go_phase2(self):
        self._phase1_widget.hide()
        self._phase2_widget.show()
        self._show_pair(0)

    def _go_phase1(self):
        self._phase2_widget.hide()
        self._phase1_widget.show()

    # ── Finish / dismiss ───────────────────────────────────────────────────────

    def _finish(self):
        ratings = self._ratings[:]
        weights = [0] * len(SUBSCALES)
        for chosen in self._pair_choices.values():
            weights[chosen] += 1

        weighted = sum(w * r for w, r in zip(weights, ratings)) / 15.0
        raw      = sum(ratings) / len(ratings)

        result = {
            "timestamp":    self._trigger_ts,
            "completed_at": time.time(),
            "ratings":  {SUBSCALES[i][0]: ratings[i] for i in range(len(SUBSCALES))},
            "weights":  {SUBSCALES[i][0]: weights[i] for i in range(len(SUBSCALES))},
            "weighted_tlx": round(weighted, 1),
            "raw_tlx":      round(raw, 1),
        }
        self.completed.emit(result)
        self.accept()

    def _dismiss(self):
        self.completed.emit(None)
        self.reject()
