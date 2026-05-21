"""
NASA Task Load Index (NASA TLX) dialog.

The NASA TLX is a standard questionnaire used in research and industry to measure
how mentally demanding a task felt. It was developed by NASA in the 1980s and is
still widely used today.

This dialog appears automatically after the app detects a sustained period of
high mental workload (detected by the pupil-dilation workload engine). It asks the
person to rate six dimensions of workload, then runs 15 pairwise comparisons to
figure out which dimensions matter most to *this individual*.

The final score is a single 0–100 number: higher = more demanding.

There are two phases:
  Phase 1 — Six sliders, one for each workload dimension.
  Phase 2 — Fifteen pairs of dimensions; click which one contributed more.

When both phases are complete the dialog fires the `completed` signal with a
result dictionary. If the user cancels, it fires `completed` with None.
"""

import itertools   # Used to generate all possible pairs from the six subscale names
import time        # Used to record when the workload spike happened and when the form was completed

# Import the GUI framework. The app normally uses PySide6, but PyQt6 also works
# (they share almost identical APIs). The try/except tries PyQt6 first; if it is
# not installed it falls back to PySide6 without any change in behaviour.
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

# ── Colour palette ─────────────────────────────────────────────────────────────
# These hex colour codes control the look of the dialog.
# They are defined once here so changing them in one place updates everything.
BG      = '#f7f8fa'   # Page background (very light grey)
SURFACE = '#ffffff'   # Card background (pure white)
BORDER  = '#d9dce3'   # Borders and inactive elements (light grey)
CORAL   = '#1a8aa3'   # Primary teal accent — used for Phase 1 header and selected buttons
BRONZE  = '#b88318'   # Gold — reserved for future use
MAUVE   = '#6e3fce'   # Violet — used as hover colour on primary buttons
GRAPE   = '#6D597A'   # Purple — used for Phase 2 header and progress bar
BLUE    = '#2872c4'   # Accent blue — reserved for future use
T1      = '#2d3340'   # Dark text — headings and important labels
T2      = '#5f6675'   # Medium text — body text and descriptions
T3      = '#8a91a1'   # Light text — hints, low/high endpoint labels

# ── The six NASA TLX subscales ────────────────────────────────────────────────
# Each entry is a tuple of: (short name, description, low-end label, high-end label)
# These are displayed as slider cards in Phase 1.
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

# Generate all 15 unique pairs from the 6 subscales (6 choose 2 = 15).
# For example: (0,1) means "Mental Demand vs Physical Demand".
# These are shown one at a time in Phase 2 so the person can pick which felt more demanding.
PAIRS = list(itertools.combinations(range(len(SUBSCALES)), 2))


# ── Small progress-bar widget used inside the Phase 2 page ────────────────────
# This is a lightweight custom widget because the built-in QProgressBar doesn't
# match the visual style of the rest of the dialog.

class _BarWidget(QWidget):
    def __init__(self, color=CORAL, parent=None):
        super().__init__(parent)
        self._value = 0            # Current fill percentage (0–100)
        self._color = color        # Fill colour (passed in from the parent)
        self.setFixedHeight(8)     # Always 8 pixels tall
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v):
        # Update the percentage and trigger a repaint.
        self._value = max(0, min(100, v))   # Clamp to valid range
        self.update()                        # Ask Qt to call paintEvent on the next frame

    def paintEvent(self, event):
        # Qt calls this automatically whenever the widget needs to be redrawn.
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)   # Smooth rounded corners
        w, h = self.width(), self.height()

        # Draw the grey background track.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(BORDER))
        p.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)

        # Draw the coloured fill on top, proportional to the current value.
        if self._value > 0:
            p.setBrush(QColor(self._color))
            p.drawRoundedRect(QRectF(0, 0, w * self._value / 100, h), 4, 4)
        p.end()


# ── Main NASA TLX dialog ───────────────────────────────────────────────────────

class NasaTLX(QDialog):
    """
    The NASA TLX questionnaire dialog.

    Usage:
        dlg = NasaTLX(parent=main_window, trigger_ts=time.time())
        dlg.completed.connect(my_callback_function)
        dlg.show()

    When the person finishes both phases, `completed` fires with a dictionary:
        {
          "ratings":       {subscale_name: 0–100, ...},   # Raw slider values
          "weights":       {subscale_name: 0–5, ...},     # How many times each was chosen in pairwise
          "weighted_tlx":  0–100,                         # Final weighted NASA TLX score
          "raw_tlx":       0–100,                         # Simple unweighted average of the six ratings
          "timestamp":     float,                         # When the workload spike was triggered
          "completed_at":  float,                         # When the person finished the form
        }

    If the person cancels, `completed` fires with None.
    """

    # Qt signal: emits either a result dict or None when the dialog closes.
    # Other parts of the app connect to this signal to receive the questionnaire result.
    completed = Signal(object)

    def __init__(self, parent=None, trigger_ts=None):
        super().__init__(parent)
        self.setWindowTitle('NASA Task Load Index')
        self.setMinimumSize(640, 600)
        self.setStyleSheet(f'background-color: {BG};')

        # ApplicationModal means the user must finish or cancel this dialog
        # before they can interact with the main dashboard window.
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self._trigger_ts   = trigger_ts or time.time()   # When the workload spike was detected
        self._ratings      = [50] * len(SUBSCALES)       # Starting slider values (all at 50 = middle)
        self._pair_choices = {}    # Stores the person's choice for each pair: pair_index → chosen subscale index
        self._current_pair = 0     # Which of the 15 pairs we are currently showing

        # Build the two phase pages.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._phase1_widget = self._build_phase1()   # The six-slider rating page
        self._phase2_widget = self._build_phase2()   # The pairwise comparison page
        root.addWidget(self._phase1_widget)
        root.addWidget(self._phase2_widget)
        self._phase2_widget.hide()   # Start on Phase 1; Phase 2 is hidden until the person clicks Next

    # ── Phase 1: six subscale rating sliders ──────────────────────────────────

    def _build_phase1(self):
        # Build the entire Phase 1 page (six slider cards + header + footer).
        page = QWidget()
        page.setStyleSheet(f'background-color: {BG};')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ─────────────────────────────────────────────────────
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

        # ── Scrollable area containing all six slider cards ─────────────
        # QScrollArea allows the cards to scroll if the window is too short to show them all.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        body_w = QWidget()
        body_w.setStyleSheet(f'background-color: {BG};')
        body   = QVBoxLayout(body_w)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(10)

        self._sliders     = []   # Stores the QSlider objects so we can read their values later
        self._slider_lbls = []   # Stores the live value labels so they update as the slider moves

        # CSS style applied to every slider so it matches the dashboard's colour scheme.
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

        # Create one card for each of the six subscales.
        for i, (name, desc, lo, hi) in enumerate(SUBSCALES):
            # A white card containing the subscale name, description, and slider.
            card = QFrame()
            card.setStyleSheet(f'background-color: {SURFACE}; border-radius: 6px;')
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 12, 16, 12)
            cl.setSpacing(6)

            # Top row: subscale name on the left, current numeric value on the right.
            top = QHBoxLayout()
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f'color: {CORAL}; font: bold 10pt "Segoe UI";')
            val_lbl  = QLabel('50')   # Shows the current slider position as a number
            val_lbl.setStyleSheet(f'color: {T1}; font: bold 10pt "Segoe UI";')
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setFixedWidth(36)
            top.addWidget(name_lbl)
            top.addWidget(val_lbl)
            cl.addLayout(top)

            # Description text explaining what the person should think about for this dimension.
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f'color: {T2}; font: 8pt "Segoe UI";')
            desc_lbl.setWordWrap(True)
            cl.addWidget(desc_lbl)

            # Slider row: low label | slider | high label
            sl_row = QHBoxLayout()
            lo_lbl = QLabel(lo)    # e.g. "Low" or "Perfect"
            lo_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
            lo_lbl.setFixedWidth(52)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)    # 0 = lowest workload, 100 = highest
            slider.setValue(50)        # Default to the midpoint
            slider.setStyleSheet(slider_style)

            hi_lbl = QLabel(hi)    # e.g. "High" or "Failure"
            hi_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
            hi_lbl.setFixedWidth(52)
            hi_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

            # Connect the slider's value-changed signal to a callback that:
            #   1. Saves the new value into self._ratings[i]
            #   2. Updates the numeric label next to the subscale name
            # The _make_cb helper is needed to correctly capture the loop variable i;
            # without it, all six sliders would update the same (last) index.
            def _make_cb(idx, lbl):
                def _cb(v):
                    self._ratings[idx] = v   # Save the rating for this subscale
                    lbl.setText(str(v))      # Update the displayed number
                return _cb
            slider.valueChanged.connect(_make_cb(i, val_lbl))

            sl_row.addWidget(lo_lbl)
            sl_row.addWidget(slider)
            sl_row.addWidget(hi_lbl)
            cl.addLayout(sl_row)

            body.addWidget(card)
            self._sliders.append(slider)
            self._slider_lbls.append(val_lbl)

        body.addStretch()   # Push all cards to the top; empty space fills the bottom
        scroll.setWidget(body_w)
        layout.addWidget(scroll, stretch=1)

        # ── Footer: Cancel and Next buttons ────────────────────────────────
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
        cancel.clicked.connect(self._dismiss)   # Clicking Cancel fires _dismiss which emits completed(None)

        nxt = QPushButton('Next: Pairwise Comparisons  →')
        nxt.setStyleSheet(f"""
            QPushButton {{ background: {CORAL}; color: white; border: none;
                padding: 10px 22px; font: bold 10pt "Segoe UI"; border-radius: 4px; }}
            QPushButton:hover {{ background: {MAUVE}; }}
        """)
        nxt.clicked.connect(self._go_phase2)   # Clicking Next switches to the pairwise comparison page

        fl.addWidget(cancel)
        fl.addStretch()
        fl.addWidget(nxt)
        layout.addWidget(foot)
        return page

    # ── Phase 2: pairwise comparisons ─────────────────────────────────────────

    def _build_phase2(self):
        # Build the Phase 2 page (shown after the person clicks Next on Phase 1).
        # The actual pair content is rendered dynamically by _show_pair().
        page = QWidget()
        page.setStyleSheet(f'background-color: {BG};')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ─────────────────────────────────────────────────────
        hdr = QFrame()
        hdr.setStyleSheet(f'background-color: {GRAPE};')
        h = QHBoxLayout(hdr)
        h.setContentsMargins(20, 12, 20, 12)
        title = QLabel('NASA Task Load Index  —  Step 2 of 2: Pairwise Comparisons')
        title.setStyleSheet('color: white; font: bold 11pt "Segoe UI";')
        self._pair_counter_lbl = QLabel('')   # Shows "Pair 3 / 15" and updates as the person progresses
        self._pair_counter_lbl.setStyleSheet('color: #c9b8d8; font: 9pt "Segoe UI";')
        h.addWidget(title)
        h.addStretch()
        h.addWidget(self._pair_counter_lbl)
        layout.addWidget(hdr)

        # ── Body area ──────────────────────────────────────────────────────
        body_w = QWidget()
        body_w.setStyleSheet(f'background-color: {BG};')
        body = QVBoxLayout(body_w)
        body.setContentsMargins(40, 20, 40, 20)
        body.setSpacing(0)

        # Brief instruction shown above the two choice buttons.
        instr = QLabel('For each pair, click the dimension that contributed MORE to your workload.')
        instr.setStyleSheet(f'color: {T2}; font: 9pt "Segoe UI";')
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(instr)
        body.addSpacing(20)

        # This frame is cleared and rebuilt every time a new pair is shown.
        # _show_pair() replaces its contents with the current pair's buttons.
        self._pair_area = QFrame()
        self._pair_area.setStyleSheet(f'background-color: {BG};')
        self._pair_area_layout = QVBoxLayout(self._pair_area)
        self._pair_area_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(self._pair_area, stretch=1)

        # Progress bar showing how many of the 15 pairs have been answered.
        body.addSpacing(16)
        prog_lbl = QLabel('Progress')
        prog_lbl.setStyleSheet(f'color: {T3}; font: 8pt "Segoe UI";')
        body.addWidget(prog_lbl)
        self._phase2_bar = _BarWidget(GRAPE)   # The purple fill bar
        body.addWidget(self._phase2_bar)
        layout.addWidget(body_w, stretch=1)

        # ── Footer: Back button ─────────────────────────────────────────────
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
        back.clicked.connect(self._go_phase1)   # Return to the slider ratings page
        fl.addWidget(back)
        fl.addStretch()
        layout.addWidget(foot)
        return page

    # ── Pair renderer ──────────────────────────────────────────────────────────

    def _show_pair(self, idx):
        """
        Display pairwise comparison number `idx` (0-indexed).
        Clears any existing buttons and builds two new ones for the current pair.
        If idx == 15 (all pairs answered), calls _finish() to compute the score.
        """
        # Remove all widgets currently inside the pair area (the old pair's buttons).
        while self._pair_area_layout.count():
            item = self._pair_area_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()   # Schedule the old widget for deletion

        # If we have gone past the last pair, all 15 comparisons are done.
        if idx >= len(PAIRS):
            self._finish()
            return

        self._current_pair = idx
        i, j = PAIRS[idx]   # The indices of the two subscales being compared this round
        already = self._pair_choices.get(idx)   # If the person already chose an answer for this pair (they went Back), highlight it

        # Update the "Pair X / 15" counter and the progress bar.
        self._pair_counter_lbl.setText(f'Pair {idx + 1} / {len(PAIRS)}')
        self._phase2_bar.set_value(int(100 * idx / len(PAIRS)))

        # The prompt question at the top of the pair area.
        prompt = QLabel('Which dimension contributed\nmore to your workload?')
        prompt.setStyleSheet(f'color: {T1}; font: 12pt "Segoe UI";')
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pair_area_layout.addWidget(prompt)
        self._pair_area_layout.addSpacing(28)

        # A row containing the two choice buttons side by side.
        row_w = QWidget()
        row_w.setStyleSheet(f'background-color: {BG};')
        row = QHBoxLayout(row_w)
        row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.setSpacing(20)

        for choice_idx in (i, j):
            name = SUBSCALES[choice_idx][0]   # The subscale name for this button

            # If the person previously chose this option, highlight it in teal.
            # Otherwise show it in white.
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

            # When the person clicks this button:
            #   1. Save their choice for this pair.
            #   2. Automatically advance to the next pair.
            # The default-argument trick (_idx=idx, _c=choice_idx) captures the
            # current loop values so all buttons don't end up referencing the last iteration.
            def _pick(checked=False, _idx=idx, _c=choice_idx):
                self._pair_choices[_idx] = _c    # Record which subscale was chosen
                self._show_pair(_idx + 1)        # Move to the next pair
            btn.clicked.connect(_pick)
            row.addWidget(btn)

            # Insert a small "vs" label between the two buttons (only after the first).
            if choice_idx == i:
                vs = QLabel('vs')
                vs.setStyleSheet(f'color: {T3}; font: italic 11pt "Segoe UI";')
                vs.setAlignment(Qt.AlignmentFlag.AlignCenter)
                vs.setFixedWidth(36)
                row.addWidget(vs)

        self._pair_area_layout.addWidget(row_w)

    # ── Page navigation ────────────────────────────────────────────────────────

    def _go_phase2(self):
        # Switch from Phase 1 (sliders) to Phase 2 (pairwise comparisons).
        self._phase1_widget.hide()
        self._phase2_widget.show()
        self._show_pair(0)   # Start from the very first pair

    def _go_phase1(self):
        # Switch back to Phase 1 (sliders). Ratings are preserved.
        self._phase2_widget.hide()
        self._phase1_widget.show()

    # ── Completion and cancellation ────────────────────────────────────────────

    def _finish(self):
        # Called when all 15 pairwise comparisons have been answered.
        # Computes the final weighted NASA TLX score and fires the `completed` signal.

        ratings = self._ratings[:]   # Copy the six slider values (0–100 each)

        # Tally how many times each subscale was chosen in the pairwise comparisons.
        # A subscale chosen 5 times out of 15 gets a weight of 5/15 = 0.333.
        weights = [0] * len(SUBSCALES)
        for chosen in self._pair_choices.values():
            weights[chosen] += 1   # Increment the tally for the chosen subscale

        # Weighted TLX score: each subscale's rating is multiplied by its weight (number of wins).
        # Dividing by 15 (total number of pairwise comparisons) normalises to 0–100.
        weighted = sum(w * r for w, r in zip(weights, ratings)) / 15.0

        # Raw (unweighted) TLX: just the simple average of all six ratings.
        raw      = sum(ratings) / len(ratings)

        # Bundle everything into a result dictionary.
        result = {
            "timestamp":    self._trigger_ts,            # When the workload spike triggered this dialog
            "completed_at": time.time(),                  # When the person finished answering
            "ratings":  {SUBSCALES[i][0]: ratings[i] for i in range(len(SUBSCALES))},   # Raw slider values by name
            "weights":  {SUBSCALES[i][0]: weights[i] for i in range(len(SUBSCALES))},   # Pairwise win counts by name
            "weighted_tlx": round(weighted, 1),          # Final weighted score (main result)
            "raw_tlx":      round(raw, 1),               # Simple average (for comparison)
        }

        # Fire the `completed` signal so connected listeners (e.g. the main window)
        # receive the result and can save it or display it.
        self.completed.emit(result)
        self.accept()   # Close the dialog normally

    def _dismiss(self):
        # Called when the person clicks Cancel before finishing.
        # Fires `completed` with None to indicate no result was collected.
        self.completed.emit(None)
        self.reject()   # Close the dialog with a "rejected" status
