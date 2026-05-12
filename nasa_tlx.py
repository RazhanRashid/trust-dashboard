"""
NASA Task Load Index dialog — Tkinter implementation.

Phase 1: six subscale sliders (0–100).
Phase 2: 15 pairwise comparisons to derive per-subscale weights.
Result:  weighted TLX score = sum(weight_i × rating_i) / 15,  range 0–100.

The dialog is modal; it blocks until the participant completes both phases
or dismisses (escape / close button).  The on_complete callback receives
a result dict or None if the dialog was dismissed early.
"""

import tkinter as tk
from tkinter import font as tkfont
import itertools
import time

# ── Palette (matches main.py) ──────────────────────────────────────────────────
BG      = '#fdf0e6'
SURFACE = '#ffffff'
BORDER  = '#e8d5c4'
CORAL   = '#c94d52'
BRONZE  = '#b87340'
MAUVE   = '#8f3e50'
GRAPE   = '#4e3e5a'
T1      = '#2a1a24'
T2      = '#6b4a5e'
T3      = '#9a7285'

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

# All C(6,2) = 15 pairs
PAIRS = list(itertools.combinations(range(len(SUBSCALES)), 2))


class NasaTLX(tk.Toplevel):
    """
    Modal NASA TLX dialog.

    Parameters
    ----------
    parent      : tk.Tk / tk.Toplevel
    on_complete : callable receiving a result dict (or None on dismiss)
    trigger_ts  : float — Unix timestamp when the workload spike ended
    """

    def __init__(self, parent, on_complete=None, trigger_ts=None):
        super().__init__(parent)
        self.title('NASA Task Load Index')
        self.configure(bg=BG)
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self._dismiss)

        self._on_complete  = on_complete
        self._trigger_ts   = trigger_ts or time.time()
        self._ratings      = [tk.IntVar(value=50) for _ in SUBSCALES]
        self._pair_choices = {}   # pair_index → chosen subscale index
        self._phase        = 1   # 1 = ratings, 2 = pairwise

        self._build_phase1()

        # Centre over parent
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h   = self.winfo_width(), self.winfo_height()
        self.geometry(f'+{px + (pw - w)//2}+{py + (ph - h)//2}')

        self.grab_set()
        self.lift()

    # ── Phase 1: subscale ratings ───────────────────────────────────────────────

    def _build_phase1(self):
        self._clear()

        hdr = tk.Frame(self, bg=CORAL)
        hdr.pack(fill='x')
        tk.Label(hdr, text='NASA Task Load Index  —  Step 1 of 2: Rate each dimension',
                 bg=CORAL, fg=SURFACE, font=('Segoe UI', 11, 'bold'),
                 padx=20, pady=10).pack(side='left')
        tk.Label(hdr, text='Workload assessment triggered by sustained high load',
                 bg=CORAL, fg='#fcd5c8', font=('Segoe UI', 8),
                 padx=20).pack(side='right')

        body = tk.Frame(self, bg=BG, padx=24, pady=16)
        body.pack(fill='both', expand=True)

        for i, (name, desc, lo, hi) in enumerate(SUBSCALES):
            card = tk.Frame(body, bg=SURFACE)
            card.pack(fill='x', pady=6)

            top = tk.Frame(card, bg=SURFACE)
            top.pack(fill='x', padx=16, pady=(12, 4))
            tk.Label(top, text=name, bg=SURFACE, fg=CORAL,
                     font=('Segoe UI', 10, 'bold')).pack(side='left')
            val_lbl = tk.Label(top, text='50', bg=SURFACE, fg=T1,
                                font=('Segoe UI', 10, 'bold'), width=4, anchor='e')
            val_lbl.pack(side='right')

            tk.Label(card, text=desc, bg=SURFACE, fg=T2,
                     font=('Segoe UI', 8), wraplength=560, justify='left',
                     padx=16).pack(anchor='w', pady=(0, 6))

            slider_row = tk.Frame(card, bg=SURFACE)
            slider_row.pack(fill='x', padx=16, pady=(0, 12))
            tk.Label(slider_row, text=lo, bg=SURFACE, fg=T3,
                     font=('Segoe UI', 8), width=8, anchor='w').pack(side='left')

            def _make_update(lbl, var):
                def _cb(*_):
                    lbl.configure(text=str(var.get()))
                return _cb

            sl = tk.Scale(slider_row, variable=self._ratings[i],
                          from_=0, to=100, orient='horizontal',
                          length=420, showvalue=False,
                          bg=SURFACE, fg=T1, troughcolor=BORDER,
                          highlightthickness=0, relief='flat',
                          activebackground=CORAL,
                          command=_make_update(val_lbl, self._ratings[i]))
            sl.pack(side='left', padx=8)
            tk.Label(slider_row, text=hi, bg=SURFACE, fg=T3,
                     font=('Segoe UI', 8), width=8, anchor='e').pack(side='left')

        btn_row = tk.Frame(body, bg=BG)
        btn_row.pack(fill='x', pady=(8, 0))
        tk.Button(btn_row, text='Cancel', command=self._dismiss,
                  bg=BORDER, fg=T2, font=('Segoe UI', 9),
                  relief='flat', padx=14, pady=6).pack(side='left')
        tk.Button(btn_row, text='Next: Pairwise Comparisons  →',
                  command=self._go_phase2,
                  bg=CORAL, fg=SURFACE, font=('Segoe UI', 10, 'bold'),
                  relief='flat', padx=20, pady=8).pack(side='right')

    # ── Phase 2: pairwise comparisons ──────────────────────────────────────────

    def _build_phase2(self):
        self._clear()
        self._current_pair = 0

        hdr = tk.Frame(self, bg=GRAPE)
        hdr.pack(fill='x')
        tk.Label(hdr, text='NASA Task Load Index  —  Step 2 of 2: Pairwise Comparisons',
                 bg=GRAPE, fg=SURFACE, font=('Segoe UI', 11, 'bold'),
                 padx=20, pady=10).pack(side='left')
        self._pair_counter_lbl = tk.Label(hdr, text='', bg=GRAPE, fg='#c9b8d8',
                                           font=('Segoe UI', 9), padx=20)
        self._pair_counter_lbl.pack(side='right')

        body = tk.Frame(self, bg=BG, padx=40, pady=20)
        body.pack(fill='both', expand=True)

        tk.Label(body,
                 text='For each pair, click the dimension that contributed MORE to your workload.',
                 bg=BG, fg=T2, font=('Segoe UI', 9), pady=(0)).pack(pady=(0, 16))

        self._pair_frame = tk.Frame(body, bg=BG)
        self._pair_frame.pack(expand=True)

        # Progress bar
        prog_wrap = tk.Frame(body, bg=BG)
        prog_wrap.pack(fill='x', pady=(20, 0))
        tk.Label(prog_wrap, text='Progress', bg=BG, fg=T3,
                 font=('Segoe UI', 8)).pack(anchor='w')
        self._pair_prog = tk.Canvas(prog_wrap, height=6, bg=BORDER, highlightthickness=0)
        self._pair_prog.pack(fill='x', pady=(3, 0))

        btn_row = tk.Frame(body, bg=BG)
        btn_row.pack(fill='x', pady=(12, 0))
        tk.Button(btn_row, text='← Back', command=self._go_phase1_back,
                  bg=BORDER, fg=T2, font=('Segoe UI', 9),
                  relief='flat', padx=14, pady=6).pack(side='left')

        self._show_pair(0)

    def _show_pair(self, idx):
        for w in self._pair_frame.winfo_children():
            w.destroy()

        if idx >= len(PAIRS):
            self._finish()
            return

        i, j       = PAIRS[idx]
        name_i     = SUBSCALES[i][0]
        name_j     = SUBSCALES[j][0]
        already    = self._pair_choices.get(idx)

        self._pair_counter_lbl.configure(text=f'Pair {idx+1} / {len(PAIRS)}')

        # Progress bar
        self._pair_prog.update_idletasks()
        pw = self._pair_prog.winfo_width()
        if pw > 2:
            self._pair_prog.delete('all')
            self._pair_prog.create_rectangle(0, 0, pw, 6, fill=BORDER, outline='')
            fw = int(pw * idx / len(PAIRS))
            if fw > 0:
                self._pair_prog.create_rectangle(0, 0, fw, 6, fill=GRAPE, outline='')

        def _choose(chosen_idx):
            self._pair_choices[idx] = chosen_idx
            self._show_pair(idx + 1)

        prompt = tk.Label(self._pair_frame,
                          text='Which of these two dimensions\ncontributed more to your workload?',
                          bg=BG, fg=T1, font=('Segoe UI', 12),
                          justify='center')
        prompt.pack(pady=(0, 28))

        btn_row = tk.Frame(self._pair_frame, bg=BG)
        btn_row.pack()

        def _btn(name, chosen_idx):
            highlight = (already == chosen_idx)
            b = tk.Button(btn_row, text=name,
                          command=lambda c=chosen_idx: _choose(c),
                          bg=CORAL if highlight else SURFACE,
                          fg=SURFACE if highlight else T1,
                          font=('Segoe UI', 12, 'bold'),
                          relief='flat', padx=32, pady=18,
                          cursor='hand2', width=20,
                          activebackground='#a83a3f', activeforeground=SURFACE)
            b.bind('<Enter>', lambda e, btn=b, h=highlight: btn.configure(
                bg='#a83a3f' if not h else CORAL, fg=SURFACE))
            b.bind('<Leave>', lambda e, btn=b, h=highlight: btn.configure(
                bg=CORAL if h else SURFACE, fg=SURFACE if h else T1))
            return b

        _btn(name_i, i).pack(side='left', padx=20)
        tk.Label(btn_row, text='vs', bg=BG, fg=T3,
                 font=('Segoe UI', 11, 'italic')).pack(side='left', padx=12)
        _btn(name_j, j).pack(side='left', padx=20)

    # ── Navigation helpers ──────────────────────────────────────────────────────

    def _go_phase2(self):
        self._phase = 2
        self._build_phase2()

    def _go_phase1_back(self):
        self._phase = 1
        self._build_phase1()

    # ── Finish ──────────────────────────────────────────────────────────────────

    def _finish(self):
        ratings = [v.get() for v in self._ratings]
        weights = [0] * len(SUBSCALES)
        for chosen in self._pair_choices.values():
            weights[chosen] += 1

        # Weighted TLX score (Hart & Staveland 1988)
        weighted = sum(w * r for w, r in zip(weights, ratings)) / 15.0
        # Raw TLX (unweighted mean, Hart 2006)
        raw = sum(ratings) / len(ratings)

        result = {
            "timestamp":    self._trigger_ts,
            "completed_at": time.time(),
            "ratings":      {SUBSCALES[i][0]: ratings[i] for i in range(len(SUBSCALES))},
            "weights":      {SUBSCALES[i][0]: weights[i] for i in range(len(SUBSCALES))},
            "weighted_tlx": round(weighted, 1),
            "raw_tlx":      round(raw, 1),
        }
        self._resolve(result)

    def _dismiss(self):
        self._resolve(None)

    def _resolve(self, result):
        self.grab_release()
        self.destroy()
        if self._on_complete:
            self._on_complete(result)

    # ── Utility ────────────────────────────────────────────────────────────────

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()
