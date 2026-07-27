SCORE_VERSION = "2.0.0"

SCORE_CONFIG = {
    "version":           SCORE_VERSION,
    "alpha":             0.2,
    "channels":          ["facial", "vocal", "gaze", "hrv"],
    # All four channels count equally toward the total — 25% each.
    "channel_active":    {"facial": True, "vocal": True, "gaze": True, "hrv": True},
    "emotion_weights":   {
        "happy": 30, "neutral": 10, "surprised": 4,
        "fearful": -30, "angry": -35, "disgusted": -30,
        "sad": -18, "contempt": -40,
    },
    "duchenne_weight":   20,
    "au_weights":        {"AU04": -12, "AU20": -10, "AU14": -8},
    "au_interaction_pen": 15,
    "pitch_stab_scale":  38,
    "tremor_weight":     32,
    "alpha_ratio_center": -10.0,
    "alpha_ratio_scale":  0.2,
    "alpha_ratio_clamp":  [-4.0, 3.0],
    "sf_baseline":        0.005,
    "sf_scale":           200.0,
    "sf_clamp":           [-5.0, 1.0],
    "ear_thresholds":    [0.14, 0.20, 0.28],
    "ear_pts":           [-28, -12, 10],
    "blink_thresholds":  [32, 23, 20, 10],
    "blink_pts":         [-22, -10, 8],
    "gaze_dev_weight":   18,

    # Population reference baseline.
    #
    # Every threshold above was written against a "typical" resting person, and
    # these are the resting readings that person was assumed to produce. They
    # are not the scoring targets — they are the values the thresholds are
    # anchored to. Calibration replaces them with the readings actually
    # measured from this user, so "narrowed eyes", "elevated blinking", or "an
    # unsteady voice" all mean *for this person* rather than for an average one.
    #
    # Chosen so that leaving them in place (calibration skipped, or no face or
    # voice captured during the window) reproduces the old fixed-threshold
    # scoring exactly.
    "reference_baseline": {
        "eye_ar":          0.27,   # Eye Aspect Ratio of a normally open eye
        "blink_rate":      15.0,   # Blinks per minute at rest
        "gaze_deviation":  0.0,    # Perfectly centred gaze
        "pitch_stability": 0.5,    # Midpoint of the 0-1 stability range
        "energy_level":    0.5,    # Middle of the comfortable speaking band
        "tremor_index":    0.0,    # No tremor
        "alpha_ratio":     -10.0,  # dB, typical relaxed conversational voice
        "spectral_flux":   0.005,  # Calm, steady spectrum
        "duchenne":        0.0,    # Not smiling
        # A resting face registering zero on every emotion and Action Unit.
        "expressions":     {"happy": 0.0, "neutral": 0.0, "surprised": 0.0,
                            "fearful": 0.0, "angry": 0.0, "disgusted": 0.0,
                            "sad": 0.0, "contempt": 0.0},
        "aus":             {"AU04": 0.0, "AU20": 0.0, "AU14": 0.0, "AU07": 0.0},
    },
    # How far a measured baseline may stretch or shrink the fixed thresholds.
    # A bad calibration window (a yawn, a half-covered camera, someone leaning
    # out of frame) should not be able to rescale a whole channel.
    "baseline_scale_clamp": [0.5, 2.0],
}


def _reference_baseline() -> dict:
    """A fresh, independently mutable copy of the population reference values."""
    return {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in SCORE_CONFIG["reference_baseline"].items()}

# TrustEngine is the "brain" of the dashboard.
# It takes live readings from the face camera, microphone, and heart-rate sensor
# and combines them into a single trust score between 0 and 100.
# 0 means very stressed or guarded; 100 means calm, open, and engaged.
class TrustEngine:
    def __init__(self):
        # 'smoothed' stores a running average for each sensor channel plus the overall total.
        # All channels start at 50, meaning "neutral / not enough data yet".
        # The keys are: total (the final score), facial (face), vocal (voice),
        # gaze (eyes), and hrv (heart rate).
        self.smoothed = {"total": 50.0, "facial": 50.0, "vocal": 50.0, "gaze": 50.0, "hrv": 50.0}

        # alpha controls how quickly the score reacts to new data.
        # Think of it as a mixing knob:
        #   0.2 means each new reading gets 20% weight, and the recent history gets 80%.
        # A smaller value makes the score change more slowly and smoothly.
        # A larger value makes it jump more quickly but also more erratically.
        self.alpha = SCORE_CONFIG["alpha"]
        self._active: dict[str, bool] = dict(SCORE_CONFIG["channel_active"])
        # Stores the raw sensor values from the previous camera/audio frame.
        # This lets the engine detect *changes* — for example, "the brow just furrowed"
        # — and tell the UI which signals caused the score to move.
        self._prev_inputs: dict = {}

        # ── Per-user calibration: two layers ──────────────────────────────── #
        #
        # input_baseline holds this user's *resting sensor readings* — how wide
        # their eyes normally sit, how often they normally blink, how steady
        # their voice normally is, what their resting face registers as. The
        # scorers below measure every signal against these, so the thresholds
        # move with the person instead of assuming an average one.
        #
        # baseline holds the resting 0–100 score each channel then produces, so
        # the final score can be shifted to put this user's resting state at
        # exactly 50 — the point every trust movement is read against.
        #
        # Both start at the population defaults, which reproduce the old fixed
        # scoring. apply_calibration() replaces them from the 30-second window.
        self.input_baseline: dict = _reference_baseline()
        self.baseline: dict[str, float] = {"facial": 50.0, "vocal": 50.0, "gaze": 50.0, "hrv": 50.0}
        self._prev_smoothed: dict[str, float] = {k: 50.0 for k in ("total", "facial", "vocal", "gaze", "hrv")}

    def update(self, face_data: dict | None, vocal_data: dict | None,
               hrv_score: int = 65) -> dict:
        # Ask each sub-scorer to convert raw sensor readings into a 0–100 number.
        # Each sub-scorer also returns a short list of the top contributors
        # (the signals that changed the score the most this frame).
        facial, facial_contribs = self._facial_score(face_data)   # Score from facial expressions
        vocal,  vocal_contribs  = self._vocal_score(vocal_data)   # Score from voice characteristics
        gaze,   gaze_contribs   = self._gaze_score(face_data)     # Score from eye movement and blinking
        # Heart-rate variability, already scored from R-R intervals by HRVAnalyzer.
        hrv    = float(hrv_score) if hrv_score is not None else 65.0

        # Combine channel scores using only active channels (equal weights among
        # active). With all four on, that is the documented 25% each.
        _raw = {"facial": facial, "vocal": vocal, "gaze": gaze, "hrv": hrv}
        _active_chs = [ch for ch in ("facial", "vocal", "gaze", "hrv") if self._active.get(ch)]
        if _active_chs:
            _w = 1.0 / len(_active_chs)
            total = sum(_raw[ch] * _w for ch in _active_chs)
        else:
            total = 50.0

        # Apply exponential smoothing to every channel so scores drift gradually
        # rather than snapping to a new value instantly.
        # Formula: new_smoothed = alpha × brand_new_value + (1 − alpha) × previous_smoothed
        for k, v in [("facial", facial), ("vocal", vocal), ("gaze", gaze),
                     ("hrv", hrv), ("total", total)]:
            self.smoothed[k] = self.alpha * v + (1 - self.alpha) * self.smoothed[k]

        # Apply per-user baseline shift: user's natural resting state maps to 50
        scores = {}
        for k in self.smoothed:
            if k == "total":
                _shift_chs = _active_chs if _active_chs else ["facial", "vocal", "gaze", "hrv"]
                avg_shift = sum(50.0 - self.baseline.get(ch, 50.0) for ch in _shift_chs) / len(_shift_chs)
                scores[k] = round(max(0, min(100, self.smoothed[k] + avg_shift)))
            else:
                scores[k] = round(max(0, min(100, self.smoothed[k] + 50.0 - self.baseline.get(k, 50.0))))

        # Attach the "why did the score change?" explanation lists so the UI
        # can show tooltips like "brow furrow: −8 pts".
        scores["contributions"] = {
            "facial": facial_contribs,
            "vocal":  vocal_contribs,
            "gaze":   gaze_contribs,
        }

        # Rate-of-change derivatives
        dscores = {k: round(self.smoothed[k] - self._prev_smoothed[k], 3)
                   for k in ("total", "facial", "vocal", "gaze", "hrv")}
        self._prev_smoothed = dict(self.smoothed)
        scores["dscores"] = dscores

        # Active channels
        scores["active_channels"] = list(_active_chs)

        return scores   # Hand the complete score dictionary back to whoever called update()

    def _facial_score(self, fd: dict | None):
        # If no face is visible in the camera frame, return a neutral 50 with nothing to explain.
        if not fd or not fd.get("detected"):
            return 50.0, []

        e   = fd["expressions"]   # Dictionary of emotion intensities, each 0.0 (absent) → 1.0 (full)
        aus = fd.get("aus", {})   # Dictionary of Action Unit intensities (individual muscle movements)
        s   = 50.0                # Start at neutral before applying emotion-based adjustments
        prev = self._prev_inputs  # Shortcut to the stored previous-frame values

        # This user's resting face, measured during calibration. Nobody's face
        # reads as a clean zero on every emotion — plenty of relaxed faces
        # register a permanent trace of sadness or a slightly furrowed brow.
        # Scoring the *difference* from their own resting face means a naturally
        # stern-looking person is not penalised for sitting still.
        base_e  = self.input_baseline["expressions"]
        base_au = self.input_baseline["aus"]

        contribs = []   # Will be filled with the signals that moved the score most this frame

        # Go through each emotion and apply its weight to the score.
        # Positive weight = that emotion raises trust. Negative = it lowers trust.
        # For example: becoming a full point happier than resting adds 30 points.
        # Becoming fully fearful subtracts 30 points.
        # Emotion definitions follow the Ekman AU classification table.
        for key, weight, label in [
            ("happy",     30,  "happy"),       # AU6+12: genuine smiling raises trust
            ("neutral",   10,  "neutral"),     # Calm neutral face is a mild positive signal
            ("surprised",  4,  "surprised"),   # AU1+2+5+26: surprise is roughly neutral (small positive)
            ("fearful",  -30,  "fearful"),     # AU1+2+4+5+7+20+26: visible fear strongly lowers trust
            ("angry",    -35,  "angry"),       # AU4+5+7+23: anger is a strong trust reducer
            ("disgusted",-30,  "disgusted"),   # AU9+15+16: disgust strongly lowers trust
            ("sad",      -18,  "sad"),         # AU1+4+15: sadness moderately lowers trust
            # Contempt is the single most trust-destructive expression — it signals
            # active disrespect or disdain toward the other person. Weighted higher
            # than anger because it is more targeted and less likely to be transient.
            ("contempt", -40,  "contempt"),    # R12A+R14A: unilateral sneer, strongest negative signal
        ]:
            cur = e.get(key, 0)                       # This frame's emotion intensity (0.0–1.0)
            rel = cur - base_e.get(key, 0.0)          # How far above (or below) this user's resting face
            prv = prev.get(f"face_{key}", cur)        # Previous frame's intensity (defaults to current on the very first frame)
            delta_pts = (cur - prv) * abs(weight)     # How many score points did this emotion add or remove compared to last frame?
            if abs(delta_pts) >= 1.0:                 # Only log it as a contributor if the change was at least 1 point (ignore tiny noise)
                contribs.append((label, prv, cur, round(delta_pts, 1)))
            s += rel * weight                         # Apply relative intensity × weight to the running score
            prev[f"face_{key}"] = cur                 # Save this frame's value so next frame can compare

        # Duchenne smile check: a *genuine* smile activates both the cheek muscles (AU06)
        # and the lip corner muscles (AU12) together. A forced smile usually only shows AU12.
        # These are combined into a single Duchenne score (0 = fake/absent, 1 = genuine).
        # Genuine smiles are a strong trust signal, worth up to +20 points.
        duchenne = fd.get("duchenne", 0)
        rel_d = duchenne - self.input_baseline["duchenne"]   # Relative to how much they were smiling at rest
        prv_d = prev.get("face_duchenne", duchenne)   # Previous frame's Duchenne smile value
        delta_d = (duchenne - prv_d) * 20             # Score change from the smile becoming more/less genuine
        if abs(delta_d) >= 1.0:
            contribs.append(("duchenne smile", prv_d, duchenne, round(delta_d, 1)))
        s += rel_d * 20              # Add up to 20 points for a full genuine smile
        prev["face_duchenne"] = duchenne

        # Action Units (AUs) are specific facial muscle movements, approximated here
        # from MediaPipe blendshapes.
        # These three are tension and stress indicators:
        #   AU04 = brow furrow (the frown between the eyebrows)
        #   AU20 = lip stretcher (lips pulled sideways under tension)
        #   AU14 = dimpler (asymmetric lip press, often a suppressed expression)
        for au, weight, label in [
            ("AU04", -12, "AU04 brow"),    # Furrowed brow subtracts up to 12 points
            ("AU20", -10, "AU20 lip"),     # Lip tension subtracts up to 10 points
            ("AU14",  -8, "AU14 dimple"),  # Suppressed/forced expression subtracts up to 8 points
        ]:
            cur = aus.get(au, 0)
            rel = cur - base_au.get(au, 0.0)   # Relative to this user's resting muscle tone
            prv = prev.get(f"face_{au}", cur)
            delta_pts = (cur - prv) * abs(weight)
            if abs(delta_pts) >= 1.0:
                contribs.append((label, prv, cur, round(delta_pts, 1)))
            s += rel * weight
            prev[f"face_{au}"] = cur

        # Interaction penalty: when the upper eyelid is tightened (AU07) AND the brow is
        # simultaneously furrowed (AU04), the combined expression looks like an intense
        # hostile stare — much more threatening than either muscle alone.
        # Multiplying them means the penalty is zero unless both are active at the same time.
        # Both sides use the amount *above* resting and are floored at zero — a
        # face that is more relaxed than its own baseline must not multiply two
        # negatives into a phantom hostile stare.
        au07 = aus.get("AU07", 0)   # Upper lid tightener intensity (0 = relaxed, 1 = fully tight)
        au04 = aus.get("AU04", 0)   # Brow furrow intensity (already scored above; re-read for the interaction)
        au07_rel = max(0.0, au07 - base_au.get("AU07", 0.0))
        au04_rel = max(0.0, au04 - base_au.get("AU04", 0.0))
        prev["face_AU07"] = au07
        s -= au07_rel * au04_rel * 15   # Extra deduction: only bites when both muscles are firing together

        # Sort the contributors by the size of their impact (biggest movers first)
        # and return only the top 2 so the UI tooltip stays readable.
        contribs.sort(key=lambda x: abs(x[3]), reverse=True)

        # Clamp the final score to the valid 0–100 range before returning.
        return max(0.0, min(100.0, s)), contribs[:2]

    def _vocal_score(self, vd: dict | None):
        prev = self._prev_inputs

        # If voice data is unavailable (microphone not started), return neutral 50.
        if not vd:
            return 50.0, []

        # When the person is silent, slowly drift the vocal score toward 50 rather than
        # snapping it there immediately. 98% weight on recent history, 2% nudge toward neutral.
        # This avoids a jarring score drop every time someone stops speaking mid-sentence.
        if not vd.get("is_speaking"):
            return self.smoothed["vocal"] * 0.98 + self.baseline.get("vocal", 50.0) * 0.02, []  # drift toward user's baseline when silent

        s = 50.0      # Neutral baseline — bonuses from pitch variation and speech energy push the score up
        contribs = [] # Will be filled with the top contributors this frame

        # Pitch stability: how consistent is the speaking pitch over time?
        # Steadier than this person's own calibrated speech → adds up to ~19 points.
        # More erratic than their own baseline → subtracts up to ~19 points.
        # Some people simply speak with more pitch variation than others, so the
        # centre point is theirs rather than a fixed 0.5.
        base_ps = self.input_baseline["pitch_stability"]
        ps = vd.get("pitch_stability", 0.5)           # 0.0 = very unstable pitch, 1.0 = perfectly stable
        prv_ps = prev.get("vocal_pitch_stab", ps)
        delta_ps = (ps - prv_ps) * 38                 # How much did pitch stability change this frame?
        if abs(delta_ps) >= 1.0:
            contribs.append(("pitch stability", prv_ps, ps, round(delta_ps, 1)))
        s += (ps - base_ps) * 38   # Below their resting steadiness subtracts; above it adds
        prev["vocal_pitch_stab"] = ps

        # Energy level: how loud is the person speaking? (0 = silent, 1 = very loud)
        # The quiet/loud thresholds are stretched by how loudly this person
        # speaks at rest, so a naturally soft talker is not scored as though
        # they were withholding — and a naturally loud one is not scored as
        # though they were shouting.
        # Too quiet may suggest hesitation or concealment → subtract 18 points.
        # Shouting suggests aggression or anxiety → subtract 6 points.
        # A comfortable middle volume → normal healthy conversation → add 8 points.
        el_scale = self._baseline_scale("energy_level")
        el = vd.get("energy_level", 0.0)
        if   el < 0.12 * el_scale: s -= 18   # Very quiet voice
        elif el > 0.88 * el_scale: s -=  6   # Shouting / very loud
        else:                      s +=  8   # Normal comfortable volume

        # Tremor index: does the voice have rapid shaking or wavering?
        # 0.0 = perfectly steady voice, 1.0 = extreme tremor (like a very nervous or fearful person).
        # Previously derived from frame-to-frame energy variance; now driven by eGeMAPS
        # jitter + shimmer + HNR composite (see VocalAnalyzer._tremor_from_features) when
        # opensmile is available — more clinically grounded and less sensitive to microphone gain.
        # High tremor is a classic anxiety indicator and subtracts up to 32 points.
        # Scored above this person's resting tremor — some voices carry a
        # permanent slight waver, and only the increase on top of it is a signal.
        tr = vd.get("tremor_index", 0.0)
        tr_rel = max(0.0, tr - self.input_baseline["tremor_index"])
        prv_tr = prev.get("vocal_tremor", tr)
        delta_tr = (tr - prv_tr) * -32   # Negative because rising tremor lowers the score
        if abs(delta_tr) >= 1.0:
            contribs.append(("tremor", prv_tr, tr, round(delta_tr, 1)))
        s -= tr_rel * 32   # Subtract up to 32 at maximum tremor
        prev["vocal_tremor"] = tr

        # Alpha ratio: log(energy 1–5 kHz / 50 Hz–1 kHz) — added from eGeMAPS feature set.
        # More negative = energy concentrated in low frequencies = normal relaxed voice.
        # Less negative (toward 0) = elevated high-frequency energy = strained or breathy voice.
        # Typical conversational speech: –15 to –5. Contribution is intentionally small (±4 pts)
        # so it only tips the score when the other signals (pitch, tremor) are already borderline.
        # Centred on this person's calibrated resting alpha ratio, since vocal
        # tract and microphone placement shift the whole range per session.
        # Skipped entirely when opensmile is unavailable (ar == 0.0 is the legacy fallback sentinel).
        base_ar = self.input_baseline["alpha_ratio"]
        ar = vd.get("alpha_ratio", 0.0)
        if ar != 0.0:   # 0.0 is the sentinel returned by the legacy path — do not score it
            ar_contrib = float(max(-4.0, min(3.0, -(ar - base_ar) * 0.2)))  # Each dB above resting subtracts 0.2 pts
            if abs(ar_contrib) >= 0.5:
                contribs.append(("alpha ratio", None, round(ar, 2), round(ar_contrib, 1)))
            s += ar_contrib
            prev["vocal_alpha_ratio"] = ar

        # Spectral flux: mean frame-to-frame spectral change — added from eGeMAPS feature set.
        # A stable, calm voice has low flux (≈ 0.002–0.005). Agitation, rapid pitch changes, or
        # vocal instability raise it toward 0.02+.  The formula subtracts up to 5 pts above this
        # person's calibrated resting flux; below it contributes a small positive nudge (capped at +1 pt).
        # Like alpha ratio, skipped when sf == 0.0 (legacy fallback sentinel).
        base_sf = self.input_baseline["spectral_flux"]
        sf = vd.get("spectral_flux", 0.0)
        if sf > 0.0:
            sf_contrib = float(max(-5.0, min(1.0, -(sf - base_sf) * 200.0)))  # Linear ramp: each 0.005 above baseline costs 1 pt
            if abs(sf_contrib) >= 0.5:
                contribs.append(("spectral flux", None, round(sf, 4), round(sf_contrib, 1)))
            s += sf_contrib

        # Return top-2 contributors and clamp to 0–100.
        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return max(0.0, min(100.0, s)), contribs[:2]

    def _gaze_score(self, fd: dict | None):
        prev = self._prev_inputs

        # If no face is detected, return neutral 50.
        if not fd or not fd.get("detected"):
            return 50.0, []

        s = 50.0      # Neutral baseline — bonuses from sustained eye contact push the score up
        contribs = []

        # Eye Aspect Ratio (EAR): a number that measures how open the eyes are.
        # Computed as the vertical eye height divided by horizontal eye width.
        # A fully open eye is roughly 0.27–0.35; a closed eye is near 0.
        # Narrowed or closed eyes are associated with suspicion or concealment.
        # Eye shape varies enormously between people, so all three cut-offs are
        # stretched by how open this person's eyes sit at rest: "narrowed" means
        # narrowed for them, not narrower than an average face.
        ear_scale = self._baseline_scale("eye_ar")
        t_shut, t_narrow, t_wide = [t * ear_scale for t in SCORE_CONFIG["ear_thresholds"]]
        ear = fd.get("eye_ar", 0.27)         # Current eye openness measurement
        prv_ear = prev.get("gaze_ear", ear)
        if   ear < t_shut:   s -= 28   # Eyes nearly shut → strong negative signal
        elif ear < t_narrow: s -= 12   # Eyes notably narrowed → moderate negative
        elif ear > t_wide:   s += 10   # Eyes wide open and alert → positive signal
        prev["gaze_ear"] = ear

        # Blink rate: how many complete blinks per minute?
        # Resting blink rate differs several-fold between people (contact lenses,
        # dry air, screen habits), so the bands are scaled to the rate measured
        # during calibration rather than a fixed 10–20 per minute.
        # Blinking much faster than their own normal is a stress/discomfort indicator.
        br_scale = self._baseline_scale("blink_rate")
        b_rapid, b_high, b_norm_hi, b_norm_lo = [t * br_scale
                                                 for t in SCORE_CONFIG["blink_thresholds"]]
        br = fd.get("blink_rate", 15.0)      # Blinks per minute
        prv_br = prev.get("gaze_blink", br)

        def _blink_pts(rate: float) -> int:
            if rate > b_rapid: return -22               # Rapid blinking → stress signal
            if rate > b_high:  return -10               # Slightly elevated blinking → mild stress
            if b_norm_lo <= rate <= b_norm_hi: return 8  # Their normal blink range → positive signal
            return 0

        # Convert the blink rate to a score impact, both for current and previous frames,
        # so we can report how much the blink rate *changed* this frame.
        delta_br = _blink_pts(br) - _blink_pts(prv_br)
        if abs(delta_br) >= 1.0:
            contribs.append(("blink rate", prv_br, br, round(delta_br, 1)))
        s += _blink_pts(br)
        prev["gaze_blink"] = br

        # Gaze deviation: how far is the person looking away from the camera centre?
        # 0.0 = looking straight ahead; higher values = looking further away.
        # Measured from where they naturally rest their gaze — almost nobody sits
        # perfectly square to the camera, and that offset is not avoidance.
        # Consistently looking away is associated with avoidance → subtracts up to 18 points.
        gd = fd.get("gaze_deviation", 0.0)   # 0.0 = straight ahead, ~1.0 = looking far to the side
        gd_rel = max(0.0, gd - self.input_baseline["gaze_deviation"])
        prv_gd = prev.get("gaze_dev", gd)
        delta_gd = (gd - prv_gd) * -18      # Negative: more deviation = lower score
        if abs(delta_gd) >= 1.0:
            contribs.append(("gaze deviation", prv_gd, gd, round(delta_gd, 1)))
        s -= gd_rel * 18   # Subtract up to 18 for maximum gaze deviation
        prev["gaze_dev"] = gd

        # Return top-2 contributors and clamp to 0–100.
        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return max(0.0, min(100.0, s)), contribs[:2]

    # ─── Calibration API ─────────────────────────────────────────────────── #

    def _baseline_scale(self, key: str) -> float:
        """How far this user's resting reading stretches a fixed threshold.

        Returns measured_baseline / population_reference, clamped, so a
        threshold written for an average person lands in the same place
        relative to *this* person. 1.0 means "no calibration data, use the
        thresholds exactly as written".
        """
        ref = SCORE_CONFIG["reference_baseline"].get(key, 0.0)
        base = self.input_baseline.get(key, ref)
        if not ref or not base or base <= 0:
            return 1.0
        lo, hi = SCORE_CONFIG["baseline_scale_clamp"]
        return max(lo, min(hi, base / ref))

    def resting_samples(self) -> tuple[dict, dict]:
        """Synthetic face and voice readings for this user sitting at rest.

        Built from the calibrated input baselines, in the same shape the
        analyzers emit, so it can be pushed through the real scorers.
        """
        ib = self.input_baseline
        face = {
            "detected":       True,
            "expressions":    dict(ib["expressions"]),
            "aus":            dict(ib["aus"]),
            "duchenne":       ib["duchenne"],
            "eye_ar":         ib["eye_ar"],
            "blink_rate":     ib["blink_rate"],
            "gaze_deviation": ib["gaze_deviation"],
        }
        vocal = {
            "is_speaking":     True,
            "pitch_stability": ib["pitch_stability"],
            "energy_level":    ib["energy_level"],
            "tremor_index":    ib["tremor_index"],
            "alpha_ratio":     ib["alpha_ratio"],
            "spectral_flux":   ib["spectral_flux"],
        }
        return face, vocal

    def apply_calibration(self, measured: dict, hrv_resting: float | None = None) -> None:
        """Adopt this user's measured resting readings as the scoring baseline.

        *measured* holds the mean sensor readings from the calibration window,
        keyed exactly like SCORE_CONFIG["reference_baseline"]. Any entry that is
        None or absent — no face seen, nobody spoke, no strap worn — keeps the
        population default for that signal alone, so a partial calibration still
        personalises everything it did capture.

        *hrv_resting* is the resting 0–100 HRV score. That channel is scored
        from R-R intervals inside HRVAnalyzer rather than here, so its resting
        value has to be handed in.
        """
        for key, val in measured.items():
            if val is None or key not in self.input_baseline:
                continue
            if isinstance(self.input_baseline[key], dict):
                if not isinstance(val, dict):
                    continue
                for sub, sub_val in val.items():
                    if sub_val is not None:
                        self.input_baseline[key][sub] = float(sub_val)
            else:
                self.input_baseline[key] = float(val)

        # Push a synthetic "this user at rest" sample through the scorers that
        # were just personalised. Whatever comes back *is* their resting score,
        # and subtracting it in update() puts their resting state at exactly 50.
        # The scorers write into _prev_inputs, so that state is saved and put
        # back — the probe must not look like a real frame to the next one.
        saved_prev = dict(self._prev_inputs)
        face_rest, vocal_rest = self.resting_samples()
        self.baseline["facial"] = self._facial_score(face_rest)[0]
        self.baseline["gaze"]   = self._gaze_score(face_rest)[0]
        self.baseline["vocal"]  = self._vocal_score(vocal_rest)[0]
        self.baseline["hrv"]    = float(hrv_resting) if hrv_resting is not None else 50.0
        self._prev_inputs = saved_prev

        # Start the session already sitting at that resting point rather than
        # ramping up from a neutral 50 that isn't this person's neutral.
        for ch in ("facial", "vocal", "gaze", "hrv"):
            self.smoothed[ch] = self.baseline[ch]
        active = [ch for ch in ("facial", "vocal", "gaze", "hrv") if self._active.get(ch)]
        active = active or ["facial", "vocal", "gaze", "hrv"]
        self.smoothed["total"] = sum(self.baseline[ch] for ch in active) / len(active)
        self._prev_smoothed = dict(self.smoothed)

    @staticmethod
    def trust_label(score: int) -> dict:
        # Converts the numeric trust score into a human-readable label and a display colour.
        # The five bands were chosen to divide 0–100 into meaningfully different behavioural states.
        if score >= 82: return {"text": "Calm + Engaged", "color": "#4ade80"}   # Green: very high trust
        if score >= 64: return {"text": "Relaxed",        "color": "#34d399"}   # Teal: above-average trust
        if score >= 46: return {"text": "Baseline",       "color": "#60a5fa"}   # Blue: neutral / normal
        if score >= 28: return {"text": "Activated",      "color": "#fb923c"}   # Orange: elevated arousal
        return                 {"text": "Heightened",     "color": "#f87171"}   # Red: high stress or tension
