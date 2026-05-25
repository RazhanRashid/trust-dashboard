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
        self.alpha = 0.2

        # Stores the raw sensor values from the previous camera/audio frame.
        # This lets the engine detect *changes* — for example, "the brow just furrowed"
        # — and tell the UI which signals caused the score to move.
        self._prev_inputs: dict = {}

    def update(self, face_data: dict | None, vocal_data: dict | None,
               hrv_score: int = 65) -> dict:
        # Ask each sub-scorer to convert raw sensor readings into a 0–100 number.
        # Each sub-scorer also returns a short list of the top contributors
        # (the signals that changed the score the most this frame).
        facial, facial_contribs = self._facial_score(face_data)   # Score from facial expressions
        vocal,  vocal_contribs  = self._vocal_score(vocal_data)   # Score from voice characteristics
        gaze,   gaze_contribs   = self._gaze_score(face_data)     # Score from eye movement and blinking
        hrv    = float(hrv_score) if hrv_score is not None else 65.0  # Heart-rate variability score (fixed placeholder for now)

        # Combine the four channel scores using equal 25% weights across all channels.
        total  = facial * 0.25 + vocal * 0.25 + gaze * 0.25 + hrv * 0.25

        # Apply exponential smoothing to every channel so scores drift gradually
        # rather than snapping to a new value instantly.
        # Formula: new_smoothed = alpha × brand_new_value + (1 − alpha) × previous_smoothed
        for k, v in [("facial", facial), ("vocal", vocal), ("gaze", gaze),
                     ("hrv", hrv), ("total", total)]:
            self.smoothed[k] = self.alpha * v + (1 - self.alpha) * self.smoothed[k]

        # Round all smoothed scores to whole numbers for cleaner display on screen.
        scores = {k: round(v) for k, v in self.smoothed.items()}

        # Attach the "why did the score change?" explanation lists so the UI
        # can show tooltips like "brow furrow: −8 pts".
        scores["contributions"] = {
            "facial": facial_contribs,
            "vocal":  vocal_contribs,
            "gaze":   gaze_contribs,
        }
        return scores   # Hand the complete score dictionary back to whoever called update()

    def _facial_score(self, fd: dict | None):
        # If no face is visible in the camera frame, return a neutral 50 with nothing to explain.
        if not fd or not fd.get("detected"):
            return 50.0, []

        e   = fd["expressions"]   # Dictionary of emotion intensities, each 0.0 (absent) → 1.0 (full)
        aus = fd.get("aus", {})   # Dictionary of Action Unit intensities (individual muscle movements)
        s   = 50.0                # Start at neutral before applying emotion-based adjustments
        prev = self._prev_inputs  # Shortcut to the stored previous-frame values

        contribs = []   # Will be filled with the signals that moved the score most this frame

        # Go through each emotion and apply its weight to the score.
        # Positive weight = that emotion raises trust. Negative = it lowers trust.
        # For example: a fully happy face (intensity 1.0) adds 30 points.
        # A fully fearful face subtracts 30 points.
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
            prv = prev.get(f"face_{key}", cur)        # Previous frame's intensity (defaults to current on the very first frame)
            delta_pts = (cur - prv) * abs(weight)     # How many score points did this emotion add or remove compared to last frame?
            if abs(delta_pts) >= 1.0:                 # Only log it as a contributor if the change was at least 1 point (ignore tiny noise)
                contribs.append((label, prv, cur, round(delta_pts, 1)))
            s += cur * weight                         # Apply intensity × weight to the running score
            prev[f"face_{key}"] = cur                 # Save this frame's value so next frame can compare

        # Duchenne smile check: a *genuine* smile activates both the cheek muscles (AU06)
        # and the lip corner muscles (AU12) together. A forced smile usually only shows AU12.
        # OpenFace combines them into a single Duchenne score (0 = fake/absent, 1 = genuine).
        # Genuine smiles are a strong trust signal, worth up to +20 points.
        duchenne = fd.get("duchenne", 0)
        prv_d = prev.get("face_duchenne", duchenne)   # Previous frame's Duchenne smile value
        delta_d = (duchenne - prv_d) * 20             # Score change from the smile becoming more/less genuine
        if abs(delta_d) >= 1.0:
            contribs.append(("duchenne smile", prv_d, duchenne, round(delta_d, 1)))
        s += duchenne * 20           # Add up to 20 points for a full genuine smile
        prev["face_duchenne"] = duchenne

        # Action Units (AUs) are specific facial muscle movements detected by OpenFace.
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
            prv = prev.get(f"face_{au}", cur)
            delta_pts = (cur - prv) * abs(weight)
            if abs(delta_pts) >= 1.0:
                contribs.append((label, prv, cur, round(delta_pts, 1)))
            s += cur * weight
            prev[f"face_{au}"] = cur

        # Interaction penalty: when the upper eyelid is tightened (AU07) AND the brow is
        # simultaneously furrowed (AU04), the combined expression looks like an intense
        # hostile stare — much more threatening than either muscle alone.
        # Multiplying them means the penalty is zero unless both are active at the same time.
        au07 = aus.get("AU07", 0)   # Upper lid tightener intensity (0 = relaxed, 1 = fully tight)
        au04 = aus.get("AU04", 0)   # Brow furrow intensity (already scored above; re-read for the interaction)
        prev["face_AU07"] = au07
        s -= au07 * au04 * 15       # Extra deduction: only bites when both muscles are firing together

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
            return self.smoothed["vocal"] * 0.98 + 50.0 * 0.02, []

        s = 55.0      # Start slightly above neutral — an active speaking voice is already a positive signal
        contribs = [] # Will be filled with the top contributors this frame

        # Pitch stability: how consistent is the speaking pitch over time?
        # A rock-steady pitch (1.0) suggests calm confidence → can add up to ~19 points.
        # Erratic pitch jumping around (0.0) suggests nervousness → can subtract up to ~19 points.
        # The formula centres around 0.5 (neutral) and scales by 38.
        ps = vd.get("pitch_stability", 0.5)           # 0.0 = very unstable pitch, 1.0 = perfectly stable
        prv_ps = prev.get("vocal_pitch_stab", ps)
        delta_ps = (ps - prv_ps) * 38                 # How much did pitch stability change this frame?
        if abs(delta_ps) >= 1.0:
            contribs.append(("pitch stability", prv_ps, ps, round(delta_ps, 1)))
        s += (ps - 0.5) * 38   # Values below 0.5 subtract; values above 0.5 add
        prev["vocal_pitch_stab"] = ps

        # Energy level: how loud is the person speaking? (0 = silent, 1 = very loud)
        # Too quiet (below 0.12) may suggest hesitation or concealment → subtract 18 points.
        # Too loud / shouting (above 0.88) suggests aggression or anxiety → subtract 6 points.
        # A comfortable middle volume → normal healthy conversation → add 8 points.
        el = vd.get("energy_level", 0.0)
        if   el < 0.12: s -= 18   # Very quiet voice
        elif el > 0.88: s -=  6   # Shouting / very loud
        else:           s +=  8   # Normal comfortable volume

        # Tremor index: does the voice have rapid shaking or wavering?
        # 0.0 = perfectly steady voice, 1.0 = extreme tremor (like a very nervous or fearful person).
        # Previously derived from frame-to-frame energy variance; now driven by eGeMAPS
        # jitter + shimmer + HNR composite (see VocalAnalyzer._tremor_from_features) when
        # opensmile is available — more clinically grounded and less sensitive to microphone gain.
        # High tremor is a classic anxiety indicator and subtracts up to 32 points.
        tr = vd.get("tremor_index", 0.0)
        prv_tr = prev.get("vocal_tremor", tr)
        delta_tr = (tr - prv_tr) * -32   # Negative because rising tremor lowers the score
        if abs(delta_tr) >= 1.0:
            contribs.append(("tremor", prv_tr, tr, round(delta_tr, 1)))
        s -= tr * 32   # Subtract up to 32 at maximum tremor
        prev["vocal_tremor"] = tr

        # Alpha ratio: log(energy 1–5 kHz / 50 Hz–1 kHz) — added from eGeMAPS feature set.
        # More negative = energy concentrated in low frequencies = normal relaxed voice.
        # Less negative (toward 0) = elevated high-frequency energy = strained or breathy voice.
        # Typical conversational speech: –15 to –5. Contribution is intentionally small (±4 pts)
        # so it only tips the score when the other signals (pitch, tremor) are already borderline.
        # Skipped entirely when opensmile is unavailable (ar == 0.0 is the legacy fallback sentinel).
        ar = vd.get("alpha_ratio", 0.0)
        if ar != 0.0:   # 0.0 is the sentinel returned by the legacy path — do not score it
            ar_contrib = float(max(-4.0, min(3.0, -(ar + 10.0) * 0.2)))  # Centred at –10 dB; each unit above –10 subtracts 0.2 pts
            if abs(ar_contrib) >= 0.5:
                contribs.append(("alpha ratio", None, round(ar, 2), round(ar_contrib, 1)))
            s += ar_contrib
            prev["vocal_alpha_ratio"] = ar

        # Spectral flux: mean frame-to-frame spectral change — added from eGeMAPS feature set.
        # A stable, calm voice has low flux (≈ 0.002–0.005). Agitation, rapid pitch changes, or
        # vocal instability raise it toward 0.02+.  The formula subtracts up to 5 pts above the
        # 0.005 baseline; below it contributes a small positive nudge (capped at +1 pt).
        # Like alpha ratio, skipped when sf == 0.0 (legacy fallback sentinel).
        sf = vd.get("spectral_flux", 0.0)
        if sf > 0.0:
            sf_contrib = float(max(-5.0, min(1.0, -(sf - 0.005) * 200.0)))  # Linear ramp: each 0.005 above baseline costs 1 pt
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

        s = 62.0      # Gaze starts above neutral — sustained eye contact is a positive trust signal
        contribs = []

        # Eye Aspect Ratio (EAR): a number that measures how open the eyes are.
        # Computed as the vertical eye height divided by horizontal eye width.
        # A fully open eye is roughly 0.27–0.35; a closed eye is near 0.
        # Narrowed or closed eyes are associated with suspicion or concealment.
        ear = fd.get("eye_ar", 0.27)         # Current eye openness measurement
        prv_ear = prev.get("gaze_ear", ear)
        if   ear < 0.14: s -= 28   # Eyes nearly shut → strong negative signal
        elif ear < 0.20: s -= 12   # Eyes notably narrowed → moderate negative
        elif ear > 0.28: s += 10   # Eyes wide open and alert → positive signal
        prev["gaze_ear"] = ear

        # Blink rate: how many complete blinks per minute?
        # Normal resting blink rate is 10–20 blinks per minute.
        # Blinking much faster than normal is a stress/discomfort indicator.
        br = fd.get("blink_rate", 15.0)      # Blinks per minute
        prv_br = prev.get("gaze_blink", br)

        # Convert the blink rate to a score impact, both for current and previous frames,
        # so we can report how much the blink rate *changed* this frame.
        br_pts  = (-22 if br > 32 else -10 if br > 23 else 8 if 10 <= br <= 20 else 0)
        prv_pts = (-22 if prv_br > 32 else -10 if prv_br > 23 else 8 if 10 <= prv_br <= 20 else 0)
        delta_br = br_pts - prv_pts
        if abs(delta_br) >= 1.0:
            contribs.append(("blink rate", prv_br, br, round(delta_br, 1)))
        if   br > 32: s -= 22              # Rapid blinking (>32/min) → stress signal
        elif br > 23: s -= 10             # Slightly elevated blinking → mild stress
        elif 10 <= br <= 20: s += 8       # Normal blink range → positive signal
        prev["gaze_blink"] = br

        # Gaze deviation: how far is the person looking away from the camera centre?
        # 0.0 = looking straight ahead; higher values = looking further away.
        # Consistently looking away is associated with avoidance → subtracts up to 18 points.
        gd = fd.get("gaze_deviation", 0.0)   # 0.0 = straight ahead, ~1.0 = looking far to the side
        prv_gd = prev.get("gaze_dev", gd)
        delta_gd = (gd - prv_gd) * -18      # Negative: more deviation = lower score
        if abs(delta_gd) >= 1.0:
            contribs.append(("gaze deviation", prv_gd, gd, round(delta_gd, 1)))
        s -= gd * 18   # Subtract up to 18 for maximum gaze deviation
        prev["gaze_dev"] = gd

        # Return top-2 contributors and clamp to 0–100.
        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return max(0.0, min(100.0, s)), contribs[:2]

    @staticmethod
    def trust_label(score: int) -> dict:
        # Converts the numeric trust score into a human-readable label and a display colour.
        # The five bands were chosen to divide 0–100 into meaningfully different behavioural states.
        if score >= 82: return {"text": "Calm + Engaged", "color": "#4ade80"}   # Green: very high trust
        if score >= 64: return {"text": "Relaxed",        "color": "#34d399"}   # Teal: above-average trust
        if score >= 46: return {"text": "Baseline",       "color": "#60a5fa"}   # Blue: neutral / normal
        if score >= 28: return {"text": "Activated",      "color": "#fb923c"}   # Orange: elevated arousal
        return                 {"text": "Heightened",     "color": "#f87171"}   # Red: high stress or tension
