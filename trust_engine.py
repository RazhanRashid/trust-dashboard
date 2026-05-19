class TrustEngine:
    def __init__(self):
        self.smoothed = {"total": 50.0, "facial": 50.0, "vocal": 50.0, "gaze": 50.0, "hrv": 50.0}
        self.alpha = 0.2

    def update(self, face_data: dict | None, vocal_data: dict | None,
               hrv_score: int = 65) -> dict:
        facial = self._facial_score(face_data)
        vocal  = self._vocal_score(vocal_data)
        gaze   = self._gaze_score(face_data)
        hrv    = float(hrv_score) if hrv_score is not None else 65.0
        # Weights: facial 35%, vocal 25%, gaze 25%, HRV 15%
        total  = facial * 0.35 + vocal * 0.25 + gaze * 0.25 + hrv * 0.15

        for k, v in [("facial", facial), ("vocal", vocal), ("gaze", gaze),
                     ("hrv", hrv), ("total", total)]:
            self.smoothed[k] = self.alpha * v + (1 - self.alpha) * self.smoothed[k]

        return {k: round(v) for k, v in self.smoothed.items()}

    def _facial_score(self, fd: dict | None) -> float:
        if not fd or not fd.get("detected"):        # Returns neutral 50 when no face is visible so the score doesn't collapse to zero
            return 50.0

        e   = fd["expressions"]                     # Deep-learning emotion probabilities (resmasknet output)
        aus = fd.get("aus", {})                     # Raw OpenFace-style AU intensities from py-feat

        s = 50.0                                    # Starts from a neutral baseline of 50

        # ── Deep-learning emotion component ───────────────────────────────────
        s += e.get("happy",     0) * 30             # Happy is a strong positive trust signal
        s += e.get("neutral",   0) * 10             # Calm neutral expression is mildly positive
        s += e.get("surprised", 0) *  4             # Surprise is weakly positive (engagement)
        s -= e.get("fearful",   0) * 30             # Fear signals discomfort or anxiety
        s -= e.get("angry",     0) * 35             # Anger is the strongest negative signal
        s -= e.get("disgusted", 0) * 30             # Disgust signals rejection or aversion
        s -= e.get("sad",       0) * 18             # Sadness signals low engagement or withdrawal

        # ── OpenFace Action Unit component (FACS-based) ───────────────────────
        # Duchenne smile: AU06 (Cheek Raiser) + AU12 (Lip Corner Puller)
        # The combination uniquely identifies a genuine (vs social/polite) smile
        duchenne = fd.get("duchenne", 0)
        s += duchenne * 20                          # Genuine smile is a strong positive trust indicator

        # AU04 (Brow Lowerer) signals concentration, concern, or hostility
        s -= aus.get("AU04", 0) * 12

        # AU07 (Lid Tightener) combined with AU04 indicates anger or suspicion
        s -= aus.get("AU07", 0) * aus.get("AU04", 0) * 15

        # AU20 (Lip Stretcher — horizontal lip pull) is a fear/stress marker
        s -= aus.get("AU20", 0) * 10

        # AU14 (Dimpler — asymmetric mouth corner) can indicate contempt
        s -= aus.get("AU14", 0) * 8

        return max(0.0, min(100.0, s))              # Clamps the result to the valid 0–100 range

    def _vocal_score(self, vd: dict | None) -> float:
        if not vd:                                  # Returns neutral 50 when no audio data has arrived yet
            return 50.0
        if not vd.get("is_speaking"):               # When the user is silent, slowly drift the vocal score back toward neutral rather than holding or resetting it
            return self.smoothed["vocal"] * 0.98 + 50.0 * 0.02
        s = 55.0                                    # Speaking at all is a slight positive signal (engagement, willingness to communicate)
        s += (vd.get("pitch_stability", 0.5) - 0.5) * 38  # Stable pitch → confident and calm; erratic pitch → nervous or deceptive
        el = vd.get("energy_level", 0.0)            # Retrieves the normalised volume level
        if   el < 0.12: s -= 18                     # Very quiet voice suggests uncertainty or evasiveness
        elif el > 0.88: s -=  6                     # Shouting suggests aggression or defensiveness
        else:           s +=  8                     # Normal conversational volume is a mild positive
        s -= vd.get("tremor_index", 0.0) * 32       # Voice tremor is a strong indicator of stress or anxiety
        return max(0.0, min(100.0, s))              # Clamps the result to the valid 0–100 range

    def _gaze_score(self, fd: dict | None) -> float:
        if not fd or not fd.get("detected"):        # Returns neutral 50 when no face is visible
            return 50.0
        s = 62.0                                    # Being present and detectable in frame is a mild positive baseline
        ear = fd.get("eye_ar", 0.27)                # Eye Aspect Ratio: how open the eyes are
        if   ear < 0.14: s -= 28                    # Nearly closed eyes suggest fatigue, drowsiness, or evasion
        elif ear < 0.20: s -= 12                    # Squinted eyes suggest discomfort or suspicion
        elif ear > 0.28: s += 10                    # Wide-open eyes indicate alertness and engagement
        br = fd.get("blink_rate", 15.0)             # Blink rate in blinks per minute
        if   br > 32: s -= 22                       # Excessive blinking is a well-known stress and anxiety indicator
        elif br > 23: s -= 10                       # Slightly elevated blinking suggests mild discomfort
        elif 10 <= br <= 20: s += 8                 # The normal blink rate (10–20/min) indicates a relaxed, comfortable state
        s -= fd.get("gaze_deviation", 0.0) * 18     # Iris offset from the eye centre: large deviation suggests looking away (avoidance)
        return max(0.0, min(100.0, s))              # Clamps the result to the valid 0–100 range

    @staticmethod
    def trust_label(score: int) -> dict:
        if score >= 82: return {"text": "Very High Trust", "color": "#4ade80"}  # Bright green: strong positive trust indicators across all channels
        if score >= 64: return {"text": "High Trust",      "color": "#34d399"}  # Teal green: generally positive signals
        if score >= 46: return {"text": "Neutral",         "color": "#60a5fa"}  # Blue: mixed or insufficient signals
        if score >= 28: return {"text": "Low Trust",       "color": "#fb923c"}  # Orange: notable negative signals present
        return                 {"text": "Very Low Trust",  "color": "#f87171"}  # Red: strong negative indicators across multiple channels
