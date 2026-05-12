class TrustEngine:
    def __init__(self):
        self.smoothed = {"total": 50.0, "facial": 50.0, "vocal": 50.0, "gaze": 50.0}  # Stores the exponentially smoothed scores; initialised to 50 (neutral) so the display starts centred
        self.alpha = 0.2   # Smoothing factor: 0.2 means each new raw score contributes 20% and the previous smoothed value contributes 80%

    def update(self, face_data: dict | None, vocal_data: dict | None) -> dict:
        facial = self._facial_score(face_data)      # Calculates a raw 0–100 facial trust score from the latest expression data
        vocal  = self._vocal_score(vocal_data)      # Calculates a raw 0–100 vocal trust score from the latest audio data
        gaze   = self._gaze_score(face_data)        # Calculates a raw 0–100 gaze trust score from the latest eye-tracking data
        total  = facial * 0.4 + vocal * 0.3 + gaze * 0.3  # Combines the three channels: facial 40%, vocal 30%, gaze 30%

        for k, v in [("facial", facial), ("vocal", vocal), ("gaze", gaze), ("total", total)]:
            self.smoothed[k] = self.alpha * v + (1 - self.alpha) * self.smoothed[k]  # Applies exponential moving average to prevent the score from jumping erratically

        return {k: round(v) for k, v in self.smoothed.items()}  # Rounds smoothed floats to integers before sending to the browser

    def _facial_score(self, fd: dict | None) -> float:
        if not fd or not fd.get("detected"):        # Returns neutral 50 when no face is visible so the score doesn't collapse to zero
            return 50.0
        e = fd["expressions"]                       # Retrieves the normalised emotion probability dict from the face analyser
        s = 50.0                                    # Starts from a neutral baseline of 50
        s += e.get("happy",     0) * 40             # Happy is the strongest positive trust signal (genuine smile = openness)
        s += e.get("neutral",   0) * 12             # Neutral expression is mildly positive (calm composure)
        s += e.get("surprised", 0) *  5             # Surprise is weakly positive (can indicate engagement)
        s -= e.get("fearful",   0) * 35             # Fear is a strong negative signal (discomfort, anxiety)
        s -= e.get("angry",     0) * 40             # Anger is the strongest negative signal (hostility, distrust)
        s -= e.get("disgusted", 0) * 35             # Disgust is a strong negative signal (rejection, aversion)
        s -= e.get("sad",       0) * 22             # Sadness is a moderate negative signal (low engagement, withdrawal)
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
