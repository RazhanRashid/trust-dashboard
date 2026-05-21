class TrustEngine:
    def __init__(self):
        self.smoothed = {"total": 50.0, "facial": 50.0, "vocal": 50.0, "gaze": 50.0, "hrv": 50.0}
        self.alpha = 0.2
        self._prev_inputs: dict = {}   # stores raw signal values from the previous tick

    def update(self, face_data: dict | None, vocal_data: dict | None,
               hrv_score: int = 65) -> dict:
        facial, facial_contribs = self._facial_score(face_data)
        vocal,  vocal_contribs  = self._vocal_score(vocal_data)
        gaze,   gaze_contribs   = self._gaze_score(face_data)
        hrv    = float(hrv_score) if hrv_score is not None else 65.0
        # Weights: facial 35%, vocal 25%, gaze 25%, HRV 15%
        total  = facial * 0.35 + vocal * 0.25 + gaze * 0.25 + hrv * 0.15

        for k, v in [("facial", facial), ("vocal", vocal), ("gaze", gaze),
                     ("hrv", hrv), ("total", total)]:
            self.smoothed[k] = self.alpha * v + (1 - self.alpha) * self.smoothed[k]

        scores = {k: round(v) for k, v in self.smoothed.items()}
        scores["contributions"] = {
            "facial": facial_contribs,
            "vocal":  vocal_contribs,
            "gaze":   gaze_contribs,
        }
        return scores

    def _facial_score(self, fd: dict | None):
        if not fd or not fd.get("detected"):
            return 50.0, []

        e   = fd["expressions"]
        aus = fd.get("aus", {})
        s   = 50.0
        prev = self._prev_inputs

        contribs = []
        for key, weight, label in [
            ("happy",     30,  "happy"),
            ("neutral",   10,  "neutral"),
            ("surprised",  4,  "surprised"),
            ("fearful",  -30,  "fearful"),
            ("angry",    -35,  "angry"),
            ("disgusted",-30,  "disgusted"),
            ("sad",      -18,  "sad"),
        ]:
            cur = e.get(key, 0)
            prv = prev.get(f"face_{key}", cur)
            delta_pts = (cur - prv) * abs(weight)
            if abs(delta_pts) >= 1.0:
                contribs.append((label, prv, cur, round(delta_pts, 1)))
            s += cur * weight
            prev[f"face_{key}"] = cur

        duchenne = fd.get("duchenne", 0)
        prv_d = prev.get("face_duchenne", duchenne)
        delta_d = (duchenne - prv_d) * 20
        if abs(delta_d) >= 1.0:
            contribs.append(("duchenne smile", prv_d, duchenne, round(delta_d, 1)))
        s += duchenne * 20
        prev["face_duchenne"] = duchenne

        for au, weight, label in [
            ("AU04", -12, "AU04 brow"),
            ("AU20", -10, "AU20 lip"),
            ("AU14",  -8, "AU14 dimple"),
        ]:
            cur = aus.get(au, 0)
            prv = prev.get(f"face_{au}", cur)
            delta_pts = (cur - prv) * abs(weight)
            if abs(delta_pts) >= 1.0:
                contribs.append((label, prv, cur, round(delta_pts, 1)))
            s += cur * weight
            prev[f"face_{au}"] = cur

        # AU07 × AU04 interaction
        au07 = aus.get("AU07", 0)
        au04 = aus.get("AU04", 0)
        prev["face_AU07"] = au07
        s -= au07 * au04 * 15

        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return max(0.0, min(100.0, s)), contribs[:2]

    def _vocal_score(self, vd: dict | None):
        prev = self._prev_inputs
        if not vd:
            return 50.0, []
        if not vd.get("is_speaking"):
            return self.smoothed["vocal"] * 0.98 + 50.0 * 0.02, []

        s = 55.0
        contribs = []

        ps = vd.get("pitch_stability", 0.5)
        prv_ps = prev.get("vocal_pitch_stab", ps)
        delta_ps = (ps - prv_ps) * 38
        if abs(delta_ps) >= 1.0:
            contribs.append(("pitch stability", prv_ps, ps, round(delta_ps, 1)))
        s += (ps - 0.5) * 38
        prev["vocal_pitch_stab"] = ps

        el = vd.get("energy_level", 0.0)
        if   el < 0.12: s -= 18
        elif el > 0.88: s -=  6
        else:           s +=  8

        tr = vd.get("tremor_index", 0.0)
        prv_tr = prev.get("vocal_tremor", tr)
        delta_tr = (tr - prv_tr) * -32
        if abs(delta_tr) >= 1.0:
            contribs.append(("tremor", prv_tr, tr, round(delta_tr, 1)))
        s -= tr * 32
        prev["vocal_tremor"] = tr

        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return max(0.0, min(100.0, s)), contribs[:2]

    def _gaze_score(self, fd: dict | None):
        prev = self._prev_inputs
        if not fd or not fd.get("detected"):
            return 50.0, []

        s = 62.0
        contribs = []

        ear = fd.get("eye_ar", 0.27)
        prv_ear = prev.get("gaze_ear", ear)
        if   ear < 0.14: s -= 28
        elif ear < 0.20: s -= 12
        elif ear > 0.28: s += 10
        prev["gaze_ear"] = ear

        br = fd.get("blink_rate", 15.0)
        prv_br = prev.get("gaze_blink", br)
        br_pts  = (-22 if br > 32 else -10 if br > 23 else 8 if 10 <= br <= 20 else 0)
        prv_pts = (-22 if prv_br > 32 else -10 if prv_br > 23 else 8 if 10 <= prv_br <= 20 else 0)
        delta_br = br_pts - prv_pts
        if abs(delta_br) >= 1.0:
            contribs.append(("blink rate", prv_br, br, round(delta_br, 1)))
        if   br > 32: s -= 22
        elif br > 23: s -= 10
        elif 10 <= br <= 20: s += 8
        prev["gaze_blink"] = br

        gd = fd.get("gaze_deviation", 0.0)
        prv_gd = prev.get("gaze_dev", gd)
        delta_gd = (gd - prv_gd) * -18
        if abs(delta_gd) >= 1.0:
            contribs.append(("gaze deviation", prv_gd, gd, round(delta_gd, 1)))
        s -= gd * 18
        prev["gaze_dev"] = gd

        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return max(0.0, min(100.0, s)), contribs[:2]

    @staticmethod
    def trust_label(score: int) -> dict:
        if score >= 82: return {"text": "Calm + Engaged", "color": "#4ade80"}
        if score >= 64: return {"text": "Relaxed",        "color": "#34d399"}
        if score >= 46: return {"text": "Baseline",       "color": "#60a5fa"}
        if score >= 28: return {"text": "Activated",      "color": "#fb923c"}
        return                 {"text": "Heightened",     "color": "#f87171"}
