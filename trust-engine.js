export class TrustEngine {
  constructor() {
    this.smoothed = { total: 50, facial: 50, vocal: 50, gaze: 50 };
    this.alpha = 0.2;
    this.history = [];
    this.MAX_HISTORY = 300;
  }

  update(faceData, vocalData) {
    const facial = this._facialScore(faceData);
    const vocal  = this._vocalScore(vocalData);
    const gaze   = this._gazeScore(faceData);
    const total  = facial * 0.4 + vocal * 0.3 + gaze * 0.3;

    this.smoothed.facial = this._ema(this.smoothed.facial, facial);
    this.smoothed.vocal  = this._ema(this.smoothed.vocal,  vocal);
    this.smoothed.gaze   = this._ema(this.smoothed.gaze,   gaze);
    this.smoothed.total  = this._ema(this.smoothed.total,  total);

    const result = {
      total:  Math.round(this.smoothed.total),
      facial: Math.round(this.smoothed.facial),
      vocal:  Math.round(this.smoothed.vocal),
      gaze:   Math.round(this.smoothed.gaze),
    };

    this.history.push({ ...result, ts: Date.now() });
    if (this.history.length > this.MAX_HISTORY) this.history.shift();
    return result;
  }

  _ema(prev, next) {
    return this.alpha * next + (1 - this.alpha) * prev;
  }

  _facialScore(fd) {
    if (!fd?.detected) return 50;
    const e = fd.expressions;
    let s = 50;
    s += e.happy      * 40;
    s += e.neutral    * 12;
    s += e.surprised  *  5;
    s -= e.fearful    * 35;
    s -= e.angry      * 40;
    s -= e.disgusted  * 35;
    s -= e.sad        * 22;
    return Math.max(0, Math.min(100, s));
  }

  _vocalScore(vd) {
    if (!vd) return 50;
    const { isSpeaking, pitchStability, energyLevel, tremorIndex } = vd;
    if (!isSpeaking) return this.smoothed.vocal * 0.98 + 50 * 0.02; // slow drift to neutral
    let s = 55;
    s += (pitchStability - 0.5) * 38;
    if      (energyLevel < 0.12) s -= 18;
    else if (energyLevel > 0.88) s -=  6;
    else                          s +=  8;
    s -= tremorIndex * 32;
    return Math.max(0, Math.min(100, s));
  }

  _gazeScore(fd) {
    if (!fd?.detected) return 50;
    const { eyeAR, blinkRate, gazeDeviation } = fd;
    let s = 62;
    if      (eyeAR < 0.14) s -= 28;
    else if (eyeAR < 0.20) s -= 12;
    else if (eyeAR > 0.28) s += 10;
    if (blinkRate !== undefined) {
      if      (blinkRate > 32) s -= 22;
      else if (blinkRate > 23) s -= 10;
      else if (blinkRate >= 10 && blinkRate <= 20) s += 8;
    }
    if (gazeDeviation !== undefined) s -= gazeDeviation * 18;
    return Math.max(0, Math.min(100, s));
  }

  trustLabel(score) {
    if (score >= 82) return { text: 'Very High Trust',   color: '#4ade80' };
    if (score >= 64) return { text: 'High Trust',        color: '#34d399' };
    if (score >= 46) return { text: 'Neutral',           color: '#60a5fa' };
    if (score >= 28) return { text: 'Low Trust',         color: '#fb923c' };
    return                  { text: 'Very Low Trust',    color: '#f87171' };
  }
}
