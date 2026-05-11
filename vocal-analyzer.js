export class VocalAnalyzer {
  constructor() {
    this.ready    = false;
    this.ctx      = null;
    this.analyser = null;
    this.pitchHist  = [];
    this.energyHist = [];
    this.SPEAK_THRESH = 0.018;
    this.isSpeaking   = false;
  }

  async init() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.8;
      this.ctx.createMediaStreamSource(stream).connect(this.analyser);
      this.ready = true;
      return true;
    } catch {
      return false;
    }
  }

  analyze() {
    const fallback = { isSpeaking: false, pitchStability: 0.5, energyLevel: 0, tremorIndex: 0, dominantHz: 0 };
    if (!this.ready) return fallback;

    const N    = this.analyser.frequencyBinCount;
    const time = new Uint8Array(N);
    const freq = new Float32Array(N);
    this.analyser.getByteTimeDomainData(time);
    this.analyser.getFloatFrequencyData(freq);

    // RMS
    let sum = 0;
    for (let i = 0; i < N; i++) { const v = (time[i] - 128) / 128; sum += v * v; }
    const rms = Math.sqrt(sum / N);
    const energy = Math.min(1, rms / 0.09);

    this.isSpeaking = rms > this.SPEAK_THRESH;

    // Dominant frequency in speech range 80-450 Hz
    const sr  = this.ctx.sampleRate;
    const res = sr / (N * 2);
    const lo  = Math.floor(80 / res);
    const hi  = Math.floor(450 / res);
    let maxDb = -Infinity, maxBin = lo;
    for (let i = lo; i < hi; i++) {
      if (freq[i] > maxDb) { maxDb = freq[i]; maxBin = i; }
    }
    const dominantHz = maxBin * res;

    if (this.isSpeaking) {
      this.pitchHist.push(dominantHz);
      this.energyHist.push(rms);
      if (this.pitchHist.length  > 60) this.pitchHist.shift();
      if (this.energyHist.length > 60) this.energyHist.shift();
    }

    return {
      isSpeaking:     this.isSpeaking,
      pitchStability: this._stability(this.pitchHist),
      energyLevel:    energy,
      tremorIndex:    this._tremor(),
      dominantHz,
    };
  }

  // Returns 0-1; higher = more stable
  _stability(arr) {
    if (arr.length < 4) return 0.5;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    if (mean === 0) return 0.5;
    const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
    return Math.max(0, Math.min(1, 1 - std / mean));
  }

  // High-frequency energy variation = tremor
  _tremor() {
    if (this.energyHist.length < 8) return 0;
    let delta = 0;
    for (let i = 1; i < this.energyHist.length; i++)
      delta += Math.abs(this.energyHist[i] - this.energyHist[i - 1]);
    return Math.min(1, delta / this.energyHist.length / 0.008);
  }

  getTimeDomainData() {
    if (!this.analyser) return new Uint8Array(512).fill(128);
    const d = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteTimeDomainData(d);
    return d;
  }

  getFreqData() {
    if (!this.analyser) return new Uint8Array(512);
    const d = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(d);
    return d;
  }
}
