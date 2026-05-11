export class FaceAnalyzer {
  constructor() {
    this.ready = false;
    this.blinkCount   = 0;
    this.blinkWindowStart = Date.now();
    this.blinkRate    = 0;
    this.isBlinking   = false;
    this.EAR_THRESH   = 0.21;
    this.gazeHist     = [];
  }

  async init(onStatus) {
    const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
    onStatus('Loading face detector…');
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    onStatus('Loading expression model…');
    await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
    onStatus('Loading landmark model…');
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
    this.ready = true;
  }

  async analyze(videoEl) {
    if (!this.ready) return null;
    const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 });
    const res = await faceapi
      .detectSingleFace(videoEl, opts)
      .withFaceLandmarks(true)
      .withFaceExpressions();

    if (!res) return { detected: false };

    const lm   = res.landmarks;
    const lEye = lm.getLeftEye();
    const rEye = lm.getRightEye();
    const lEAR = this._ear(lEye);
    const rEAR = this._ear(rEye);
    const eyeAR = (lEAR + rEAR) / 2;

    this._trackBlink(eyeAR);
    const gazeDeviation = this._gazeDeviation(lEye, rEye, res.detection.box);

    return {
      detected: true,
      expressions: {
        happy:     res.expressions.happy,
        sad:       res.expressions.sad,
        angry:     res.expressions.angry,
        fearful:   res.expressions.fearful,
        disgusted: res.expressions.disgusted,
        surprised: res.expressions.surprised,
        neutral:   res.expressions.neutral,
      },
      dominant: this._dominant(res.expressions),
      eyeAR,
      lEAR,
      rEAR,
      blinkRate: this.blinkRate,
      gazeDeviation,
      box: res.detection.box,
      lEye,
      rEye,
    };
  }

  // Eye Aspect Ratio — Soukupova & Cech (2016)
  _ear(pts) {
    const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const h = d(pts[0], pts[3]);
    if (h === 0) return 0;
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2 * h);
  }

  _trackBlink(ear) {
    if (ear < this.EAR_THRESH && !this.isBlinking) {
      this.isBlinking = true;
    } else if (ear >= this.EAR_THRESH && this.isBlinking) {
      this.isBlinking = false;
      this.blinkCount++;
    }
    const elapsed = (Date.now() - this.blinkWindowStart) / 1000;
    if (elapsed >= 10) {
      this.blinkRate = (this.blinkCount / elapsed) * 60;
      this.blinkCount = 0;
      this.blinkWindowStart = Date.now();
    }
  }

  _gazeDeviation(lEye, rEye, box) {
    const cx = pts => pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const eyeCx = (cx(lEye) + cx(rEye)) / 2;
    const boxCx = box.x + box.width / 2;
    const dev = Math.abs(eyeCx - boxCx) / (box.width || 1);
    this.gazeHist.push(dev);
    if (this.gazeHist.length > 30) this.gazeHist.shift();
    return this.gazeHist.reduce((s, v) => s + v, 0) / this.gazeHist.length;
  }

  _dominant(expr) {
    return Object.entries(expr).sort((a, b) => b[1] - a[1])[0][0];
  }
}
