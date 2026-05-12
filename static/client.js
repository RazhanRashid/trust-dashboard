// Browser capture layer — camera frames + audio sent to Python backend via WebSocket.
// All analysis (face expressions, eye tracking, vocal) runs in Python.

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

let ws          = null;
let histChart   = null;
let lastChartAt = 0;
let demoMode    = false;
let demoT       = 0;

const GAUGE_LEN    = 251.2;
const MAX_CHART_PTS = 60;

// ─── Boot ────────────────────────────────────────────────────────────────────

async function boot() {
  const overlay = $('loading-overlay');
  const txt     = $('loading-text');
  const bar     = $('loading-bar');

  try {
    setBar(bar, 10);
    txt.textContent = 'Connecting to Python backend…';

    const connected = await connectWS(3000);
    if (!connected) throw new Error('no-server');
    setBar(bar, 35);

    txt.textContent = 'Requesting camera…';
    const vid = $('video');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
    });
    vid.srcObject = stream;
    await new Promise(r => vid.addEventListener('loadedmetadata', r, { once: true }));
    setBar(bar, 65);

    txt.textContent = 'Requesting microphone…';
    let hasAudio = false;
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      setupAudio(audioStream);
      hasAudio = true;
    } catch { /* mic optional */ }

    setBar(bar, 90);
    initChart();
    setBar(bar, 100);
    await delay(300);

    overlay.style.display = 'none';
    dot('face',  'loading');
    dot('gaze',  'loading');
    dot('voice', hasAudio ? 'idle' : 'off');

    startFrameCapture(vid);

  } catch (err) {
    if (err.message === 'no-server' || !navigator.mediaDevices) {
      // Fall back to demo mode so the UI is visible in preview environments
      initChart();
      await delay(300);
      overlay.style.display = 'none';
      demoMode = true;
      showDemoBanner();
      dot('face',  'active');
      dot('gaze',  'active');
      dot('voice', 'active');
      loopDemo();
    } else {
      txt.textContent = `Error: ${err.message}`;
      bar.style.background = '#f87171';
    }
  }
}

// ─── WebSocket ────────────────────────────────────────────────────────────────

function connectWS(timeoutMs = 3000) {
  return new Promise(resolve => {
    ws = new WebSocket(WS_URL);
    const timer = setTimeout(() => resolve(false), timeoutMs);
    ws.onopen    = () => { clearTimeout(timer); resolve(true); };
    ws.onerror   = () => { clearTimeout(timer); resolve(false); };
    ws.onmessage = handleMessage;
    ws.onclose   = () => {
      // Auto-reconnect after 2s if we were in live mode
      if (!demoMode) setTimeout(() => connectWS().catch(() => {}), 2000);
    };
  });
}

function handleMessage(event) {
  const msg = JSON.parse(event.data);
  if (!msg.scores) return;
  const { scores, label, metrics } = msg;
  renderScores(scores, label);
  renderFaceMetrics(metrics.face);
  renderVocalMetrics(metrics.vocal);
  drawFaceOverlay(metrics.face);
  updateDots(metrics.face, metrics.vocal);
  const now = Date.now();
  if (now - lastChartAt > 500) { pushChart(scores); lastChartAt = now; }
}

// ─── Frame capture ────────────────────────────────────────────────────────────

let captureCanvas = null;

function startFrameCapture(vid) {
  captureCanvas       = document.createElement('canvas');
  captureCanvas.width  = 320;
  captureCanvas.height = 240;
  setInterval(() => sendFrame(vid), 80); // ~12 fps
}

function sendFrame(vid) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(vid, 0, 0, 320, 240);
  const b64 = captureCanvas.toDataURL('image/jpeg', 0.72).split(',')[1];
  ws.send(JSON.stringify({ t: 'frame', d: b64 }));
}

// ─── Audio capture ────────────────────────────────────────────────────────────

function setupAudio(stream) {
  const ctx  = new (window.AudioContext || window.webkitAudioContext)();
  const src  = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = e => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const samples = Array.from(e.inputBuffer.getChannelData(0));
    ws.send(JSON.stringify({ t: 'audio', d: samples, sr: Math.round(ctx.sampleRate) }));
  };
  src.connect(proc);
  proc.connect(ctx.destination);
}

// ─── Face overlay (drawn from Python-computed coordinates) ────────────────────

function drawFaceOverlay(fd) {
  const canvas = $('face-canvas');
  const vid    = $('video');
  if (canvas.width !== vid.videoWidth && vid.videoWidth > 0) {
    canvas.width  = vid.videoWidth;
    canvas.height = vid.videoHeight;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!fd?.detected) return;

  const W = canvas.width, H = canvas.height;
  const [bx, by, bw, bh] = fd.box_norm;

  // Bounding box
  ctx.strokeStyle = 'rgba(0,212,255,0.85)';
  ctx.lineWidth   = 2;
  ctx.strokeRect(bx * W, by * H, bw * W, bh * H);
  ctx.fillStyle   = 'rgba(0,212,255,0.05)';
  ctx.fillRect(bx * W, by * H, bw * W, bh * H);

  // Dominant expression label
  ctx.font      = 'bold 13px system-ui';
  ctx.fillStyle = '#00d4ff';
  ctx.fillText(fd.dominant, bx * W + 4, by * H - 8);

  // Eye outlines
  if (fd.eye_norm) {
    drawEyePoly(ctx, fd.eye_norm.l, fd.l_ear, W, H);
    drawEyePoly(ctx, fd.eye_norm.r, fd.r_ear, W, H);
  }
}

function drawEyePoly(ctx, pts, ear, W, H) {
  if (!pts?.length) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0] * W, pts[0][1] * H);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * W, pts[i][1] * H);
  ctx.closePath();
  const open = (ear ?? 0.3) > 0.21;
  ctx.strokeStyle = open ? '#4ade80' : '#fbbf24';
  ctx.lineWidth   = 1.5;
  ctx.stroke();
  ctx.fillStyle   = open ? 'rgba(74,222,128,0.1)' : 'rgba(251,191,36,0.08)';
  ctx.fill();
}

// ─── Score rendering ──────────────────────────────────────────────────────────

function renderScores(scores, label) {
  const arc = $('gauge-arc');
  const num = $('gauge-num');
  const lbl = $('trust-label');

  arc.setAttribute('stroke-dashoffset', GAUGE_LEN * (1 - scores.total / 100));
  arc.setAttribute('stroke', label.color);
  num.textContent = scores.total;
  num.setAttribute('fill', label.color);
  lbl.textContent = label.text;
  lbl.style.color = label.color;

  ['facial', 'vocal', 'gaze'].forEach(k => {
    $(`bar-${k}`).style.width = `${scores[k]}%`;
    $(`num-${k}`).textContent = scores[k];
  });
}

function renderFaceMetrics(fd) {
  if (fd?.detected) {
    set('expr-val',     fd.dominant);
    set('ear-val',      (fd.eye_ar * 100).toFixed(0) + '%');
    set('blink-val',    fd.blink_rate.toFixed(0) + '/min');
    set('gaze-dev-val', (fd.gaze_deviation * 100).toFixed(0) + '%');
  } else {
    ['expr-val', 'ear-val', 'blink-val', 'gaze-dev-val'].forEach(id => set(id, '—'));
  }
}

function renderVocalMetrics(vd) {
  if (!vd) return;
  set('pitch-val',    (vd.pitch_stability * 100).toFixed(0) + '%');
  set('energy-val',   (vd.energy_level    * 100).toFixed(0) + '%');
  set('tremor-val',   (vd.tremor_index    * 100).toFixed(0) + '%');
  set('hz-val',       vd.dominant_hz ? vd.dominant_hz.toFixed(0) + ' Hz' : '—');
  set('speaking-val', vd.is_speaking ? 'Yes' : 'No');
  $('speaking-val').style.color = vd.is_speaking ? '#4ade80' : '#94a3b8';
}

// ─── History chart ────────────────────────────────────────────────────────────

function initChart() {
  const ctx = $('hist-chart').getContext('2d');
  histChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Trust',  data: [], borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.07)', borderWidth: 2.5, pointRadius: 0, tension: 0.4, fill: true },
        { label: 'Facial', data: [], borderColor: '#4ade80', borderWidth: 1.5, pointRadius: 0, tension: 0.4, borderDash: [5,4] },
        { label: 'Vocal',  data: [], borderColor: '#818cf8', borderWidth: 1.5, pointRadius: 0, tension: 0.4, borderDash: [5,4] },
        { label: 'Gaze',   data: [], borderColor: '#fbbf24', borderWidth: 1.5, pointRadius: 0, tension: 0.4, borderDash: [5,4] },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 18, font: { size: 11 } } } },
      scales: {
        x: { display: false },
        y: { min: 0, max: 100,
             ticks: { color: '#475569', stepSize: 25 },
             grid:  { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });
}

function pushChart(scores) {
  if (!histChart) return;
  const hd = histChart.data;
  const t  = new Date().toLocaleTimeString('en', { hour12: false });
  hd.labels.push(t);
  hd.datasets[0].data.push(scores.total);
  hd.datasets[1].data.push(scores.facial);
  hd.datasets[2].data.push(scores.vocal);
  hd.datasets[3].data.push(scores.gaze);
  if (hd.labels.length > MAX_CHART_PTS) {
    hd.labels.shift();
    hd.datasets.forEach(ds => ds.data.shift());
  }
  histChart.update('none');
}

// ─── Status dots ─────────────────────────────────────────────────────────────

function updateDots(fd, vd) {
  dot('face',  fd?.detected  ? 'active' : 'loading');
  dot('gaze',  fd?.detected  ? 'active' : 'loading');
  dot('voice', vd?.is_speaking ? 'active' : 'idle');
}

// ─── Demo mode (preview / no server) ─────────────────────────────────────────

let demoFeedT = 0;

function loopDemo() {
  demoT += 0.04; demoFeedT += 0.02;
  const fd = _demoFace();
  const vd = _demoVocal();

  // Fake trust engine in JS for demo
  const scores = {
    total:  Math.round(50 + 28 * Math.sin(demoT * 0.18)),
    facial: Math.round(50 + 25 * Math.sin(demoT * 0.22)),
    vocal:  Math.round(50 + 20 * Math.sin(demoT * 0.31)),
    gaze:   Math.round(50 + 22 * Math.sin(demoT * 0.14)),
  };
  const label = _demoLabel(scores.total);

  _drawDemoFeed();
  renderScores(scores, label);
  renderFaceMetrics(fd);
  renderVocalMetrics(vd);
  updateDots(fd, vd);

  const now = Date.now();
  if (now - lastChartAt > 500) { pushChart(scores); lastChartAt = now; }
  requestAnimationFrame(loopDemo);
}

function _demoFace() {
  const ear = 0.27 + 0.06 * Math.sin(demoT * 0.4);
  return {
    detected:       true,
    dominant:       Math.sin(demoT * 0.3) > 0.3 ? 'happy' : 'neutral',
    eye_ar:         ear,
    l_ear:          ear + 0.01,
    r_ear:          ear - 0.01,
    blink_rate:     14 + 4 * Math.sin(demoT * 0.2),
    gaze_deviation: 0.06 + 0.04 * Math.abs(Math.sin(demoT * 0.15)),
  };
}

function _demoVocal() {
  const speaking = Math.sin(demoT * 0.25) > 0.1;
  return {
    is_speaking:     speaking,
    pitch_stability: 0.65 + 0.18 * Math.sin(demoT * 0.4),
    energy_level:    speaking ? 0.35 + 0.25 * Math.abs(Math.sin(demoT * 0.8)) : 0.02,
    tremor_index:    0.08 + 0.07 * Math.abs(Math.sin(demoT * 1.2)),
    dominant_hz:     speaking ? 160 + 40 * Math.sin(demoT * 0.35) : 0,
  };
}

function _demoLabel(score) {
  if (score >= 82) return { text: 'Very High Trust', color: '#4ade80' };
  if (score >= 64) return { text: 'High Trust',      color: '#34d399' };
  if (score >= 46) return { text: 'Neutral',         color: '#60a5fa' };
  if (score >= 28) return { text: 'Low Trust',       color: '#fb923c' };
  return                  { text: 'Very Low Trust',  color: '#f87171' };
}

function _drawDemoFeed() {
  const canvas = $('face-canvas');
  canvas.width = 640; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0a0a16';
  ctx.fillRect(0, 0, 640, 480);

  const cx = 320, cy = 210;
  const ear  = 0.27 + 0.06 * Math.sin(demoFeedT * 0.4);

  ctx.beginPath();
  ctx.ellipse(cx, cy, 90, 110, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(0,212,255,0.22)';
  ctx.lineWidth = 1.5; ctx.stroke();

  for (const [ex, lEar] of [[cx - 32, ear + 0.01], [cx + 32, ear - 0.01]]) {
    ctx.beginPath();
    ctx.ellipse(ex, cy - 20, 16, 16 * lEar, 0, 0, Math.PI * 2);
    ctx.strokeStyle = lEar > 0.21 ? 'rgba(74,222,128,0.6)' : 'rgba(251,191,36,0.6)';
    ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = lEar > 0.21 ? 'rgba(74,222,128,0.08)' : 'rgba(251,191,36,0.06)';
    ctx.fill();
  }

  ctx.strokeStyle = 'rgba(0,212,255,0.45)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(cx - 110, cy - 130, 220, 260);
  ctx.font = '12px system-ui';
  ctx.fillStyle = 'rgba(251,191,36,0.8)';
  ctx.fillText('DEMO — start server for live analysis', cx - 110, cy - 142);
}

function showDemoBanner() {
  const b = document.createElement('div');
  b.style.cssText = `position:fixed;bottom:14px;left:50%;transform:translateX(-50%);
    background:#1a1a2e;border:1px solid #fbbf24;border-radius:8px;
    padding:8px 18px;font-size:0.75rem;color:#fbbf24;z-index:50;
    display:flex;align-items:center;gap:8px;white-space:nowrap`;
  b.innerHTML = `<span>⚠</span> Demo mode &mdash; run
    <code style="background:#0f0f1a;padding:2px 6px;border-radius:4px">
      pip install -r requirements.txt &amp;&amp; python main.py
    </code> for live analysis`;
  document.body.appendChild(b);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const $    = id  => document.getElementById(id);
const set  = (id, v) => { const el = $(id); if (el) el.textContent = v; };
const setBar = (el, p) => { el.style.width = p + '%'; };
const delay  = ms => new Promise(r => setTimeout(r, ms));
function dot(id, state) {
  const el = $('dot-' + id);
  if (el) el.className = 'dot ' + state;
}

boot();
