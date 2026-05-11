import { FaceAnalyzer } from './face-analyzer.js';
import { VocalAnalyzer } from './vocal-analyzer.js';
import { TrustEngine }   from './trust-engine.js';

const face  = new FaceAnalyzer();
const vocal = new VocalAnalyzer();
const trust = new TrustEngine();

let histChart    = null;
let lastFaceAt   = 0;
let lastFaceData = null;
let lastChartAt  = 0;
let demoMode     = false;
let demoT        = 0;

const FACE_INTERVAL  = 67;
const CHART_INTERVAL = 500;
const MAX_CHART_PTS  = 60;

// ─── Boot ────────────────────────────────────────────────────────────────────

async function boot() {
  const overlay = $('loading-overlay');
  const txt     = $('loading-text');
  const bar     = $('loading-bar');

  // Check for secure context / mediaDevices support
  const hasMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  try {
    setBar(bar, 5);

    if (!hasMedia) {
      // Skip model loading in demo mode — just animate the progress
      txt.textContent = 'Demo mode (no camera/mic available)…';
      await delay(400); setBar(bar, 35);
      await delay(400); setBar(bar, 70);
      await delay(300); setBar(bar, 100);
    } else {
      await face.init(msg => { txt.textContent = msg; });
      setBar(bar, 65);

      txt.textContent = 'Requesting camera…';
      const vid = $('video');
      const cam = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
      });
      vid.srcObject = cam;
      await new Promise(r => vid.addEventListener('loadedmetadata', r, { once: true }));
      setBar(bar, 82);

      txt.textContent = 'Requesting microphone…';
      await vocal.init();
      setBar(bar, 100);
    }

    initChart();
    await delay(350);
    overlay.style.display = 'none';

    if (!hasMedia) {
      demoMode = true;
      showDemoBanner();
      dot('face',  'active');
      dot('gaze',  'active');
      dot('voice', 'active');
      loopDemo();
    } else {
      dot('face',  'loading');
      dot('gaze',  'loading');
      dot('voice', vocal.ready ? 'active' : 'off');
      loop();
    }

  } catch (err) {
    // Any error → fall back to demo mode
    txt.textContent = 'Live access unavailable — starting demo mode…';
    bar.style.background = '#fbbf24';
    await delay(1200);
    initChart();
    overlay.style.display = 'none';
    demoMode = true;
    showDemoBanner();
    dot('face',  'active');
    dot('gaze',  'active');
    dot('voice', 'active');
    loopDemo();
  }
}

// ─── Demo mode ───────────────────────────────────────────────────────────────
// Generates realistic sinusoidal signals so every panel is interactive.

function demoFaceData() {
  const t   = demoT;
  const happy   = 0.45 + 0.3  * Math.sin(t * 0.7)  * Math.cos(t * 0.3);
  const neutral = 0.35 - 0.15 * Math.sin(t * 0.5);
  const fearful = Math.max(0, 0.05 + 0.08 * Math.sin(t * 1.1));
  const angry   = Math.max(0, 0.02 + 0.04 * Math.sin(t * 0.9 + 1));
  const sad     = Math.max(0, 0.03 + 0.05 * Math.sin(t * 0.6));
  const sum     = happy + neutral + fearful + angry + sad + 0.02 + 0.01;
  const ear     = 0.27 + 0.06  * Math.sin(t * 0.4);
  const blink   = 14   + 4     * Math.sin(t * 0.2);

  // Slow gaze drift
  const gazeDeviation = 0.06 + 0.05 * Math.abs(Math.sin(t * 0.15));

  const expr = {
    happy:     happy   / sum,
    neutral:   neutral / sum,
    fearful:   fearful / sum,
    angry:     angry   / sum,
    sad:       sad     / sum,
    disgusted: 0.01    / sum,
    surprised: 0.02    / sum,
  };

  const dominant = Object.entries(expr).sort((a, b) => b[1] - a[1])[0][0];

  return {
    detected: true,
    expressions: expr,
    dominant,
    eyeAR: ear,
    lEAR: ear + 0.01 * Math.sin(t),
    rEAR: ear - 0.01 * Math.sin(t),
    blinkRate: blink,
    gazeDeviation,
    // No box/lEye/rEye — overlay skipped in demo
  };
}

function demoVocalData() {
  const t   = demoT;
  const speaking = (Math.sin(t * 0.25) > 0.1);
  return {
    isSpeaking:     speaking,
    pitchStability: 0.68 + 0.18 * Math.sin(t * 0.4),
    energyLevel:    speaking ? (0.35 + 0.25 * Math.abs(Math.sin(t * 0.8))) : 0.02,
    tremorIndex:    0.08 + 0.07 * Math.abs(Math.sin(t * 1.2)),
    dominantHz:     speaking ? (160 + 40 * Math.sin(t * 0.35)) : 0,
  };
}

function loopDemo() {
  demoT += 0.04;
  const now = Date.now();

  const fd = demoFaceData();
  const vd = demoVocalData();
  const scores = trust.update(fd, vd);

  // Overlay skipped (no real video), but draw placeholder
  drawDemoFeed();
  renderScores(scores);
  renderFaceMetrics(fd);
  renderVocalMetrics(vd);
  drawDemoWaveform(vd);
  updateDots(fd, vd);

  if (now - lastChartAt > CHART_INTERVAL) {
    pushChart(scores);
    lastChartAt = now;
  }

  requestAnimationFrame(loopDemo);
}

// Animated placeholder for the camera feed in demo mode
let demoFeedT = 0;
function drawDemoFeed() {
  demoFeedT += 0.02;
  const vid    = $('video');
  const canvas = $('face-canvas');
  canvas.width  = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');

  // Dark background
  ctx.fillStyle = '#0a0a16';
  ctx.fillRect(0, 0, 640, 480);

  // Subtle face silhouette
  const cx = 320, cy = 210;
  const pulse = 1 + 0.01 * Math.sin(demoFeedT * 2);

  // Head outline
  ctx.beginPath();
  ctx.ellipse(cx, cy, 90 * pulse, 110 * pulse, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(0,212,255,0.25)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Eyes
  const eyeY = cy - 20;
  const lx = cx - 32, rx = cx + 32;
  const eyeOpen = 0.27 + 0.06 * Math.sin(demoFeedT * 0.4);

  for (const ex of [lx, rx]) {
    ctx.beginPath();
    ctx.ellipse(ex, eyeY, 16, 16 * eyeOpen, 0, 0, Math.PI * 2);
    ctx.strokeStyle = eyeOpen > 0.21 ? 'rgba(74,222,128,0.6)' : 'rgba(251,191,36,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = eyeOpen > 0.21 ? 'rgba(74,222,128,0.08)' : 'rgba(251,191,36,0.06)';
    ctx.fill();
  }

  // Face bounding box
  ctx.strokeStyle = 'rgba(0,212,255,0.5)';
  ctx.lineWidth   = 1.5;
  ctx.strokeRect(cx - 110, cy - 130, 220, 260);

  // Demo label
  ctx.font      = '12px system-ui';
  ctx.fillStyle = 'rgba(0,212,255,0.6)';
  ctx.fillText('DEMO MODE', cx - 110, cy - 142);

  // Expression label
  ctx.font      = 'bold 13px system-ui';
  ctx.fillStyle = 'rgba(0,212,255,0.85)';
  ctx.fillText('happy', cx - 110, cy - 142 + 14);
}

function drawDemoWaveform(vd) {
  const wc   = $('wave-canvas');
  const wCtx = wc.getContext('2d');
  const W = wc.width, H = wc.height;
  wCtx.fillStyle = '#0d0d1c';
  wCtx.fillRect(0, 0, W, H);

  wCtx.beginPath();
  wCtx.strokeStyle = vd.isSpeaking ? '#60a5fa' : '#2d3748';
  wCtx.lineWidth   = 1.5;
  const amp = vd.isSpeaking ? vd.energyLevel * 0.7 : 0.03;
  for (let i = 0; i < W; i++) {
    const t   = (i / W) * Math.PI * 6 + demoT * 4;
    const noise = amp * (Math.sin(t) + 0.3 * Math.sin(t * 2.5) + 0.15 * Math.sin(t * 5.3));
    const y = H / 2 + noise * H * 0.45;
    i === 0 ? wCtx.moveTo(0, y) : wCtx.lineTo(i, y);
  }
  wCtx.stroke();

  const sc   = $('spec-canvas');
  const sCtx = sc.getContext('2d');
  const SW = sc.width, SH = sc.height;
  sCtx.fillStyle = '#0d0d1c';
  sCtx.fillRect(0, 0, SW, SH);
  const bars = 80;
  const bw   = SW / bars;
  for (let i = 0; i < bars; i++) {
    const freq  = (i / bars);
    const base  = vd.isSpeaking ? Math.exp(-Math.pow((freq - 0.15) * 6, 2)) * 0.9 : 0.05;
    const noise = base * (0.8 + 0.2 * Math.sin(demoT * 3 + i * 0.4));
    const bh    = noise * SH;
    sCtx.fillStyle = `hsl(${185 + noise * 55}, 80%, 52%)`;
    sCtx.fillRect(i * bw, SH - bh, bw - 1, bh);
  }
}

function showDemoBanner() {
  const banner = document.createElement('div');
  banner.style.cssText = `
    position:fixed; bottom:14px; left:50%; transform:translateX(-50%);
    background:#1a1a2e; border:1px solid #fbbf24; border-radius:8px;
    padding:8px 18px; font-size:0.75rem; color:#fbbf24; z-index:50;
    display:flex; align-items:center; gap:8px;
  `;
  banner.innerHTML = `<span style="font-size:1rem">⚠</span>
    Demo mode — run via <code style="background:#0f0f1a;padding:2px 6px;border-radius:4px">npx serve trust-dashboard -p 3000</code>
    for live camera &amp; microphone`;
  document.body.appendChild(banner);
}

// ─── Live mode loop ──────────────────────────────────────────────────────────

async function loop() {
  const now    = Date.now();
  const vid    = $('video');
  const canvas = $('face-canvas');

  if (canvas.width !== vid.videoWidth) {
    canvas.width  = vid.videoWidth  || 640;
    canvas.height = vid.videoHeight || 480;
  }

  if (now - lastFaceAt > FACE_INTERVAL) {
    lastFaceData = await face.analyze(vid);
    lastFaceAt   = now;
  }

  const vd     = vocal.analyze();
  const scores = trust.update(lastFaceData, vd);

  drawOverlay(canvas, lastFaceData);
  renderScores(scores);
  renderFaceMetrics(lastFaceData);
  renderVocalMetrics(vd);
  drawWaveform(vd);
  updateDots(lastFaceData, vd);

  if (now - lastChartAt > CHART_INTERVAL) {
    pushChart(scores);
    lastChartAt = now;
  }

  requestAnimationFrame(loop);
}

// ─── Face overlay (live) ─────────────────────────────────────────────────────

function drawOverlay(canvas, fd) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!fd?.detected) return;

  const { box, lEye, rEye, lEAR, rEAR, dominant } = fd;

  ctx.strokeStyle = 'rgba(0,212,255,0.85)';
  ctx.lineWidth   = 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.fillStyle   = 'rgba(0,212,255,0.06)';
  ctx.fillRect(box.x, box.y, box.width, box.height);

  ctx.font      = 'bold 13px system-ui';
  ctx.fillStyle = '#00d4ff';
  ctx.fillText(dominant, box.x + 4, box.y - 8);

  drawEye(ctx, lEye, lEAR);
  drawEye(ctx, rEye, rEAR);
}

function drawEye(ctx, pts, ear) {
  if (!pts) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  const open = ear > 0.21;
  ctx.strokeStyle = open ? '#4ade80' : '#fbbf24';
  ctx.lineWidth   = 1.5;
  ctx.stroke();
  ctx.fillStyle   = open ? 'rgba(74,222,128,0.12)' : 'rgba(251,191,36,0.1)';
  ctx.fill();
}

// ─── Score UI ────────────────────────────────────────────────────────────────

const GAUGE_LEN = 251.2;

function renderScores(scores) {
  const arc = $('gauge-arc');
  const num = $('gauge-num');
  const lbl = $('trust-label');
  const { text, color } = trust.trustLabel(scores.total);

  arc.setAttribute('stroke-dashoffset', GAUGE_LEN * (1 - scores.total / 100));
  arc.setAttribute('stroke', color);
  num.textContent = scores.total;
  num.setAttribute('fill', color);
  lbl.textContent = text;
  lbl.style.color = color;

  ['facial', 'vocal', 'gaze'].forEach(k => {
    $(`bar-${k}`).style.width = `${scores[k]}%`;
    $(`num-${k}`).textContent = scores[k];
  });
}

function renderFaceMetrics(fd) {
  if (fd?.detected) {
    set('expr-val',     fd.dominant);
    set('ear-val',      (fd.eyeAR * 100).toFixed(0) + '%');
    set('blink-val',    fd.blinkRate.toFixed(0) + '/min');
    set('gaze-dev-val', (fd.gazeDeviation * 100).toFixed(0) + '%');
  } else {
    ['expr-val', 'ear-val', 'blink-val', 'gaze-dev-val'].forEach(id => set(id, '—'));
  }
}

function renderVocalMetrics(vd) {
  set('pitch-val',    (vd.pitchStability * 100).toFixed(0) + '%');
  set('energy-val',   (vd.energyLevel    * 100).toFixed(0) + '%');
  set('tremor-val',   (vd.tremorIndex    * 100).toFixed(0) + '%');
  set('hz-val',       vd.dominantHz ? vd.dominantHz.toFixed(0) + ' Hz' : '—');
  set('speaking-val', vd.isSpeaking ? 'Yes' : 'No');
  $('speaking-val').style.color = vd.isSpeaking ? '#4ade80' : '#94a3b8';
}

// ─── Waveform + Spectrum (live) ──────────────────────────────────────────────

function drawWaveform(vd) {
  const wc   = $('wave-canvas');
  const wCtx = wc.getContext('2d');
  const W = wc.width, H = wc.height;
  wCtx.fillStyle = '#0d0d1c';
  wCtx.fillRect(0, 0, W, H);
  if (!vocal.ready) return;

  const td = vocal.getTimeDomainData();
  wCtx.beginPath();
  wCtx.strokeStyle = vd.isSpeaking ? '#60a5fa' : '#2d3748';
  wCtx.lineWidth   = 1.5;
  const step = W / td.length;
  for (let i = 0; i < td.length; i++) {
    const y = ((td[i] - 128) / 128) * (H / 2) + H / 2;
    i === 0 ? wCtx.moveTo(0, y) : wCtx.lineTo(i * step, y);
  }
  wCtx.stroke();

  const sc   = $('spec-canvas');
  const sCtx = sc.getContext('2d');
  const SW = sc.width, SH = sc.height;
  sCtx.fillStyle = '#0d0d1c';
  sCtx.fillRect(0, 0, SW, SH);
  const fd   = vocal.getFreqData();
  const bars = 80;
  const bw   = SW / bars;
  const skip = Math.floor(fd.length / bars);
  for (let i = 0; i < bars; i++) {
    const v  = fd[i * skip] / 255;
    const bh = v * SH;
    sCtx.fillStyle = `hsl(${185 + v * 55}, 80%, 52%)`;
    sCtx.fillRect(i * bw, SH - bh, bw - 1, bh);
  }
}

// ─── History chart ───────────────────────────────────────────────────────────

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
        y: {
          min: 0, max: 100,
          ticks: { color: '#475569', stepSize: 25 },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
      },
    },
  });
}

function pushChart(scores) {
  if (!histChart) return;
  const t  = new Date().toLocaleTimeString('en', { hour12: false });
  const hd = histChart.data;
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
  dot('face',  fd?.detected ? 'active' : 'loading');
  dot('gaze',  fd?.detected ? 'active' : 'loading');
  dot('voice', vd?.isSpeaking ? 'active' : (vocal.ready ? 'idle' : 'off'));
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const $    = id  => document.getElementById(id);
const set  = (id, v) => { const el = $(id); if (el) el.textContent = v; };
const setBar = (el, pct) => { el.style.width = pct + '%'; };
const delay  = ms => new Promise(r => setTimeout(r, ms));

function dot(id, state) {
  const el = $('dot-' + id);
  if (el) el.className = 'dot ' + state;
}

boot();
