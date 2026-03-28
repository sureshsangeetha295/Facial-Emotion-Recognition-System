// ── Data ──
const EMOTIONS = [
  { key: 'happiness', label: 'Happiness', icon: '😊', color: '#f59e0b' },
  { key: 'neutral',   label: 'Neutral',   icon: '😐', color: '#64748b' },
  { key: 'sadness',   label: 'Sadness',   icon: '😢', color: '#6366f1' },
  { key: 'anger',     label: 'Anger',     icon: '😠', color: '#ef4444' },
  { key: 'fear',      label: 'Fear',      icon: '😨', color: '#8b5cf6' },
  { key: 'disgust',   label: 'Disgust',   icon: '🤢', color: '#10b981' },
  { key: 'surprise',  label: 'Surprise',  icon: '😲', color: '#f97316' },
];

const EMOTION_MESSAGES = {
  happiness: "You're radiating positive energy — keep smiling! 🌟",
  neutral:   "Calm and composed. You seem focused and steady. 👍",
  sadness:   "It's okay to feel low sometimes. Take a deep breath. 💙",
  anger:     "Tension detected. Try taking a moment to cool down. 🧘",
  fear:      "Something seems unsettling. You're safe here. 🫂",
  disgust:   "Something's clearly not sitting right with you! 😅",
  surprise:  "Caught off guard! Something unexpected just happened? 🎉",
};

// Must match EMOTION_LABELS order in backend: ["Anger","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
const PROB_KEYS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'];

const API_URL = 'http://localhost:8000/predict/';

let isLive = false;
let liveInterval = null;
let timeline = [];
let currentProbs = { Anger:0, Disgust:0, Fear:0, Happiness:0, Neutral:0, Sadness:0, Surprise:0 };
let topEmotion = null;

// ── Capture frame from video as JPEG Blob ──
function captureFrame() {
  return new Promise((resolve, reject) => {
    const video = document.getElementById('videoFeed');
    if (!video || !video.srcObject) return reject(new Error('No camera stream'));
    const canvas = document.createElement('canvas');
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      if (blob) resolve(blob);
      else reject(new Error('Canvas toBlob failed'));
    }, 'image/jpeg', 0.92);
  });
}

// ── Call real model API ──
async function callPredict() {
  const t0 = performance.now();
  const blob = await captureFrame();

  const form = new FormData();
  form.append('file', blob, 'frame.jpg');

  const res = await fetch(API_URL, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || `HTTP ${res.status}`);
  }

  const data = await res.json();
  const elapsed = Math.round(performance.now() - t0);

  // data.emotion          → "Happiness"
  // data.confidence       → 0–1
  // data.all_probabilities → 7  in PROB_KEYS order

  const probs = {};
  PROB_KEYS.forEach((k, i) => {
    const raw = data.all_probabilities?.[i] ?? 0;
    // backend returns 0–1; convert to 0–100 rounded integer
    probs[k] = Math.round((raw > 1 ? raw : raw * 100));
  });

  // Normalise so they sum to 100
  const total = Object.values(probs).reduce((a, b) => a + b, 0);
  if (total > 0) PROB_KEYS.forEach(k => probs[k] = Math.round(probs[k] / total * 100));

  const emotion = data.emotion; // "Happiness"
  const conf = Math.round(data.confidence > 1 ? data.confidence : data.confidence * 100);

  return { probs, emotion, conf, elapsed };
}

// ── Build probability list ──
function buildProbList() {
  const el = document.getElementById('probList');
  el.innerHTML = '';
  PROB_KEYS.forEach(k => {
    const emo = EMOTIONS.find(e => e.label === k);
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-label"><span class="p-icon">${emo ? emo.icon : ''}</span>${k}</span>
      <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar-${k}" style="width:0%"></div></div>
      <span class="prob-pct" id="pct-${k}">0%</span>`;
    el.appendChild(row);
  });
}

// ── Update all UI panels ──
function updateUI(probs, top, conf) {
  const topLower = top?.toLowerCase();

  // Result panel
  document.getElementById('resultDot').style.background =
    EMOTIONS.find(e => e.label.toLowerCase() === topLower)?.color || '#94a3b8';
  document.getElementById('resultEmotion').textContent =
    top ? (top.charAt(0).toUpperCase() + top.slice(1)) : '—';
  document.getElementById('resultConf').textContent =
    top ? `${conf}% confidence` : 'No detection yet';

  // Probabilities
  PROB_KEYS.forEach(k => {
    const v = probs[k] || 0;
    const isActive = k.toLowerCase() === topLower;
    const bar = document.getElementById(`bar-${k}`);
    const pct = document.getElementById(`pct-${k}`);
    bar.style.width = v + '%';
    bar.className = 'prob-bar-fill' + (isActive ? ' active' : '');
    pct.textContent = v + '%';
    pct.className = 'prob-pct' + (isActive ? ' active' : '');
  });

  // Dominant breakdown card
  if (top) {
    const emo = EMOTIONS.find(e => e.label.toLowerCase() === topLower);
    const msg = EMOTION_MESSAGES[topLower] || '';
    const v = probs[emo ? emo.label : ''] || conf;
    const card = document.getElementById('dominantCard');
    card.className = 'dominant-card';
    card.innerHTML = `
      <div class="dom-icon">${emo ? emo.icon : '😐'}</div>
      <div class="dom-info">
        <div class="dom-name">${top.charAt(0).toUpperCase() + top.slice(1)}</div>
        <div class="dom-conf">${v}% confidence</div>
        <div class="dom-bar-bg"><div class="dom-bar-fill" style="width:${v}%"></div></div>
        <div class="dom-message">${top.charAt(0).toUpperCase() + top.slice(1)}: ${msg}</div>
      </div>`;
  }

  // Detection bar on camera
  if (top) {
    const bar = document.getElementById('detectionBar');
    bar.textContent = `Detected: ${top.charAt(0).toUpperCase() + top.slice(1)} ${conf}% confidence`;
    bar.classList.add('visible');
    document.getElementById('lastDetected').textContent =
      `Last detected: ${top.charAt(0).toUpperCase() + top.slice(1)}`;
  }
}

// ── Timeline dot ──
function addTimelineDot(emotion) {
  timeline.push(emotion);
  const container = document.getElementById('timelineDots');
  container.querySelectorAll('.t-dot.current').forEach(d => d.classList.remove('current'));
  const dot = document.createElement('div');
  dot.className = `t-dot ${emotion.toLowerCase()} current`;
  container.appendChild(dot);

  const countMap = {};
  timeline.forEach(e => countMap[e] = (countMap[e] || 0) + 1);
  const dominant = Object.entries(countMap).sort((a, b) => b[1] - a[1])[0];
  document.getElementById('timelineTag').innerHTML =
    `<span class="timeline-tag">${dominant[0].charAt(0).toUpperCase() + dominant[0].slice(1)} ${dominant[1]}</span>`;
}

// ── Show inline error on camera ──
function showError(msg) {
  const bar = document.getElementById('detectionBar');
  bar.textContent = `⚠ ${msg}`;
  bar.classList.add('visible');
  setTimeout(() => bar.classList.remove('visible'), 3000);
}

// ── Detection (calls real model) ──
async function runDetection() {
  try {
    const { probs, emotion, conf, elapsed } = await callPredict();

    topEmotion  = emotion;
    currentProbs = probs;

    updateUI(probs, emotion.toLowerCase(), conf);
    addTimelineDot(emotion.toLowerCase());

    const badge = document.getElementById('msBadge');
    badge.style.display = 'block';
    badge.textContent = elapsed + ' ms';
  } catch (err) {
    console.error('[EmotionAI] Detection error:', err);
    showError(err.message || 'Detection failed');
  }
}

// ── Button handlers ──
function doDetect() {
  runDetection();
}

function startLive() {
  if (isLive) return;
  isLive = true;
  document.getElementById('liveTag').classList.add('visible');
  document.getElementById('liveBadge').classList.add('active');
  document.getElementById('btnGoLive').style.display = 'none';
  document.getElementById('btnStop').style.display = 'flex';
  document.getElementById('btnDetect').disabled = true;
  liveInterval = setInterval(runDetection, 1500);
}

function stopLive() {
  if (!isLive) return;
  isLive = false;
  clearInterval(liveInterval);
  document.getElementById('liveTag').classList.remove('visible');
  document.getElementById('liveBadge').classList.remove('active');
  document.getElementById('btnGoLive').style.display = '';
  document.getElementById('btnStop').style.display = 'none';
  document.getElementById('btnDetect').disabled = false;
  document.getElementById('detectionBar').classList.remove('visible');
}

// ── Camera ──
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('videoFeed').srcObject = stream;
  } catch(e) {
    console.warn('[EmotionAI] Camera unavailable:', e);
  }
}

// ── Keyboard shortcuts ──
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); if (!isLive) doDetect(); }
  if (e.key === 'l' || e.key === 'L') { isLive ? stopLive() : startLive(); }
});

// ── Init ──
buildProbList();
startCamera();