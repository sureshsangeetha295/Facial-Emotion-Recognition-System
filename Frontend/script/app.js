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

const API_URL = '/predict/';

let isLive        = false;
let liveInterval  = null;
let timeline      = [];
let currentProbs  = { Anger:0, Disgust:0, Fear:0, Happiness:0, Neutral:0, Sadness:0, Surprise:0 };
let topEmotion    = null;

// ── FIX 1: AbortController — cancels ALL in-flight requests when Stop is pressed ──
let currentAbortController = null;

// ── FIX 2: Guard flag — prevents overlapping live detections ──
// (If inference takes 2 s and interval fires every 1.5 s, requests pile up.)
let detectionInProgress = false;


// ── Capture frame from video as JPEG Blob ──
function captureFrame(quality = 0.75) {
  return new Promise((resolve, reject) => {
    const video = document.getElementById('videoFeed');
    if (!video || !video.srcObject) return reject(new Error('No camera stream'));
    const canvas = document.createElement('canvas');
    // FIX 3: Capture at 320×240 instead of native resolution.
    // The backend resizes to 320×240 anyway for Haar; sending a smaller blob
    // cuts network/encode time meaningfully on localhost too.
    canvas.width  = 320;
    canvas.height = 240;
    canvas.getContext('2d').drawImage(video, 0, 0, 320, 240);
    canvas.toBlob(blob => {
      if (blob) resolve(blob);
      else reject(new Error('Canvas toBlob failed'));
    }, 'image/jpeg', quality);
  });
}

// ── Call real model API ──
// signal: AbortSignal from the current AbortController (lets stopLive cancel this)
// fast:   true = Haar (live), false = MTCNN (single detect, higher quality)
async function callPredict(signal, fast = true) {
  const t0 = performance.now();

  const blob = await captureFrame(fast ? 0.75 : 0.90);

  const form = new FormData();
  form.append('file', blob, 'frame.jpg');

  // Use ?save=false in live mode (high-FPS) — only save single-shot detections
  const saveParam = fast ? 'false' : 'true';
  const url = `${API_URL}?fast=${fast}&save=${saveParam}`;

  // Auth.apiFetch automatically attaches Bearer token + refreshes if expired
  const res = await Auth.apiFetch(url, { method: 'POST', body: form, signal });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || `HTTP ${res.status}`);
  }

  const data    = await res.json();
  const elapsed = Math.round(performance.now() - t0);

  const probs = {};
  PROB_KEYS.forEach((k, i) => {
    const raw = data.all_probabilities?.[i] ?? 0;
    probs[k] = Math.round((raw > 1 ? raw : raw * 100));
  });

  const total = Object.values(probs).reduce((a, b) => a + b, 0);
  if (total > 0) PROB_KEYS.forEach(k => probs[k] = Math.round(probs[k] / total * 100));

  const emotion = data.emotion;
  const conf    = Math.round(data.confidence > 1 ? data.confidence : data.confidence * 100);

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

  document.getElementById('resultDot').style.background =
    EMOTIONS.find(e => e.label.toLowerCase() === topLower)?.color || '#94a3b8';
  document.getElementById('resultEmotion').textContent =
    top ? (top.charAt(0).toUpperCase() + top.slice(1)) : '—';
  document.getElementById('resultConf').textContent =
    top ? `${conf}% confidence` : 'No detection yet';

  PROB_KEYS.forEach(k => {
    const v        = probs[k] || 0;
    const isActive = k.toLowerCase() === topLower;
    const bar      = document.getElementById(`bar-${k}`);
    const pct      = document.getElementById(`pct-${k}`);
    bar.style.width = v + '%';
    bar.className   = 'prob-bar-fill' + (isActive ? ' active' : '');
    pct.textContent = v + '%';
    pct.className   = 'prob-pct' + (isActive ? ' active' : '');
  });

  // Dominant breakdown card — animated
  if (top) {
    const emo   = EMOTIONS.find(e => e.label.toLowerCase() === topLower);
    const conf2 = probs[emo ? emo.label : ''] || conf;
    animateBreakdownCard(top, emo, conf2);
  }

  if (top) {
    const bar = document.getElementById('detectionBar');
    bar.textContent = `Detected: ${top.charAt(0).toUpperCase() + top.slice(1)} ${conf}% confidence`;
    bar.classList.add('visible');
    document.getElementById('lastDetected').textContent =
      `Last detected: ${top.charAt(0).toUpperCase() + top.slice(1)}`;
  }
}



// ── Avatar faces data ──
const AVATAR_FACES = {
  happiness: {
    faceColor:'#FDDFA4', cheekColor:'#F9A8A8', eyeShape:'happy',
    browAngle:0, mouth:'M 30 62 Q 44 76 58 62', mouthColor:'#C0392B',
    showTeeth:true, eyeColor:'#3d2800', pupilSize:5,
  },
  neutral: {
    faceColor:'#FDDFA4', cheekColor:'transparent', eyeShape:'normal',
    browAngle:0, mouth:'M 30 64 Q 44 66 58 64', mouthColor:'#8B6555',
    showTeeth:false, eyeColor:'#3d2800', pupilSize:4.5,
  },
  sadness: {
    faceColor:'#C9D8F0', cheekColor:'#A8C0E8', eyeShape:'sad',
    browAngle:-12, mouth:'M 30 68 Q 44 58 58 68', mouthColor:'#5563A0',
    showTeeth:false, eyeColor:'#2a3a6b', pupilSize:4,
  },
  anger: {
    faceColor:'#F4A09A', cheekColor:'#E86050', eyeShape:'angry',
    browAngle:18, mouth:'M 30 68 Q 44 60 58 68', mouthColor:'#8B1A1A',
    showTeeth:true, eyeColor:'#5a0000', pupilSize:3.5,
  },
  fear: {
    faceColor:'#D8C8EE', cheekColor:'transparent', eyeShape:'wide',
    browAngle:-8, mouth:'M 32 66 Q 44 72 56 66', mouthColor:'#6a40b0',
    showTeeth:false, eyeColor:'#2d1a5e', pupilSize:6.5,
  },
  disgust: {
    faceColor:'#B8E8C8', cheekColor:'transparent', eyeShape:'squint',
    browAngle:10, mouth:'M 28 64 Q 36 70 44 64 Q 52 58 58 64', mouthColor:'#2d7a4a',
    showTeeth:false, eyeColor:'#1a4d2e', pupilSize:4,
  },
  surprise: {
    faceColor:'#FDDFA4', cheekColor:'#FFB347', eyeShape:'wide',
    browAngle:-15, mouth:'M 36 62 Q 44 78 52 62', mouthColor:'#C0392B',
    showTeeth:false, eyeColor:'#3d2800', pupilSize:7,
  },
};

function _buildEyes(f) {
  const { eyeColor, pupilSize, eyeShape } = f;
  if (eyeShape === 'happy') return `
    <path d="M 26 42 Q 30 36 34 42" fill="none" stroke="${eyeColor}" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M 54 42 Q 58 36 62 42" fill="none" stroke="${eyeColor}" stroke-width="2.5" stroke-linecap="round"/>`;
  if (eyeShape === 'sad') return `
    <ellipse cx="30" cy="42" rx="5.5" ry="5" fill="white"/>
    <ellipse cx="58" cy="42" rx="5.5" ry="5" fill="white"/>
    <ellipse cx="30" cy="43" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="58" cy="43" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="28.5" cy="41.5" rx="1.5" ry="1.5" fill="white" opacity="0.7"/>
    <ellipse cx="56.5" cy="41.5" rx="1.5" ry="1.5" fill="white" opacity="0.7"/>
    <ellipse cx="34" cy="50" rx="2" ry="3" fill="#a8c0e8" opacity="0.7"/>`;
  if (eyeShape === 'angry') return `
    <ellipse cx="30" cy="42" rx="5" ry="4.5" fill="white"/>
    <ellipse cx="58" cy="42" rx="5" ry="4.5" fill="white"/>
    <ellipse cx="30" cy="43" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="58" cy="43" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>`;
  if (eyeShape === 'wide') return `
    <ellipse cx="30" cy="42" rx="7" ry="7" fill="white"/>
    <ellipse cx="58" cy="42" rx="7" ry="7" fill="white"/>
    <ellipse cx="30" cy="42" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="58" cy="42" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="28.5" cy="40.5" rx="1.8" ry="1.8" fill="white" opacity="0.75"/>
    <ellipse cx="56.5" cy="40.5" rx="1.8" ry="1.8" fill="white" opacity="0.75"/>`;
  if (eyeShape === 'squint') return `
    <path d="M 25 42 Q 30 38 35 42" fill="white" stroke="${eyeColor}" stroke-width="1"/>
    <path d="M 53 42 Q 58 38 63 42" fill="white" stroke="${eyeColor}" stroke-width="1"/>
    <ellipse cx="30" cy="41" rx="${pupilSize-1}" ry="${pupilSize-1.5}" fill="${eyeColor}"/>
    <ellipse cx="58" cy="41" rx="${pupilSize-1}" ry="${pupilSize-1.5}" fill="${eyeColor}"/>`;
  return `
    <ellipse cx="30" cy="42" rx="5.5" ry="5.5" fill="white"/>
    <ellipse cx="58" cy="42" rx="5.5" ry="5.5" fill="white"/>
    <ellipse cx="30" cy="42" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="58" cy="42" rx="${pupilSize}" ry="${pupilSize}" fill="${eyeColor}"/>
    <ellipse cx="28.5" cy="40.5" rx="1.5" ry="1.5" fill="white" opacity="0.7"/>
    <ellipse cx="56.5" cy="40.5" rx="1.5" ry="1.5" fill="white" opacity="0.7"/>`;
}

function _buildBrow(cx, cy, angle, emotionKey) {
  const browColor = emotionKey === 'anger' ? '#8B1A1A' : '#3d2800';
  return `<rect x="${cx-8}" y="${cy-1.5}" width="16" height="3" rx="2" fill="${browColor}"
    transform="rotate(${angle} ${cx} ${cy})"/>`;
}

function _buildAvatarSVG(emotionKey) {
  const f = AVATAR_FACES[emotionKey] || AVATAR_FACES.neutral;
  const cheeks = f.cheekColor !== 'transparent'
    ? `<ellipse cx="20" cy="58" rx="10" ry="6" fill="${f.cheekColor}" opacity="0.45"/>
       <ellipse cx="68" cy="58" rx="10" ry="6" fill="${f.cheekColor}" opacity="0.45"/>`
    : '';
  const mouthFill = f.showTeeth
    ? `<path d="${f.mouth}" fill="${f.mouthColor}" stroke="${f.mouthColor}" stroke-width="1.5" stroke-linecap="round"/>
       <ellipse cx="44" cy="67" rx="8" ry="4" fill="white" opacity="0.85"/>`
    : `<path d="${f.mouth}" fill="none" stroke="${f.mouthColor}" stroke-width="2.5" stroke-linecap="round"/>`;
  return `<svg viewBox="0 0 88 88" xmlns="http://www.w3.org/2000/svg" style="width:88px;height:88px;display:block;">
  <defs>
    <radialGradient id="fg_${emotionKey}" cx="45%" cy="40%" r="55%">
      <stop offset="0%" stop-color="white" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="${f.faceColor}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <ellipse cx="44" cy="46" rx="34" ry="36" fill="${f.faceColor}"/>
  <ellipse cx="44" cy="46" rx="34" ry="36" fill="url(#fg_${emotionKey})"/>
  <ellipse cx="10" cy="46" rx="5" ry="7" fill="${f.faceColor}"/>
  <ellipse cx="78" cy="46" rx="5" ry="7" fill="${f.faceColor}"/>
  <ellipse cx="44" cy="14" rx="32" ry="14" fill="#3d2800"/>
  <rect x="12" y="10" width="64" height="12" fill="#3d2800" rx="4"/>
  ${cheeks}
  ${_buildBrow(22, 30, f.browAngle, emotionKey)}
  ${_buildBrow(54, 30, -f.browAngle, emotionKey)}
  ${_buildEyes(f)}
  <ellipse cx="44" cy="56" rx="3.5" ry="2.5" fill="rgba(0,0,0,0.1)"/>
  ${mouthFill}
</svg>`;
}

// ── Animated breakdown card — avatar version ──
let _typewriterTimer = null;
let _particleTimers  = [];

function animateBreakdownCard(emotion, emo, conf) {
  const card    = document.getElementById('dominantCard');
  const label   = emotion.charAt(0).toUpperCase() + emotion.slice(1);
  const fullMsg = EMOTION_MESSAGES[emotion.toLowerCase()] || '';
  const emoKey  = emotion.toLowerCase();
  const accentC = emo?.color || '#64748b';

  clearTimeout(_typewriterTimer);
  _particleTimers.forEach(clearTimeout);
  _particleTimers = [];

  card.className = 'dominant-card animating';
  card.innerHTML = `
    <div class="avatar-col" data-emotion="${emoKey}">
      <div class="avatar-glow"></div>
      <div class="avatar-face" id="avatarFace">${_buildAvatarSVG(emoKey)}</div>
      <div class="avatar-label" id="avatarLabel">${label}</div>
    </div>
    <div class="message-col">
      <div class="dom-name" id="domName" data-emotion="${emoKey}" style="color:${accentC}">${label}</div>
      <div class="dom-conf-pill" id="domPill">⚡ ${conf}% confidence</div>
      <div class="dom-message" id="domMsg"><span class="dom-cursor"></span></div>
    </div>`;

  setTimeout(() => card.classList.remove('animating'), 750);

  requestAnimationFrame(() => {
    const face = document.getElementById('avatarFace');
    if (face) {
      face.classList.add('visible');
      setTimeout(() => face.classList.add('idle'), 550);
    }
  });

  setTimeout(() => {
    const lbl  = document.getElementById('avatarLabel');
    const name = document.getElementById('domName');
    if (lbl)  lbl.classList.add('visible');
    if (name) name.classList.add('revealed');
  }, 220);

  setTimeout(() => {
    const pill = document.getElementById('domPill');
    if (pill) pill.classList.add('revealed');
  }, 380);

  setTimeout(() => typewriteMessage(fullMsg, 0), 560);

  [300, 650, 1000].forEach((delay, i) => {
    const t = setTimeout(() => {
      const avatarCol = card.querySelector('.avatar-col');
      if (avatarCol) _spawnParticle(avatarCol, emo?.icon || '😐', i);
    }, delay);
    _particleTimers.push(t);
  });
}

function typewriteMessage(text, index) {
  const msgEl = document.getElementById('domMsg');
  if (!msgEl) return;
  if (index === 0) msgEl.innerHTML = '<span class="dom-cursor"></span>';
  if (index >= text.length) {
    _typewriterTimer = setTimeout(() => {
      const cursor = msgEl.querySelector('.dom-cursor');
      if (cursor) cursor.style.display = 'none';
    }, 900);
    return;
  }
  const cursor = msgEl.querySelector('.dom-cursor');
  if (cursor) cursor.insertAdjacentText('beforebegin', text[index]);
  const ch    = text[index];
  const delay = ch === ' ' ? 52 : (ch === ',' || ch === '.' || ch === '!') ? 155 : 36;
  _typewriterTimer = setTimeout(() => typewriteMessage(text, index + 1), delay);
}

function _spawnParticle(container, icon, idx) {
  const p = document.createElement('span');
  p.className   = 'dom-particle';
  p.textContent = icon;
  p.style.left   = (15 + idx * 25) + '%';
  p.style.bottom = '8px';
  container.appendChild(p);
  setTimeout(() => p.remove(), 1500);
}


// ── Timeline chip-row ──
function addTimelineDot(emotion) {
  timeline.push(emotion);

  const container = document.getElementById('timelineDots');
  const tagEl     = document.getElementById('timelineTag');
  if (!container) return;

  const colorMap = {};
  EMOTIONS.forEach(e => { colorMap[e.key] = e.color; });

  // Rebuild chip row from full timeline history
  container.innerHTML = timeline.map((e, i) => {
    const c      = colorMap[e] || '#6b7280';
    const isLast = i === timeline.length - 1;
    const label  = e.charAt(0).toUpperCase() + e.slice(1);
    const connector = i < timeline.length - 1
      ? `<div class="tl-connector"></div>` : '';

    if (isLast) {
      return `<div class="tl-chip tl-chip--active"
                   style="border-color:${c};background:${c}22;">
                <div class="tl-chip-dot" style="background:${c};"></div>
                <span class="tl-chip-label" style="color:${c};">${label}</span>
              </div>${connector}`;
    }
    return `<div class="tl-chip tl-chip--done">
              <div class="tl-chip-dot" style="background:${c};opacity:0.7;"></div>
              <span class="tl-chip-label">${label}</span>
            </div>${connector}`;
  }).join('');

  // Scroll to show latest chip
  container.scrollLeft = container.scrollWidth;

  // Summary count chips
  if (tagEl) {
    const countMap = {};
    timeline.forEach(e => { countMap[e] = (countMap[e] || 0) + 1; });
    const top3 = Object.entries(countMap).sort((a, b) => b[1] - a[1]).slice(0, 3);
    tagEl.innerHTML = top3.map(([e, n]) => {
      const c = colorMap[e] || '#6b7280';
      return `<span class="tl-summary-chip"
                    style="background:${c}18;border-color:${c}44;color:${c};">
                ${e.charAt(0).toUpperCase() + e.slice(1)} <b>${n}</b>
              </span>`;
    }).join('');
  }
}


// ── Show inline error on camera ──
function showError(msg) {
  const bar = document.getElementById('detectionBar');
  bar.textContent = `⚠ ${msg}`;
  bar.classList.add('visible');
  setTimeout(() => bar.classList.remove('visible'), 3000);
}


// ── Core detection runner ──
// fast=true  → Haar + 320×240 (live mode)
// fast=false → MTCNN + higher quality (single detect button)
async function runDetection(fast = true) {
  // Abort any in-flight request from a previous call
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  try {
    const { probs, emotion, conf, elapsed } = await callPredict(
      currentAbortController.signal,
      fast
    );

    // If we are no longer live and this was a live call, discard result
    if (fast && !isLive) return;

    topEmotion   = emotion;
    currentProbs = probs;
    _frameCount++;

    updateUI(probs, emotion.toLowerCase(), conf);
    addTimelineDot(emotion.toLowerCase());

    const badge = document.getElementById('msBadge');
    badge.style.display = 'block';
    badge.textContent   = elapsed + ' ms';

  } catch (err) {
    if (err.name === 'AbortError') return;  // Intentionally cancelled — not an error
    console.error('[EmotionAI] Detection error:', err);
    showError(err.message || 'Detection failed');
  } finally {
    detectionInProgress = false;
  }
}


// ── Button: single detect (high quality, MTCNN) ──
function doDetect() {
  if (detectionInProgress) return;
  detectionInProgress = true;
  runDetection(false);   // fast=false → MTCNN for better single-shot accuracy
}


// ── Live mode: schedules repeated detections safely ──
function scheduleLiveDetection() {
  // FIX: Use a recursive setTimeout instead of setInterval.
  // This ensures we only fire the NEXT request AFTER the previous one finishes,
  // so requests never pile up regardless of how slow inference is.
  liveInterval = setTimeout(async () => {
    if (!isLive) return;    // Stopped — don't fire

    if (!detectionInProgress) {
      detectionInProgress = true;
      await runDetection(true);  // fast=true → Haar
    }

    scheduleLiveDetection();    // Schedule next only after this one completes
  }, 300);   // 300 ms delay between completions (~3 fps max, adjustable)
}


let _sessionId    = null;
let _frameCount   = 0;

async function startLive() {
  if (isLive) return;
  isLive = true;
  detectionInProgress = false;
  _frameCount = 0;

  document.getElementById('liveTag').classList.add('visible');
  document.getElementById('liveBadge').classList.add('active');
  document.getElementById('btnGoLive').style.display  = 'none';
  document.getElementById('btnStop').style.display    = 'flex';
  document.getElementById('btnDetect').disabled       = true;

  // Start a DB session for this live run
  try {
    const res  = await Auth.apiFetch('/sessions/start/', { method: 'POST' });
    const data = await res.json();
    _sessionId = data.session_id;
  } catch (e) {
    console.warn('[EmotionAI] Could not start session:', e);
  }

  scheduleLiveDetection();
}


async function stopLive() {
  if (!isLive) return;
  isLive = false;

  clearTimeout(liveInterval);
  liveInterval = null;

  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }

  detectionInProgress = false;

  document.getElementById('liveTag').classList.remove('visible');
  document.getElementById('liveBadge').classList.remove('active');
  document.getElementById('btnGoLive').style.display = '';
  document.getElementById('btnStop').style.display   = 'none';
  document.getElementById('btnDetect').disabled      = false;
  document.getElementById('detectionBar').classList.remove('visible');

  // End the DB session
  if (_sessionId) {
    try {
      await Auth.apiFetch('/sessions/end/', {
        method: 'POST',
        body:   JSON.stringify({ session_id: _sessionId, total_frames: _frameCount }),
      });
    } catch (e) {
      console.warn('[EmotionAI] Could not end session:', e);
    }
    _sessionId  = null;
    _frameCount = 0;
  }
}


// ── Camera ──
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('videoFeed').srcObject = stream;
  } catch (e) {
    console.warn('[EmotionAI] Camera unavailable:', e);
  }
}


// ── Keyboard shortcuts ──
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); if (!isLive) doDetect(); }
  if (e.key  === 'l' || e.key === 'L') { isLive ? stopLive() : startLive(); }
});


// ── Init ──
// Redirect to login if not authenticated
if (!Auth.requireAuth()) throw new Error('Not authenticated');

// Show logged-in user's name in topbar if element exists
const _user = Auth.getUser();
const _userEl = document.getElementById('topbarUser');
if (_userEl && _user) _userEl.textContent = _user.username || _user.email || '';

buildProbList();
startCamera();