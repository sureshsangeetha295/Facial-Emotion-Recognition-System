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
  happiness: "Student is highly engaged and motivated — ideal moment to introduce a harder concept or go deeper into the topic. Keep the energy alive! 🌟",
  neutral:   "Student appears calm and attentive but passively so. Pose a direct question or invite them to explain a concept back to you to boost active participation. 👍",
  sadness:   "Student seems low in mood or unmotivated. Pause the content briefly, offer a warm check-in, and give genuine encouragement before continuing. 💙",
  anger:     "Frustration detected — the student may be struggling or feeling stuck. Slow down, simplify the current explanation, and validate their effort before moving forward. 🧘",
  fear:      "Anxiety signals present — student may feel overwhelmed. Break the task into very small achievable steps, reassure them, and confirm understanding at each stage. 🫂",
  disgust:   "Student appears disengaged or uninterested. Try switching the delivery method, linking the topic to something they care about, or introducing a hands-on activity. 😅",
  surprise:  "Attention is at a peak — the student is highly alert and receptive right now. Use this moment to clarify, reinforce, or introduce the key concept clearly. 🎉",
};

// Must match EMOTION_LABELS order in backend: ["Anger","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
const PROB_KEYS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'];

// Client-side engagement score map (mirrors backend ENGAGEMENT_SCORES)
const ENGAGEMENT_MAP = {
  Happiness: 1.0, Surprise: 1.0,
  Neutral:   0.6,
  Sadness:   0.3, Fear: 0.3,
  Anger:     0.1, Disgust: 0.1,
};

const API_URL = '/predict/';

let isLive        = false;
let liveInterval  = null;
let timeline      = [];
let currentProbs  = { Anger:0, Disgust:0, Fear:0, Happiness:0, Neutral:0, Sadness:0, Surprise:0 };
let topEmotion    = null;

let currentAbortController = null;
let detectionInProgress = false;

let _engagementScores = [];
let _sessionTimeline  = [];


// ── Capture frame from video as JPEG Blob ──
function captureFrame(quality = 0.75) {
  return new Promise((resolve, reject) => {
    const video = document.getElementById('videoFeed');
    if (!video || !video.srcObject) return reject(new Error('No camera stream'));
    const canvas = document.createElement('canvas');
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
async function callPredict(signal, fast = true) {
  const t0 = performance.now();

  const blob = await captureFrame(fast ? 0.75 : 0.90);
  const form = new FormData();
  form.append('file', blob, 'frame.jpg');

  const saveParam = fast ? 'false' : 'true';
  const sidParam  = (typeof _sessionId === 'string' && _sessionId) ? '&session_id=' + encodeURIComponent(_sessionId) : '';
  const url = `${API_URL}?fast=${fast}&save=${saveParam}${sidParam}`;

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

  const emotion    = data.emotion;
  const conf       = Math.round(data.confidence > 1 ? data.confidence : data.confidence * 100);
  const engagement = (data.engagement != null) ? data.engagement : (ENGAGEMENT_MAP[emotion] ?? 0.5);

  return { probs, emotion, conf, elapsed, engagement };
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
      <span class="prob-label">
        <span class="p-icon">${emo ? emo.icon : ''}</span>${k}
      </span>
      <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar-${k}" style="width:0%"></div></div>
      <span class="prob-pct" id="pct-${k}">0%</span>`;
    el.appendChild(row);
  });
}


// ── Update all UI panels ──
function updateUI(probs, top, conf, engagement) {
  const topLower = top?.toLowerCase();
  const engPct   = (engagement != null) ? Math.round(engagement * 100) : null;
  const emoData  = EMOTIONS.find(e => e.label.toLowerCase() === topLower);

  // Camera overlay elements are ghost (hidden) — safe null-guarded writes only
  const rdot = document.getElementById('resultDot');
  const remo = document.getElementById('resultEmotion');
  const rcon = document.getElementById('resultConf');
  if (rdot) rdot.style.background = emoData?.color || '#94a3b8';
  if (remo) remo.textContent = top ? (top.charAt(0).toUpperCase() + top.slice(1)) : '—';
  if (rcon) rcon.textContent = top
    ? (engPct != null ? `${conf}% conf · Eng ${engPct}%` : `${conf}% confidence`)
    : 'No detection yet';

  // Probabilities
  PROB_KEYS.forEach(k => {
    const v        = probs[k] || 0;
    const isActive = k.toLowerCase() === topLower;
    const bar      = document.getElementById(`bar-${k}`);
    const pct      = document.getElementById(`pct-${k}`);
    if (bar) { bar.style.width = v + '%'; bar.className = 'prob-bar-fill' + (isActive ? ' active' : ''); }
    if (pct) { pct.textContent = v + '%'; pct.className = 'prob-pct' + (isActive ? ' active' : ''); }
  });

  // Dominant insight card
  if (top) {
    const conf2 = probs[emoData ? emoData.label : ''] || conf;
    animateBreakdownCard(top, emoData, conf2);
  }

  // Detection bar — brief flash only, auto-hides after 2.5s
  if (top) {
    const bar = document.getElementById('detectionBar');
    if (bar) {
      bar.textContent = engPct != null
        ? `Detected: ${top.charAt(0).toUpperCase() + top.slice(1)} · ${conf}% conf · Engagement ${engPct}%`
        : `Detected: ${top.charAt(0).toUpperCase() + top.slice(1)} · ${conf}% confidence`;
      bar.classList.add('visible');
      clearTimeout(bar._hideTimer);
      bar._hideTimer = setTimeout(() => bar.classList.remove('visible'), 2500);
    }
    const lastEl = document.getElementById('lastDetected');
    if (lastEl) lastEl.textContent = `Last: ${top.charAt(0).toUpperCase() + top.slice(1)}`;
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
  return `<svg viewBox="0 0 88 88" xmlns="http://www.w3.org/2000/svg" style="width:56px;height:56px;display:block;">
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


// ── Animated breakdown card ──
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
    document.getElementById('avatarLabel')?.classList.add('visible');
    document.getElementById('domName')?.classList.add('revealed');
  }, 220);

  setTimeout(() => {
    document.getElementById('domPill')?.classList.add('revealed');
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
function addTimelineDot(emotion, engagement) {
  timeline.push({ emotion, engagement: engagement ?? null });

  const container = document.getElementById('timelineDots');
  const tagEl     = document.getElementById('timelineTag');
  if (!container) return;

  const colorMap = {};
  EMOTIONS.forEach(e => { colorMap[e.key] = e.color; });

  container.innerHTML = timeline.map((entry, i) => {
    const e      = entry.emotion;
    const c      = colorMap[e] || '#6b7280';
    const isLast = i === timeline.length - 1;
    const label  = e.charAt(0).toUpperCase() + e.slice(1);
    const engTxt = (entry.engagement != null)
      ? ` · Eng ${Math.round(entry.engagement * 100)}%` : '';
    const connector = i < timeline.length - 1
      ? `<div class="tl-connector"></div>` : '';

    if (isLast) {
      return `<div class="tl-chip tl-chip--active"
                   style="border-color:${c};background:${c}22;"
                   title="${label}${engTxt}">
                <div class="tl-chip-dot" style="background:${c};"></div>
                <span class="tl-chip-label" style="color:${c};">${label}</span>
              </div>${connector}`;
    }
    return `<div class="tl-chip tl-chip--done" title="${label}${engTxt}">
              <div class="tl-chip-dot" style="background:${c};opacity:0.7;"></div>
              <span class="tl-chip-label">${e}</span>
            </div>${connector}`;
  }).join('');

  container.scrollLeft = container.scrollWidth;

  if (tagEl) {
    const countMap = {};
    timeline.forEach(entry => {
      countMap[entry.emotion] = (countMap[entry.emotion] || 0) + 1;
    });
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
async function runDetection(fast = true) {
  if (currentAbortController) currentAbortController.abort();
  currentAbortController = new AbortController();

  try {
    const { probs, emotion, conf, elapsed, engagement } = await callPredict(
      currentAbortController.signal, fast
    );

    if (fast && !isLive) return;

    topEmotion   = emotion;
    currentProbs = probs;
    _frameCount++;

    _engagementScores.push(engagement);
    _sessionTimeline.push({ emotion, engagement, time: Date.now() });

    updateUI(probs, emotion.toLowerCase(), conf, engagement);
    addTimelineDot(emotion.toLowerCase(), engagement);

    const badge = document.getElementById('msBadge');
    badge.style.display = 'block';
    badge.textContent   = elapsed + ' ms';

  } catch (err) {
    if (err.name === 'AbortError') return;
    console.error('[EmotionAI] Detection error:', err);
    showError(err.message || 'Detection failed');
  } finally {
    detectionInProgress = false;
  }
}


// ── Button: single detect ──
function doDetect() {
  if (detectionInProgress) return;
  detectionInProgress = true;
  runDetection(false);
}


// ── Live mode ──
function scheduleLiveDetection() {
  liveInterval = setTimeout(async () => {
    if (!isLive) return;
    if (!detectionInProgress) {
      detectionInProgress = true;
      await runDetection(true);
    }
    scheduleLiveDetection();
  }, 300);
}


let _sessionId  = null;
let _frameCount = 0;

async function startLive() {
  if (isLive) return;
  isLive = true;
  detectionInProgress = false;
  _frameCount       = 0;
  _engagementScores = [];
  _sessionTimeline  = [];

  document.getElementById('liveTag').classList.add('visible');
  document.getElementById('liveBadge').classList.add('active');
  document.getElementById('btnGoLive').style.display = 'none';
  document.getElementById('btnStop').style.display   = 'flex';
  document.getElementById('btnDetect').disabled      = true;

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

  // Update avg engagement in insight panel pill (no camera overlay text)
  if (_engagementScores.length > 0) {
    const avg = Math.round(
      (_engagementScores.reduce((a, b) => a + b, 0) / _engagementScores.length) * 100
    );

    // ghost elements — safe null-guarded writes
    const engFloat  = document.getElementById('engFloat');
    const engNumEl  = document.getElementById('engNum');
    const engPctSuf = document.getElementById('engPctSuffix');
    if (engFloat)  engFloat.classList.add('show');
    if (engPctSuf) engPctSuf.textContent = '%';
    if (engNumEl)  engNumEl.textContent  = avg;

    const avgEl = document.getElementById('avgEngDisplay');
    if (avgEl) avgEl.textContent = avg + '%';

    const lastEl = document.getElementById('lastDetected');
    if (lastEl) lastEl.textContent = `Avg: ${avg}%`;
  }

  if (_sessionId) {
    try {
      await Auth.apiFetch('/sessions/end/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body:   JSON.stringify({ session_id: _sessionId, total_frames: _frameCount }),
      });
      const report = await API.getSessionReport(_sessionId);
      console.info('[EmotionAI] Session report:', report);
    } catch (e) {
      console.warn('[EmotionAI] Could not end session / fetch report:', e);
    }
    _sessionId        = null;
    _frameCount       = 0;
    _engagementScores = [];
    _sessionTimeline  = [];
  }
}


// ── Video upload handler ──
async function handleVideoUpload(file) {
  if (!file) return;

  const bar = document.getElementById('detectionBar');
  bar.textContent = '⏳ Analyzing video…';
  bar.classList.add('visible');

  try {
    const result = await API.analyzeVideo(file);

    const emotion  = result.dominant_emotion || '—';
    const engPct   = Math.round((result.average_engagement || 0) * 100);
    const emoLower = emotion.toLowerCase();
    const emo      = EMOTIONS.find(e => e.label.toLowerCase() === emoLower);

    document.getElementById('resultDot').style.background = emo?.color || '#94a3b8';
    document.getElementById('resultEmotion').textContent  = emotion;

    const confEl = document.getElementById('resultConf');
    if (confEl) confEl.textContent = `Video · Avg Engagement ${engPct}%`;

    bar.textContent = `Video done · Dominant: ${emotion} · Avg Engagement ${engPct}%`;

    document.getElementById('lastDetected').textContent =
      `Video: ${emotion} (${engPct}% avg)`;

    if (result.timeline && result.timeline.length) {
      result.timeline.forEach(entry => {
        addTimelineDot((entry.emotion || '').toLowerCase(), entry.engagement);
      });
    }

    if (emo) animateBreakdownCard(emoLower, emo, engPct);

  } catch (err) {
    console.error('[EmotionAI] Video analysis error:', err);
    showError(err.message || 'Video analysis failed');
  }
}


// ── Wire up file input ──
document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('videoFileInput');
  if (input) {
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) handleVideoUpload(input.files[0]);
      input.value = '';
    });
  }
});


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
if (!Auth.requireAuth()) throw new Error('Not authenticated');

const _user   = Auth.getUser();
const _userEl = document.getElementById('topbarUser');
if (_userEl && _user) _userEl.textContent = _user.username || _user.email || '';

buildProbList();
startCamera();

/* ══════════════════════════════════════════
   TUTOR MESSAGES + TIMELINE STATS + NOTES
   ══════════════════════════════════════════ */

const _TUTOR = {
  happiness: { icon:'🌟', tag:'Happy',      text:'Student is highly engaged — great moment to deepen the topic or raise difficulty.' },
  neutral:   { icon:'👍', tag:'Neutral',    text:'Calm focus detected. Try an interactive question to boost active participation.' },
  sadness:   { icon:'💙', tag:'Low Mood',   text:'Low mood signals detected. A quick check-in or encouragement can help re-engage.' },
  anger:     { icon:'🧘', tag:'Frustrated', text:'Frustration detected. Simplify the current explanation or offer a short break.' },
  fear:      { icon:'🫂', tag:'Anxious',    text:'Anxiety signals present. Break tasks into smaller steps to rebuild confidence.' },
  disgust:   { icon:'😅', tag:'Disengaged', text:'Disengagement signals detected. Try varying the approach or connecting to interests.' },
  surprise:  { icon:'🎉', tag:'Surprised',  text:'Attention is at peak — clarify and reinforce the concept now.' },
};

let _sessionStart  = null;
let _durationTimer = null;
let _detCount      = 0;
let _notesList     = [];
let _lastConf      = 0;

function _sessionTime() {
  if (!_sessionStart) return '00:00';
  const s = Math.floor((Date.now() - _sessionStart) / 1000);
  return String(Math.floor(s / 60)).padStart(2,'0') + ':' + String(s % 60).padStart(2,'0');
}

function _ensureTimer() {
  if (_durationTimer) return;
  _sessionStart = Date.now();
  _durationTimer = setInterval(() => {
    const el = document.getElementById('tlDuration');
    if (el) el.textContent = _sessionTime();
  }, 1000);
}

function _addTutorMsg(emotionLower, conf) {
  const m   = _TUTOR[emotionLower] || _TUTOR.neutral;
  const log = document.getElementById('tutorLog');
  if (!log) return;
  const div = document.createElement('div');
  div.className = 'tutor-msg';
  div.innerHTML = `<div class="tutor-msg-icon">${m.icon}</div>
    <div class="tutor-msg-body">
      <div class="tutor-msg-meta">
        <span class="tutor-msg-tag">${m.tag}</span>
        <span class="tutor-msg-conf">${conf}% conf</span>
        <span class="tutor-msg-time">${_sessionTime()}</span>
      </div>
      <div>${m.text}</div>
    </div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function _updateTimelineStats(conf) {
  _detCount++;
  _lastConf = conf;
  _ensureTimer();

  // detection count
  const cEl = document.getElementById('tlDetCount');
  if (cEl) cEl.textContent = _detCount;

  // engagement bar + avg
  if (_engagementScores.length) {
    const avg = Math.round(_engagementScores.reduce((a,b)=>a+b,0) / _engagementScores.length * 100);
    const engBar = document.getElementById('tlEngBar');
    const engVal = document.getElementById('tlEngVal');
    const avgEl  = document.getElementById('avgEngDisplay');
    if (engBar) engBar.style.width = avg + '%';
    if (engVal) engVal.textContent = avg + '%';
    if (avgEl)  avgEl.textContent  = avg + '%';

    const avgStatEl = document.getElementById('tlAvgEng');
    if (avgStatEl) avgStatEl.textContent = avg + '%';
  }

  // confidence bar
  const cBar = document.getElementById('tlConfBar');
  const cVal = document.getElementById('tlConfVal');
  if (cBar) cBar.style.width = conf + '%';
  if (cVal) cVal.textContent = conf + '%';

  // top emotion
  if (timeline.length) {
    const counts = {};
    timeline.forEach(t => counts[t.emotion] = (counts[t.emotion]||0) + 1);
    const topKey = Object.keys(counts).sort((a,b) => counts[b]-counts[a])[0];
    const emo    = EMOTIONS.find(e => e.key === topKey);
    const topEl  = document.getElementById('tlTopEmo');
    if (topEl && emo) topEl.textContent = emo.icon + ' ' + emo.label;
  }
}

/* Patch updateUI to also fire tutor + stats */
const _origUpdateUI = updateUI;
window.updateUI = function(probs, top, conf, engagement) {
  _origUpdateUI(probs, top, conf, engagement);
  if (top) {
    const confPct = Math.round(conf > 1 ? conf : conf * 100);
    _addTutorMsg(top, confPct);
    _updateTimelineStats(confPct);
  }
};

/* ── NOTES ── */
function toggleNoteInput() {
  const wrap = document.getElementById('noteInputWrap');
  const open = wrap.style.display !== 'none';
  wrap.style.display = open ? 'none' : 'block';
  if (!open) document.getElementById('noteTextarea').focus();
}

function saveNote() {
  const ta   = document.getElementById('noteTextarea');
  const text = ta.value.trim();
  if (!text) return;
  _notesList.unshift({ text, ts: _sessionTime(), id: Date.now() });
  ta.value = '';
  document.getElementById('noteInputWrap').style.display = 'none';
  _renderNotes();
}

function deleteNote(id) {
  _notesList = _notesList.filter(n => n.id !== id);
  _renderNotes();
}

function _renderNotes() {
  const list  = document.getElementById('notesList');
  const empty = document.getElementById('notesEmpty');
  if (!_notesList.length) {
    empty.style.display = 'block';
    list.innerHTML = '';
    list.appendChild(empty);
    return;
  }
  empty.style.display = 'none';
  list.innerHTML = _notesList.map(n => `
    <div class="note-item">
      <span class="note-ts">${n.ts}</span>
      <span class="note-text">${n.text.replace(/</g,'&lt;')}</span>
      <button class="note-delete" onclick="deleteNote(${n.id})" title="Delete">✕</button>
    </div>`).join('');
}

/* Allow Ctrl+Enter in textarea to save */
document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('noteTextarea');
  if (ta) ta.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') saveNote();
  });
});