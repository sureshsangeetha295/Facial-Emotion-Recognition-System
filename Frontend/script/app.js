// ═══════════════════════════════════════════════════════
//  EmotionAI — app.js  (fully integrated with dashboard)
// ═══════════════════════════════════════════════════════

// ── Emotion data ──
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

// Backend label order: ["Anger","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
const PROB_KEYS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'];

const ENGAGEMENT_MAP = {
  Happiness: 1.0, Surprise: 1.0,
  Neutral:   0.6,
  Sadness:   0.3, Fear: 0.3,
  Anger:     0.1, Disgust: 0.1,
};

// ── Dashboard emotion meta (for detected panel, insight, speedometer) ──
const DETECT_META = {
  happiness: {
    label:'Happy', color:'#f59e0b', chip:'chip-green', chipTxt:'● Actively Engaged',
    desc:'Joy is a warm feeling of pleasure — your brain is flooded with dopamine.',
    head:'Riding the happy wave',
    tips:'Channel this energy into your most creative task. Joy supercharges problem-solving.',
    motivate:'', motivateCls:'',
  },
  neutral: {
    label:'Neutral', color:'#6b7280', chip:'chip-yellow', chipTxt:'◑ Partially Focused',
    desc:'A calm, balanced state — neither highs nor lows. Your mind is clear and steady.',
    head:'Making the most of calm',
    tips:'This steady state is ideal for deep analytical thinking and careful decision-making.',
    motivate:'💡 Try taking a quick note — writing helps lock in your attention!', motivateCls:'yellow',
  },
  sadness: {
    label:'Sad', color:'#3b82f6', chip:'chip-red', chipTxt:'● Disengaged',
    desc:'A deep ache of loss or disappointment — a natural signal asking for reflection.',
    head:'Navigating through sadness',
    tips:'Be gentle with yourself. Take a short break and reach out to someone you trust.',
    motivate:'❤️ It\'s okay to feel this way. One step at a time!', motivateCls:'red',
  },
  anger: {
    label:'Frustrated', color:'#ef4444', chip:'chip-red', chipTxt:'● Disengaged',
    desc:'Anger is intense energy triggered by a perceived threat — it sharpens focus but clouds judgment.',
    head:'Cooling and redirecting anger',
    tips:'Step away briefly and try box breathing: 4 counts in, hold 4, out 4, hold 4.',
    motivate:'🧘 Breathe. Frustration means you care — you\'re closer than you think!', motivateCls:'red',
  },
  fear: {
    label:'Anxious', color:'#8b5cf6', chip:'chip-red', chipTxt:'● Stressed / Anxious',
    desc:'Fear is your mind\'s alarm — it detects uncertainty and floods your body with alertness.',
    head:'Moving through fear',
    tips:'Name the fear and break your challenge into the smallest next step.',
    motivate:'💙 Slow breath in, slow breath out. You\'ve got this! 🙌', motivateCls:'red',
  },
  disgust: {
    label:'Disengaged', color:'#10b981', chip:'chip-red', chipTxt:'● Very Low Interest',
    desc:'Very low engagement detected — try switching the delivery or connecting to something interesting.',
    head:'Re-engaging your interest',
    tips:'Find just one surprising fact about this topic to spark curiosity.',
    motivate:'🔥 Even 2 focused minutes can reset the entire session. Go!', motivateCls:'red',
  },
  surprise: {
    label:'Surprised', color:'#f97316', chip:'chip-yellow', chipTxt:'◑ Attention Spike',
    desc:'Surprise is a brief jolt — your brain snaps to full alertness and curiosity instantly.',
    head:'Harness the attention spike',
    tips:'Your brain is at peak receptivity right now. Use this to absorb something new.',
    motivate:'', motivateCls:'',
  },
};

const API_URL = '/predict/';

// ── State ──
let isLive               = false;
let liveInterval         = null;
let timeline             = [];
let currentProbs         = { Anger:0, Disgust:0, Fear:0, Happiness:0, Neutral:0, Sadness:0, Surprise:0 };
let topEmotion           = null;
let currentAbortController = null;
let detectionInProgress  = false;
let _engagementScores    = [];
let _sessionTimeline     = [];
let _sessionId           = null;
let _frameCount          = 0;

// ── Session / timeline state ──
let _sessionStart    = null;
let _durationTimer   = null;
let _detCount        = 0;
let _notesList       = [];
let _lastConf        = 0;

// ── Engagement smoothing (for speedometer) ──
let _currentEngScore   = 0;
let _attentiveTime     = 0;
let _partialTime       = 0;
let _disengagedTime    = 0;


// ════════════════════════════════════════════
//  CAMERA
// ════════════════════════════════════════════

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const vid = document.getElementById('videoFeed');
    vid.srcObject = stream;
    vid.style.display = 'block';
    // Hide idle overlay once camera is ready
    vid.onloadedmetadata = () => {
      const idle = document.getElementById('camIdle');
      if (idle) idle.style.opacity = '0';
    };
  } catch (e) {
    console.warn('[EmotionAI] Camera unavailable:', e);
  }
}


// ════════════════════════════════════════════
//  CAPTURE + API
// ════════════════════════════════════════════

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

async function callPredict(signal, fast = true) {
  const t0   = performance.now();
  const blob = await captureFrame(fast ? 0.75 : 0.90);
  const form = new FormData();
  form.append('file', blob, 'frame.jpg');

  const saveParam = fast ? 'false' : 'true';
  const sidParam  = (typeof _sessionId === 'string' && _sessionId)
    ? '&session_id=' + encodeURIComponent(_sessionId) : '';
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
    probs[k]  = Math.round((raw > 1 ? raw : raw * 100));
  });
  const total = Object.values(probs).reduce((a, b) => a + b, 0);
  if (total > 0) PROB_KEYS.forEach(k => probs[k] = Math.round(probs[k] / total * 100));

  const emotion    = data.emotion;
  const conf       = Math.round(data.confidence > 1 ? data.confidence : data.confidence * 100);
  const engagement = (data.engagement != null) ? data.engagement : (ENGAGEMENT_MAP[emotion] ?? 0.5);

  return { probs, emotion, conf, elapsed, engagement };
}


// ════════════════════════════════════════════
//  BUILD PROBABILITY LIST
// ════════════════════════════════════════════

function buildProbList() {
  const el = document.getElementById('probList');
  if (!el) return;
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


// ════════════════════════════════════════════
//  SPEEDOMETER
// ════════════════════════════════════════════

function setSpeedometer(score) {
  const angle = (score / 100) * 180 - 90;
  const n = document.getElementById('speedoNeedle');
  if (n) n.style.transform = `rotate(${angle}deg)`;

  const cx = 115, cy = 112, r = 95;
  const sR = Math.PI, eR = Math.PI - (score / 100) * Math.PI;
  const x1 = cx + r * Math.cos(sR), y1 = cy + r * Math.sin(sR);
  const x2 = cx + r * Math.cos(eR), y2 = cy + r * Math.sin(eR);
  const largeArc = score > 50 ? 1 : 0;
  const col = score >= 70 ? '#16a34a' : score >= 40 ? '#ca8a04' : '#dc2626';

  const arc = document.getElementById('speedoArc');
  if (arc) {
    arc.setAttribute('d', `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`);
    arc.setAttribute('stroke', col);
  }

  const sv = document.getElementById('speedoValue');
  const ss = document.getElementById('speedoStatus');
  if (sv) { sv.textContent = score > 0 ? Math.round(score) : '—'; sv.style.color = col; }
  if (ss) ss.textContent = score >= 70 ? '🟢 Actively Listening'
                          : score >= 40 ? '🟡 Partially Engaged'
                          : score > 0   ? '🔴 Low Engagement'
                          : 'Awaiting detection';
}


// ════════════════════════════════════════════
//  DASHBOARD PANELS (detected emotion + insight)
// ════════════════════════════════════════════

function _updateDashboardPanels(top, conf, engagement) {
  const emoLower = (top || '').toLowerCase();
  const meta     = DETECT_META[emoLower];
  const color    = meta ? meta.color : '#6b7280';

  // ── Detected emotion panel ──
  const dn  = document.getElementById('detName');
  const da  = document.getElementById('detAwaitTxt');
  const dc  = document.getElementById('detConf');
  const dcv = document.getElementById('detConfVal');
  const dd  = document.getElementById('detDesc');
  if (dn)  { dn.textContent = meta ? meta.label : (top || 'Awaiting'); dn.style.color = color; }
  if (da)  da.style.display = 'none';
  if (dc)  dc.classList.remove('hidden');
  if (dcv) dcv.textContent = `${conf}% confidence`;
  if (dd)  dd.textContent  = meta ? meta.desc : '';

  // ── Detected avatar mouth ──
  const mouths = {
    happiness: 'M 63 115 Q 75 125 87 115',
    neutral:   'M 63 110 Q 75 112 87 110',
    sadness:   'M 63 115 Q 75 105 87 115',
    anger:     'M 63 113 Q 75 105 87 113',
    fear:      'M 65 112 Q 75 118 85 112',
    disgust:   'M 60 112 Q 68 118 76 112 Q 84 106 90 112',
    surprise:  'M 67 110 Q 75 122 83 110',
  };
  const dm = document.getElementById('detMouth');
  if (dm) dm.setAttribute('d', mouths[emoLower] || mouths.neutral);

  // ── Insight panel ──
  const chip = document.getElementById('insightChip');
  const itxt = document.getElementById('insightText');
  const mbox = document.getElementById('motivateBox');
  if (chip && meta) {
    chip.className   = `insight-chip visible ${meta.chip}`;
    chip.textContent = meta.chipTxt;
  }
  if (itxt && meta) {
    itxt.innerHTML = `<div class="insight-head">${meta.head}</div><span>${meta.tips}</span>`;
  }
  if (mbox && meta) {
    if (meta.motivate) {
      mbox.textContent = meta.motivate;
      mbox.className   = `motivate-box show ${meta.motivateCls}`;
    } else {
      mbox.className = 'motivate-box';
    }
  }

  // ── Speedometer (smooth) ──
  const engVal = engagement != null ? engagement : (ENGAGEMENT_MAP[top] || 0.5);
  _currentEngScore += ((engVal * 100 - _currentEngScore) * 0.4);
  setSpeedometer(_currentEngScore);

  // ── Track attentive / partial / disengaged ──
  const score = _currentEngScore;
  if (score >= 70) _attentiveTime++;
  else if (score >= 40) _partialTime++;
  else _disengagedTime++;

  const totalT = _attentiveTime + _partialTime + _disengagedTime;
  if (totalT > 0) {
    const sa = document.getElementById('statAttentive');
    const sp = document.getElementById('statPartial');
    const sd = document.getElementById('statDisengaged');
    if (sa) sa.textContent = Math.round((_attentiveTime / totalT) * 100) + '%';
    if (sp) sp.textContent = Math.round((_partialTime   / totalT) * 100) + '%';
    if (sd) sd.textContent = Math.round((_disengagedTime/ totalT) * 100) + '%';
  }

  // ── Hide cam idle overlay ──
  const idle = document.getElementById('camIdle');
  if (idle) idle.style.display = 'none';
}


// ════════════════════════════════════════════
//  TIMELINE (new scroll list)
// ════════════════════════════════════════════

function _addTLScrollItem(label, color, icon) {
  const empty = document.getElementById('tlEmpty');
  if (empty) empty.remove();
  const sc = document.getElementById('timelineScroll');
  if (!sc) return;
  const el  = document.createElement('div');
  el.className = 'tl-item';
  const elapsed = _sessionStart ? Date.now() - _sessionStart : 0;
  const s = Math.floor(elapsed / 1000), m = Math.floor(s / 60);
  const ts = `${String(m).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;
  el.innerHTML = `<div class="tl-dot" style="background:${color}"></div>
    <div class="tl-time">${ts}</div>
    <div class="tl-msg">${label} detected</div>
    <div class="tl-icon">${icon}</div>`;
  sc.appendChild(el);
  sc.scrollTop = sc.scrollHeight;
}


// ════════════════════════════════════════════
//  TIMELINE CHIPS (original dot row)
// ════════════════════════════════════════════

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
    const connector = i < timeline.length - 1 ? `<div class="tl-connector"></div>` : '';

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
    timeline.forEach(entry => { countMap[entry.emotion] = (countMap[entry.emotion] || 0) + 1; });
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


// ════════════════════════════════════════════
//  SESSION TIMER
// ════════════════════════════════════════════

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


// ════════════════════════════════════════════
//  TIMELINE STATS
// ════════════════════════════════════════════

function _updateTimelineStats(conf) {
  _detCount++;
  _lastConf = conf;
  _ensureTimer();

  const cEl = document.getElementById('tlDetCount');
  if (cEl) cEl.textContent = _detCount;

  if (_engagementScores.length) {
    const avg    = Math.round(_engagementScores.reduce((a,b)=>a+b,0) / _engagementScores.length * 100);
    const engBar = document.getElementById('tlEngBar');
    const engVal = document.getElementById('tlEngVal');
    const avgEl  = document.getElementById('avgEngDisplay');
    if (engBar) engBar.style.width  = avg + '%';
    if (engVal) engVal.textContent  = avg + '%';
    if (avgEl)  avgEl.textContent   = avg + '%';
    const avgStatEl = document.getElementById('tlAvgEng');
    if (avgStatEl) avgStatEl.textContent = avg + '%';
  }

  const cBar = document.getElementById('tlConfBar');
  const cVal = document.getElementById('tlConfVal');
  if (cBar) cBar.style.width = conf + '%';
  if (cVal) cVal.textContent = conf + '%';

  if (timeline.length) {
    const counts = {};
    timeline.forEach(t => counts[t.emotion] = (counts[t.emotion]||0) + 1);
    const topKey = Object.keys(counts).sort((a,b)=>counts[b]-counts[a])[0];
    const emo    = EMOTIONS.find(e => e.key === topKey);
    const topEl  = document.getElementById('tlTopEmo');
    if (topEl && emo) topEl.textContent = emo.icon + ' ' + emo.label;
  }
}


// ════════════════════════════════════════════
//  AVATAR FACES (for animateBreakdownCard)
// ════════════════════════════════════════════

const AVATAR_FACES = {
  happiness: { faceColor:'#FDDFA4', cheekColor:'#F9A8A8', eyeShape:'happy',   browAngle:0,   mouth:'M 30 62 Q 44 76 58 62', mouthColor:'#C0392B', showTeeth:true,  eyeColor:'#3d2800', pupilSize:5   },
  neutral:   { faceColor:'#FDDFA4', cheekColor:'transparent', eyeShape:'normal', browAngle:0, mouth:'M 30 64 Q 44 66 58 64', mouthColor:'#8B6555', showTeeth:false, eyeColor:'#3d2800', pupilSize:4.5 },
  sadness:   { faceColor:'#C9D8F0', cheekColor:'#A8C0E8', eyeShape:'sad',     browAngle:-12, mouth:'M 30 68 Q 44 58 58 68', mouthColor:'#5563A0', showTeeth:false, eyeColor:'#2a3a6b', pupilSize:4   },
  anger:     { faceColor:'#F4A09A', cheekColor:'#E86050', eyeShape:'angry',   browAngle:18,  mouth:'M 30 68 Q 44 60 58 68', mouthColor:'#8B1A1A', showTeeth:true,  eyeColor:'#5a0000', pupilSize:3.5 },
  fear:      { faceColor:'#D8C8EE', cheekColor:'transparent', eyeShape:'wide', browAngle:-8, mouth:'M 32 66 Q 44 72 56 66', mouthColor:'#6a40b0', showTeeth:false, eyeColor:'#2d1a5e', pupilSize:6.5 },
  disgust:   { faceColor:'#B8E8C8', cheekColor:'transparent', eyeShape:'squint', browAngle:10, mouth:'M 28 64 Q 36 70 44 64 Q 52 58 58 64', mouthColor:'#2d7a4a', showTeeth:false, eyeColor:'#1a4d2e', pupilSize:4 },
  surprise:  { faceColor:'#FDDFA4', cheekColor:'#FFB347', eyeShape:'wide',    browAngle:-15, mouth:'M 36 62 Q 44 78 52 62', mouthColor:'#C0392B', showTeeth:false, eyeColor:'#3d2800', pupilSize:7   },
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
       <ellipse cx="68" cy="58" rx="10" ry="6" fill="${f.cheekColor}" opacity="0.45"/>` : '';
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


// ════════════════════════════════════════════
//  ANIMATED BREAKDOWN CARD (dominantCard)
// ════════════════════════════════════════════

let _typewriterTimer = null;
let _particleTimers  = [];

function animateBreakdownCard(emotion, emo, conf) {
  const card = document.getElementById('dominantCard');
  if (!card) return;
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
    if (face) { face.classList.add('visible'); setTimeout(() => face.classList.add('idle'), 550); }
  });
  setTimeout(() => {
    document.getElementById('avatarLabel')?.classList.add('visible');
    document.getElementById('domName')?.classList.add('revealed');
  }, 220);
  setTimeout(() => document.getElementById('domPill')?.classList.add('revealed'), 380);
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


// ════════════════════════════════════════════
//  ERROR DISPLAY
// ════════════════════════════════════════════

function showError(msg) {
  const bar = document.getElementById('detectionBar');
  if (!bar) return;
  bar.textContent = `⚠ ${msg}`;
  bar.classList.add('visible');
  setTimeout(() => bar.classList.remove('visible'), 3000);
}


// ════════════════════════════════════════════
//  MASTER updateUI
// ════════════════════════════════════════════

function updateUI(probs, top, conf, engagement) {
  const topLower = top?.toLowerCase();
  const engPct   = (engagement != null) ? Math.round(engagement * 100) : null;
  const emoData  = EMOTIONS.find(e => e.label.toLowerCase() === topLower);

  // Ghost overlay elements (safe null-guarded)
  const rdot = document.getElementById('resultDot');
  const remo = document.getElementById('resultEmotion');
  const rcon = document.getElementById('resultConf');
  if (rdot) rdot.style.background = emoData?.color || '#94a3b8';
  if (remo) remo.textContent = top ? (top.charAt(0).toUpperCase() + top.slice(1)) : '—';
  if (rcon) rcon.textContent = top
    ? (engPct != null ? `${conf}% conf · Eng ${engPct}%` : `${conf}% confidence`) : 'No detection yet';

  // Probability bars
  PROB_KEYS.forEach(k => {
    const v        = probs[k] || 0;
    const isActive = k.toLowerCase() === topLower;
    const bar      = document.getElementById(`bar-${k}`);
    const pct      = document.getElementById(`pct-${k}`);
    if (bar) { bar.style.width = v + '%'; bar.className = 'prob-bar-fill' + (isActive ? ' active' : ''); }
    if (pct) { pct.textContent = v + '%'; pct.className = 'prob-pct' + (isActive ? ' active' : ''); }
  });

  // Dominant card
  if (top) {
    const conf2 = probs[emoData ? emoData.label : ''] || conf;
    animateBreakdownCard(top, emoData, conf2);
  }

  // Detection bar flash
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

  // ── Dashboard panels (detected emotion, insight, speedometer) ──
  if (top) _updateDashboardPanels(top, conf, engagement);

  // ── Guide steps ──
  const s1 = document.getElementById('step1');
  const s2 = document.getElementById('step2');
  const s3 = document.getElementById('step3');
  const s4 = document.getElementById('step4');
  if (s1) s1.classList.add('done');
  if (s2) s2.classList.add('done');
  if (s3) s3.classList.add('done');
  if (s4) s4.classList.add('active');

  // ── Cam status ──
  const dot = document.getElementById('camStatusDot');
  const txt = document.getElementById('camStatusTxt');
  if (!isLive) {
    if (dot) dot.className = 'cam-status-dot ready';
    if (txt) txt.textContent = 'DETECTED';
  }

  // ── Add to scroll timeline ──
  if (top) {
    const meta  = DETECT_META[top.toLowerCase()];
    const color = meta ? meta.color : '#6b7280';
    const score = _currentEngScore;
    const icon  = score >= 70 ? '😊' : score >= 40 ? '😐' : '😞';
    _addTLScrollItem(meta ? meta.label : top, color, icon);
  }

  // ── Timeline stats ──
  if (top) _updateTimelineStats(conf);
}


// ════════════════════════════════════════════
//  CORE DETECTION RUNNER
// ════════════════════════════════════════════

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

    // ms badge (ghost element)
    const badge = document.getElementById('msBadge');
    if (badge) { badge.style.display = 'block'; badge.textContent = elapsed + ' ms'; }

  } catch (err) {
    if (err.name === 'AbortError') return;
    console.error('[EmotionAI] Detection error:', err);
    showError(err.message || 'Detection failed');
  } finally {
    detectionInProgress = false;
  }
}


// ════════════════════════════════════════════
//  SINGLE DETECT BUTTON
// ════════════════════════════════════════════

function doDetect() {
  if (detectionInProgress) return;
  detectionInProgress = true;

  // Scan animation
  const scan = document.getElementById('camScan');
  if (scan) { scan.className = 'cam-scan'; void scan.offsetWidth; scan.className = 'cam-scan active'; }

  // Status
  const dot = document.getElementById('camStatusDot');
  const txt = document.getElementById('camStatusTxt');
  if (dot) dot.className = 'cam-status-dot processing';
  if (txt) txt.textContent = 'SCANNING…';

  runDetection(false);
}


// ════════════════════════════════════════════
//  LIVE MODE
// ════════════════════════════════════════════

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

async function startLive() {
  if (isLive) return;
  isLive = true;
  detectionInProgress  = false;
  _frameCount          = 0;
  _engagementScores    = [];
  _sessionTimeline     = [];
  _currentEngScore     = 0;
  _attentiveTime       = 0;
  _partialTime         = 0;
  _disengagedTime      = 0;
  _detCount            = 0;
  _sessionStart        = Date.now();

  // Reset duration timer
  if (_durationTimer) { clearInterval(_durationTimer); _durationTimer = null; }
  _ensureTimer();

  // UI changes
  document.getElementById('liveTag')?.classList.add('visible');
  document.getElementById('liveBadge')?.classList.add('active');
  document.getElementById('btnGoLive').style.display = 'none';
  document.getElementById('btnStop').style.display   = 'flex';
  document.getElementById('btnDetect').disabled      = true;
  document.getElementById('camBox')?.classList.add('live-active');

  // Status
  const dot = document.getElementById('camStatusDot');
  const txt = document.getElementById('camStatusTxt');
  if (dot) dot.className = 'cam-status-dot live';
  if (txt) txt.textContent = 'LIVE';

  // Scan line
  const scan = document.getElementById('camScan');
  if (scan) scan.classList.add('live');

  // Reaction box
  document.getElementById('reactionBox')?.classList.add('show');

  // Clear and reset session summary
  document.getElementById('endNoteRow')?.classList.remove('show');
  const statA = document.getElementById('statAttentive');
  const statP = document.getElementById('statPartial');
  const statD = document.getElementById('statDisengaged');
  if (statA) statA.textContent = '—';
  if (statP) statP.textContent = '—';
  if (statD) statD.textContent = '—';

  // Clear timeline scroll for new session
  const sc = document.getElementById('timelineScroll');
  if (sc) {
    sc.innerHTML = '';
    const e = document.createElement('div');
    e.className = 'tl-empty'; e.id = 'tlEmpty';
    e.textContent = 'Session started. Monitoring…';
    sc.appendChild(e);
  }
  _addTLScrollItem('Session started', '#16a34a', '🎬');

  // Guide steps
  document.getElementById('step1')?.classList.add('done');
  document.getElementById('step2')?.classList.add('done');
  document.getElementById('step3')?.classList.add('active');

  // Start backend session
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
  if (currentAbortController) { currentAbortController.abort(); currentAbortController = null; }
  detectionInProgress = false;

  // UI changes
  document.getElementById('liveTag')?.classList.remove('visible');
  document.getElementById('liveBadge')?.classList.remove('active');
  document.getElementById('btnGoLive').style.display = '';
  document.getElementById('btnStop').style.display   = 'none';
  document.getElementById('btnDetect').disabled      = false;
  document.getElementById('detectionBar')?.classList.remove('visible');
  document.getElementById('camBox')?.classList.remove('live-active');
  document.getElementById('reactionBox')?.classList.remove('show');

  const scan = document.getElementById('camScan');
  if (scan) scan.classList.remove('live');

  // Status
  const dot = document.getElementById('camStatusDot');
  const txt = document.getElementById('camStatusTxt');
  if (dot) dot.className = 'cam-status-dot ready';
  if (txt) txt.textContent = 'STOPPED';

  // Guide
  document.getElementById('step4')?.classList.add('active');

  // Stop timer
  if (_durationTimer) { clearInterval(_durationTimer); _durationTimer = null; }

  // Avg engagement display
  if (_engagementScores.length > 0) {
    const avg = Math.round((_engagementScores.reduce((a,b)=>a+b,0) / _engagementScores.length) * 100);
    const avgEl  = document.getElementById('avgEngDisplay');
    const lastEl = document.getElementById('lastDetected');
    const avgSt  = document.getElementById('tlAvgEng');
    if (avgEl)  avgEl.textContent  = avg + '%';
    if (lastEl) lastEl.textContent = `Avg: ${avg}%`;
    if (avgSt)  avgSt.textContent  = avg + '%';

    // Ghost elements (safe)
    const engFloat  = document.getElementById('engFloat');
    const engNumEl  = document.getElementById('engNum');
    const engPctSuf = document.getElementById('engPctSuffix');
    if (engFloat)  engFloat.classList.add('show');
    if (engPctSuf) engPctSuf.textContent = '%';
    if (engNumEl)  engNumEl.textContent  = avg;
  }

  // End note row
  const total   = _attentiveTime + _partialTime + _disengagedTime || 1;
  const attPct  = Math.round((_attentiveTime / total) * 100);
  const note    = attPct >= 60 ? 'Outstanding session! 🌟'
                : attPct >= 40 ? 'Good effort — keep it up! 💪'
                : 'Keep pushing — every session counts! 🔥';
  const enr = document.getElementById('endNoteRow');
  const ebt = document.getElementById('endBadgeTime');
  const ebn = document.getElementById('endBadgeNote');
  if (ebt) ebt.textContent = _sessionTime();
  if (ebn) ebn.textContent = note;
  if (enr) enr.classList.add('show');

  _addTLScrollItem('Session ended', '#6b7280', '🏁');

  // End backend session
  if (_sessionId) {
    try {
      await Auth.apiFetch('/sessions/end/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: _sessionId, total_frames: _frameCount }),
      });
      const report = await API.getSessionReport(_sessionId);
      console.info('[EmotionAI] Session report:', report);
    } catch (e) {
      console.warn('[EmotionAI] Could not end session / fetch report:', e);
    }
    _sessionId = null;
    _frameCount = 0;
    _engagementScores = [];
    _sessionTimeline  = [];
  }
}


// ════════════════════════════════════════════
//  VIDEO UPLOAD
// ════════════════════════════════════════════

async function handleVideoUpload(file) {
  if (!file) return;
  const bar = document.getElementById('detectionBar');
  if (bar) { bar.textContent = '⏳ Analyzing video…'; bar.classList.add('visible'); }

  try {
    const result    = await API.analyzeVideo(file);
    const emotion   = result.dominant_emotion || '—';
    const engPct    = Math.round((result.average_engagement || 0) * 100);
    const emoLower  = emotion.toLowerCase();
    const emo       = EMOTIONS.find(e => e.label.toLowerCase() === emoLower);

    const rdot = document.getElementById('resultDot');
    const remo = document.getElementById('resultEmotion');
    const rcon = document.getElementById('resultConf');
    if (rdot) rdot.style.background = emo?.color || '#94a3b8';
    if (remo) remo.textContent = emotion;
    if (rcon) rcon.textContent = `Video · Avg Engagement ${engPct}%`;

    if (bar) bar.textContent = `Video done · Dominant: ${emotion} · Avg Engagement ${engPct}%`;

    const lastEl = document.getElementById('lastDetected');
    if (lastEl) lastEl.textContent = `Video: ${emotion} (${engPct}% avg)`;

    if (result.timeline && result.timeline.length) {
      result.timeline.forEach(entry => addTimelineDot((entry.emotion || '').toLowerCase(), entry.engagement));
    }

    if (emo) animateBreakdownCard(emoLower, emo, engPct);

  } catch (err) {
    console.error('[EmotionAI] Video analysis error:', err);
    showError(err.message || 'Video analysis failed');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('videoFileInput');
  if (input) {
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) handleVideoUpload(input.files[0]);
      input.value = '';
    });
  }
});


// ════════════════════════════════════════════
//  NOTES
// ════════════════════════════════════════════

function toggleNoteInput() {
  const wrap = document.getElementById('noteInputWrap');
  const open = wrap.style.display !== 'none';
  wrap.style.display = open ? 'none' : 'block';
  if (!open) document.getElementById('noteTextarea')?.focus();
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
    if (empty) empty.style.display = 'block';
    list.innerHTML = '';
    if (empty) list.appendChild(empty);
    return;
  }
  if (empty) empty.style.display = 'none';
  list.innerHTML = _notesList.map(n => `
    <div class="note-item">
      <span class="note-ts">${n.ts}</span>
      <span class="note-text">${n.text.replace(/</g,'&lt;')}</span>
      <button class="note-delete" onclick="deleteNote(${n.id})" title="Delete">✕</button>
    </div>`).join('');
}

document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('noteTextarea');
  if (ta) ta.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') saveNote();
  });
});


// ════════════════════════════════════════════
//  REACTIONS (float-up emoji on camera)
// ════════════════════════════════════════════

function sendReaction(type, btn) {
  btn.classList.add('active');
  setTimeout(() => btn.classList.remove('active'), 600);

  const labels = { thumbsup:'👍 Liked content', thumbsdown:'👎 Unclear', handraise:'✋ Doubt raised', confused:'😕 Confused', clap:'👏 Appreciated' };
  const colors = { thumbsup:'#16a34a', thumbsdown:'#dc2626', handraise:'#e8440a', confused:'#ca8a04', clap:'#3b82f6' };
  const icons  = { thumbsup:'👍', thumbsdown:'👎', handraise:'✋', confused:'😕', clap:'👏' };

  _addTLScrollItem(labels[type], colors[type] || '#6b7280', icons[type] || '⭐');

  const camBox = document.getElementById('camBox');
  for (let i = 0; i < 3; i++) {
    setTimeout(() => {
      const el = document.createElement('div');
      el.className  = 'float-reaction';
      el.textContent = icons[type];
      el.style.cssText = `left:${15 + Math.random()*50}%;bottom:70px;position:absolute;font-size:22px;pointer-events:none;z-index:20;animation:floatUp 1.6s ease-out forwards`;
      camBox.appendChild(el);
      setTimeout(() => el.remove(), 1800);
    }, i * 200);
  }
}


// ════════════════════════════════════════════
//  KEYBOARD SHORTCUTS
// ════════════════════════════════════════════

document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); if (!isLive) doDetect(); }
  if (e.key  === 'l' || e.key === 'L') { isLive ? stopLive() : startLive(); }
});


// ════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════

if (!Auth.requireAuth()) throw new Error('Not authenticated');

const _user   = Auth.getUser();
const _userEl = document.getElementById('topbarUser');
if (_userEl && _user) _userEl.textContent = _user.username || _user.email || '';

buildProbList();
startCamera();
setSpeedometer(0);

// ════════════════════════════════════════════
//  RESPONSIVE: resize / orientation handling
// ════════════════════════════════════════════

(function () {
  // Re-measure cam-pane on resize so the video feed fills correctly
  function onResize() {
    const camPane = document.querySelector('.cam-pane');
    const feed    = document.getElementById('camFeed') || document.querySelector('.cam-feed');
    if (!camPane || !feed) return;

    // In landscape on mobile, cap cam height to viewport height - topbar
    const isMobileLandscape =
      window.innerWidth <= 768 && window.innerHeight < window.innerWidth;

    if (isMobileLandscape) {
      const topbarH = document.querySelector('.topbar')?.offsetHeight || 44;
      camPane.style.maxHeight = (window.innerHeight - topbarH) + 'px';
    } else {
      camPane.style.maxHeight = '';
    }
  }

  // Debounced resize listener
  let _resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(onResize, 120);
  });

  // Also fire on orientation change (iOS fires this before resize)
  window.addEventListener('orientationchange', () => {
    setTimeout(onResize, 300);
  });

  // Initial call
  document.addEventListener('DOMContentLoaded', onResize);

  // Mobile hamburger: close drawer on nav link click (if not already wired)
  document.addEventListener('DOMContentLoaded', () => {
    const drawer  = document.getElementById('mobileNavDrawer');
    const overlay = document.getElementById('mobileNavOverlay');
    const btn     = document.getElementById('hamburgerBtn');
    if (!drawer) return;

    function closeDrawer() {
      drawer.classList.remove('open');
      if (overlay) overlay.classList.remove('open');
      if (btn) { btn.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); }
      document.body.style.overflow = '';
    }

    // Close on any internal anchor click
    drawer.querySelectorAll('a, button').forEach(el => {
      el.addEventListener('click', closeDrawer);
    });

    // Close on Escape
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeDrawer();
    });
  });
})();