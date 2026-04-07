// ═══════════════════════════════════════════════════════════════
//  studytimer.js  —  Emotion-Aware Adaptive Study Timer
//  Depends on: auth.js (Auth.apiFetch, Auth.requireAuth, Auth.getUser)
//  Backend:    /predict/  /sessions/start/  /sessions/end/
// ═══════════════════════════════════════════════════════════════

// ── Emotion → Study State mapping ──────────────────────────────
//  Backend returns: Anger,Disgust,Fear,Happiness,Neutral,Sadness,Surprise
//  We map them into 4 study states: focused | bored | frustrated | distracted
const STUDY_STATE_MAP = {
  Happiness: 'focused',
  Neutral:   'focused',
  Surprise:  'focused',    // attention spike = good
  Sadness:   'bored',
  Disgust:   'bored',
  Anger:     'frustrated',
  Fear:      'frustrated',
};

// Engagement scores per emotion (mirrors main.py ENGAGEMENT_SCORES)
const ENGAGEMENT_SCORES = {
  Happiness: 0.90, Surprise: 0.85, Neutral: 0.75,
  Sadness: 0.30,   Fear: 0.50,    Anger: 0.60, Disgust: 0.20,
};

// Study state metadata
const STATE_META = {
  focused: {
    label: 'Focused',
    color: '#16a34a',
    chip: 'chip-green',
    chipTxt: '● Focused & On Track',
    mouth: 'M 34 56 Q 40 62 46 56',
    suggestion: null,   // no suggestion needed
    scanClass: '',
    camClass: 'study-active',
    camDotClass: 'study',
    camTxt: 'STUDYING',
  },
  bored: {
    label: 'Bored',
    color: '#3b82f6',
    chip: 'chip-blue',
    chipTxt: '◑ Boredom Detected',
    mouth: 'M 34 60 Q 40 58 46 60',
    suggestion: '😴 You seem bored. Try switching to a different topic, or take a short break to reset your focus.',
    suggestionClass: 'bored',
    scanClass: '',
    camClass: 'study-active',
    camDotClass: 'study',
    camTxt: 'STUDYING',
  },
  frustrated: {
    label: 'Frustrated',
    color: '#ca8a04',
    chip: 'chip-yellow',
    chipTxt: '⚡ Frustration Detected',
    mouth: 'M 34 62 Q 40 57 46 62',
    suggestion: '🧘 Take a slow breath. Inhale 4s → Hold 4s → Exhale 4s. You\'re closer to understanding than you think!',
    suggestionClass: 'frustrated',
    scanClass: '',
    camClass: 'paused-active',
    camDotClass: 'paused',
    camTxt: 'PAUSED',
  },
  distracted: {
    label: 'Distracted',
    color: '#dc2626',
    chip: 'chip-red',
    chipTxt: '● Distracted — Paused',
    mouth: 'M 34 59 Q 40 56 46 59',
    suggestion: '📵 Timer paused — distraction detected. Close other tabs, silence notifications, and refocus.',
    suggestionClass: 'distracted',
    scanClass: '',
    camClass: 'paused-active',
    camDotClass: 'paused',
    camTxt: 'PAUSED',
  },
};

// ── Timer phase ─────────────────────────────────────────────────
const PHASE = { IDLE: 'idle', STUDY: 'study', BREAK: 'break', PAUSED: 'paused' };

// ── State variables ─────────────────────────────────────────────
let phase             = PHASE.IDLE;
let isRunning         = false;

// Timer counters (all in seconds)
let studyDuration     = 25 * 60;   // from config, default 25 min
let breakDuration     = 5  * 60;   // from config, default 5 min
let blockSecondsLeft  = 0;
let blockTotalSeconds = 0;
let totalStudySecs    = 0;
let totalBreakSecs    = 0;
let sessionStartTime  = null;
let focusBlocksDone   = 0;
let breakCount        = 0;
let pauseCount        = 0;
let adaptiveBonusSecs = 0;        // extra time added for sustained focus

// Tick interval handles
let timerTick         = null;
let totalTimeTick     = null;
let totalSecsElapsed  = 0;

// Emotion detection loop
let detectionLoop     = null;
let detectionActive   = false;
let abortCtrl         = null;

// Session backend
let _sessionId        = null;
let _frameCount       = 0;
let _engagementScores = [];

// Consecutive state counters (for adaptive logic)
let consecutiveBored      = 0;
let consecutiveFrustrated = 0;
let consecutiveLowEng     = 0;    // for distracted detection
let consecutiveFocused    = 0;    // for bonus extension
let boredAlertCount       = 0;   // auto-break after 3 bored alerts
let lastStudyState        = null;
let alertShowing          = false;

// Emotion frequency counters (for breakdown)
let emotionCounts = { focused: 0, bored: 0, frustrated: 0, distracted: 0 };
let totalDetections = 0;

// Speedometer smooth value
let _engScore = 0;

// ── Thresholds (in detection cycles ~300ms each) ────────────────
const BORED_THRESHOLD       = 4;   // ~1.2s sustained bored → suggest
const FRUSTRATED_THRESHOLD  = 4;   // ~1.2s frustrated → pause
const DISTRACTED_THRESHOLD  = 5;   // ~1.5s low-eng → pause + alert
const FOCUSED_BONUS_FRAMES  = 20;  // ~6s sustained focus → +2 min bonus (once per block)
let bonusGivenThisBlock     = false;

// Detection: frames per second target (~3fps via 300ms loop)
const PROB_KEYS = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'];

// ── Typewriter lines ────────────────────────────────────────────
const TW = [
  'Adaptive timer adjusts to your emotions…',
  'Stay focused and earn bonus study time…',
  'Boredom detected? A break is suggested…',
  'Your emotions guide your study rhythm…',
  'Frustration? Breathe — the timer pauses…',
];
let twIdx=0, twChar=0, twDir=1;

function initGreeting() {
  const h = new Date().getHours();
  const el = document.getElementById('greetingLine');
  if (el) el.textContent = h < 12 ? 'Good Morning! Let\'s study 🌅'
                         : h < 17 ? 'Good Afternoon! Stay focused 📚'
                         :          'Good Evening! Study session time 🌙';
  runTW();
}

function runTW() {
  const el = document.getElementById('typewriterText');
  if (!el) return;
  const line = TW[twIdx];
  if (twDir === 1) {
    el.textContent = line.slice(0, ++twChar);
    if (twChar >= line.length) { twDir = -1; setTimeout(runTW, 1800); return; }
  } else {
    el.textContent = line.slice(0, --twChar);
    if (twChar <= 0) { twDir = 1; twIdx = (twIdx + 1) % TW.length; }
  }
  setTimeout(runTW, twDir === 1 ? 55 : 28);
}

// ── Helpers ─────────────────────────────────────────────────────
function fmtSecs(s) {
  const m = Math.floor(s / 60);
  return String(m).padStart(2,'0') + ':' + String(s % 60).padStart(2,'0');
}
function totalElapsed() {
  return sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
}

// ── Speedometer ─────────────────────────────────────────────────
function setSpeedometer(score) {
  const angle = (score / 100) * 180 - 90;
  const needle = document.getElementById('speedoNeedle');
  if (needle) needle.style.transform = `rotate(${angle}deg)`;

  const cx = 115, cy = 107, r = 95;
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
  if (ss) ss.textContent = score >= 70 ? '🟢 Fully Focused'
                         : score >= 40 ? '🟡 Partially Focused'
                         : score > 0   ? '🔴 Low Focus'
                         :               'Awaiting session start';
}

// ── Timeline helper ─────────────────────────────────────────────
function addTL(msg, color, icon) {
  const empty = document.getElementById('tlEmpty');
  if (empty) empty.remove();
  const sc = document.getElementById('timelineScroll');
  if (!sc) return;
  const el = document.createElement('div');
  el.className = 'tl-item';
  el.innerHTML = `<div class="tl-dot" style="background:${color}"></div>
    <div class="tl-time">${fmtSecs(totalSecsElapsed)}</div>
    <div class="tl-msg">${msg}</div>
    <div class="tl-icon">${icon}</div>`;
  sc.appendChild(el);
  sc.scrollTop = sc.scrollHeight;
}

// ── Emotion breakdown bars ───────────────────────────────────────
function renderBreakdownBars() {
  const el = document.getElementById('emotionBreakdownBars');
  if (!el || totalDetections === 0) return;
  const entries = [
    { key: 'focused',    label: 'Focused',    color: '#16a34a' },
    { key: 'bored',      label: 'Bored',      color: '#3b82f6' },
    { key: 'frustrated', label: 'Frustrated', color: '#ca8a04' },
    { key: 'distracted', label: 'Distracted', color: '#dc2626' },
  ];
  el.innerHTML = entries.map(e => {
    const pct = Math.round((emotionCounts[e.key] / totalDetections) * 100);
    return `<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
      <span style="font-size:9.5px;font-weight:600;color:var(--text2);width:64px;flex-shrink:0">${e.label}</span>
      <div style="flex:1;height:6px;background:var(--elevated);border-radius:99px;overflow:hidden">
        <div style="width:${pct}%;height:100%;background:${e.color};border-radius:99px;transition:width 0.6s"></div>
      </div>
      <span style="font-size:9.5px;font-weight:700;color:${e.color};min-width:28px;text-align:right">${pct}%</span>
    </div>`;
  }).join('');
}

// ── Update live stats ────────────────────────────────────────────
function updateLiveStats() {
  if (totalDetections === 0) return;
  const pct = k => Math.round((emotionCounts[k] / totalDetections) * 100) + '%';
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set('lstatFocused',    pct('focused'));
  set('lstatBored',      pct('bored'));
  set('lstatFrustrated', pct('frustrated'));
  set('lstatDistracted', pct('distracted'));
}

// ── Update summary stats ─────────────────────────────────────────
function updateSummaryStats() {
  const total = totalElapsed();
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set('sumDuration',  fmtSecs(total));
  set('sumStudy',     fmtSecs(totalStudySecs));
  set('sumBreaks',    fmtSecs(totalBreakSecs));
  set('sumBlocks',    focusBlocksDone);
  const avgFocus = totalDetections > 0
    ? Math.round((emotionCounts.focused / totalDetections) * 100) + '%'
    : '—';
  set('sumAvgFocus', avgFocus);
  // Nav timer
  const nd = document.getElementById('navTimerDisplay');
  if (nd) nd.textContent = fmtSecs(total);
  const ts = document.getElementById('tlTimerSide');
  if (ts) ts.textContent = fmtSecs(total);
}

// ── Timer card (left panel) ──────────────────────────────────────
function updateTimerCard() {
  const tb     = document.getElementById('timerBig');
  const pb     = document.getElementById('timerProgressBar');
  const pl     = document.getElementById('timerPhaseLabel');
  const bbt    = document.getElementById('blockTimerBig');
  const bph    = document.getElementById('blockPhase');
  const bsub   = document.getElementById('blockSubtitle');
  const bbpb   = document.getElementById('blockProgressBar');
  const statTT = document.getElementById('statTotalTime');
  const banote = document.getElementById('blockAdaptiveNote');

  const timeStr = fmtSecs(blockSecondsLeft);
  const pct     = blockTotalSeconds > 0
    ? Math.round(((blockTotalSeconds - blockSecondsLeft) / blockTotalSeconds) * 100)
    : 0;

  if (tb) tb.textContent = timeStr;
  if (bbt) bbt.textContent = timeStr;
  if (statTT) statTT.textContent = fmtSecs(totalSecsElapsed);

  if (phase === PHASE.STUDY) {
    if (pl) { pl.textContent = 'STUDY BLOCK'; pl.className = 'timer-phase-label study'; }
    if (tb) tb.className = 'timer-big';
    if (pb) { pb.style.width = pct + '%'; pb.className = 'timer-progress-bar'; }
    if (bph) bph.textContent = `Study Block ${focusBlocksDone + 1}`;
    if (bsub) bsub.textContent = `${fmtSecs(blockSecondsLeft)} remaining`;
    if (bbpb) { bbpb.style.width = pct + '%'; bbpb.style.background = 'var(--grad)'; }
    if (banote) banote.textContent = adaptiveBonusSecs > 0
      ? `⭐ +${adaptiveBonusSecs / 60}min bonus earned for sustained focus!` : '';
  } else if (phase === PHASE.BREAK) {
    if (pl) { pl.textContent = 'BREAK'; pl.className = 'timer-phase-label break'; }
    if (tb) tb.className = 'timer-big break-color';
    if (pb) { pb.style.width = pct + '%'; pb.className = 'timer-progress-bar break-bar'; }
    if (bph) bph.textContent = `Break ${breakCount}`;
    if (bsub) bsub.textContent = `${fmtSecs(blockSecondsLeft)} until next study block`;
    if (bbpb) { bbpb.style.width = pct + '%'; bbpb.style.background = 'linear-gradient(90deg,#3b82f6,#60a5fa)'; }
    // Update break overlay timer
    const bot = document.getElementById('breakOverlayTimer');
    if (bot) bot.textContent = fmtSecs(blockSecondsLeft);
  } else if (phase === PHASE.PAUSED) {
    if (pl) { pl.textContent = 'PAUSED'; pl.className = 'timer-phase-label paused'; }
    if (tb) tb.className = 'timer-big paused-color';
    if (pb) { pb.style.width = pct + '%'; pb.className = 'timer-progress-bar paused-bar'; }
    if (bph) bph.textContent = 'Timer Paused';
    if (bsub) bsub.textContent = 'Paused — regain focus to resume';
    if (bbpb) { bbpb.style.width = pct + '%'; bbpb.style.background = 'linear-gradient(90deg,#ca8a04,#fcd34d)'; }
  }
}

// ── Apply study state to all UI panels ───────────────────────────
function applyStudyState(backendEmotion, conf, engRaw) {
  const studyState = STUDY_STATE_MAP[backendEmotion] || 'focused';
  const meta       = STATE_META[studyState];
  const engScore   = (engRaw != null) ? engRaw * 100 : (ENGAGEMENT_SCORES[backendEmotion] || 0.5) * 100;

  // Smooth engagement
  _engScore += ((engScore - _engScore) * 0.4);
  setSpeedometer(_engScore);

  // Counters
  totalDetections++;
  _engagementScores.push(engRaw != null ? engRaw : engScore / 100);

  // Distracted = low engagement sustained regardless of emotion label
  const isLowEng = _engScore < 35;
  if (isLowEng) {
    consecutiveLowEng++;
    consecutiveFocused = 0;
    bonusGivenThisBlock = false;
  } else {
    consecutiveLowEng = 0;
  }

  // State-specific consecutive counts
  if (studyState === 'focused') {
    consecutiveBored = 0;
    consecutiveFrustrated = 0;
    consecutiveFocused++;
    emotionCounts.focused++;
  } else if (studyState === 'bored') {
    consecutiveBored++;
    consecutiveFrustrated = 0;
    consecutiveFocused = 0;
    emotionCounts.bored++;
  } else if (studyState === 'frustrated') {
    consecutiveFrustrated++;
    consecutiveBored = 0;
    consecutiveFocused = 0;
    emotionCounts.frustrated++;
  }

  // Emotion name panel
  const enb  = document.getElementById('emotionNameBig');
  const chip = document.getElementById('emotionStateChip');
  const conf_ = document.getElementById('emotionConf');
  const sam  = document.getElementById('stateAvatarMouth');
  if (enb)  { enb.textContent = meta.label; enb.style.color = meta.color; }
  if (chip) { chip.style.display = 'inline-flex'; chip.className = `emotion-state-chip ${meta.chip}`; chip.textContent = meta.chipTxt; }
  if (conf_) conf_.textContent = `${backendEmotion} · ${conf}% confidence`;
  if (sam)   sam.setAttribute('d', meta.mouth);

  // Suggestion box
  const sbox = document.getElementById('suggestionBox');
  if (sbox) {
    if (meta.suggestion) {
      sbox.textContent = meta.suggestion;
      sbox.className = `suggestion-box show ${meta.suggestionClass}`;
    } else {
      sbox.className = 'suggestion-box';
    }
  }

  updateLiveStats();
  renderBreakdownBars();
  updateSummaryStats();

  // ── Adaptive timer logic ─────────────────────────────────────
  if (!isRunning) return;

  // 1. Focused bonus: +2 min if sustained focus for FOCUSED_BONUS_FRAMES
  if (phase === PHASE.STUDY && studyState === 'focused' && consecutiveFocused >= FOCUSED_BONUS_FRAMES && !bonusGivenThisBlock) {
    bonusGivenThisBlock = true;
    const bonus = 120; // 2 minutes
    blockSecondsLeft  = Math.min(blockSecondsLeft + bonus, blockTotalSeconds);
    adaptiveBonusSecs += bonus;
    addTL('⭐ Focus bonus! +2 min added to study block', '#16a34a', '⭐');
    consecutiveFocused = 0;
  }

  // 2. Bored: suggest break after threshold
  if (phase === PHASE.STUDY && studyState === 'bored' && consecutiveBored >= BORED_THRESHOLD) {
    consecutiveBored = 0;
    boredAlertCount++;
    addTL('😴 Boredom detected — break suggested', '#3b82f6', '😴');
    if (boredAlertCount >= 3) {
      // Auto-start break after 3 bored alerts
      boredAlertCount = 0;
      addTL('☕ Auto-break started (3 boredom alerts)', '#3b82f6', '☕');
      startBreak();
      return;
    }
    // Show bored alert (non-blocking)
    showAlert('😴', 'Feeling Bored?',
      'You\'ve looked bored for a while. Consider switching topics, doing a quick exercise, or taking a short break!',
      'Got it, I\'ll refocus', false);
    emotionCounts.distracted++; // count bored-pause as distracted for stats
  }

  // 3. Frustrated: pause timer
  if (phase === PHASE.STUDY && studyState === 'frustrated' && consecutiveFrustrated >= FRUSTRATED_THRESHOLD) {
    consecutiveFrustrated = 0;
    pauseCount++;
    document.getElementById('statPauseCount').textContent = pauseCount;
    addTL('⚡ Frustration detected — timer paused (30s pause)', '#ca8a04', '⚡');
    showAlert('🧘', 'Take a Breath',
      'Frustration detected. Your timer is paused. Take 30 seconds: inhale 4s → hold 4s → exhale 4s.',
      'I\'m calmer now ✓', false);
    setPhasePaused();
    // Auto-resume after 30 seconds
    setTimeout(() => {
      if (phase === PHASE.PAUSED) {
        dismissAlert();
        setPhaseStudy();
        addTL('▶ Auto-resumed after 30s frustration pause', '#ca8a04', '▶');
      }
    }, 30000);
  }

  // 4. Distracted: pause + show overlay alert
  if (phase === PHASE.STUDY && consecutiveLowEng >= DISTRACTED_THRESHOLD) {
    consecutiveLowEng = 0;
    pauseCount++;
    document.getElementById('statPauseCount').textContent = pauseCount;
    emotionCounts.distracted++;
    addTL('📵 Distraction detected — timer paused', '#dc2626', '📵');
    showAlert('📵', 'You Seem Distracted',
      'Your engagement dropped significantly. Close other tabs, silence notifications, and come back fully.',
      'I\'m focused now ✓', true);
    setPhasePaused();
  }
}

// ── Phase transitions ────────────────────────────────────────────
function setPhaseStudy() {
  phase = PHASE.STUDY;
  const cb = document.getElementById('camBox');
  const cd = document.getElementById('camStatusDot');
  const ct = document.getElementById('camStatusTxt');
  const sl = document.getElementById('camScanLive');
  if (cb) cb.className = 'cam-box study-active';
  if (cd) cd.className = 'cam-status-dot study';
  if (ct) ct.textContent = 'STUDYING';
  if (sl) { sl.className = 'cam-scan-live active'; }
  document.getElementById('breakOverlay')?.classList.remove('show');
}

function setPhasePaused() {
  phase = PHASE.PAUSED;
  const cb = document.getElementById('camBox');
  const cd = document.getElementById('camStatusDot');
  const ct = document.getElementById('camStatusTxt');
  const sl = document.getElementById('camScanLive');
  if (cb) cb.className = 'cam-box paused-active';
  if (cd) cd.className = 'cam-status-dot paused';
  if (ct) ct.textContent = 'PAUSED';
  if (sl) sl.className = 'cam-scan-live';
}

function startBreak() {
  phase = PHASE.BREAK;
  breakCount++;
  focusBlocksDone++;
  bonusGivenThisBlock = false;
  adaptiveBonusSecs   = 0;
  consecutiveFocused  = 0;
  consecutiveBored    = 0;
  consecutiveFrustrated = 0;
  boredAlertCount     = 0;

  blockTotalSeconds = parseInt(document.getElementById('cfgBreak').value) * 60;
  blockSecondsLeft  = blockTotalSeconds;

  const cb = document.getElementById('camBox');
  const cd = document.getElementById('camStatusDot');
  const ct = document.getElementById('camStatusTxt');
  const sl = document.getElementById('camScanLive');
  if (cb) cb.className = 'cam-box break-active';
  if (cd) cd.className = 'cam-status-dot break';
  if (ct) ct.textContent = 'BREAK';
  if (sl) sl.className = 'cam-scan-live active break-scan';
  document.getElementById('breakOverlay')?.classList.add('show');
  document.getElementById('statBreakCount').textContent = breakCount;
  document.getElementById('statFocusBlocks').textContent = focusBlocksDone;
  dismissAlert();
}

function endBreak() {
  document.getElementById('breakOverlay')?.classList.remove('show');
  blockTotalSeconds = studyDuration;
  blockSecondsLeft  = studyDuration;
  setPhaseStudy();
  addTL('📚 Back to studying — new block started', '#16a34a', '📚');
}

// ── Alert overlay ─────────────────────────────────────────────────
function showAlert(emoji, title, msg, btnTxt, isDistraction) {
  if (alertShowing) return;
  alertShowing = true;
  document.getElementById('alertEmoji').textContent = emoji;
  document.getElementById('alertTitle').textContent = title;
  document.getElementById('alertMsg').textContent   = msg;
  const btn = document.getElementById('alertBtn');
  if (btn) {
    btn.textContent = btnTxt;
    btn.className = isDistraction ? 'cam-alert-btn' : 'cam-alert-btn blue';
  }
  document.getElementById('camAlertOverlay')?.classList.add('show');
}

function dismissAlert() {
  alertShowing = false;
  document.getElementById('camAlertOverlay')?.classList.remove('show');
  // Resume study phase if we were paused
  if (phase === PHASE.PAUSED && isRunning) {
    setPhaseStudy();
  }
}

// ── Timer tick (every second) ────────────────────────────────────
function onTick() {
  totalSecsElapsed++;

  // Accumulate phase-specific time
  if (phase === PHASE.STUDY) {
    totalStudySecs++;
    blockSecondsLeft = Math.max(0, blockSecondsLeft - 1);
  } else if (phase === PHASE.BREAK) {
    totalBreakSecs++;
    blockSecondsLeft = Math.max(0, blockSecondsLeft - 1);
  }
  // PAUSED: only total ticks, block does NOT count down

  updateTimerCard();
  updateSummaryStats();

  // Block complete
  if (blockSecondsLeft <= 0) {
    if (phase === PHASE.STUDY) {
      addTL('✅ Study block complete — starting break', '#16a34a', '✅');
      startBreak();
    } else if (phase === PHASE.BREAK) {
      addTL('⏰ Break over — resuming study', '#3b82f6', '⏰');
      endBreak();
    }
  }
}

// ── Detection loop ───────────────────────────────────────────────
function scheduleDetection() {
  if (!isRunning) return;
  detectionLoop = setTimeout(async () => {
    if (!isRunning) return;
    if (!detectionActive) {
      detectionActive = true;
      try {
        if (abortCtrl) abortCtrl.abort();
        abortCtrl = new AbortController();

        const blob = await captureFrame();
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        const sidParam = _sessionId ? '&session_id=' + encodeURIComponent(_sessionId) : '';
        const url = `/predict/?fast=true&save=true${sidParam}`;

        const res  = await Auth.apiFetch(url, { method: 'POST', body: form, signal: abortCtrl.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        _frameCount++;
        const emotion    = data.emotion;
        const conf       = Math.round(data.confidence > 1 ? data.confidence : data.confidence * 100);
        const engagement = data.engagement != null ? data.engagement : (ENGAGEMENT_SCORES[emotion] || 0.5);

        applyStudyState(emotion, conf, engagement);

      } catch (err) {
        if (err.name !== 'AbortError') {
          console.warn('[StudyTimer] Detection error:', err.message);
        }
      } finally {
        detectionActive = false;
      }
    }
    scheduleDetection();
  }, 300);
}

// ── Capture frame ─────────────────────────────────────────────────
function captureFrame() {
  return new Promise((resolve, reject) => {
    const video = document.getElementById('videoEl');
    if (!video || !video.srcObject) return reject(new Error('No camera stream'));
    const canvas = document.createElement('canvas');
    canvas.width  = 320;
    canvas.height = 240;
    canvas.getContext('2d').drawImage(video, 0, 0, 320, 240);
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('toBlob failed')), 'image/jpeg', 0.75);
  });
}

// ── Session start ─────────────────────────────────────────────────
async function doStart() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const v = document.getElementById('videoEl');
    v.srcObject = stream;
    v.style.display = 'block';
    document.getElementById('camIdle').style.display = 'none';
  } catch (e) {
    alert('Camera access denied or unavailable. Please allow camera permission and try again.');
    return;
  }

  // Read config
  studyDuration     = parseInt(document.getElementById('cfgStudy').value) * 60;
  breakDuration     = parseInt(document.getElementById('cfgBreak').value) * 60;
  blockTotalSeconds = studyDuration;
  blockSecondsLeft  = studyDuration;

  // Reset all state
  isRunning            = true;
  phase                = PHASE.STUDY;
  totalStudySecs       = 0;
  totalBreakSecs       = 0;
  totalSecsElapsed     = 0;
  focusBlocksDone      = 0;
  breakCount           = 0;
  pauseCount           = 0;
  adaptiveBonusSecs    = 0;
  consecutiveBored     = 0;
  consecutiveFrustrated = 0;
  consecutiveLowEng    = 0;
  consecutiveFocused   = 0;
  boredAlertCount      = 0;
  bonusGivenThisBlock  = false;
  lastStudyState       = null;
  alertShowing         = false;
  emotionCounts        = { focused: 0, bored: 0, frustrated: 0, distracted: 0 };
  totalDetections      = 0;
  _engScore            = 0;
  _engagementScores    = [];
  _frameCount          = 0;
  _sessionId           = null;
  sessionStartTime     = Date.now();

  // Reset stats display
  ['lstatFocused','lstatBored','lstatFrustrated','lstatDistracted'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = '—';
  });
  ['statFocusBlocks','statBreakCount','statPauseCount'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = '0';
  });
  document.getElementById('endNoteRow')?.classList.remove('show');
  document.getElementById('emotionBreakdownBars').innerHTML =
    '<div style="font-size:10.5px;color:var(--text3);font-style:italic">Detecting emotions…</div>';

  // Reset timeline
  const sc = document.getElementById('timelineScroll');
  if (sc) {
    sc.innerHTML = '';
    const e = document.createElement('div'); e.className = 'tl-empty'; e.id = 'tlEmpty';
    e.textContent = 'Session started. Monitoring…'; sc.appendChild(e);
  }

  // Camera & nav UI
  setPhaseStudy();
  document.getElementById('navLiveBadge').style.display  = 'flex';
  document.getElementById('navTimerDisplay').style.display = 'block';
  document.getElementById('camScanLive').classList.add('active');

  // Buttons
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnBreak').disabled = false;
  document.getElementById('btnStop').disabled  = false;
  document.getElementById('cfgStudy').disabled = true;
  document.getElementById('cfgBreak').disabled = true;

  setSpeedometer(0);
  updateTimerCard();

  // Start timers
  timerTick = setInterval(onTick, 1000);

  // Backend session
  try {
    const res  = await Auth.apiFetch('/sessions/start/', { method: 'POST' });
    const data = await res.json();
    _sessionId = data.session_id;
  } catch (e) {
    console.warn('[StudyTimer] Session start failed:', e);
  }

  addTL(`📚 Study session started (${studyDuration / 60}min blocks)`, '#16a34a', '🎬');
  scheduleDetection();
}

// ── Manual break ──────────────────────────────────────────────────
function manualBreak() {
  if (!isRunning || phase === PHASE.BREAK) return;
  addTL('☕ Manual break started', '#3b82f6', '☕');
  startBreak();
}

// ── Resume early from break ───────────────────────────────────────
function resumeEarly() {
  if (phase !== PHASE.BREAK) return;
  addTL('▶ Break ended early', '#3b82f6', '▶');
  endBreak();
}

// ── Session stop ──────────────────────────────────────────────────
async function doStop() {
  if (!isRunning) return;
  isRunning = false;

  clearInterval(timerTick);
  timerTick = null;
  clearTimeout(detectionLoop);
  detectionLoop = null;
  if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }
  detectionActive = false;

  // Stop camera
  const v = document.getElementById('videoEl');
  if (v && v.srcObject) { v.srcObject.getTracks().forEach(t => t.stop()); v.srcObject = null; v.style.display = 'none'; }
  document.getElementById('camIdle').style.display = 'flex';
  document.getElementById('breakOverlay')?.classList.remove('show');
  document.getElementById('camAlertOverlay')?.classList.remove('show');

  // UI cleanup
  const cb = document.getElementById('camBox');
  if (cb) cb.className = 'cam-box';
  document.getElementById('camStatusDot').className = 'cam-status-dot ready';
  document.getElementById('camStatusTxt').textContent = 'STOPPED';
  document.getElementById('camScanLive').className = 'cam-scan-live';
  document.getElementById('navLiveBadge').style.display   = 'none';
  document.getElementById('navTimerDisplay').style.display = 'none';
  document.getElementById('timerPhaseLabel').textContent   = 'SESSION ENDED';
  document.getElementById('timerPhaseLabel').className     = 'timer-phase-label';
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnBreak').disabled = true;
  document.getElementById('btnStop').disabled  = true;
  document.getElementById('cfgStudy').disabled = false;
  document.getElementById('cfgBreak').disabled = false;

  if (focusBlocksDone === 0 && phase === PHASE.STUDY) focusBlocksDone = 1;
  const totalSecs = totalElapsed();

  addTL(`🏁 Session ended — ${fmtSecs(totalSecs)} total`, '#6b7280', '🏁');
  updateSummaryStats();
  renderBreakdownBars();

  // End note
  const focusPct = totalDetections > 0
    ? Math.round((emotionCounts.focused / totalDetections) * 100) : 0;
  const note = focusPct >= 65 ? 'Outstanding focus! 🌟'
             : focusPct >= 40 ? 'Good session — keep it up! 💪'
             :                  'Keep going — each session builds the habit! 🔥';
  const enr  = document.getElementById('endNoteRow');
  const etb  = document.getElementById('endTimeBadge');
  const ent  = document.getElementById('endNoteText');
  if (etb) etb.textContent = fmtSecs(totalSecs);
  if (ent) ent.textContent = note;
  if (enr) enr.classList.add('show');

  // Backend session end
  if (_sessionId) {
    try {
      await Auth.apiFetch('/sessions/end/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: _sessionId, total_frames: _frameCount }),
      });
    } catch (e) {
      console.warn('[StudyTimer] Session end error:', e);
    }
    _sessionId = null;
  }

  // Save study timer specific summary to backend
  if (_engagementScores.length > 0) {
    const avgEng = _engagementScores.reduce((a, b) => a + b, 0) / _engagementScores.length;
    try {
      await Auth.apiFetch('/studytimer/summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_seconds:     totalSecs,
          study_seconds:     totalStudySecs,
          break_seconds:     totalBreakSecs,
          focus_blocks:      focusBlocksDone,
          break_count:       breakCount,
          pause_count:       pauseCount,
          avg_engagement:    Math.round(avgEng * 100) / 100,
          focused_pct:       Math.round((emotionCounts.focused      / Math.max(totalDetections, 1)) * 100),
          bored_pct:         Math.round((emotionCounts.bored        / Math.max(totalDetections, 1)) * 100),
          frustrated_pct:    Math.round((emotionCounts.frustrated   / Math.max(totalDetections, 1)) * 100),
          distracted_pct:    Math.round((emotionCounts.distracted   / Math.max(totalDetections, 1)) * 100),
        }),
      });
    } catch (e) {
      console.warn('[StudyTimer] Summary save error:', e);
    }
  }
}

// ── Auth guard + init ─────────────────────────────────────────────
if (!Auth.requireAuth()) throw new Error('Not authenticated');
window.addEventListener('load', () => {
  initGreeting();
  setSpeedometer(0);
});