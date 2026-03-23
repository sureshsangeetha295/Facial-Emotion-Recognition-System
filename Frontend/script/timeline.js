// timeline.js — rolling emotion timeline bar

const Timeline = (() => {
  const history = [];

  function push(emotion, confidence) {
    history.push({ emotion, confidence, time: Date.now() });
    if (history.length > CONFIG.TIMELINE_MAX_POINTS) history.shift();
    render();
  }

  function render() {
    const el = document.getElementById("timeline");
    if (!el) return;
    if (history.length === 0) {
      el.innerHTML = `<span class="timeline-empty">No detections yet</span>`;
      return;
    }
    el.innerHTML = history.map(h => {
      const c = CONFIG.EMOTION_COLORS[h.emotion] || "#6b7280";
      const pct = Math.round(h.confidence * 100);
      return `<div class="tl-dot" style="background:${c};opacity:${0.4 + pct/140}" title="${h.emotion} ${pct}%"></div>`;
    }).join("");

    // Summary counts
    const counts = {};
    history.forEach(h => { counts[h.emotion] = (counts[h.emotion] || 0) + 1; });
    const top = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,3);
    const summaryEl = document.getElementById("timeline-summary");
    if (summaryEl) {
      summaryEl.innerHTML = top.map(([e, n]) =>
        `<span class="tl-chip" style="background:${CONFIG.EMOTION_COLORS[e]}22;border-color:${CONFIG.EMOTION_COLORS[e]}44;color:${CONFIG.EMOTION_COLORS[e]}">${e} <b>${n}</b></span>`
      ).join("");
    }
  }

  function clear() {
    history.length = 0;
    render();
  }

  return { push, clear };
})();