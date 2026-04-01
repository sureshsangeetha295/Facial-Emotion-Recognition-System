// timeline.js — chip-row emotion timeline

const Timeline = (() => {
  const history = [];

  function push(emotion, confidence) {
    history.push({ emotion, confidence, time: Date.now() });
    if (history.length > CONFIG.TIMELINE_MAX_POINTS) history.shift();
    render();
  }

  function render() {
    const el    = document.getElementById("timelineDots");
    const tagEl = document.getElementById("timelineTag");
    if (!el) return;

    if (history.length === 0) {
      el.innerHTML = `<span class="timeline-empty">No detections yet</span>`;
      if (tagEl) tagEl.innerHTML = "";
      return;
    }

    // Chip row — all entries, newest is active
    el.innerHTML = history.map((h, i) => {
      const c      = CONFIG.EMOTION_COLORS[h.emotion] || "#6b7280";
      const pct    = Math.round(h.confidence * 100);
      const isLast = i === history.length - 1;

      const connector = i < history.length - 1
        ? `<div class="tl-connector"></div>`
        : "";

      if (isLast) {
        return `
          <div class="tl-chip tl-chip--active"
               style="border-color:${c};background:${c}22;"
               title="${h.emotion} ${pct}%">
            <div class="tl-chip-dot" style="background:${c};"></div>
            <span class="tl-chip-label" style="color:${c};">${h.emotion}</span>
          </div>${connector}`;
      }

      return `
        <div class="tl-chip tl-chip--done" title="${h.emotion} ${pct}%">
          <div class="tl-chip-dot" style="background:${c};opacity:0.7;"></div>
          <span class="tl-chip-label">${h.emotion}</span>
        </div>${connector}`;
    }).join("");

    // Summary count chips in #timelineTag
    if (tagEl) {
      const counts = {};
      history.forEach(h => { counts[h.emotion] = (counts[h.emotion] || 0) + 1; });
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3);
      tagEl.innerHTML = top.map(([e, n]) => {
        const c = CONFIG.EMOTION_COLORS[e] || "#6b7280";
        return `<span class="tl-summary-chip"
                      style="background:${c}18;border-color:${c}44;color:${c};">
                  ${e} <b>${n}</b>
                </span>`;
      }).join("");
    }
  }

  function clear() {
    history.length = 0;
    render();
  }

  return { push, clear };
})();