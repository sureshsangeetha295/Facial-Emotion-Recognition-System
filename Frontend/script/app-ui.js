// app-ui.js — UI hooks for camera start/stop (LIVE badge, card resets, live toggle)

(function () {
  function wait(name, cb) {
    const t = setInterval(() => {
      if (typeof window[name] !== "undefined") { clearInterval(t); cb(); }
    }, 40);
  }

  wait("startCamera", () => {
    const _start = startCamera;
    const _stop  = stopCamera;

    window.startCamera = async function () {
      await _start();

      // Show live badge
      const live = document.getElementById("live-badge");
      if (live) live.classList.add("on");

      // Enable detect + go live buttons
      const btnDetect = document.getElementById("btn-detect");
      const btnLive   = document.getElementById("btn-live");
      if (btnDetect) btnDetect.disabled = false;
      if (btnLive)   btnLive.disabled   = false;

      // Show countdown wrap
      const cw = document.getElementById("countdown-wrap");
      if (cw) cw.style.display = "none"; // only show when live is active
    };

    window.stopCamera = function () {
      // If live was running, stop it cleanly
      if (window._liveActive) {
        window._liveActive = false;
        const btnLive = document.getElementById("btn-live");
        if (btnLive) {
          btnLive.classList.remove("active");
          btnLive.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>
            </svg>
            Go Live`;
        }
        const cw = document.getElementById("countdown-wrap");
        if (cw) cw.style.display = "none";
      }

      _stop();

      // Reset badges
      const live = document.getElementById("live-badge");
      const ms   = document.getElementById("ms-badge");
      const msg  = document.getElementById("emotion-msg");
      const sub  = document.getElementById("bottom-subtitle");

      if (live) live.classList.remove("on");
      if (ms)   ms.style.display = "none";
      if (msg)  msg.style.display = "none";
      if (sub)  sub.textContent = "Detect an emotion to see the breakdown";

      // Disable buttons again
      const btnDetect = document.getElementById("btn-detect");
      const btnLive   = document.getElementById("btn-live");
      if (btnDetect) btnDetect.disabled = true;
      if (btnLive)   btnLive.disabled   = true;

      // Reset emotion cards
      document.querySelectorAll(".e-card").forEach(c => {
        c.classList.remove("active");
        c.style.cssText = "";
      });
      document.querySelectorAll(".e-bar").forEach(b => b.style.width = "0%");
      document.querySelectorAll("[id^='ep-']").forEach(p => p.textContent = "—");
    };
  });

  // ── TOGGLE LIVE ──
  window.toggleLive = function () {
    if (!window._liveActive) {
      // Start live
      window._liveActive = true;
      startLiveDetection();

      const btnLive = document.getElementById("btn-live");
      if (btnLive) {
        btnLive.classList.add("active");
        btnLive.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
          </svg>
          Stop Live`;
      }

      const cw = document.getElementById("countdown-wrap");
      if (cw) cw.style.display = "grid";

    } else {
      // Stop live
      window._liveActive = false;
      stopLiveDetection();

      const btnLive = document.getElementById("btn-live");
      if (btnLive) {
        btnLive.classList.remove("active");
        btnLive.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>
          </svg>
          Go Live`;
      }

      const cw = document.getElementById("countdown-wrap");
      if (cw) cw.style.display = "none";
    }
  };

  // ── KEYBOARD SHORTCUTS ──
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.code === "Space") {
      e.preventDefault();
      if (typeof detect === "function") detect();
    }
    if (e.key === "l" || e.key === "L") {
      if (typeof toggleLive === "function" && !document.getElementById("btn-live")?.disabled) {
        toggleLive();
      }
    }
  });

})();