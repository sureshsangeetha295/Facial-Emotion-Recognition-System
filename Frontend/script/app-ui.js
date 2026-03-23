// app-ui.js — UI hooks for camera start/stop (LIVE badge, card resets)

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
      const live = document.getElementById("live-badge");
      if (live) live.classList.add("on");
    };

    window.stopCamera = function () {
      _stop();
      const live = document.getElementById("live-badge");
      const ms   = document.getElementById("ms-badge");
      const msg  = document.getElementById("emotion-msg");
      const sub  = document.getElementById("bottom-subtitle");

      if (live) live.classList.remove("on");
      if (ms)   ms.style.display = "none";
      if (msg)  msg.style.display = "none";
      if (sub)  sub.textContent = "Detect an emotion to see the breakdown";

      document.querySelectorAll(".e-card").forEach(c => {
        c.classList.remove("active");
        c.style.cssText = "";
      });
      document.querySelectorAll(".e-bar").forEach(b => b.style.width = "0%");
      document.querySelectorAll("[id^='ep-']").forEach(p => p.textContent = "—");
    };
  });
})();