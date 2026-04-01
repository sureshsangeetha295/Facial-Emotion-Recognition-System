// api.js — all backend communication

const API = (() => {

  async function checkHealth() {
    try {
      const res = await fetch(CONFIG.HEALTH_URL, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      return data.status === "ok";
    } catch {
      return false;
    }
  }

  async function predict(blob) {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);

    try {
      // Auth.apiFetch attaches Bearer token automatically
      const res = await Auth.apiFetch(CONFIG.PREDICT_URL, {
        method: "POST",
        body:   formData,
        signal: controller.signal,
      });

      clearTimeout(timer);

      const data = await res.json();

      if (!res.ok || data.error) {
        const msg = data.message || data.error || `HTTP ${res.status}`;
        const err = new Error(msg);
        err.code  = data.error;
        throw err;
      }

      return data;
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  }

  return { checkHealth, predict };
})();