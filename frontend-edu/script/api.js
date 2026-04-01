// api.js — EngageAI backend communication (UPDATED)

const API = (() => {

  // ✅ Check backend health
  async function checkHealth() {
    try {
      const res = await fetch(CONFIG.HEALTH_URL, {
        signal: AbortSignal.timeout(3000)
      });

      const data = await res.json();
      return data.status === "ok";

    } catch {
      return false;
    }
  }

  // ✅ Send frame to backend
  async function predict(blob) {

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch(CONFIG.PREDICT_URL, {
        method: "POST",
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timer);

      const data = await res.json();

      // ❗ Handle backend errors
      if (!res.ok || data.error) {
        const msg = data.message || data.error || `HTTP ${res.status}`;
        const err = new Error(msg);
        err.code = data.error;
        throw err;
      }

      // ✅ EXPECTED RESPONSE FORMAT (your backend)
      /*
        {
          emotion: "Happiness",
          confidence: 0.87,
          probabilities: {...},
          engagement: "Engaged"   // ✅ NEW FIELD
        }
      */

      return data;

    } catch (err) {
      clearTimeout(timer);
      console.error("Predict error:", err);
      throw err;
    }
  }

  return {
    checkHealth,
    predict
  };

})();