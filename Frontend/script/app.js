// app.js

const state = {
  stream: null,
  detecting: false,
  cameraReady: false,
  backendOnline: false
};

let video, canvas, btnStart, btnDetect;

window.addEventListener("DOMContentLoaded", async () => {
  video     = document.getElementById("cam-video");
  canvas    = document.getElementById("cam-canvas");
  btnStart  = document.getElementById("btn-start");
  btnDetect = document.getElementById("btn-detect");

  await checkBackend();
  setInterval(checkBackend, CONFIG.HEALTH_INTERVAL_MS);

  document.addEventListener("keydown", e => {
    if (e.code === "Space" && state.cameraReady && !state.detecting) {
      e.preventDefault();
      detect();
    }
  });
});

async function checkBackend() {
  const online = await API.checkHealth();
  state.backendOnline = online;
  Renderer.setBackendStatus(online);
}

async function startCamera() {
  try {
    // First check if any video devices exist at all
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");

    if (videoDevices.length === 0) {
      Renderer.setStatus("error", "No camera detected — check Device Manager or enable camera in Windows Privacy settings");
      return;
    }

    // Try ideal constraints first, fall back to basic if needed
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
      });
    } catch (constraintErr) {
      // Fall back to minimal constraints
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
    }

    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    state.cameraReady = true;

    document.getElementById("cam-col").classList.add("active");
    document.getElementById("cam-placeholder").style.display = "none";
    video.style.display = "block";

    btnStart.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/></svg> Stop`;
    btnStart.classList.add("active");
    btnDetect.disabled = false;

    if (!state.backendOnline) {
      Renderer.setStatus("error", "Backend offline — run python main.py");
    } else {
      Renderer.setStatus("info", "Ready — press Space or click Detect");
    }

  } catch (err) {
    console.error("[Camera error]", err.name, err.message);

    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      Renderer.setStatus("error", "Camera permission denied — click the camera icon in the address bar and allow access");
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      Renderer.setStatus("error", "Camera not found — make sure no other app is using it (Teams, Zoom, etc.) and try again");
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      Renderer.setStatus("error", "Camera is in use by another app — close Teams, Zoom, or any other camera app and try again");
    } else if (err.name === "OverconstrainedError") {
      Renderer.setStatus("error", "Camera settings not supported — trying basic mode");
      // Try one more time with no constraints
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        state.stream = stream;
        video.srcObject = stream;
        await video.play();
        state.cameraReady = true;
        document.getElementById("cam-col").classList.add("active");
        document.getElementById("cam-placeholder").style.display = "none";
        video.style.display = "block";
        btnStart.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/></svg> Stop`;
        btnStart.classList.add("active");
        btnDetect.disabled = false;
        Renderer.setStatus("info", "Ready — press Space or click Detect");
      } catch (e) {
        Renderer.setStatus("error", "Camera failed: " + e.message);
      }
    } else {
      Renderer.setStatus("error", "Camera error: " + err.message + " — try refreshing the page");
    }
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  state.cameraReady = false;
  video.srcObject   = null;
  video.style.display = "none";

  const placeholder = document.getElementById("cam-placeholder");
  if (placeholder) placeholder.style.display = "";

  const camCol = document.getElementById("cam-col");
  if (camCol) camCol.classList.remove("active");

  btnStart.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg> Start Camera`;
  btnStart.classList.remove("active");
  btnDetect.disabled = true;

  const statusBar = document.getElementById("status-bar");
  if (statusBar) statusBar.style.display = "none";

  const msBadge = document.getElementById("ms-badge");
  if (msBadge) msBadge.style.display = "none";
}

function toggleCamera() {
  state.cameraReady ? stopCamera() : startCamera();
}

async function detect() {
  if (!state.cameraReady || state.detecting) return;

  if (!state.backendOnline) {
    Renderer.setStatus("error", "Backend offline — run python main.py");
    return;
  }

  state.detecting = true;
  Renderer.showSpinner(true);
  Renderer.setStatus("info", "Analyzing...");

  try {
    const ctx = canvas.getContext("2d");
    canvas.width  = video.videoWidth  || CONFIG.CAPTURE_WIDTH;
    canvas.height = video.videoHeight || CONFIG.CAPTURE_HEIGHT;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(res =>
      canvas.toBlob(res, "image/jpeg", CONFIG.JPEG_QUALITY)
    );

    if (!blob) {
      throw new Error("Frame capture failed — try again");
    }

    const t0   = performance.now();
    const data = await API.predict(blob);
    const ms   = Math.round(performance.now() - t0);

    Renderer.showResult({ ...data, ms });
    Timeline.push(data.emotion, data.confidence);

    const msBadge = document.getElementById("ms-badge");
    if (msBadge) { msBadge.textContent = ms + " ms"; msBadge.style.display = "inline"; }

  } catch (err) {
    if (err.name === "AbortError") {
      Renderer.setStatus("error", "Request timed out — backend took too long");
    } else if (err.code === "no_face") {
      Renderer.setStatus("error", "No face detected — face the camera directly");
    } else if (err.code === "offline") {
      Renderer.setStatus("error", "Backend offline — run python main.py");
      state.backendOnline = false;
      Renderer.setBackendStatus(false);
    } else {
      Renderer.setStatus("error", err.message || "Detection failed — try again");
    }
  } finally {
    state.detecting = false;
    Renderer.showSpinner(false, state.cameraReady);
  }
}