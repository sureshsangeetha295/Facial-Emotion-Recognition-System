import os
import io
import asyncio
import webbrowser
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
executor       = ThreadPoolExecutor(max_workers=1)


# Background preload
def _preload():
    try:
        import testing
        testing.get_detector()
        testing.get_model()
    except Exception as e:
        print(f"[EmotionAI] Preload error: {e}")


# Lifespan (modern FastAPI startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _preload)
    yield


# App setup
app = FastAPI(title="EmotionAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Predict pipeline
def run_pipeline(img_rgb: np.ndarray):
    import testing
    idx, confidence, probs = testing.predict_emotion_from_image(img_rgb)

    conf = float(confidence)
    if conf > 1.0:
        conf /= 100.0

    norm_probs = [
        round(float(p) / 100.0 if float(p) > 1.0 else float(p), 4)
        for p in probs
    ]

    return {
        "emotion": EMOTION_LABELS[idx],
        "confidence": round(conf, 4),
        "all_probabilities": norm_probs,
    }, None


# API routes

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_pil = img_pil.resize((640, 480), Image.Resampling.LANCZOS)
        img     = np.array(img_pil)

        loop        = asyncio.get_event_loop()
        result, err = await loop.run_in_executor(executor, run_pipeline, img)

        if err:
            return JSONResponse(status_code=400, content={
                "error":   "no_face",
                "message": "No face detected. Please face the camera directly with good lighting.",
            })

        return result

    except Exception as exc:
        print(f"[EmotionAI] Predict error: {exc}")
        return JSONResponse(status_code=500, content={
            "error":   "server_error",
            "message": str(exc),
        })


# Frontend directory
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

FRONTEND_DIR: str = ""
for _name in ("frontend", "Frontend", "front-end", "Front-End"):
    _candidate = os.path.join(PROJECT_ROOT, _name)
    if os.path.isdir(_candidate):
        FRONTEND_DIR = _candidate
        break

if not FRONTEND_DIR:
    raise RuntimeError(
        f"\n\n  Frontend folder not found under: {PROJECT_ROOT}\n"
        "  Expected one of: frontend / Frontend / front-end / Front-End\n"
    )

print(f"[EmotionAI] Frontend: {FRONTEND_DIR}  exists={os.path.isdir(FRONTEND_DIR)}")


# Frontend pages (must be before static mount)

# "/" now opens login first
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

# Landing page (after login)
@app.get("/home")
@app.get("/index.html")
async def landing_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/app")
@app.get("/app.html")
async def app_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.html"))

@app.get("/login")
@app.get("/login.html")
async def login_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

@app.get("/faq")
@app.get("/faq.html")
async def faq_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "faq.html"))

@app.get("/feedback")
@app.get("/feedback.html")
async def feedback_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "feedback.html"))

# Logout — clears session cookie and redirects back to login
@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    # Clear the session cookie (name matches what login.html sets)
    response.delete_cookie("ea_session")
    response.delete_cookie("ea_cookie_consent")
    return response


# Static catch-all — must be last
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# Entry point
if __name__ == "__main__":
    webbrowser.open("http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)