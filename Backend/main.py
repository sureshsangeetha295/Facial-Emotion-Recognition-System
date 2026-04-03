import os
os.environ["TF_USE_LEGACY_KERAS"]  = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io
import json
import uuid
import re
import asyncio
import psycopg2
import psycopg2.extras
import webbrowser
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt


# ── ENV ───────────────────────────────────────────────────────────────────────

load_dotenv()

APP_HOST     = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT     = int(os.getenv("APP_PORT", "8000"))
SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM    = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS   = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",   "30"))

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@1234")

DB_DSN = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT",  "5432")),
    "dbname":   os.getenv("DB_NAME",     "emotionai"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


# ── PASSWORD ──────────────────────────────────────────────────────────────────

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def validate_password_strength(pw: str) -> Optional[str]:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return "Password must include at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", pw):
        return "Password must include at least one lowercase letter (a-z)."
    if not re.search(r"[0-9]", pw):
        return "Password must include at least one number (0-9)."
    if not re.search(r"[^A-Za-z0-9]", pw):
        return "Password must include at least one special character (!@#$...)."
    return None

def validate_username(username: str) -> Optional[str]:
    """
    Allowed: letters (a-z A-Z), digits (0-9), underscore (_), dot (.), hyphen (-), space.
    Rules:
      - 3 to 30 characters
      - Only letters, numbers, spaces, _ . -
      - Must start and end with a letter or number (no leading/trailing spaces)
      - Must contain at least one letter (not digits/symbols only)
    Examples: "john doe", "Dayana Priya", "john_doe", "JohnDoe99", "j.doe-24"
    """
    u = username.strip()
    if len(u) < 3:
        return "Username must be at least 3 characters."
    if len(u) > 30:
        return "Username must be 30 characters or fewer."
    if not re.match(r"^[A-Za-z0-9._ -]+$", u):
        return "Username can only contain letters, numbers, spaces, _ . and -"
    if not re.match(r"^[A-Za-z0-9]", u):
        return "Username must start with a letter or number."
    if not re.search(r"[A-Za-z0-9]$", u):
        return "Username must end with a letter or number."
    if not re.search(r"[A-Za-z]", u):
        return "Username must contain at least one letter."
    return None


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token(user_id: int, is_admin: bool = False) -> str:
    return create_token({"sub": str(user_id), "admin": is_admin, "type": "access"},
                        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

def create_refresh_token(user_id: int) -> str:
    return create_token({"sub": str(user_id), "type": "refresh"},
                        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ── DB ────────────────────────────────────────────────────────────────────────

def db_conn():
    con = psycopg2.connect(**DB_DSN)
    con.autocommit = False
    return con

def fetchone(cur) -> Optional[dict]:
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))

def fetchall(cur) -> list:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def init_db():
    """Create all tables and sync admin account from .env."""
    con = db_conn()
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            BIGSERIAL    PRIMARY KEY,
            username      VARCHAR(80)  NOT NULL UNIQUE,
            email         VARCHAR(254) NOT NULL UNIQUE,
            password_hash TEXT         NOT NULL,
            is_admin      BOOLEAN      NOT NULL DEFAULT FALSE,
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            last_login    TIMESTAMPTZ
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id         BIGSERIAL    PRIMARY KEY,
            user_id    BIGINT       REFERENCES users(id) ON DELETE SET NULL,
            emotion    VARCHAR(40)  NOT NULL,
            confidence REAL         NOT NULL,
            engagement REAL         NOT NULL,
            source     VARCHAR(20)  NOT NULL DEFAULT 'webcam',
            all_probs  TEXT,
            created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_det_user    ON detections(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_det_created ON detections(created_at)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_timeline (
            id                 BIGSERIAL   PRIMARY KEY,
            session_id         TEXT        NOT NULL,
            user_id            BIGINT      REFERENCES users(id) ON DELETE SET NULL,
            source             VARCHAR(20) NOT NULL DEFAULT 'webcam',
            time_offset        REAL        NOT NULL DEFAULT 0.0,
            emotion            VARCHAR(40) NOT NULL,
            engagement         REAL        NOT NULL,
            average_engagement REAL,
            dominant_emotion   VARCHAR(40),
            frame_count        INTEGER     NOT NULL DEFAULT 1,
            started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at           TIMESTAMPTZ
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stl_session ON session_timeline(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stl_user    ON session_timeline(user_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         BIGSERIAL    PRIMARY KEY,
            user_id    BIGINT       REFERENCES users(id) ON DELETE SET NULL,
            username   VARCHAR(80)  NOT NULL DEFAULT 'Guest',
            email      VARCHAR(254),
            rating     SMALLINT     CHECK (rating BETWEEN 1 AND 5),
            category   VARCHAR(60)  NOT NULL DEFAULT 'General',
            message    TEXT         NOT NULL,
            status     VARCHAR(20)  NOT NULL DEFAULT 'new',
            created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    # ── Migrate existing feedback table ──────────────────────────────────────
    # Adds any columns that were introduced after the initial deploy.
    # Each ALTER runs in isolation so an already-existing column is just skipped.
    _fb_migrations = [
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS username  VARCHAR(80)  NOT NULL DEFAULT 'Guest'",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS email     VARCHAR(254)",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS rating    SMALLINT",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS category  VARCHAR(60)  NOT NULL DEFAULT 'General'",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS status    VARCHAR(20)  NOT NULL DEFAULT 'new'",
    ]
    for _sql in _fb_migrations:
        try:
            cur.execute(_sql)
            con.commit()
        except Exception as _e:
            con.rollback()
            print(f"[EmotionAI] feedback migration skipped (already applied): {_e}")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fb_user    ON feedback(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fb_created ON feedback(created_at)")

    # Sync admin account from .env (create or update password)
    cur.execute("SELECT id FROM users WHERE username=%s AND is_admin=TRUE", (ADMIN_USERNAME,))
    existing_admin = cur.fetchone()
    if not existing_admin:
        cur.execute(
            "INSERT INTO users (username, email, password_hash, is_admin) VALUES (%s,%s,%s,TRUE)",
            (ADMIN_USERNAME, f"{ADMIN_USERNAME}@emotionai.local", hash_password(ADMIN_PASSWORD))
        )
        print(f"[EmotionAI] Default admin created  ->  username: {ADMIN_USERNAME}")
        print("[EmotionAI] Please change the admin password after first login!")
    else:
        cur.execute(
            "UPDATE users SET password_hash=%s WHERE username=%s AND is_admin=TRUE",
            (hash_password(ADMIN_PASSWORD), ADMIN_USERNAME)
        )
        print(f"[EmotionAI] Admin credentials synced from .env  ->  username: {ADMIN_USERNAME}")

    con.commit()
    cur.close()
    con.close()
    print("[EmotionAI] Database initialised")


# ── LIFESPAN ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        import testing
        testing.get_haar_detector()
        testing.get_model()
        print("[EmotionAI] Preload complete")
    except Exception as e:
        print(f"[EmotionAI] Preload error (non-fatal): {e}")
    webbrowser.open(f"http://localhost:{APP_PORT}")
    yield


# ── APP (must be created before auth dependencies and route decorators) ────────

app = FastAPI(title="EmotionAI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── AUTH DEPENDENCIES ─────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    row = fetchone(cur)
    cur.close(); con.close()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return row

async def get_admin_user(current: dict = Depends(get_current_user)):
    if not current.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current


# ── SCHEMAS ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class FeedbackRequest(BaseModel):
    username: str                    # mandatory — display name shown in admin
    email:    Optional[str] = None   # optional
    rating:   Optional[int] = None
    category: str           = "General"
    message:  str

class SessionEndRequest(BaseModel):
    session_id:         str
    average_engagement: float
    dominant_emotion:   str

class SessionStartRequest(BaseModel):
    source: str = "webcam"

class SessionStopRequest(BaseModel):
    session_id:   str
    total_frames: int = 0

class AdminCreateUserRequest(BaseModel):
    username: str
    email:    str
    password: str
    is_admin: bool = False


# ── ML ────────────────────────────────────────────────────────────────────────

EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

ENGAGEMENT_SCORES: dict[str, float] = {
    "Happiness": 1.0, "Surprise": 1.0, "Neutral": 0.6,
    "Sadness": 0.3,   "Fear": 0.3,    "Anger": 0.1, "Disgust": 0.1,
}

def emotion_to_engagement(emotion: str) -> float:
    return ENGAGEMENT_SCORES.get(emotion, 0.5)

executor = ThreadPoolExecutor(max_workers=1)

def run_pipeline(img_rgb: np.ndarray, use_mtcnn: bool = False):
    import testing
    idx, confidence, probs = testing.predict_emotion_from_image(img_rgb, use_mtcnn=use_mtcnn)
    conf = float(confidence)
    if conf > 1.0:
        conf /= 100.0
    norm_probs = [round(float(p)/100.0 if float(p)>1.0 else float(p), 4) for p in probs]
    emotion = EMOTION_LABELS[idx]
    return {
        "emotion":           emotion,
        "confidence":        round(conf, 4),
        "all_probabilities": norm_probs,
        "engagement":        emotion_to_engagement(emotion),
        "timestamp":         datetime.utcnow().isoformat(),
    }, None


# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def register(body: RegisterRequest):
    err = validate_password_strength(body.password)
    if err:
        raise HTTPException(status_code=400, detail=err)
    un_err = validate_username(body.username)
    if un_err:
        raise HTTPException(status_code=400, detail=un_err)
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s OR email=%s",
                    (body.username.strip(), body.email.strip()))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already taken.")
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s,%s,%s)",
                    (body.username.strip(), body.email.strip(), hash_password(body.password)))
        con.commit()
        return {"message": "Account created successfully"}
    finally:
        cur.close(); con.close()


@app.post("/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    FIX: Removed AND is_admin=FALSE — now works for all active users.
    is_admin is returned so the frontend can route to admin panel if needed.
    """
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT * FROM users WHERE username=%s",
            (form.username,)
        )
        row = fetchone(cur)
        if not row or not verify_password(form.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (row["id"],))
        con.commit()
        return {
            "access_token":  create_access_token(row["id"], is_admin=bool(row["is_admin"])),
            "refresh_token": create_refresh_token(row["id"]),
            "token_type":    "bearer",
            "is_admin":      bool(row["is_admin"]),
        }
    finally:
        cur.close(); con.close()


@app.post("/auth/admin/login")
async def admin_login(form: OAuth2PasswordRequestForm = Depends()):
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT * FROM users WHERE username=%s AND is_admin=TRUE",
            (form.username,)
        )
        row = fetchone(cur)
        if not row or not verify_password(form.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid admin credentials")
        cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (row["id"],))
        con.commit()
        return {
            "access_token":  create_access_token(row["id"], is_admin=True),
            "refresh_token": create_refresh_token(row["id"]),
            "token_type":    "bearer",
        }
    finally:
        cur.close(); con.close()


@app.post("/auth/refresh")
async def refresh_token(body: RefreshRequest):
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    row = fetchone(cur)
    cur.close(); con.close()
    if not row:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return {
        "access_token": create_access_token(row["id"], is_admin=bool(row["is_admin"])),
        "token_type":   "bearer",
    }


@app.get("/auth/me")
async def me(current: dict = Depends(get_current_user)):
    return {
        "id":         current["id"],
        "username":   current["username"],
        "email":      current["email"],
        "is_admin":   current["is_admin"],
        "created_at": current["created_at"],
        "last_login": current["last_login"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# In-memory store of active sessions per user: user_id -> session_id
# This ensures every /predict call within a continuous webcam session shares
# the same session_id even when the frontend does not call /sessions/start/ first.
_active_sessions: dict[int, str] = {}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    fast: bool = Query(False),
    save: bool = Query(True),
    session_id: Optional[str] = Query(None),
    current: dict = Depends(get_current_user),
):
    """
    Accepts webcam frames for real-time emotion detection.

    Session-ID resolution order:
      1. Use the session_id query param if the frontend passed one.
      2. Reuse the in-memory active session for this user (so consecutive frames
         that don't carry a session_id still group into the same session).
      3. Create a new UUID and cache it as the active session for this user.

    This means detections will always have a session_id in the DB — no more
    null/dash rows in the admin detections table.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(img)
        loop = asyncio.get_event_loop()
        use_mtcnn = not fast
        result, _ = await loop.run_in_executor(executor, lambda: run_pipeline(img_rgb, use_mtcnn=use_mtcnn))

        uid = current["id"]
        if session_id:
            # Frontend provided explicit session — update the cache
            sid = session_id
            _active_sessions[uid] = sid
        elif uid in _active_sessions:
            # Reuse the existing active session for this user
            sid = _active_sessions[uid]
        else:
            # No session at all — create one and cache it
            sid = str(uuid.uuid4())
            _active_sessions[uid] = sid

        if save:
            _save_detection(uid, sid, result, source="webcam")
        return {**result, "session_id": sid, "user_id": uid}
    except Exception as exc:
        print(f"[EmotionAI] /predict error: {exc}")
        return JSONResponse(status_code=500, content={"error": "server_error", "message": str(exc)})


@app.post("/analyze")
async def analyze_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None),
    current: dict = Depends(get_current_user),
):
    """Legacy alias — kept for backwards compatibility."""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(img)
        loop = asyncio.get_event_loop()
        result, _ = await loop.run_in_executor(executor, lambda: run_pipeline(img_rgb))
        sid = session_id or str(uuid.uuid4())
        _save_detection(current["id"], sid, result, source="webcam")
        return {**result, "session_id": sid, "user_id": current["id"]}
    except Exception as exc:
        print(f"[EmotionAI] analyze error: {exc}")
        return JSONResponse(status_code=500, content={"error": "server_error", "message": str(exc)})


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    current: dict = Depends(get_current_user),
):
    try:
        import cv2, tempfile
        sid = str(uuid.uuid4())
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents); tmp_path = tmp.name

        cap            = cv2.VideoCapture(tmp_path)
        fps            = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int(fps))
        timeline_data: list[dict] = []
        frame_idx      = 0
        loop           = asyncio.get_event_loop()

        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % frame_interval == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    result, _ = await loop.run_in_executor(
                        executor, lambda f=img_rgb: run_pipeline(f, use_mtcnn=False)
                    )
                    t_offset = round(frame_idx / fps, 2)
                    timeline_data.append({
                        "time":       t_offset,
                        "emotion":    result["emotion"],
                        "engagement": result["engagement"],
                    })
                    _save_detection(current["id"], sid, result, source="upload")
                except Exception:
                    pass
            frame_idx += 1

        cap.release(); os.unlink(tmp_path)

        if not timeline_data:
            return JSONResponse(status_code=400, content={"error": "no_data", "message": "No frames analyzed."})

        avg_eng  = round(sum(t["engagement"] for t in timeline_data) / len(timeline_data), 3)
        counts: dict[str, int] = {}
        for t in timeline_data:
            counts[t["emotion"]] = counts.get(t["emotion"], 0) + 1
        dominant = max(counts, key=lambda k: counts[k])

        _save_session_timeline(sid, current["id"], "upload", timeline_data, avg_eng, dominant)
        return {
            "session_id":         sid,
            "timeline":           timeline_data,
            "average_engagement": avg_eng,
            "dominant_emotion":   dominant,
        }

    except Exception as exc:
        print(f"[EmotionAI] analyze-video error: {exc}")
        return JSONResponse(status_code=500, content={"error": "server_error", "message": str(exc)})


@app.get("/session-report/{session_id}")
async def session_report(session_id: str, current: dict = Depends(get_current_user)):
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT * FROM detections WHERE user_id=%s ORDER BY created_at DESC LIMIT 500",
            (current["id"],)
        )
        data = fetchall(cur)
        if not data:
            raise HTTPException(status_code=404, detail="Session not found")
        avg = round(sum(r["engagement"] for r in data) / len(data), 3)
        counts: dict[str, int] = {}
        for r in data:
            counts[r["emotion"]] = counts.get(r["emotion"], 0) + 1
        dominant = max(counts, key=lambda k: counts[k])
        return {
            "session_id":         session_id,
            "frame_count":        len(data),
            "average_engagement": avg,
            "dominant_emotion":   dominant,
            "emotion_counts":     counts,
            "detections":         data,
        }
    finally:
        cur.close(); con.close()


# ── Session start / end ───────────────────────────────────────────────────────

@app.post("/sessions/start/")
async def session_start(current: dict = Depends(get_current_user)):
    """Create a new session UUID and return it so the frontend can tag frames."""
    sid = str(uuid.uuid4())
    return {"session_id": sid}


@app.post("/sessions/end/")
async def session_stop(body: SessionStopRequest, current: dict = Depends(get_current_user)):
    # Clear the in-memory active session so the next start gets a fresh UUID
    _active_sessions.pop(current["id"], None)
    """Compute summary from stored detections and write to session_timeline."""
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT emotion, engagement FROM detections "
            "WHERE user_id=%s ORDER BY created_at DESC LIMIT 500",
            (current["id"],)
        )
        rows = fetchall(cur)
        if not rows:
            return {"message": "No detections found for session", "session_id": body.session_id}

        timeline_data = [
            {"time": r.get("time_offset", 0), "emotion": r["emotion"], "engagement": r["engagement"]}
            for r in rows
        ]
        avg_eng = round(sum(r["engagement"] for r in rows) / len(rows), 3)
        counts: dict = {}
        for r in rows:
            counts[r["emotion"]] = counts.get(r["emotion"], 0) + 1
        dominant = max(counts, key=lambda k: counts[k])
        _save_session_timeline(body.session_id, current["id"], "webcam", timeline_data, avg_eng, dominant)
        return {
            "message":            "Session saved",
            "session_id":         body.session_id,
            "average_engagement": avg_eng,
            "dominant_emotion":   dominant,
        }
    finally:
        cur.close(); con.close()


@app.post("/session-end")
async def session_end(body: SessionEndRequest, current: dict = Depends(get_current_user)):
    _active_sessions.pop(current["id"], None)
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT emotion, engagement FROM detections "
            "WHERE user_id=%s ORDER BY created_at DESC LIMIT 500",
            (current["id"],)
        )
        rows = fetchall(cur)
        timeline_data = [
            {"time": r.get("time_offset", 0), "emotion": r["emotion"], "engagement": r["engagement"]}
            for r in rows
        ]
        _save_session_timeline(
            body.session_id, current["id"], "webcam",
            timeline_data, body.average_engagement, body.dominant_emotion
        )
        return {"message": "Session saved"}
    finally:
        cur.close(); con.close()


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

async def _do_save_feedback(body: FeedbackRequest, request: Request):
    """Shared: resolve user from Bearer token (or guest) and insert feedback row."""
    if not body.username or not body.username.strip():
        raise HTTPException(status_code=400, detail="Username is required.")

    # Resolve user_id from token — non-fatal, falls back to guest
    user_id = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            payload = decode_token(auth_header.split(" ", 1)[1])
            if payload.get("type") == "access":
                uid = int(payload["sub"])
                con_check = db_conn()
                cur_check = con_check.cursor()
                cur_check.execute("SELECT id FROM users WHERE id=%s", (uid,))
                if cur_check.fetchone():
                    user_id = uid
                cur_check.close(); con_check.close()
        except Exception:
            pass  # treat as guest

    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO feedback (user_id, username, email, rating, category, message) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                user_id,
                body.username.strip(),
                body.email or None,
                body.rating if body.rating and 1 <= body.rating <= 5 else None,
                body.category or "General",
                body.message.strip(),
            )
        )
        con.commit()
        return {"message": "Feedback received — thank you!"}
    except Exception as e:
        con.rollback()
        print(f"[EmotionAI] feedback insert error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {str(e)}")
    finally:
        cur.close(); con.close()


# Register BOTH /feedback and /api/feedback so the POST works regardless of
# whether the deployed feedback.html has been updated to use /api/feedback yet.
# FastAPI explicit POST routes ALWAYS take priority over the catch-all
# StaticFiles mount, so /feedback POST is never intercepted by static serving.
@app.post("/feedback")
async def submit_feedback_compat(body: FeedbackRequest, request: Request):
    return await _do_save_feedback(body, request)


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest, request: Request):
    return await _do_save_feedback(body, request)


@app.post("/api/feedback/guest")
@app.post("/feedback/guest")
async def submit_feedback_guest(body: FeedbackRequest, request: Request):
    """Guest alias (no auth required, user_id always NULL)."""
    if not body.username or not body.username.strip():
        raise HTTPException(status_code=400, detail="Username is required.")
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO feedback (user_id, username, email, rating, category, message) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (None, body.username.strip(), body.email, body.rating, body.category, body.message)
        )
        con.commit()
        return {"message": "Feedback received — thank you!"}
    finally:
        cur.close(); con.close()


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN — STATS + FULL CRUD FOR ALL 4 TABLES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/stats")
async def admin_stats(admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM users WHERE is_admin=FALSE")
        total_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM detections")
        total_detections = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM session_timeline")
        total_sessions = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cur.fetchone()[0]
        cur.execute("SELECT AVG(engagement) FROM detections")
        avg_row = cur.fetchone()[0]
        cur.execute("SELECT emotion, COUNT(*) as cnt FROM detections GROUP BY emotion ORDER BY cnt DESC")
        emotion_rows = fetchall(cur)
        return {
            "total_users":      total_users,
            "total_detections": total_detections,
            "total_sessions":   total_sessions,
            "total_feedback":   total_feedback,
            "avg_engagement":   round(avg_row, 3) if avg_row else 0,
            "emotion_counts":   {r["emotion"]: r["cnt"] for r in emotion_rows},
        }
    finally:
        cur.close(); con.close()


# ── Users ─────────────────────────────────────────────────────────────────────

@app.get("/admin/users")
async def admin_list_users(admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT id,username,email,is_admin,created_at,last_login FROM users ORDER BY id")
    rows = fetchall(cur)
    cur.close(); con.close()
    return rows


@app.post("/admin/users")
async def admin_create_user(body: AdminCreateUserRequest, admin: dict = Depends(get_admin_user)):
    """
    FIX: This endpoint was missing entirely. Without it the admin dashboard had
    no API to call, so users were added directly to DB without a bcrypt hash,
    causing all login attempts to fail with 'Incorrect username or password'.
    """
    err = validate_password_strength(body.password)
    if err:
        raise HTTPException(status_code=400, detail=err)
    un_err = validate_username(body.username)
    if un_err:
        raise HTTPException(status_code=400, detail=un_err)
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s OR email=%s",
                    (body.username.strip(), body.email.strip()))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already taken.")
        cur.execute(
            "INSERT INTO users (username, email, password_hash, is_admin) VALUES (%s,%s,%s,%s) RETURNING id",
            (body.username.strip(), body.email.strip(), hash_password(body.password), body.is_admin)
        )
        new_id = cur.fetchone()[0]
        con.commit()
        return {"message": "User created successfully", "user_id": new_id}
    finally:
        cur.close(); con.close()


@app.patch("/admin/users/{user_id}/toggle-admin")
async def toggle_admin(user_id: int, admin: dict = Depends(get_admin_user)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot modify your own admin status")
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT is_admin FROM users WHERE id=%s", (user_id,))
        row = fetchone(cur)
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        new_val = not row["is_admin"]
        cur.execute("UPDATE users SET is_admin=%s WHERE id=%s", (new_val, user_id))
        con.commit()
        return {"user_id": user_id, "is_admin": new_val}
    finally:
        cur.close(); con.close()


@app.patch("/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: int, body: dict, admin: dict = Depends(get_admin_user)):
    """Allow admin to reset any user's password."""
    new_password = body.get("password", "")
    err = validate_password_strength(new_password)
    if err:
        raise HTTPException(status_code=400, detail=err)
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE id=%s", (user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="User not found")
        cur.execute("UPDATE users SET password_hash=%s WHERE id=%s",
                    (hash_password(new_password), user_id))
        con.commit()
        return {"message": f"Password reset for user {user_id}"}
    finally:
        cur.close(); con.close()


@app.delete("/admin/users/{user_id}")
async def deactivate_user(user_id: int, admin: dict = Depends(get_admin_user)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("DELETE FROM users WHERE id=%s AND is_admin=FALSE", (user_id,))
        con.commit()
        return {"message": f"User {user_id} deleted"}
    finally:
        cur.close(); con.close()


# ── Detections ────────────────────────────────────────────────────────────────

@app.get("/admin/detections")
async def admin_list_detections(admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        SELECT d.*, u.username
        FROM detections d
        LEFT JOIN users u ON d.user_id = u.id
        ORDER BY d.created_at DESC
        LIMIT 2000
    """)
    rows = fetchall(cur)
    cur.close(); con.close()
    return rows


@app.delete("/admin/detections/{detection_id}")
async def delete_detection(detection_id: int, admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("DELETE FROM detections WHERE id=%s", (detection_id,))
        con.commit()
        return {"message": "Deleted"}
    finally:
        cur.close(); con.close()


# ── Feedback ──────────────────────────────────────────────────────────────────

@app.get("/admin/feedback")
async def admin_list_feedback(admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        SELECT f.id, f.user_id, f.username, f.email, f.rating,
               f.category, f.message, f.status, f.created_at,
               u.username AS registered_username
        FROM feedback f
        LEFT JOIN users u ON f.user_id = u.id
        ORDER BY f.created_at DESC
    """)
    rows = fetchall(cur)
    cur.close(); con.close()
    return rows


@app.delete("/admin/feedback/{feedback_id}")
async def delete_feedback(feedback_id: int, admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("DELETE FROM feedback WHERE id=%s", (feedback_id,))
        con.commit()
        return {"message": "Deleted"}
    finally:
        cur.close(); con.close()


@app.patch("/admin/feedback/{feedback_id}/status")
async def update_feedback_status(feedback_id: int, body: dict, admin: dict = Depends(get_admin_user)):
    """Allow admin to mark feedback as new / read / resolved."""
    status = body.get("status", "read")
    if status not in ("new", "read", "resolved"):
        raise HTTPException(status_code=400, detail="status must be new, read, or resolved")
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("UPDATE feedback SET status=%s WHERE id=%s RETURNING id", (status, feedback_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Feedback not found")
        con.commit()
        return {"id": feedback_id, "status": status}
    finally:
        cur.close(); con.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/admin/sessions")
async def admin_list_sessions(admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        SELECT st.*, u.username
        FROM session_timeline st
        LEFT JOIN users u ON st.user_id = u.id
        ORDER BY st.started_at DESC
        LIMIT 2000
    """)
    rows = fetchall(cur)
    cur.close(); con.close()
    return rows


@app.delete("/admin/sessions/{session_row_id}")
async def delete_session_row(session_row_id: int, admin: dict = Depends(get_admin_user)):
    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute("DELETE FROM session_timeline WHERE id=%s", (session_row_id,))
        con.commit()
        return {"message": "Deleted"}
    finally:
        cur.close(); con.close()


# ══════════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_detection(user_id: int, session_id: Optional[str], result: dict, source: str = "webcam"):
    try:
        con = db_conn()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO detections "
            "(user_id, emotion, confidence, engagement, source, all_probs) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (user_id, result["emotion"], result["confidence"],
             result["engagement"], source, json.dumps(result.get("all_probabilities", [])))
        )
        con.commit(); cur.close(); con.close()
    except Exception as e:
        print(f"[EmotionAI] _save_detection error: {e}")


def _save_session_timeline(session_id, user_id, source, timeline_data, avg_engagement, dominant_emotion):
    try:
        con = db_conn()
        cur = con.cursor()
        for entry in timeline_data:
            cur.execute(
                """INSERT INTO session_timeline
                   (session_id, user_id, source, time_offset, emotion, engagement,
                    average_engagement, dominant_emotion, frame_count)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (session_id, user_id, source, entry.get("time", 0), entry["emotion"],
                 entry["engagement"], avg_engagement, dominant_emotion, 1)
            )
        con.commit(); cur.close(); con.close()
    except Exception as e:
        print(f"[EmotionAI] _save_session_timeline error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  FRONTEND ROUTES
# ══════════════════════════════════════════════════════════════════════════════

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

print(f"[EmotionAI] Frontend: {FRONTEND_DIR}")


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

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

@app.get("/admin")
@app.get("/admin.html")
async def admin_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("ea_session")
    response.delete_cookie("ea_cookie_consent")
    return response

# Static mount MUST be last — it catches everything not matched above.
# If /predict/ were not defined before this line, POST requests to it would
# hit StaticFiles which only handles GET → 405 Method Not Allowed.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ── ENTRY ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)