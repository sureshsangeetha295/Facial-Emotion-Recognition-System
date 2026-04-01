import os
os.environ["TF_USE_LEGACY_KERAS"]  = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io
import json
import asyncio
from typing import cast
import webbrowser
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from PIL import Image
from pydantic import BaseModel, ConfigDict          # ← FIX: import ConfigDict
from sqlalchemy import (
    create_engine, Integer, String, Float,
    DateTime, Text, ForeignKey, Boolean, text       # ← FIX: import text for health check
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, Session,
    relationship, Mapped, mapped_column
)
from dotenv import load_dotenv
import hashlib
import base64
from passlib.context import CryptContext
from jose import JWTError, jwt


# LOAD .env
load_dotenv()

DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

SECRET_KEY                  = cast(str, os.getenv("SECRET_KEY"))
ALGORITHM                   = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS   = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",   "7"))

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

_missing = [v for v in ("DB_NAME", "DB_USER", "DB_PASSWORD", "SECRET_KEY") if not os.getenv(v)]
if _missing:
    raise RuntimeError(
        f"\n\n  Missing required .env variables: {', '.join(_missing)}\n"
        "  Copy .env.example → .env and fill in the values.\n"
    )


# PASSWORD HASHING & JWT
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def _prehash(plain: str) -> str:
    """SHA-256 → base64 so bcrypt never sees more than 72 bytes."""
    digest = hashlib.sha256(plain.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("utf-8")

def hash_password(plain: str) -> str:
    return pwd_context.hash(_prehash(plain))

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(_prehash(plain), hashed)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"]  = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload["type"] = "access"
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> tuple[str, datetime]:
    expires = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = data.copy()
    payload["exp"]  = expires
    payload["type"] = "refresh"
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM), expires

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# DATABASE
engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


# Models 

class User(Base):
    __tablename__ = "users"

    id              : Mapped[int]      = mapped_column(Integer,     primary_key=True, index=True)
    username        : Mapped[str]      = mapped_column(String(100), unique=True, nullable=False, index=True)
    email           : Mapped[str]      = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password : Mapped[str]      = mapped_column(String(255), nullable=False)
    is_active       : Mapped[bool]     = mapped_column(Boolean,     default=True)
    is_admin        : Mapped[bool]     = mapped_column(Boolean,     default=False)
    created_at      : Mapped[datetime] = mapped_column(DateTime,    default=datetime.utcnow)

    detections     = relationship("Detection",    back_populates="user")
    sessions       = relationship("SessionLog",   back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")
    feedbacks      = relationship("Feedback",     back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         : Mapped[int]      = mapped_column(Integer,     primary_key=True, index=True)
    user_id    : Mapped[int]      = mapped_column(Integer,     ForeignKey("users.id"), nullable=False)
    token      : Mapped[str]      = mapped_column(String(512), unique=True, nullable=False, index=True)
    expires_at : Mapped[datetime] = mapped_column(DateTime,    nullable=False)
    revoked    : Mapped[bool]     = mapped_column(Boolean,     default=False)
    created_at : Mapped[datetime] = mapped_column(DateTime,    default=datetime.utcnow)

    user = relationship("User", back_populates="refresh_tokens")


class Detection(Base):
    __tablename__ = "detections"

    id          : Mapped[int]        = mapped_column(Integer,    primary_key=True, index=True)
    user_id     : Mapped[int | None] = mapped_column(Integer,    ForeignKey("users.id"), nullable=True)
    emotion     : Mapped[str]        = mapped_column(String(50), nullable=False)
    confidence  : Mapped[float]      = mapped_column(Float,      nullable=False)
    all_scores  : Mapped[str | None] = mapped_column(Text,       nullable=True)
    detected_at : Mapped[datetime]   = mapped_column(DateTime,   default=datetime.utcnow)

    user = relationship("User", back_populates="detections")


class SessionLog(Base):
    __tablename__ = "sessions"

    id           : Mapped[int]             = mapped_column(Integer,  primary_key=True, index=True)
    user_id      : Mapped[int | None]      = mapped_column(Integer,  ForeignKey("users.id"), nullable=True)
    started_at   : Mapped[datetime]        = mapped_column(DateTime, default=datetime.utcnow)
    ended_at     : Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_frames : Mapped[int]             = mapped_column(Integer,  default=0)

    user = relationship("User", back_populates="sessions")


class Feedback(Base):
    __tablename__ = "feedback"

    id         : Mapped[int]             = mapped_column(Integer,     primary_key=True, index=True)
    user_id    : Mapped[int | None]      = mapped_column(Integer,     ForeignKey("users.id"), nullable=True)
    name       : Mapped[str | None]      = mapped_column(String(100), nullable=True)
    email      : Mapped[str | None]      = mapped_column(String(255), nullable=True)
    rating     : Mapped[int | None]      = mapped_column(Integer,     nullable=True)   # 1–5
    message    : Mapped[str]             = mapped_column(Text,        nullable=False)
    created_at : Mapped[datetime]        = mapped_column(DateTime,    default=datetime.utcnow)

    user = relationship("User", back_populates="feedbacks")


# PYDANTIC SCHEMAS

class UserRegister(BaseModel):
    username: str
    email:    str
    password: str

class UserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

class SessionEnd(BaseModel):
    session_id:   int
    total_frames: int = 0

class FeedbackIn(BaseModel):
    name:    str | None = None
    email:   str | None = None
    rating:  int | None = None          # 1–5
    message: str

class FeedbackOut(BaseModel):
    id:         int
    user_id:    int | None
    name:       str | None
    email:      str | None
    rating:     int | None
    message:    str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AdminUserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    is_active:  bool
    is_admin:   bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# FASTAPI DEPENDENCIES

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    token: str     = Depends(oauth2_scheme),
    db:    Session = Depends(get_db),
) -> User:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Wrong token type")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user

oauth2_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

def get_optional_user(
    token: str | None = Depends(oauth2_optional),
    db:    Session    = Depends(get_db),
) -> User | None:
    if not token:
        return None
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        user_id = payload.get("sub")
        if not user_id:
            return None
        return db.query(User).filter(
            User.id == int(user_id), User.is_active == True
        ).first()
    except HTTPException:
        return None


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# APP SETUP

EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

executor = ThreadPoolExecutor(max_workers=1)


#  DB connectivity check with clear error message 
def _check_db_connection(retries: int = 3, delay: float = 2.0) -> None:
    """
    Try to reach PostgreSQL before creating tables.
    Raises RuntimeError with an actionable message if the server is unreachable.
    """
    import time
    from sqlalchemy import text as sa_text

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(sa_text("SELECT 1"))
            return  # success
        except Exception as exc:
            last_exc = exc
            print(
                f"[EmotionAI] DB connection attempt {attempt}/{retries} failed. "
                f"Retrying in {delay}s…"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Database connection failed. Check PostgreSQL and .env settings.\n"
        f"Error: {last_exc}"
    )


def _auto_migrate() -> None:
    """
    Safely apply schema changes to tables that already exist in the DB.
    Uses IF NOT EXISTS / conditional logic so re-runs are always safe.
    """
    from sqlalchemy import text as sa_text

    migrations = [
        # Add is_admin to users (was missing in older deployments)
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='users' AND column_name='is_admin'
            ) THEN
                ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT FALSE;
            END IF;
        END $$;
        """,
        # Drop image_path from detections if it still exists
        """
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='detections' AND column_name='image_path'
            ) THEN
                ALTER TABLE detections DROP COLUMN image_path;
            END IF;
        END $$;
        """,
    ]

    with engine.begin() as conn:
        for stmt in migrations:
            conn.execute(sa_text(stmt))

    print("[EmotionAI] Auto-migration complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[EmotionAI] DB → {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"[EmotionAI] JWT algorithm={ALGORITHM}  "
          f"access_ttl={ACCESS_TOKEN_EXPIRE_MINUTES}m")

    #  check DB is reachable BEFORE calling create_all 
    _check_db_connection()

    # Auto-migrate existing tables for columns added after initial creation.
    # These are safe no-ops if the columns already exist (IF NOT EXISTS).
    _auto_migrate()

    Base.metadata.create_all(bind=engine)
    print("[EmotionAI] Database tables ready")

    try:
        import testing
        testing.get_haar_detector()
        testing.get_model()
        print("[EmotionAI] Preload complete")
    except Exception as e:
        print(f"[EmotionAI] Preload error (non-fatal): {e}")

    # Open browser only after server is fully ready
    webbrowser.open(f"http://localhost:{APP_PORT}")

    yield   # Server runs here


app = FastAPI(title="EmotionAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# AUTH ROUTES

@app.post("/auth/register", response_model=UserOut, summary="Register a new user")
def register(payload: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(
        (User.username == payload.username) | (User.email == payload.email)
    ).first():
        raise HTTPException(status_code=400, detail="Username or email already taken")

    user = User(
        username        = payload.username,
        email           = payload.email,
        hashed_password = hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=TokenResponse, summary="Login")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   Session                   = Depends(get_db),
):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, str(user.hashed_password)):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    token_data             = {"sub": str(user.id)}
    access_token           = create_access_token(token_data)
    refresh_token_str, exp = create_refresh_token(token_data)

    db.add(RefreshToken(user_id=user.id, token=refresh_token_str, expires_at=exp))
    db.commit()

    return TokenResponse(access_token=access_token, refresh_token=refresh_token_str)


@app.post("/auth/refresh", response_model=TokenResponse, summary="Refresh token pair")
def refresh_tokens(payload: RefreshRequest, db: Session = Depends(get_db)):
    data = decode_token(payload.refresh_token)
    if data.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    stored = db.query(RefreshToken).filter(
        RefreshToken.token   == payload.refresh_token,
        RefreshToken.revoked == False,
    ).first()
    if not stored or stored.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token expired or already used")

    user = db.query(User).filter(User.id == int(data["sub"]), User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    stored.revoked         = True
    token_data             = {"sub": str(user.id)}
    new_access             = create_access_token(token_data)
    new_refresh_str, exp   = create_refresh_token(token_data)
    db.add(RefreshToken(user_id=user.id, token=new_refresh_str, expires_at=exp))
    db.commit()

    return TokenResponse(access_token=new_access, refresh_token=new_refresh_str)


@app.post("/auth/logout", summary="Logout")
def logout_api(payload: RefreshRequest, db: Session = Depends(get_db)):
    stored = db.query(RefreshToken).filter(RefreshToken.token == payload.refresh_token).first()
    if stored:
        stored.revoked = True
        db.commit()
    return {"message": "Logged out successfully"}


@app.get("/auth/me", response_model=UserOut, summary="Get current user")
def me(current_user: User = Depends(get_current_user)):
    return current_user


# ML PIPELINE

def run_pipeline(img_rgb: np.ndarray, use_mtcnn: bool = False):
    import testing
    idx, confidence, probs = testing.predict_emotion_from_image(
        img_rgb, use_mtcnn=use_mtcnn
    )
    conf = float(confidence)
    if conf > 1.0:
        conf /= 100.0

    norm_probs = [
        round(float(p) / 100.0 if float(p) > 1.0 else float(p), 4)
        for p in probs
    ]
    return {
        "emotion":           EMOTION_LABELS[idx],
        "confidence":        round(conf, 4),
        "all_probabilities": norm_probs,
    }, None


# HEALTH + PREDICT

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict/", summary="Detect emotion from image")
async def predict(
    file:         UploadFile  = File(...),
    fast:         bool        = Query(default=True),
    save:         bool        = Query(default=True),
    db:           Session     = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    try:
        contents = await file.read()
        img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
        img_pil  = img_pil.resize((320, 240), Image.Resampling.LANCZOS)
        img      = np.array(img_pil)

        loop = asyncio.get_event_loop()
        result, err = await loop.run_in_executor(
            executor,
            lambda: run_pipeline(img, use_mtcnn=False),
        )

        if err:
            return JSONResponse(status_code=400, content={
                "error":   "no_face",
                "message": "No face detected. Please face the camera directly.",
            })

        if save:
            record = Detection(
                user_id    = current_user.id if current_user else None,
                emotion    = result["emotion"],
                confidence = result["confidence"],
                all_scores = json.dumps(
                    {EMOTION_LABELS[i]: p for i, p in enumerate(result["all_probabilities"])}
                ),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            result["detection_id"] = record.id

        return result

    except Exception as exc:
        print(f"[EmotionAI] Predict error: {exc}")
        return JSONResponse(status_code=500, content={
            "error": "server_error", "message": str(exc),
        })


# HISTORY & STATS

@app.get("/history/", summary="Detection history")
def get_history(
    limit:        int     = Query(default=20, le=200),
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    records = (
        db.query(Detection)
        .filter(Detection.user_id == current_user.id)
        .order_by(Detection.detected_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":          r.id,
            "emotion":     r.emotion,
            "confidence":  r.confidence,
            "all_scores":  json.loads(str(r.all_scores)) if r.all_scores else {},
            "detected_at": r.detected_at,
        }
        for r in records
    ]


@app.get("/history/stats/", summary="Emotion distribution stats")
def get_stats(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    records = db.query(Detection).filter(Detection.user_id == current_user.id).all()
    if not records:
        return {"total": 0, "distribution": {}}

    counts: dict[str, int] = {}
    for r in records:
        counts[str(r.emotion)] = counts.get(str(r.emotion), 0) + 1

    total = len(records)
    return {
        "total": total,
        "distribution": {
            k: {"count": v, "percent": round(v / total * 100, 1)}
            for k, v in counts.items()
        },
    }


# SESSION ROUTES

@app.post("/sessions/start/", summary="Start a detection session")
def start_session(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    session = SessionLog(user_id=current_user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": session.id, "started_at": session.started_at}


@app.post("/sessions/end/", summary="End a detection session")
def end_session(
    payload:      SessionEnd,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    session = db.query(SessionLog).filter(
        SessionLog.id      == payload.session_id,
        SessionLog.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.ended_at     = datetime.utcnow()
    session.total_frames = payload.total_frames
    db.commit()
    db.refresh(session)
    return {
        "session_id":   session.id,
        "started_at":   session.started_at,
        "ended_at":     session.ended_at,
        "total_frames": session.total_frames,
    }

# FEEDBACK

@app.post("/feedback/", response_model=FeedbackOut, summary="Submit feedback")
def submit_feedback(
    payload:      FeedbackIn,
    db:           Session     = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    fb = Feedback(
        user_id = current_user.id if current_user else None,
        name    = payload.name  or (current_user.username if current_user else None),
        email   = payload.email or (current_user.email    if current_user else None),
        rating  = payload.rating,
        message = payload.message,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


# ADMIN ROUTES

@app.get("/admin/users", response_model=list[AdminUserOut], summary="[Admin] List all users")
def admin_list_users(
    db:    Session = Depends(get_db),
    admin: User    = Depends(get_current_admin),
):
    return db.query(User).order_by(User.id).all()


@app.patch("/admin/users/{user_id}/toggle", response_model=AdminUserOut, summary="[Admin] Toggle user active state")
def admin_toggle_user(
    user_id: int,
    db:      Session = Depends(get_db),
    admin:   User    = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    user.is_active = not user.is_active
    db.commit()
    db.refresh(user)
    return user


@app.get("/admin/detections", summary="[Admin] List all detections")
def admin_list_detections(
    limit: int     = Query(default=100, le=1000),
    db:    Session = Depends(get_db),
    admin: User    = Depends(get_current_admin),
):
    records = (
        db.query(Detection)
        .order_by(Detection.detected_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":          r.id,
            "user_id":     r.user_id,
            "emotion":     r.emotion,
            "confidence":  r.confidence,
            "all_scores":  json.loads(str(r.all_scores)) if r.all_scores else {},
            "detected_at": r.detected_at,
        }
        for r in records
    ]


@app.get("/admin/feedback", response_model=list[FeedbackOut], summary="[Admin] List all feedback")
def admin_list_feedback(
    db:    Session = Depends(get_db),
    admin: User    = Depends(get_current_admin),
):
    return db.query(Feedback).order_by(Feedback.created_at.desc()).all()


@app.delete("/admin/feedback/{feedback_id}", summary="[Admin] Delete a feedback entry")
def admin_delete_feedback(
    feedback_id: int,
    db:          Session = Depends(get_db),
    admin:       User    = Depends(get_current_admin),
):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    db.delete(fb)
    db.commit()
    return {"message": f"Feedback {feedback_id} deleted"}


# FRONTEND ROUTES

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

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("ea_session")
    response.delete_cookie("ea_cookie_consent")
    return response


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ENTRY POINT

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)