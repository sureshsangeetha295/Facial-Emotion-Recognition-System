# Emotion Analysis — AI-Based Facial Emotion Recognition System

Emotion Analysis is a full-stack web application that detects and classifies human facial emotions in real time using a MobileNetV2-based deep learning model. It serves a live webcam detection dashboard, multi-user simultaneous detection, video upload analysis, session tracking, LLM-generated coaching feedback, and an admin panel — all from a single FastAPI server that also hosts the frontend.

---

## Overview

The system captures facial expressions frame by frame via webcam or from an uploaded video, runs each frame through a two-phase trained CNN model, and computes an Exponential Moving Average (EMA) engagement score across the session. After each session the Groq LLM API (LLaMA 3.1 8B Instant) generates a personalized coaching summary and three structured insight cards based on the engagement and emotion data.

Everything lives in a single Python process. The FastAPI backend handles the REST API, serves the static HTML/CSS/JS frontend, and auto-opens the browser on startup.

---

## Features

- Real-time facial emotion detection via webcam at configurable FPS (1–30, default 5)
- **Multi-user simultaneous detection** — `/predict-multi/` endpoint detects and tracks up to 12 faces in a single frame, each with their own emotion, confidence, and engagement card
- Video file upload analysis with per-second EMA timeline breakdown
- MobileNetV2 model trained in two phases; phase 2 model used for inference
- Haar Cascade face detector (default, fast) with MTCNN as a higher-accuracy alternative
- Standalone desktop webcam app (`webcam_main.py`) with animated UI, face tracking, and multi-face ID assignment
- EMA-based engagement scoring with confidence weighting across 7 emotion classes
- **SpikeDetector** — per-session z-score spike detection on engagement values, flagging anomalous engagement surges in real time
- LLM-generated session summary and 3 insight cards via Groq (LLaMA 3.1 8B Instant)
- JWT authentication (access + refresh tokens) with bcrypt password hashing
- **Google OAuth 2.0** — one-click sign-in via Google; existing email accounts are linked automatically
- **OTP-based password reset** — 3-step email OTP flow (send → verify → reset) via Gmail SMTP, replacing the legacy security-question flow
- **reCAPTCHA v2** support on login/register to block bots (optional, activated by env vars)
- Session history, detection records, and timeline stored in PostgreSQL
- Admin dashboard with full CRUD over users, detections, sessions, feedback, and FAQs
- Dynamic FAQ management — admins create/edit/delete FAQ entries; users vote on them
- Responsive multi-page frontend (landing, login, live detection, multi-user detection, video upload, results, FAQ, feedback, admin)
- Modular CSS split into base, layout, components, responsive, and per-page stylesheets
- **Network alert banner** — `network-alert.js` shows a non-blocking banner when the backend is unreachable
- **Spike detector frontend** — `spike-detector.js` subscribes to real-time engagement events and visually highlights emotion spikes on the live detection page

---

## Tech Stack

**Backend**

| Layer | Technology |
|---|---|
| API Framework | FastAPI, Uvicorn |
| Deep Learning | TensorFlow 2.15, Keras 2.15 |
| Face Detection | OpenCV Haar Cascade (primary), MTCNN (primary for desktop app) |
| Image Processing | Pillow, NumPy |
| LLM Integration | Groq API — LLaMA 3.1 8B Instant |
| Authentication | JWT via python-jose, bcrypt via passlib |
| OAuth | Google OAuth 2.0 (via requests + Google Identity endpoints) |
| Email / OTP | Gmail SMTP via smtplib (HTML OTP emails) |
| Database | PostgreSQL via psycopg2 |
| ML Utilities | Scikit-learn, SciPy, Matplotlib |

**Frontend**

| Layer | Technology |
|---|---|
| UI | Vanilla HTML, CSS, JavaScript |
| Fonts | Fraunces (display), Plus Jakarta Sans (body) |
| Architecture | Multi-page static files served by FastAPI |
| CSS Structure | Modular — base, layout, components, responsive, and per-page files |
| JS Structure | Modular — shared utilities, per-page scripts, app sub-modules |

---

## Project Structure

```
project/
│
├── Backend/
│   ├── main.py                  # FastAPI app — all routes, DB, auth, ML pipeline, LLM
│   ├── testing.py               # Model loading, face detection, prediction pipeline (web)
│   ├── face_pipeline.py         # Face detection & emotion prediction for desktop app
│   ├── draw_utils.py            # Rendering helpers (emoji overlays, animated UI elements)
│   ├── screen_states.py         # Animated screen states (waiting, aligning, result flash)
│   ├── webcam_config.py         # Config constants for desktop webcam app (window size,
│   │                            #   MTCNN, EMA alpha, prior boosts, colour map, etc.)
│   ├── webcam_main.py           # Standalone desktop webcam app with multi-face tracking
│   ├── Webcam_test.py           # Local webcam test script
│   ├── init_db.py               # Standalone DB initialisation script
│   ├── classweights.py          # Class imbalance analysis utility
│   ├── preprocessing.py         # Dataset preparation
│   ├── predict.py               # Standalone prediction script
│   ├── rgbvsgray.py             # Grayscale vs RGB input comparison
│   ├── Model_1.py               # Phase 1 model training script
│   ├── Model_2.py               # Phase 2 / fine-tuned model training script
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment config — do not commit, add to .gitignore
│   │
│   ├── routes/                  # Route modules (imported by main.py)
│   │   ├── app_setup.py         # FastAPI app factory, CORS, lifespan
│   │   ├── auth_routes.py       # /auth/* endpoints
│   │   ├── auth_helpers.py      # JWT helpers, password hashing, validation
│   │   ├── detection_routes.py  # /predict/, /analyze, /analyze-video, /sessions/*
│   │   ├── insights_routes.py   # /generate-summary, /generate-insights
│   │   ├── llm_helpers.py       # LLM prompt builders, engagement band logic
│   │   ├── misc_routes.py       # /feedback, /api/faq-feedback, /admin/* routes
│   │   ├── db.py                # DB connection, fetchone, fetchall, init_db
│   │   ├── config.py            # ENV loading, shared constants (SMTP, Google OAuth,
│   │   │                        #   reCAPTCHA keys, emotion labels, EMA config)
│   │   └── otp_email.py         # OTP generator + Gmail SMTP email sender (NEW)
│   │
│   └── Models/
│       ├── phase1_best_model.keras      # Phase 1 trained model
│       ├── phase1_training_log.csv      # Phase 1 training history
│       ├── phase2_best_model.keras      # Phase 2 model used for inference
│       └── phase2_training_log.csv      # Phase 2 training history
│
└── frontend/
    ├── index.html               # Landing page
    ├── login.html               # Login / register / Google OAuth (served at /)
    ├── app.html                 # Detection dashboard (redirects to /livecam)
    ├── livecam.html             # Live webcam detection (single user)
    ├── multiuser.html           # Multi-user simultaneous detection (NEW)
    ├── live-results.html        # Session results and LLM coaching feedback
    ├── detect.html              # Video upload detection
    ├── feedback.html            # User feedback form
    ├── faq.html                 # FAQ page (dynamic, loaded from /api/faqs)
    ├── admin.html               # Admin dashboard
    │
    ├── script/
    │   ├── config.js            # API base URL, FPS limits, emotion labels and colours
    │   ├── auth.js              # Login, register, Google OAuth, token storage and refresh
    │   ├── api.js               # Fetch wrappers for all API endpoints
    │   ├── landing.js           # Landing page interactions
    │   ├── renderer.js          # Shared UI rendering helpers
    │   ├── timeline.js          # Legacy timeline chart helper
    │   ├── network-alert.js     # Non-blocking banner when backend is unreachable (NEW)
    │   │
    │   ├── app/                 # Modular live-detection sub-scripts
    │   │   ├── camera.js        # Webcam capture and frame loop
    │   │   ├── constants.js     # Emotion metadata, engagement map
    │   │   ├── detection.js     # Frame analysis and result processing
    │   │   ├── init.js          # Page bootstrap and session initialisation
    │   │   ├── session.js       # Session start/stop lifecycle
    │   │   ├── ui-avatar.js     # Avatar / emotion indicator rendering
    │   │   └── ui-core.js       # Core UI updates (stats, charts)
    │   │
    │   ├── pages/               # Per-page script entry points
    │   │   ├── admin.js         # Admin dashboard logic
    │   │   ├── analysis.js      # Video analysis results
    │   │   ├── app.js           # Live detection page orchestrator
    │   │   ├── charts.js        # Chart.js wrappers
    │   │   ├── detect.js        # Video upload page
    │   │   ├── faq.js           # Dynamic FAQ loader and vote handler
    │   │   ├── feedback.js      # Feedback form submission
    │   │   ├── index.js         # Landing page
    │   │   ├── live-results.js  # Results page (summary + insights)
    │   │   ├── livecam.js       # Live webcam page entry (single user)
    │   │   ├── login.js         # Login / register / Google OAuth page
    │   │   ├── multiuser.js     # Multi-user detection page (NEW)
    │   │   └── timeline.js      # Session timeline chart
    │   │
    │   └── shared/
    │       ├── drawer.js        # Mobile navigation drawer
    │       └── spike-detector.js  # Real-time engagement spike visualiser (NEW)
    │
    └── style/
        ├── base.css             # CSS reset and design tokens
        ├── components.css       # Reusable component styles
        ├── layout.css           # Page layout grid and nav
        ├── responsive.css       # Breakpoints and media queries
        └── pages/               # Per-page overrides
            ├── admin.css
            ├── app.css
            ├── detect.css
            ├── faq.css
            ├── feedback.css
            ├── index.css
            ├── live-results.css
            ├── livecam.css
            ├── login.css
            └── multiuser.css    # Multi-user detection page styles (NEW)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14 or higher
- A [Groq API key](https://console.groq.com/) for LLM-powered summaries and insights
- A Gmail account with an [App Password](https://myaccount.google.com/apppasswords) for OTP email delivery
- *(Optional)* A Google Cloud project with OAuth 2.0 credentials for Google sign-in
- *(Optional)* A Google reCAPTCHA v2 site/secret key pair

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sureshsangeetha295/Facial-Emotion-Recognition-System.git
cd Facial-Emotion-Recognition-System

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 3. Install dependencies
cd Backend
pip install -r requirements.txt

# 4. Configure environment variables
# Create a .env file in Backend/ using the Environment Variables table below as reference

# 5. Update the model path in testing.py (web server) and webcam_config.py (desktop app)
# Change MODEL_PATH to a relative or absolute path on your machine:
#   MODEL_PATH = os.path.join(os.path.dirname(__file__), "Models", "phase2_best_model.keras")

# 6. Run the web application
python main.py

# Optional: run the standalone desktop webcam app
python webcam_main.py
```

The web app starts at `http://localhost:8000`, opens automatically in your browser, and initialises all database tables on first run. The default landing route (`/`) serves `login.html`.

---

## Environment Variables

Create a `.env` file in the `Backend/` directory. On startup the application prints each key it finds and masks values for debugging.

| Key | Description |
|---|---|
| `APP_HOST` | Host to bind the server to |
| `APP_PORT` | Port to run on (default 8000) |
| `DB_HOST` | PostgreSQL host |
| `DB_PORT` | PostgreSQL port |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `SECRET_KEY` | Long random string used to sign JWT tokens |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime in minutes |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime in days |
| `ADMIN_USERNAME` | Admin account username (auto-created on first run) |
| `ADMIN_PASSWORD` | Admin account password |
| `ADMIN_EMAIL` | Admin account email |
| `GROQ_API_KEY` | Groq API key for LLM features |
| `SMTP_EMAIL` | Gmail address used to send OTP emails |
| `SMTP_APP_PASSWORD` | Gmail App Password (not your account password) |
| `GOOGLE_CLIENT_ID` | Google OAuth 2.0 client ID (optional) |
| `GOOGLE_CLIENT_SECRET` | Google OAuth 2.0 client secret (optional) |
| `GOOGLE_REDIRECT_URI` | OAuth callback URL (default: `http://localhost:8000/auth/google/callback`) |
| `RECAPTCHA_SITE_KEY` | Google reCAPTCHA v2 site key (optional) |
| `RECAPTCHA_SECRET_KEY` | Google reCAPTCHA v2 secret key (optional) |

Never commit the `.env` file. Add it to `.gitignore` before your first push.

---

## API Endpoints

**Authentication**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create a new user account |
| POST | `/auth/login` | Login and receive JWT tokens |
| POST | `/auth/admin/login` | Admin-only login |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/auth/me` | Get current user profile |
| GET | `/auth/google/login` | Initiate Google OAuth 2.0 sign-in flow |
| GET | `/auth/google/callback` | Google OAuth callback — issues JWT on success |
| POST | `/auth/otp/send` | Step 1 — send a 6-digit OTP to the user's email |
| POST | `/auth/otp/verify` | Step 2 — verify the OTP is correct and not expired |
| POST | `/auth/otp/reset-password` | Step 3 — verify OTP again and set a new password |
| POST | `/auth/reset/verify-email` | Legacy Step 1 — verify email and return security questions |
| POST | `/auth/reset/verify-answers` | Legacy Step 2 — verify security question answers |
| POST | `/auth/reset/password` | Legacy Step 3 — set new password |

**Detection** — requires JWT

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict/` | Predict emotion from a single image frame (webcam, single user) |
| POST | `/predict-multi/` | Detect and classify emotions for all faces in a frame (multi-user) |
| POST | `/analyze` | Analyze a frame and save to active session |
| POST | `/analyze-video` | Upload and analyze a video file with EMA timeline |
| GET | `/session-report/{session_id}` | Get full session detection report |
| POST | `/sessions/start/` | Start a new session (returns UUID) |
| POST | `/sessions/end/` | End session and persist timeline |
| POST | `/session-end` | Alternative session end with dominant emotion override |

**LLM** — requires JWT

| Method | Endpoint | Description |
|---|---|---|
| POST | `/generate-summary` | Generate a chat-style coaching summary via Groq |
| POST | `/generate-insights` | Generate 3 structured insight cards via Groq |

**FAQ (Public)**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/faqs` | List all FAQ entries (used by faq.html) |
| POST | `/api/faq-feedback` | Submit a liked/disliked vote on a FAQ question |

**Feedback**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/feedback` | Submit user feedback (authenticated or guest) |
| POST | `/api/feedback` | Submit user feedback (authenticated or guest) |
| POST | `/api/feedback/guest` | Submit feedback without authentication |

**Admin** — requires admin JWT

| Method | Endpoint | Description |
|---|---|---|
| GET | `/admin/stats` | Platform-wide usage statistics |
| GET/POST | `/admin/users` | List or create users |
| PATCH | `/admin/users/{id}/toggle-admin` | Toggle admin status |
| PATCH | `/admin/users/{id}/reset-password` | Force reset a user password |
| DELETE | `/admin/users/{id}` | Delete a non-admin user |
| GET/DELETE | `/admin/detections` | List or delete detection records |
| GET/DELETE | `/admin/sessions` | List or delete session timeline rows |
| GET/DELETE | `/admin/feedback` | List or delete user feedback |
| GET/DELETE | `/admin/faq-feedback` | List or delete FAQ votes |
| GET/POST | `/admin/faqs` | List all FAQs or create a new FAQ entry |
| PUT | `/admin/faqs/{id}` | Update an existing FAQ entry |
| DELETE | `/admin/faqs/{id}` | Delete a FAQ entry |

---

## Multi-User Detection

The `/predict-multi/` endpoint processes a single frame and returns results for every face detected in it. It is designed for classroom, group, or panel scenarios where multiple participants are visible simultaneously.

**Request** — `POST /predict-multi/` (multipart/form-data)

| Parameter | Type | Description |
|---|---|---|
| `file` | blob | JPEG/PNG frame from the webcam |
| `fast` | bool (query) | Use Haar Cascade instead of MTCNN (default: true) |
| `save` | bool (query) | Persist detections to the DB (default: false) |
| `session_id` | str (query) | Active session UUID to associate detections with |

**Response**

```json
{
  "session_id": "uuid",
  "user_id": 42,
  "face_count": 3,
  "faces": [
    {
      "face_index": 0,
      "bbox": { "x": 120, "y": 80, "w": 90, "h": 90 },
      "emotion": "Happiness",
      "confidence": 0.92,
      "all_probabilities": [0.01, 0.01, 0.01, 0.92, 0.03, 0.01, 0.01],
      "engagement": 0.782,
      "timestamp": "2025-01-15T10:30:00.000Z"
    }
  ]
}
```

The frontend (`multiuser.html` + `pages/multiuser.js`) renders one card per detected face, capped at 12 simultaneous faces. The page polls at 400 ms intervals (slightly relaxed vs the single-user 200 ms cadence).

---

## Engagement Spike Detection

A `SpikeDetector` class (defined in `main.py`) runs per active session. It uses a rolling window z-score calculation: if an incoming frame's engagement value deviates more than 1.8 standard deviations from the window mean, it is flagged as a spike.

- Window size: 10 frames
- Z-score threshold: 1.8
- Spikes are returned inline with `/predict/` responses and surfaced visually via `spike-detector.js` on the frontend

---

## OTP Password Reset

The legacy security-question reset flow is still available but the recommended flow is now OTP-based:

1. **Send OTP** — `POST /auth/otp/send` with `{ "email": "..." }`. A 6-digit numeric code is stored (hashed) against the user row with a 10-minute expiry, and an HTML-formatted email is dispatched via Gmail SMTP.
2. **Verify OTP** — `POST /auth/otp/verify` with `{ "email": "...", "otp": "..." }`. Returns 200 if the code matches and has not expired.
3. **Reset password** — `POST /auth/otp/reset-password` with `{ "email": "...", "otp": "...", "new_password": "..." }`. Re-verifies the OTP, updates the password hash, and clears the OTP columns.

Requires `SMTP_EMAIL` and `SMTP_APP_PASSWORD` in `.env`.

---

## Google OAuth 2.0

When `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set, a **Sign in with Google** button appears on `login.html`.

1. The browser navigates to `GET /auth/google/login`, which redirects to Google's consent screen.
2. Google redirects back to `GET /auth/google/callback` with an authorization code.
3. The backend exchanges the code for an ID token, extracts the email and Google subject ID, and either links the Google account to an existing user (matching by email) or creates a new one.
4. A standard JWT access + refresh token pair is issued and the user is redirected to the app.

The `users` table stores a nullable `google_id` column for this linkage.

---

## Emotion Classes and Engagement Scoring

The model classifies faces into 7 emotion categories. Each emotion maps to a base engagement weight used in the EMA session score calculation:

| Emotion | Engagement Weight | Tone |
|---|---|---|
| Happiness | 0.85 | Positive |
| Surprise | 0.70 | Positive |
| Neutral | 0.55 | Neutral |
| Fear | 0.35 | Negative |
| Sadness | 0.20 | Negative |
| Anger | 0.15 | Negative |
| Disgust | 0.05 | Negative |

Session engagement is computed using an EMA with alpha = 0.20, meaning recent frames carry more weight than earlier ones. A learner who finishes strongly will score higher than one who was engaged only at the start. The final EMA score is bucketed into five bands: Very Low, Low, Moderate, Good, and Excellent.

A confidence-weighted average is also tracked in parallel and passed to the LLM prompt alongside tone breakdown (% positive / neutral / negative frames) for richer coaching context.

---

## Model

The inference pipeline uses a MobileNetV2-based CNN trained in two phases. `testing.py` handles model loading and prediction for the web server; `face_pipeline.py` handles the same for the standalone desktop app.

- Input size: 224 × 224 RGB
- Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input`
- Face detection (web): Haar Cascade (default, ~5–15 ms per frame); MTCNN available via `use_mtcnn=True`
- Face detection (desktop): MTCNN primary, with temperature scaling and per-class prior boosts configured in `webcam_config.py`
- If no face is detected in a frame, the full frame is used as fallback
- Models are loaded once as lazy singletons with thread locks; a dummy forward pass warms up the model on startup
- Inference runs in a `ThreadPoolExecutor` (1 worker) to avoid blocking the async event loop

The model files are included in the `Backend/Models/` directory. Before running, update `MODEL_PATH` in `testing.py` (web) and `webcam_config.py` (desktop) to point to the correct path on your system.

---

## Standalone Desktop Webcam App

In addition to the web interface, the project ships a self-contained desktop application (`webcam_main.py`) that runs directly from the command line without a browser or server.

Key capabilities beyond the web version:

- MTCNN-based face detection with multi-face tracking and persistent face IDs
- Animated UI built with OpenCV — hex grid background, glow effects, scan-line animation
- Distinct screen states for waiting, face alignment, detection flash, and result display
- Temperature scaling and per-class prior boosts for calibrated confidence outputs
- Configurable via `webcam_config.py` (window layout, EMA alpha, smoothing frames, colour map)

To run:

```bash
cd Backend
python webcam_main.py
# Press ENTER to scan, Q to quit
```

---

## Database Schema

Seven tables are auto-created on startup:

- `users` — accounts, hashed passwords, security questions, admin flag, `google_id` (for OAuth linking), `otp_code` and `otp_expires` (for OTP password reset)
- `detections` — per-frame emotion, confidence, engagement, and source (`webcam` / `upload` / `multiuser`)
- `session_timeline` — aggregated session rows with EMA engagement and dominant emotion
- `feedback` — user-submitted ratings and messages (includes `username`, `email`, `rating`, `category`)
- `password_reset_log` — 3-step reset audit trail with IP address
- `faq_feedback` — per-question liked/disliked votes with optional complaint text
- `faqs` — admin-managed FAQ entries with category, question, and answer fields

---

## Author

**Suresh Sangeetha**  
GitHub: [sureshsangeetha295/Facial-Emotion-Recognition-System](https://github.com/sureshsangeetha295/Facial-Emotion-Recognition-System)