# EmotionAI — AI-Based Facial Emotion Recognition System

EmotionAI is a full-stack web application that detects and classifies human facial emotions in real time using a MobileNetV2-based deep learning model. It serves a live webcam detection dashboard, video upload analysis, session tracking, LLM-generated coaching feedback, and an admin panel — all from a single FastAPI server that also hosts the frontend.

---

## Overview

The system captures facial expressions frame by frame via webcam or from an uploaded video, runs each frame through a two-phase trained CNN model, and computes an Exponential Moving Average (EMA) engagement score across the session. After each session the Groq LLM API (LLaMA 3.1 8B Instant) generates a personalized coaching summary and three structured insight cards based on the engagement and emotion data.

Everything lives in a single Python process. The FastAPI backend handles the REST API, serves the static HTML/CSS/JS frontend, and auto-opens the browser on startup.

---

## Features

- Real-time facial emotion detection via webcam at configurable FPS (1–30, default 5)
- Video file upload analysis with per-second EMA timeline breakdown
- MobileNetV2 model trained in two phases; phase 2 model used for inference
- Haar Cascade face detector (default, fast) with MTCNN as a higher-accuracy alternative
- Standalone desktop webcam app (`webcam_main.py`) with animated UI, face tracking, and multi-face ID assignment
- EMA-based engagement scoring with confidence weighting across 7 emotion classes
- LLM-generated session summary and 3 insight cards via Groq (LLaMA 3.1 8B Instant)
- JWT authentication (access + refresh tokens) with bcrypt password hashing
- Security question-based self-service password reset (3-step flow)
- Session history, detection records, and timeline stored in PostgreSQL
- Admin dashboard with full CRUD over users, detections, sessions, feedback, and FAQs
- Dynamic FAQ management — admins create/edit/delete FAQ entries; users vote on them
- Responsive multi-page frontend (landing, login, live detection, video upload, results, FAQ, feedback, admin)
- Modular CSS split into base, layout, components, responsive, and per-page stylesheets

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
│   │   └── config.py            # ENV loading, shared constants (re-exports main config)
│   │
│   └── Models/
│       ├── phase1_best_model.keras      # Phase 1 trained model
│       ├── phase1_training_log.csv      # Phase 1 training history
│       ├── phase2_best_model.keras      # Phase 2 model used for inference
│       └── phase2_training_log.csv      # Phase 2 training history
│
└── frontend/
    ├── index.html               # Landing page
    ├── login.html               # Login / register (served at /)
    ├── app.html                 # Detection dashboard (redirects to /livecam)
    ├── livecam.html             # Live webcam detection
    ├── live-results.html        # Session results and LLM coaching feedback
    ├── detect.html              # Video upload detection
    ├── feedback.html            # User feedback form
    ├── faq.html                 # FAQ page (dynamic, loaded from /api/faqs)
    ├── admin.html               # Admin dashboard
    │
    ├── script/
    │   ├── config.js            # API base URL, FPS limits, emotion labels and colours
    │   ├── auth.js              # Login, register, token storage and refresh
    │   ├── api.js               # Fetch wrappers for all API endpoints
    │   ├── landing.js           # Landing page interactions
    │   ├── renderer.js          # Shared UI rendering helpers
    │   ├── timeline.js          # Legacy timeline chart helper
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
    │   │   ├── livecam.js       # Live webcam page entry
    │   │   ├── login.js         # Login / register page
    │   │   └── timeline.js      # Session timeline chart
    │   │
    │   └── shared/
    │       └── drawer.js        # Mobile navigation drawer
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
            └── login.css
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14 or higher
- A [Groq API key](https://console.groq.com/) for LLM-powered summaries and insights

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/emotionai.git
cd emotionai

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

Create a `.env` file in the `Backend/` directory. On startup the application prints each key it finds and masks values for debugging. Common causes of misconfiguration are printed as warnings.

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
| POST | `/auth/reset/verify-email` | Step 1 — verify email and return security questions |
| POST | `/auth/reset/verify-answers` | Step 2 — verify security question answers |
| POST | `/auth/reset/password` | Step 3 — set new password |

**Detection** — requires JWT

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict/` | Predict emotion from a single image frame (webcam) |
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

Six tables are auto-created on startup:

- `users` — accounts, hashed passwords, security questions, admin flag
- `detections` — per-frame emotion, confidence, engagement, and source (webcam / upload)
- `session_timeline` — aggregated session rows with EMA engagement and dominant emotion
- `feedback` — user-submitted ratings and messages
- `password_reset_log` — 3-step reset audit trail with IP address
- `faq_feedback` — per-question liked/disliked votes with optional complaint text
- `faqs` — admin-managed FAQ entries with category, question, and answer fields

---

## Author

**Your Name**  
GitHub: [Suresh Sangeetha](https://github.com/sureshsangeetha295/Facial-Emotion-Recognition-System.git)