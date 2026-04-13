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
- EMA-based engagement scoring with confidence weighting across 7 emotion classes
- LLM-generated session summary and 3 insight cards via Groq (LLaMA 3.1 8B Instant)
- JWT authentication (access + refresh tokens) with bcrypt password hashing
- Security question-based self-service password reset (3-step flow)
- Session history, detection records, and timeline stored in PostgreSQL
- Admin dashboard with full CRUD over users, detections, sessions, and feedback
- Responsive multi-page frontend (landing, login, live detection, video upload, results, FAQ, feedback, admin)

---

## Tech Stack

**Backend**

| Layer | Technology |
|---|---|
| API Framework | FastAPI, Uvicorn |
| Deep Learning | TensorFlow 2.15, Keras 2.15 |
| Face Detection | OpenCV Haar Cascade (primary), MTCNN (optional) |
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

---

## Project Structure

```
project/
│
├── Backend/
│   ├── main.py                  # FastAPI app — all routes, DB, auth, ML pipeline, LLM
│   ├── testing.py               # Model loading, face detection, prediction pipeline
│   ├── init_db.py               # Standalone DB initialisation script
│   ├── classweights.py          # Class imbalance analysis utility
│   ├── preprocessing.py         # Dataset preparation
│   ├── predict.py               # Standalone prediction script
│   ├── rgbvsgray.py             # Grayscale vs RGB input comparison
│   ├── Webcam_test.py           # Local webcam test script
│   ├── Model_1.py               # Phase 1 model training script
│   ├── Model_2.py               # Phase 2 / fine-tuned model training script
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment config — do not commit, add to .gitignore
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
    ├── faq.html                 # FAQ page
    ├── admin.html               # Admin dashboard
    │
    ├── script/
    │   ├── config.js            # API base URL, FPS limits, emotion labels and colors
    │   ├── app.js               # Live detection logic, emotion metadata, engagement map
    │   ├── auth.js              # Login, register, token storage and refresh
    │   ├── api.js               # Fetch wrappers for all API endpoints
    │   ├── landing.js           # Landing page interactions
    │   ├── renderer.js          # UI rendering helpers
    │   └── timeline.js          # Session timeline chart
    │
    └── style/
        └── app.css              # Application styles
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

# 5. Update the model path in testing.py
# Change MODEL_PATH to a relative or absolute path on your machine:
#   MODEL_PATH = os.path.join(os.path.dirname(__file__), "Models", "phase2_best_model.keras")

# 6. Run the application
python main.py
```

The app starts at `http://localhost:8000`, opens automatically in your browser, and initialises all database tables on first run. The default landing route (`/`) serves `login.html`.

---

## Environment Variables

Create a `.env` file in the `Backend/` directory. The application expects the following keys — refer to the `.env.example` file (not committed) for the full template:

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

Never commit the `.env` file. Add it to `.gitignore` before your first push. The app will print a warning on startup for any missing or malformed keys.

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

The inference pipeline uses a MobileNetV2-based CNN trained in two phases. `testing.py` handles all model loading and prediction logic. Key details:

- Input size: 224 x 224 RGB
- Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input`
- Face detection: Haar Cascade (default, ~5–15 ms per frame); MTCNN available via `use_mtcnn=True`
- If no face is detected in a frame, the full frame is used as fallback
- Models are loaded once as lazy singletons with thread locks; a dummy forward pass warms up the model on startup
- Inference runs in a `ThreadPoolExecutor` (1 worker) to avoid blocking the async event loop

The model files are included in the `Backend/Models/` directory. Before running, update `MODEL_PATH` in `testing.py` to point to the correct path on your system.

---

## Database Schema

Five tables are auto-created on startup:

- `users` — accounts, hashed passwords, security questions, admin flag
- `detections` — per-frame emotion, confidence, engagement, and source (webcam / upload)
- `session_timeline` — aggregated session rows with EMA engagement and dominant emotion
- `feedback` — user-submitted ratings and messages
- `password_reset_log` — 3-step reset audit trail with IP address
- `faq_feedback` — per-question liked/disliked votes with optional complaint text

---

## Author

**Your Name**
GitHub: [Suresh Sangeetha](https://github.com/sureshsangeetha295/Facial-Emotion-Recognition-System.git)