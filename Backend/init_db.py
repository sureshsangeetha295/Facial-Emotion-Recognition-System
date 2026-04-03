"""
init_db.py — EmotionAI database initialiser
Run once (or re-run safely — all CREATE TABLE statements use IF NOT EXISTS).

Usage:
    python init_db.py

Reads DATABASE_URL from .env (defaults to SQLite: emotionai.db).
Set DATABASE_URL=postgresql://user:pass@host/db for Postgres.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///emotionai.db")

IS_SQLITE   = DATABASE_URL.startswith("sqlite")
IS_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")

print(f"[init_db] Connecting to: {DATABASE_URL}")

if IS_SQLITE:
    import sqlite3, re
    _db_path = re.sub(r"^sqlite:///", "", DATABASE_URL)

    con = sqlite3.connect(_db_path)
    cur = con.cursor()
    cur.executescript("""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- 1. users
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    username      TEXT     NOT NULL UNIQUE,
    email         TEXT     NOT NULL UNIQUE,
    password_hash TEXT     NOT NULL,
    is_admin      INTEGER  NOT NULL DEFAULT 0,
    created_at    TEXT     NOT NULL DEFAULT (datetime('now')),
    last_login    TEXT
);

-- 2. detections
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER  REFERENCES users(id) ON DELETE SET NULL,
    emotion       TEXT     NOT NULL,
    confidence    REAL     NOT NULL,
    engagement    REAL     NOT NULL,
    source        TEXT     NOT NULL DEFAULT 'webcam',
    all_probs     TEXT,
    created_at    TEXT     NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_user    ON detections(user_id);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at);

-- 3. feedback
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER  REFERENCES users(id) ON DELETE SET NULL,
    username      TEXT     NOT NULL DEFAULT 'Guest',
    email         TEXT,
    rating        INTEGER,
    category      TEXT     NOT NULL DEFAULT 'General',
    message       TEXT     NOT NULL,
    status        TEXT     NOT NULL DEFAULT 'new',
    created_at    TEXT     NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_user    ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);

-- 4. session_timeline
CREATE TABLE IF NOT EXISTS session_timeline (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT     NOT NULL,
    user_id           INTEGER  REFERENCES users(id) ON DELETE SET NULL,
    source            TEXT     NOT NULL DEFAULT 'webcam',
    time_offset       REAL     NOT NULL DEFAULT 0.0,
    emotion           TEXT     NOT NULL,
    engagement        REAL     NOT NULL,
    average_engagement REAL,
    dominant_emotion  TEXT,
    frame_count       INTEGER  NOT NULL DEFAULT 1,
    started_at        TEXT     NOT NULL DEFAULT (datetime('now')),
    ended_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_stl_session ON session_timeline(session_id);
CREATE INDEX IF NOT EXISTS idx_stl_user    ON session_timeline(user_id);
CREATE INDEX IF NOT EXISTS idx_stl_started ON session_timeline(started_at);
    """)
    # ── Migrate existing tables (add columns introduced after initial deploy) ──
    sqlite_migrations = [
        "ALTER TABLE feedback ADD COLUMN username  TEXT NOT NULL DEFAULT 'Guest'",
        "ALTER TABLE feedback ADD COLUMN email     TEXT",
        "ALTER TABLE feedback ADD COLUMN rating    INTEGER",
        "ALTER TABLE feedback ADD COLUMN category  TEXT NOT NULL DEFAULT 'General'",
        "ALTER TABLE feedback ADD COLUMN status    TEXT NOT NULL DEFAULT 'new'",
    ]
    for _sql in sqlite_migrations:
        try:
            cur.execute(_sql)
        except Exception:
            pass  # column already exists — safe to ignore
    con.commit()
    con.close()
    print("[init_db] SQLite tables created/verified:", _db_path)

elif IS_POSTGRES:
    import psycopg2

    con = psycopg2.connect(DATABASE_URL)
    con.autocommit = False
    cur = con.cursor()

    ddl = """
-- 1. users
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL    PRIMARY KEY,
    username      VARCHAR(80)  NOT NULL UNIQUE,
    email         VARCHAR(254) NOT NULL UNIQUE,
    password_hash TEXT         NOT NULL,
    is_admin      BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

-- 2. detections
CREATE TABLE IF NOT EXISTS detections (
    id         BIGSERIAL    PRIMARY KEY,
    user_id    BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    emotion    VARCHAR(40)  NOT NULL,
    confidence REAL         NOT NULL,
    engagement REAL         NOT NULL,
    source     VARCHAR(20)  NOT NULL DEFAULT 'webcam',
    all_probs  JSONB,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detections_user    ON detections(user_id);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at);

-- 3. feedback
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
);

CREATE INDEX IF NOT EXISTS idx_feedback_user    ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);

-- 4. session_timeline
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
);

CREATE INDEX IF NOT EXISTS idx_stl_session ON session_timeline(session_id);
CREATE INDEX IF NOT EXISTS idx_stl_user    ON session_timeline(user_id);
CREATE INDEX IF NOT EXISTS idx_stl_started ON session_timeline(started_at);
    """

    cur.execute(ddl)
    con.commit()

    # ── Migrate existing tables (add columns introduced after initial deploy) ──
    pg_migrations = [
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS username  VARCHAR(80)  NOT NULL DEFAULT 'Guest'",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS email     VARCHAR(254)",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS rating    SMALLINT",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS category  VARCHAR(60)  NOT NULL DEFAULT 'General'",
        "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS status    VARCHAR(20)  NOT NULL DEFAULT 'new'",
    ]
    for _sql in pg_migrations:
        try:
            cur.execute(_sql)
            con.commit()
        except Exception as _e:
            con.rollback()
            print(f"[init_db] migration skipped (already applied): {_e}")

    cur.close()
    con.close()
    print("[init_db] Postgres tables created/verified.")

else:
    raise ValueError(f"Unsupported DATABASE_URL scheme: {DATABASE_URL}")

print("[init_db] Done — all 4 tables ready: users, detections, feedback, session_timeline")