import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
import time
import math
import warnings
warnings.filterwarnings("ignore")
from collections import deque

#Config 
MODEL_PATH    = r"models/phase2_best_model.keras"
LABELS        = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
HAAR_PATH     = r"D:\Facial Emotion Detection\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

COLORS = {
    "Angry":    (60,  60,  255),
    "Disgust":  (60,  200, 60),
    "Fear":     (180, 60,  220),
    "Happy":    (60,  255, 180),
    "Neutral":  (200, 200, 200),
    "Sad":      (255, 160, 60),
    "Surprise": (60,  230, 255),
}

RESULT_LINES = {
    "Angry":    ("ANGRY",    "Take a deep breath!"),
    "Disgust":  ("DISGUSTED","Something smell funnyq?"),
    "Fear":     ("FEARFUL",  "It's okay, you're safe!"),
    "Happy":    ("HAPPY",    "Love that energy!"),
    "Neutral":  ("NEUTRAL",  "Poker face activated."),
    "Sad":      ("SAD",      "Sending good vibes!"),
    "Surprise": ("SURPRISED","Didn't see that coming!"),
}

WIN_W      = 520
FACE_H     = 460
BOTTOM_H   = 140
WIN_H      = FACE_H + BOTTOM_H
LOAD_SECS  = 2.2
SMOOTH_N   = 14
ALIGN_SECS = 2.5   # how long to show "look straight" before capturing

#Load model 
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Ready. Press Q to quit.\n")

detector = cv2.CascadeClassifier(HAAR_PATH)

def preprocess(face_bgr):
    face = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    return np.expand_dims(face, 0)

def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=8,
        minSize=(80, 80), maxSize=(500, 500))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    pad = int(min(w, h) * 0.28)
    fh, fw = frame.shape[:2]
    crop = frame[max(0,y-pad):min(fh,y+h+pad), max(0,x-pad):min(fw,x+w+pad)]
    return crop if crop.size > 0 else None

#Gradient background 
def make_gradient(h, w, c1, c2):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        t = i / h
        canvas[i] = tuple(int(c1[k]*(1-t) + c2[k]*t) for k in range(3))
    return canvas

#Corner brackets 
def draw_brackets(canvas, x, y, w, h, color, size=28, thick=3):
    for (cx, cy, sx, sy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(canvas, (cx, cy), (cx+sx*size, cy), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy+sy*size), color, thick, cv2.LINE_AA)

#Spinner 
def draw_spinner(canvas, cx, cy, t, radius=34, color=(255,255,255)):
    n = 12
    for i in range(n):
        angle = (t * 3.5 + i * (2*math.pi / n)) % (2*math.pi)
        alpha = (i + 1) / n
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        r = max(2, int(2 + alpha * 4))
        brightness = int(50 + alpha * 205)
        c = tuple(int(brightness * cc / 255) for cc in color)
        cv2.circle(canvas, (x, y), r, c, -1, cv2.LINE_AA)

#Checkmark 
def draw_checkmark(canvas, cx, cy, color, size=40):
    p1 = (cx - size,     cy + 2)
    p2 = (cx - size//4,  cy + size*3//4)
    p3 = (cx + size,     cy - size*3//4)
    cv2.line(canvas, p1, p2, color, 5, cv2.LINE_AA)
    cv2.line(canvas, p2, p3, color, 5, cv2.LINE_AA)

#Text helpers 
def put_centered(canvas, text, y, font, scale, color, thick=2):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, text, (x, y), font, scale, (0,0,0), thick+3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color,   thick,   cv2.LINE_AA)
    return th

#Scan line effect 
def add_scanlines(canvas, alpha=0.08):
    for y in range(0, canvas.shape[0], 4):
        canvas[y] = (canvas[y] * (1 - alpha)).astype(np.uint8)


#SCREEN BUILDERS

def screen_waiting(t):
    canvas = make_gradient(WIN_H, WIN_W, (8,8,18), (18,8,28))
    # Pulsing circle
    pulse = int(60 + 20 * math.sin(t * 2))
    cv2.circle(canvas, (WIN_W//2, FACE_H//2 - 30), pulse, (40,40,80), 2, cv2.LINE_AA)
    cv2.circle(canvas, (WIN_W//2, FACE_H//2 - 30), pulse//2, (30,30,60), 2, cv2.LINE_AA)
    # Face icon outline
    cx, cy = WIN_W//2, FACE_H//2 - 30
    cv2.circle(canvas, (cx, cy), 55, (60,60,100), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy+30), (35,20), 0, 0, 180, (60,60,100), 2)

    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "STEP INTO FRAME", FACE_H//2 + 60,  font, 0.85, (160,140,255), 2)
    put_centered(canvas, "to begin",        FACE_H//2 + 92,  font, 0.55, (100,90,160),  1)

    # Bottom bar
    cv2.rectangle(canvas, (0, FACE_H), (WIN_W, WIN_H), (12,12,20), -1)
    cv2.line(canvas, (0, FACE_H), (WIN_W, FACE_H), (50,50,90), 1)
    put_centered(canvas, "EMOTION DETECTOR", FACE_H + 45, font, 0.65, (70,70,110), 1)
    put_centered(canvas, "AI Powered  |  RAF-DB Model", FACE_H + 78, font, 0.42, (50,50,80), 1)
    add_scanlines(canvas)
    return canvas

def screen_align(face_bgr, t, elapsed):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    # Slightly desaturated
    gray_face = cv2.cvtColor(face_panel, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
    canvas[:FACE_H] = cv2.addWeighted(face_panel, 0.55, gray_face, 0.45, 0)

    # Animated corner brackets
    blink = int(255 * (0.6 + 0.4 * math.sin(t * 4)))
    bracket_color = (blink, blink, 80)
    margin = 60
    draw_brackets(canvas, margin, margin, WIN_W-2*margin, FACE_H-2*margin,
                  bracket_color, size=36, thick=3)

    # Center crosshair
    cx, cy = WIN_W//2, FACE_H//2
    cl = 18
    cv2.line(canvas, (cx-cl, cy), (cx+cl, cy), bracket_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy-cl), (cx, cy+cl), bracket_color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 28, bracket_color, 1, cv2.LINE_AA)

    # Progress bar (time remaining)
    progress = min(elapsed / ALIGN_SECS, 1.0)
    bar_y = FACE_H - 8
    cv2.rectangle(canvas, (0, bar_y), (WIN_W, FACE_H), (20,20,20), -1)
    fill = int(progress * WIN_W)
    grad_bar = np.zeros((8, WIN_W, 3), dtype=np.uint8)
    for x in range(WIN_W):
        t2 = x / WIN_W
        grad_bar[:, x] = (int(80+t2*60), int(200+t2*55), int(120+t2*80))
    if fill > 0:
        canvas[bar_y:FACE_H, :fill] = grad_bar[:, :fill]

    # Bottom panel
    cv2.rectangle(canvas, (0, FACE_H), (WIN_W, WIN_H), (10,12,18), -1)
    cv2.line(canvas, (0, FACE_H), (WIN_W, FACE_H), (80,200,120), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "LOOK STRAIGHT", FACE_H + 48, font, 1.0, (80,230,140), 2)
    put_centered(canvas, "Hold still  -  capturing your expression", FACE_H + 82, font, 0.40, (60,140,80), 1)
    add_scanlines(canvas)
    return canvas

def screen_snapshot_flash(face_bgr, flash_alpha):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    canvas[:FACE_H] = face_panel
    # White flash overlay
    flash = np.ones((FACE_H, WIN_W, 3), dtype=np.uint8) * 255
    cv2.addWeighted(flash, flash_alpha, canvas[:FACE_H], 1-flash_alpha, 0, canvas[:FACE_H])
    cv2.rectangle(canvas, (0, FACE_H), (WIN_W, WIN_H), (10,12,18), -1)
    cv2.line(canvas, (0, FACE_H), (WIN_W, FACE_H), (255,255,255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "CAPTURED!", FACE_H + 52, font, 1.1, (255,255,255), 2)
    return canvas

def screen_loading(face_bgr, t, progress):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    dark = (face_panel * 0.22).astype(np.uint8)
    canvas[:FACE_H] = dark

    # Spinner
    cx, cy = WIN_W//2, FACE_H//2
    spin_color = (100, 220, 180)
    draw_spinner(canvas, cx, cy, t, radius=42, color=spin_color)

    # Pct text inside spinner
    pct = f"{int(progress*100)}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(pct, font, 0.65, 1)
    cv2.putText(canvas, pct, ((WIN_W-tw)//2, cy+th//2), font, 0.65, spin_color, 1, cv2.LINE_AA)

    put_centered(canvas, "ANALYSING...", cy + 70, font, 0.85, (180,180,180), 1)

    # Animated dots
    n_dots = 3
    dot_spacing = 18
    dot_x0 = WIN_W//2 - dot_spacing
    for i in range(n_dots):
        phase = (t * 2.5 + i * 0.5) % (2*math.pi)
        alpha = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(phase))
        brightness = int(alpha * 220)
        cv2.circle(canvas, (dot_x0 + i*dot_spacing, cy + 95), 5,
                   (brightness, brightness, brightness), -1, cv2.LINE_AA)

    # Progress bar
    bar_y = FACE_H - 8
    cv2.rectangle(canvas, (0, bar_y), (WIN_W, FACE_H), (20,20,20), -1)
    fill = int(progress * WIN_W)
    grad_bar = np.zeros((8, WIN_W, 3), dtype=np.uint8)
    for x in range(WIN_W):
        t2 = x / WIN_W
        grad_bar[:, x] = (int(60+t2*160), int(180+t2*40), int(140+t2*60))
    if fill > 0:
        canvas[bar_y:FACE_H, :fill] = grad_bar[:, :fill]

    # Bottom
    cv2.rectangle(canvas, (0, FACE_H), (WIN_W, WIN_H), (10,12,18), -1)
    cv2.line(canvas, (0, FACE_H), (WIN_W, FACE_H), (60,60,60), 1)
    put_centered(canvas, "Reading your expression...", FACE_H + 52, font, 0.55, (80,80,80), 1)
    put_centered(canvas, "Powered by MobileNetV2  |  RAF-DB", FACE_H + 85, font, 0.38, (50,50,50), 1)
    add_scanlines(canvas)
    return canvas

def screen_confirm(face_bgr, label, color, alpha):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    dark = (face_panel * 0.28).astype(np.uint8)
    canvas[:FACE_H] = dark

    # Glowing circle behind checkmark
    cx, cy = WIN_W//2, FACE_H//2
    glow_r = int(70 * alpha)
    for r in range(glow_r, 0, -8):
        a = (r / glow_r) * 0.15 * alpha
        overlay = canvas.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, a, canvas, 1-a, 0, canvas)

    draw_checkmark(canvas, cx, cy, color, size=int(44*alpha))

    # Bottom
    cv2.rectangle(canvas, (0, FACE_H), (WIN_W, WIN_H), (10,12,18), -1)
    cv2.line(canvas, (0, FACE_H), (WIN_W, FACE_H), color, 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "GOT IT!", FACE_H + 52, font, 1.1, color, 2)
    put_centered(canvas, "Revealing result...", FACE_H + 88, font, 0.45, (100,100,100), 1)
    add_scanlines(canvas)
    return canvas

def screen_result(face_bgr, label, color, t):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    canvas[:FACE_H] = face_panel

    # Colored border on face
    pulse = int(180 + 75 * math.sin(t * 2))
    pc = tuple(min(255, int(c * pulse/255)) for c in color)
    cv2.rectangle(canvas, (0,0), (WIN_W-1, FACE_H-1), pc, 3)

    # Bottom gradient panel
    c1 = tuple(max(0, int(c*0.15)) for c in color)
    c2 = (8, 8, 14)
    for i in range(BOTTOM_H):
        t2 = i / BOTTOM_H
        row_color = tuple(int(c1[k]*(1-t2) + c2[k]*t2) for k in range(3))
        canvas[FACE_H+i, :] = row_color

    # Accent line
    cv2.line(canvas, (0, FACE_H), (WIN_W, FACE_H), color, 3)

    title, subtitle = RESULT_LINES.get(label, (label.upper(), ""))
    font = cv2.FONT_HERSHEY_DUPLEX

    # Main emotion word — large
    put_centered(canvas, title, FACE_H + 58, font, 1.55, color, 3)
    # Subtitle
    put_centered(canvas, subtitle, FACE_H + 96, font, 0.52, (160,160,160), 1)
    # Small label tag top-left of face
    tag = f"  {label}  "
    cv2.rectangle(canvas, (0, 0), (int(len(tag)*10)+10, 28), color, -1)
    cv2.putText(canvas, label, (6, 20), font, 0.55, (0,0,0), 1, cv2.LINE_AA)

    add_scanlines(canvas)
    return canvas

#States
S_WAIT     = 0   # no face
S_ALIGN    = 1   # face found, asking to look straight
S_FLASH    = 2   # camera flash effect
S_LOAD     = 3   # analysing
S_CONFIRM  = 4   # checkmark
S_RESULT   = 5   # showing emotion

state         = S_WAIT
history       = deque(maxlen=SMOOTH_N)
phase_start   = 0.0
final_label   = "Neutral"
final_color   = (200,200,200)
last_face     = np.zeros((200,200,3), dtype=np.uint8)
snapshot_face = None
t             = 0.0
prev_time     = time.time()

FLASH_SECS   = 0.35
CONFIRM_SECS = 0.9

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam."); exit()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    now  = time.time()
    dt   = now - prev_time
    prev_time = now
    t   += dt

    face = get_face(frame)
    elapsed = now - phase_start

    #Transitions 
    if state == S_WAIT:
        if face is not None:
            state = S_ALIGN
            phase_start = now
            last_face = face

    elif state == S_ALIGN:
        if face is None:
            state = S_WAIT
        else:
            last_face = face
            if elapsed >= ALIGN_SECS:
                snapshot_face = face.copy()
                state = S_FLASH
                phase_start = now

    elif state == S_FLASH:
        if elapsed >= FLASH_SECS:
            state = S_LOAD
            phase_start = now
            history.clear()

    elif state == S_LOAD:
        if snapshot_face is not None:
            probs = model.predict(preprocess(snapshot_face), verbose=0)[0]
            history.append(probs)
        progress = min(elapsed / LOAD_SECS, 1.0)
        if elapsed >= LOAD_SECS and len(history) >= 5:
            avg = np.mean(history, axis=0)
            final_label = LABELS[np.argmax(avg)]
            final_color = COLORS[final_label]
            state = S_CONFIRM
            phase_start = now

    elif state == S_CONFIRM:
        if elapsed >= CONFIRM_SECS:
            state = S_RESULT
            phase_start = now

    elif state == S_RESULT:
        if face is not None:
            last_face = face
        if face is None and elapsed > 1.5:
            state = S_WAIT
        # Re-analyze if face changes significantly
        elif face is not None and elapsed > 3.0:
            new_probs = model.predict(preprocess(face), verbose=0)[0]
            new_label = LABELS[np.argmax(new_probs)]
            if new_label != final_label and np.max(new_probs) > 0.60:
                snapshot_face = face.copy()
                state = S_FLASH
                phase_start = now

    #Render 
    if state == S_WAIT:
        canvas = screen_waiting(t)

    elif state == S_ALIGN:
        canvas = screen_align(last_face, t, elapsed)

    elif state == S_FLASH:
        flash_alpha = max(0.0, 1.0 - elapsed/FLASH_SECS)
        canvas = screen_snapshot_flash(snapshot_face, flash_alpha)

    elif state == S_LOAD:
        progress = min(elapsed / LOAD_SECS, 1.0)
        canvas = screen_loading(snapshot_face, t, progress)

    elif state == S_CONFIRM:
        alpha = min(elapsed / 0.25, 1.0)
        canvas = screen_confirm(snapshot_face, final_label, final_color, alpha)

    elif state == S_RESULT:
        canvas = screen_result(last_face, final_label, final_color, t)

    cv2.imshow("Emotion Detection  |  Q = quit", canvas)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")