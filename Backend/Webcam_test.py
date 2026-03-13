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
from PIL import Image, ImageDraw, ImageFont
import platform

# ─────────────────────────── CONFIG ────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "models", "phase2_best_model.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.normpath(os.path.join(_DIR, "..", "models", "phase2_best_model.keras"))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model not found at: {MODEL_PATH}")
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


TEMPERATURE = 0.75   # sharpen softmax output

PRIOR_BOOST = {
    "Angry":    1.55,   # 452 samples — was wrongly penalized
    "Disgust":  1.52,   # 460 samples — nearly same as Angry
    "Fear":     3.09,   # 180 samples — rarest, was severely under-boosted
    "Happy":    0.23,   # 3055 samples — dominant class
    "Neutral":  0.44,   # 1616 samples
    "Sad":      0.55,   # 1269 samples
    "Surprise": 1.06,   # 826 samples — was wrongly penalized
}
_BOOST_VEC = np.array([PRIOR_BOOST[l] for l in LABELS], dtype=np.float32)


def apply_calibration(raw_probs):
    
    eps   = 1e-7
    probs = np.clip(raw_probs, eps, 1.0)

    # Step 1 — temperature scaling (log-prob space)
    log_p  = np.log(probs) / TEMPERATURE
    log_p -= np.max(log_p)
    scaled  = np.exp(log_p)
    scaled /= scaled.sum()

    # Step 2 — prior boost
    adjusted = scaled * _BOOST_VEC

    # Step 3 — re-normalise
    total = adjusted.sum()
    return adjusted / total if total > 0 else scaled


# ── Temporal smoothing ───────────────────────────────────────────
SMOOTH_N = 10


def preprocess(face_bgr):
    face = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    return np.expand_dims(face, 0)


def predict_single(face_bgr):
    
    raw = model.predict(preprocess(face_bgr), verbose=0)[0]
    return apply_calibration(raw)


HAAR_PATH = r"D:\Facial Emotion Detection\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

COLORS = {
    "Angry":    (40,  40,  255),
    "Disgust":  (40,  220,  40),
    "Fear":     (200,  40, 255),
    "Happy":    (40,  255, 160),
    "Neutral":  (210, 210, 210),
    "Sad":      (255, 140,  40),
    "Surprise": (40,  210, 255),
}

EMOJIS = {
    "Angry":    ("ANGRY",     "Take a deep breath!",     "😠"),
    "Disgust":  ("DISGUSTED", "Something smell funny?",  "🤢"),
    "Fear":     ("FEARFUL",   "It's okay, you're safe!", "😨"),
    "Happy":    ("HAPPY",     "Love that energy!",       "😄"),
    "Neutral":  ("NEUTRAL",   "Poker face activated.",   "😐"),
    "Sad":      ("SAD",       "Sending good vibes!",     "😢"),
    "Surprise": ("SURPRISED", "Didn't see that coming!", "😲"),
}

WIN_W        = 560
FACE_H       = 460
BOTTOM_H     = 160
WIN_H        = FACE_H + BOTTOM_H
LOAD_SECS    = 2.5      # time window for SMOOTH_N real passes
ALIGN_SECS   = 2.0
FLASH_SECS   = 0.40
CONFIRM_SECS = 0.9

# ─────────────────────── LOAD MODEL ────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Ready.  Press ENTER for next scan, Q to quit.\n")

detector = cv2.CascadeClassifier(HAAR_PATH)


def get_face(frame):

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4,
        minSize=(80, 80), maxSize=(600, 600))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    fh, fw = frame.shape[:2]
    cx, cy = x + w//2, y + h//2
    half   = int(max(w, h) * 0.70)
    mx1, my1 = max(0, cx-half), max(0, cy-half)
    mx2, my2 = min(fw, cx+half), min(fh, cy+half)
    model_crop = frame[my1:my2, mx1:mx2]
    if model_crop.size == 0:
        return None
    return frame, model_crop, (x, y, w, h)


# ─────────────────────── EMOJI / PIL HELPERS ───────────────────
def get_emoji_font(size=52):
    system     = platform.system()
    font_paths = []
    if system == "Windows":
        font_paths = [r"C:\Windows\Fonts\seguiemj.ttf",
                      r"C:\Windows\Fonts\seguisym.ttf"]
    elif system == "Darwin":
        font_paths = ["/System/Library/Fonts/Apple Color Emoji.ttc"]
    else:
        font_paths = ["/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                      "/usr/share/fonts/noto/NotoColorEmoji.ttf"]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def put_emoji_centered(canvas_bgr, emoji_char, y_top, font_size=56):

    pil_img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    font    = get_emoji_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), emoji_char, font=font)
        tw   = bbox[2] - bbox[0]
    except AttributeError:
        tw = font_size
    x = (canvas_bgr.shape[1] - tw) // 2
    draw.text((x, y_top), emoji_char, font=font, embedded_color=True)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─────────────────────── DRAWING HELPERS ───────────────────────
def make_gradient(h, w, c1, c2, vertical=True):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    n = h if vertical else w
    for i in range(n):
        t2  = i / n
        col = tuple(int(c1[k]*(1-t2) + c2[k]*t2) for k in range(3))
        if vertical:
            canvas[i] = col
        else:
            canvas[:, i] = col
    return canvas


def draw_brackets(canvas, x, y, w, h, color, size=30, thick=3):
    for (cx, cy, sx, sy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(canvas, (cx, cy), (cx+sx*size, cy), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy+sy*size), color, thick, cv2.LINE_AA)


def draw_spinner(canvas, cx, cy, t, radius=38, color=(255,255,255)):
    n = 14
    for i in range(n):
        angle = (t*4.0 + i*(2*math.pi/n)) % (2*math.pi)
        alpha = (i+1) / n
        px    = int(cx + radius*math.cos(angle))
        py    = int(cy + radius*math.sin(angle))
        r     = max(2, int(2+alpha*5))
        bright = int(60+alpha*195)
        c = tuple(int(bright*cc/255) for cc in color)
        cv2.circle(canvas, (px, py), r, c, -1, cv2.LINE_AA)


def draw_checkmark(canvas, cx, cy, color, size=44):
    p1 = (cx-size,    cy+4)
    p2 = (cx-size//4, cy+size*3//4)
    p3 = (cx+size,    cy-size*3//4)
    cv2.line(canvas, p1, p2, color, 6, cv2.LINE_AA)
    cv2.line(canvas, p2, p3, color, 6, cv2.LINE_AA)


def put_centered(canvas, text, y, font, scale, color, thick=2, shadow=True):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (canvas.shape[1] - tw) // 2
    if shadow:
        cv2.putText(canvas, text, (x+2, y+2), font, scale, (0,0,0), thick+4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return th


def add_scanlines(canvas, alpha=0.06):
    for y in range(0, canvas.shape[0], 4):
        canvas[y] = (canvas[y] * (1-alpha)).astype(np.uint8)


def add_noise(canvas, intensity=6):
    noise = np.random.randint(-intensity, intensity, canvas.shape, dtype=np.int16)
    canvas[:] = np.clip(canvas.astype(np.int16)+noise, 0, 255).astype(np.uint8)


def draw_glowing_circle(canvas, cx, cy, radius, color, layers=5):
    for i in range(layers, 0, -1):
        r = radius + i*6
        a = 0.04*(layers-i+1)
        ov = canvas.copy()
        cv2.circle(ov, (cx, cy), r, color, 2)
        cv2.addWeighted(ov, a, canvas, 1-a, 0, canvas)
    cv2.circle(canvas, (cx, cy), radius, color, 2, cv2.LINE_AA)


def draw_robot_face(canvas, cx, cy, t, color=(100,220,180), blink=False):
    hw, hh = 70, 58
    head_pts = np.array([[cx-hw,cy-hh],[cx+hw,cy-hh],[cx+hw,cy+hh],[cx-hw,cy+hh]], dtype=np.int32)
    cv2.polylines(canvas, [head_pts], True, color, 2, cv2.LINE_AA)
    for ox, oy in [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]:
        cv2.circle(canvas, (cx+ox, cy+oy), 10, color, 2, cv2.LINE_AA)
    eye_y = cy - 10
    for side in [-1, 1]:
        ex = cx + side*28
        if blink:
            cv2.line(canvas, (ex-10, eye_y), (ex+10, eye_y), color, 3, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (ex, eye_y), 12, color, 2, cv2.LINE_AA)
            px = int(ex + 6*math.sin(t*3.2))
            py = int(eye_y + 4*math.cos(t*2.1))
            cv2.circle(canvas, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (ex, eye_y), int(8+2*math.sin(t*5)),
                       tuple(min(255,c+80) for c in color), 1, cv2.LINE_AA)
    mouth_y  = cy + 28
    bar_w, bars = 10, 9
    start_x  = cx - bars*(bar_w+3)//2
    for i in range(bars):
        phase = t*6 + i*0.7
        bh    = int(6+12*abs(math.sin(phase)))
        bx    = start_x + i*(bar_w+3)
        bc    = tuple(int(c*(0.5+0.5*abs(math.sin(phase)))) for c in color)
        cv2.rectangle(canvas, (bx, mouth_y-bh), (bx+bar_w, mouth_y), bc, -1)
    ant_y, ant_top = cy-hh, cy-hh-22
    cv2.line(canvas, (cx, ant_y), (cx, ant_top), color, 2, cv2.LINE_AA)
    bp = abs(math.sin(t*8))
    cv2.circle(canvas, (cx, ant_top), 6, tuple(int(c*bp) for c in color), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, ant_top), 8, color, 1, cv2.LINE_AA)
    for side in [-1, 1]:
        bx = cx + side*hw
        for off in [-20, 0, 20]:
            cv2.circle(canvas, (bx, cy+off), 3, color, -1, cv2.LINE_AA)
    chars         = "01 FER CNN PRED 10 LSTM 01 RAF"
    scroll_offset = int(t*40) % len(chars)
    scrolled      = (chars[scroll_offset:] + chars[:scroll_offset])[:30]
    (tw, _), _    = cv2.getTextSize(scrolled, cv2.FONT_HERSHEY_PLAIN, 0.65, 1)
    cv2.putText(canvas, scrolled, (cx-tw//2, cy+hh+20),
                cv2.FONT_HERSHEY_PLAIN, 0.65,
                tuple(int(c*0.45) for c in color), 1, cv2.LINE_AA)


# ─────────────────────── SCREEN BUILDERS ───────────────────────
def screen_waiting(t):
    canvas = make_gradient(WIN_H, WIN_W, (4,4,14), (14,6,28))
    add_noise(canvas, 3)
    for row in range(-1, WIN_H//28+2):
        for col in range(-1, WIN_W//28+2):
            hs  = 24
            hcx = int(col*hs*1.73)
            hcy = int(row*hs*2) + (hs if col%2==1 else 0)
            pts = [(int(hcx+hs*math.cos(math.radians(60*i+30))),
                    int(hcy+hs*math.sin(math.radians(60*i+30)))) for i in range(6)]
            phase  = (hcx/WIN_W + hcy/WIN_H + t*0.3) % 1.0
            bright = int(12+6*math.sin(phase*math.pi*2))
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], True, (bright,bright,bright+10), 1)

    cx, cy  = WIN_W//2, FACE_H//2-20
    pulse_r = int(90+12*math.sin(t*1.8))
    draw_glowing_circle(canvas, cx, cy, pulse_r,    (60,40,120))
    draw_glowing_circle(canvas, cx, cy, pulse_r-22, (40,30,90))
    cv2.ellipse(canvas, (cx,cy-8),  (38,46), 0,0,360, (70,55,130), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx,cy+32), (30,18), 0,0,180, (70,55,130), 2, cv2.LINE_AA)
    scan_y = cy-46 + int(92*((math.sin(t*1.4)+1)/2))
    cv2.line(canvas, (cx-38, scan_y), (cx+38, scan_y), (80,60,180), 1, cv2.LINE_AA)
    for i in range(6):
        angle = t*0.9 + i*(2*math.pi/6)
        ox, oy = int(cx+pulse_r*math.cos(angle)), int(cy+pulse_r*math.sin(angle))
        ci = 0.5+0.5*math.sin(t*2+i)
        cv2.circle(canvas, (ox,oy), 3 if i%2==0 else 2,
                   tuple(int(c*ci) for c in (120,80,220)), -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "STEP INTO FRAME",            cy+88,  font, 0.88, (160,130,255), 2)
    put_centered(canvas, "Position your face to begin", cy+118, font, 0.45, (80,65,130),  1, shadow=False)
    corner_size = 22
    for (bx,by),(sx,sy) in zip([(12,12),(WIN_W-12,12),(12,FACE_H-12),(WIN_W-12,FACE_H-12)],
                                [(1,1),(-1,1),(1,-1),(-1,-1)]):
        cv  = int(60+40*abs(math.sin(t*1.5)))
        cc  = (cv, cv//2, cv*2)
        cv2.line(canvas, (bx,by), (bx+sx*corner_size,by),   cc, 2, cv2.LINE_AA)
        cv2.line(canvas, (bx,by), (bx,by+sy*corner_size),   cc, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0,FACE_H), (WIN_W,WIN_H), (8,6,16), -1)
    cv2.line(canvas, (0,FACE_H), (WIN_W,FACE_H), (50,35,100), 2)
    aw = int(WIN_W*(0.5+0.5*abs(math.sin(t*0.7))))
    ax = (WIN_W-aw)//2
    cv2.line(canvas, (ax,FACE_H+1), (ax+aw,FACE_H+1), (90,60,180), 1)
    put_centered(canvas, "EMOTION DETECTOR",                    FACE_H+52,  font, 0.78, (110,80,200), 2)
    put_centered(canvas, "NEURAL NETWORK  |  RAF-DB  MODEL",   FACE_H+88,  font, 0.38, (55,42,100),  1, shadow=False)
    put_centered(canvas, "MobileNetV2  +  Prior Calibration",  FACE_H+114, font, 0.36, (40,32,80),   1, shadow=False)
    add_scanlines(canvas)
    return canvas


def screen_align(face_bgr, t, elapsed, face_rect=None):
    canvas     = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    gray_face  = cv2.cvtColor(cv2.cvtColor(face_panel, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    canvas[:FACE_H] = cv2.addWeighted(face_panel, 0.60, gray_face, 0.40, 0)

    progress      = min(elapsed/ALIGN_SECS, 1.0)
    blink_v       = int(180+75*math.sin(t*5))
    green_v       = int(220*(1-progress)+255*progress)
    bracket_color = (40, green_v, blink_v//2)
    cx, cy        = WIN_W//2, FACE_H//2

    if face_rect is not None:
        fh_orig, fw_orig = face_bgr.shape[:2]
        rx, ry, rw, rh = face_rect
        dx  = int(rx*WIN_W/fw_orig);  dy  = int(ry*FACE_H/fh_orig)
        dw  = int(rw*WIN_W/fw_orig);  dh  = int(rh*FACE_H/fh_orig)
        bx1 = max(0,      dx-int(dw*0.18));  by1 = max(0,       dy-int(dh*0.18))
        bx2 = min(WIN_W-1,dx+dw+int(dw*0.18)); by2 = min(FACE_H-1,dy+dh+int(dh*0.18))
        for gi in range(4, 0, -1):
            ov = canvas.copy()
            cv2.rectangle(ov, (bx1-gi*3,by1-gi*3), (bx2+gi*3,by2+gi*3), bracket_color, 1)
            cv2.addWeighted(ov, 0.08*gi*progress, canvas, 1-0.08*gi*progress, 0, canvas)
        draw_brackets(canvas, bx1, by1, bx2-bx1, by2-by1, bracket_color, size=28, thick=3)
        fcx, fcy = (bx1+bx2)//2, (by1+by2)//2
        cv2.line(canvas, (fcx-14,fcy), (fcx+14,fcy), bracket_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (fcx,fcy-14), (fcx,fcy+14), bracket_color, 1, cv2.LINE_AA)
    else:
        m = 50
        draw_brackets(canvas, m, m, WIN_W-2*m, FACE_H-2*m, bracket_color, size=38, thick=3)
        cv2.ellipse(canvas, (cx,cy), (120,155), 0,0,360, bracket_color, 2, cv2.LINE_AA)

    ring_r = 44
    draw_glowing_circle(canvas, cx, cy+FACE_H//3+20, ring_r,
                        tuple(int(c*progress) for c in (60,255,120)))
    cv2.ellipse(canvas, (cx,cy+FACE_H//3+20), (ring_r,ring_r),
                -90, -90, int(-90+360*progress), (60,255,120), 3, cv2.LINE_AA)
    remain = f"{max(0,ALIGN_SECS-elapsed):.1f}s"
    font   = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(remain, font, 0.55, 1)
    cv2.putText(canvas, remain, (cx-tw//2, cy+FACE_H//3+20+th//2),
                font, 0.55, (60,255,120), 1, cv2.LINE_AA)

    bar_y = FACE_H-6
    cv2.rectangle(canvas, (0,bar_y), (WIN_W,FACE_H), (15,18,15), -1)
    for xi in range(int(progress*WIN_W)):
        t2 = xi/WIN_W
        cv2.line(canvas, (xi,bar_y), (xi,FACE_H),
                 (int(40+t2*80),int(200+t2*55),int(80+t2*100)), 1)

    cv2.rectangle(canvas, (0,FACE_H), (WIN_W,WIN_H), (8,12,10), -1)
    cv2.line(canvas, (0,FACE_H), (WIN_W,FACE_H), (60,220,100), 3)
    put_centered(canvas, "LOOK  STRAIGHT",                       FACE_H+55, font, 1.05, (60,240,120), 3)
    put_centered(canvas, "Keep still  —  capturing your expression",
                 FACE_H+90, font, 0.40, (40,130,60), 1, shadow=False)
    dx0 = WIN_W//2 - 5*14//2
    for i in range(5):
        da = 0.2+0.8*abs(math.sin((t*3+i*0.4) % (2*math.pi)))
        cv2.circle(canvas, (dx0+i*14,FACE_H+112), 3,
                   (int(60*da),int(220*da),int(90*da)), -1, cv2.LINE_AA)
    add_scanlines(canvas)
    return canvas


def screen_snapshot_flash(face_bgr, flash_alpha, face_rect=None):
    canvas     = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    canvas[:FACE_H] = face_panel
    flash = np.ones((FACE_H,WIN_W,3), dtype=np.uint8)*255
    cv2.addWeighted(flash, flash_alpha, canvas[:FACE_H], 1-flash_alpha, 0, canvas[:FACE_H])
    cv2.rectangle(canvas, (0,FACE_H), (WIN_W,WIN_H), (20,22,18), -1)
    cv2.line(canvas, (0,FACE_H), (WIN_W,FACE_H), (255,255,255), 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "CAPTURED!", FACE_H+58, font, 1.2,  (255,255,255), 3)
    put_centered(canvas, "Snap!",     FACE_H+96, font, 0.55, (180,180,180), 1, shadow=False)
    return canvas


def screen_loading(face_bgr, t, progress, pass_count):
    canvas     = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    canvas[:FACE_H] = (face_panel*0.15).astype(np.uint8)

    for row in range(-1, FACE_H//22+2):
        for col in range(-1, WIN_W//22+2):
            hs  = 20
            hcx = int(col*hs*1.73);  hcy = int(row*hs*2)+(hs if col%2==1 else 0)
            pts = [(int(hcx+hs*math.cos(math.radians(60*i+30))),
                    int(hcy+hs*math.sin(math.radians(60*i+30)))) for i in range(6)]
            phase = (hcx/WIN_W + hcy/FACE_H + t*0.5) % 1.0
            if hcy < FACE_H:
                bright = int(14+10*abs(math.sin(phase*math.pi*2)))
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], True, (bright,bright+10,bright+5), 1)

    rcx, rcy = WIN_W//2, FACE_H//2-30
    rc        = (80,220,170)
    for ri in range(3):
        rp     = (t*2+ri*0.5) % 3.0
        ring_r = int(80+ri*35+rp*15)
        ring_a = max(0.0, 1.0-rp/3.0)*0.3
        ov     = canvas.copy()
        cv2.circle(ov, (rcx,rcy), ring_r, rc, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, ring_a, canvas, 1-ring_a, 0, canvas)

    draw_robot_face(canvas, rcx, rcy, t, rc, blink=(int(t*2)%5==0))
    draw_spinner(canvas, rcx, rcy, t, radius=120, color=rc)

    font = cv2.FONT_HERSHEY_DUPLEX
    pct  = f"{int(progress*100)}%"
    (tw,th),_ = cv2.getTextSize(pct, font, 0.7, 1)
    cv2.putText(canvas, pct, (rcx-tw//2, rcy+105+th), font, 0.7, rc, 1, cv2.LINE_AA)

    # Show actual pass count (honest — same real image each pass)
    status = f"Pass {min(pass_count, SMOOTH_N)} / {SMOOTH_N}  —  real image"
    put_centered(canvas, status, rcy+128, font, 0.40, (80,150,120), 1, shadow=False)

    bar_y = FACE_H-8
    cv2.rectangle(canvas, (0,bar_y), (WIN_W,FACE_H), (12,16,14), -1)
    for xi in range(int(progress*WIN_W)):
        t2 = xi/WIN_W
        cv2.line(canvas, (xi,bar_y+1), (xi,FACE_H-1),
                 (int(40+t2*60),int(180+t2*40),int(120+t2*50)), 1)

    cv2.rectangle(canvas, (0,FACE_H), (WIN_W,WIN_H), (6,10,8), -1)
    cv2.line(canvas, (0,FACE_H), (WIN_W,FACE_H), (50,100,70), 2)
    put_centered(canvas, "ANALYSING...", FACE_H+52, font, 1.0, (80,200,140), 2)

    dx0 = WIN_W//2 - 5*16//2
    for i in range(5):
        da = 0.15+0.85*abs(math.sin((t*3.5+i*0.5) % (2*math.pi)))
        cv2.circle(canvas, (dx0+i*16,FACE_H+85), 4,
                   tuple(int(c*da) for c in (60,200,140)), -1, cv2.LINE_AA)

    put_centered(canvas, "Temperature Scaling  +  Prior Calibration",
                 FACE_H+116, font, 0.35, (35,65,48), 1, shadow=False)
    add_scanlines(canvas)
    return canvas


def screen_confirm(face_bgr, label, color, alpha, t):
    canvas     = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    canvas[:FACE_H] = (face_panel*0.20).astype(np.uint8)
    cx, cy = WIN_W//2, FACE_H//2
    for ri in range(5):
        glow_r = int((50+ri*25)*alpha)
        glow_a = max(0,0.12-ri*0.02)*alpha
        if glow_r > 0:
            ov = canvas.copy()
            cv2.circle(ov, (cx,cy), glow_r, color, -1)
            cv2.addWeighted(ov, glow_a, canvas, 1-glow_a, 0, canvas)
    cv2.circle(canvas, (cx,cy), int(65*alpha), color, 2, cv2.LINE_AA)
    ring_r = int(80*alpha)
    if ring_r > 0:
        cv2.ellipse(canvas, (cx,cy), (ring_r,ring_r), t*90, 0, int(300*alpha), color, 2, cv2.LINE_AA)
    draw_checkmark(canvas, cx, cy, color, size=int(48*alpha))
    for i in range(12):
        ang = i*(2*math.pi/12)+t*2
        dist = int(90*alpha)
        cv2.circle(canvas, (cx+int(dist*math.cos(ang)), cy+int(dist*math.sin(ang))),
                   max(1,int(4*(1-alpha)+2)), color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0,FACE_H), (WIN_W,WIN_H), (8,8,12), -1)
    cv2.line(canvas, (0,FACE_H), (WIN_W,FACE_H), color, 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    put_centered(canvas, "GOT IT!",             FACE_H+56, font, 1.15, color,       2)
    put_centered(canvas, "Revealing result...", FACE_H+94, font, 0.45, (100,100,100), 1, shadow=False)
    add_scanlines(canvas)
    return canvas


def screen_result(face_bgr, label, color, t, avg_probs, result_elapsed=0.0):
    
    PHASE2_START  = 1.2    # seconds before caption appears
    TYPEWR_SPEED  = 18     # characters per second

    canvas     = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    face_panel = cv2.resize(face_bgr, (WIN_W, FACE_H))
    canvas[:FACE_H] = face_panel

    # ── Vignette ──
    vy   = ((np.arange(FACE_H)-FACE_H/2)/(FACE_H/2))**2
    vx   = ((np.arange(WIN_W) -WIN_W/2) /(WIN_W/2)) **2
    vign = np.clip(1 - np.outer(vy, np.ones(WIN_W))
                     - np.outer(np.ones(FACE_H), vx)*0.55, 0, 1).astype(np.float32)
    for ch in range(3):
        canvas[:FACE_H,:,ch] = (canvas[:FACE_H,:,ch]*vign).astype(np.uint8)

    # ── Pulsing border ──
    pulse = 0.65+0.35*math.sin(t*2.2)
    pc    = tuple(min(255,int(c*pulse)) for c in color)
    cv2.rectangle(canvas, (0,0),  (WIN_W-1,FACE_H-1), pc, 5)
    cv2.rectangle(canvas, (6,6),  (WIN_W-7, FACE_H-7), tuple(int(c*0.3) for c in color), 1)

    # ── Scan line ──
    scan_y = int(FACE_H//2+(FACE_H//3)*math.sin(t*1.3))
    sl = canvas[:FACE_H].copy()
    cv2.line(sl, (0,scan_y), (WIN_W,scan_y), color, 1, cv2.LINE_AA)
    cv2.addWeighted(sl, 0.2, canvas[:FACE_H], 0.8, 0, canvas[:FACE_H])

    # ── Bottom panel ──
    dark_base = tuple(max(0,int(c*0.12)) for c in color)
    for i in range(BOTTOM_H):
        f = 1-i/BOTTOM_H
        canvas[FACE_H+i,:] = tuple(int(dark_base[k]*f) for k in range(3))
    cv2.line(canvas, (0,FACE_H),   (WIN_W,FACE_H),   color, 3)
    cv2.line(canvas, (0,FACE_H+4), (WIN_W,FACE_H+4), tuple(int(c*0.4) for c in color), 1)

    font  = cv2.FONT_HERSHEY_DUPLEX
    title, subtitle, emoji_char = EMOJIS.get(label, (label.upper(),"",""))
    conf  = float(np.max(avg_probs)) if avg_probs is not None else 0.0

    # ── Phase progress ──
    p1     = min(result_elapsed / PHASE2_START, 1.0)          # 0→1 during phase 1
    in_p2  = result_elapsed >= PHASE2_START
    p2_t   = max(0.0, result_elapsed - PHASE2_START)          # seconds into phase 2

    # Spring-ease helper  (overshoot then settle)
    def spring(x):
        x = max(0.0, min(x, 1.0))
        return 1 - math.exp(-6*x) * math.cos(8*x)

    # ── PHASE 1: emotion name slides in ──
    # slides from FACE_H+5 down to FACE_H+40 with spring
    name_target_y = FACE_H + 40
    name_slide     = spring(p1)
    name_y         = int(FACE_H + 5 + 35*name_slide)
    name_alpha     = p1
    name_scale     = 0.85 + 0.50*spring(p1)   # grows 0.85→1.35

    # Draw name with fade (blend trick via overlay)
    name_canvas = canvas.copy()
    put_centered(name_canvas, title, name_y, font,
                 min(name_scale, 1.35), color, 3)
    cv2.addWeighted(name_canvas, name_alpha, canvas, 1-name_alpha, 0, canvas)

    # ── Confidence — fades in ──
    conf_str = f"{conf*100:.1f}%  confidence"
    (cw,_),_ = cv2.getTextSize(conf_str, font, 0.48, 1)
    cx2 = (WIN_W-cw)//2
    conf_alpha = min(p1*1.5, 1.0)
    bright_col = tuple(min(255,c+80) for c in color)
    conf_cv = canvas.copy()
    cv2.putText(conf_cv, conf_str, (cx2+1,FACE_H+75+1), font, 0.48, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(conf_cv, conf_str, (cx2,  FACE_H+75),   font, 0.48, bright_col, 1, cv2.LINE_AA)
    cv2.addWeighted(conf_cv, conf_alpha, canvas, 1-conf_alpha, 0, canvas)

    # ── Emoji ──
    if emoji_char:
        if not in_p2:
            # Phase 1 — large, centred, bouncing
            bounce    = math.sin(t*5)*4 * (1-p1*0.5)          # settle the bounce
            emoji_sz  = int(52 + 18*spring(p1))                # 52→70 px
            emoji_y   = int(FACE_H + 90 + bounce)
            ov_emoji  = canvas.copy()
            ov_emoji  = put_emoji_centered(ov_emoji, emoji_char, emoji_y, font_size=emoji_sz)
            cv2.addWeighted(ov_emoji, p1, canvas, 1-p1, 0, canvas)
        else:
            # Phase 2 — shrink + shift left, stable
            # Smooth transition over 0.4 s
            shrink_t  = min(p2_t / 0.4, 1.0)
            emoji_sz  = int(70 - 22*shrink_t)                  # 70→48 px
            # x: centred → shifted left so caption fits right of it
            centre_x  = WIN_W//2 - emoji_sz//2
            left_x    = WIN_W//4 - emoji_sz//2
            emoji_x   = int(centre_x + (left_x - centre_x)*shrink_t)
            emoji_y   = FACE_H + 92
            # Render at computed x (put_emoji_centered only centres, so we
            # do a custom x placement via a temporary canvas)
            tmp = canvas.copy()
            pil_img   = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
            draw_pil  = ImageDraw.Draw(pil_img)
            efont     = get_emoji_font(emoji_sz)
            try:
                bbox  = draw_pil.textbbox((0,0), emoji_char, font=efont)
                etw   = bbox[2]-bbox[0]
            except AttributeError:
                etw   = emoji_sz
            draw_pil.text((emoji_x, emoji_y), emoji_char, font=efont, embedded_color=True)
            canvas    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # ── Caption pill + typewriter ──
            cap_alpha  = min(p2_t / 0.5, 1.0)
            n_chars    = int(p2_t * TYPEWR_SPEED)
            shown_text = subtitle[:n_chars]

            if shown_text:
                # Pill background
                (tw, th), _ = cv2.getTextSize(shown_text, font, 0.46, 1)
                pad_x, pad_y = 14, 8
                pill_x1 = WIN_W//2 - tw//2 - pad_x
                pill_y1 = FACE_H + 88
                pill_x2 = WIN_W//2 + tw//2 + pad_x
                pill_y2 = FACE_H + 88 + th + pad_y*2

                # Draw pill (rounded via overlapping rects + circles)
                pill_ov = canvas.copy()
                pill_color_bg = tuple(max(0, min(255, int(c*0.22))) for c in color)
                pill_color_border = tuple(max(0, min(255, int(c*0.75))) for c in color)
                cv2.rectangle(pill_ov, (pill_x1+8, pill_y1),
                              (pill_x2-8, pill_y2), pill_color_bg, -1)
                cv2.rectangle(pill_ov, (pill_x1, pill_y1+8),
                              (pill_x2, pill_y2-8), pill_color_bg, -1)
                for (cx_p, cy_p) in [(pill_x1+8, pill_y1+8), (pill_x2-8, pill_y1+8),
                                      (pill_x1+8, pill_y2-8), (pill_x2-8, pill_y2-8)]:
                    cv2.circle(pill_ov, (cx_p, cy_p), 8, pill_color_bg, -1)
                cv2.addWeighted(pill_ov, cap_alpha*0.82, canvas, 1-cap_alpha*0.82, 0, canvas)

                # Pill border glow
                pill_border_ov = canvas.copy()
                cv2.rectangle(pill_border_ov, (pill_x1, pill_y1),
                              (pill_x2, pill_y2), pill_color_border, 1, cv2.LINE_AA)
                cv2.addWeighted(pill_border_ov, cap_alpha*0.6, canvas, 1-cap_alpha*0.6, 0, canvas)

                # Caption text — bright white with colored glow shadow
                text_y = pill_y1 + th + pad_y
                text_x = WIN_W//2 - tw//2
                glow_col = tuple(min(255,c+120) for c in color)
                cap_ov = canvas.copy()
                # glow shadow
                cv2.putText(cap_ov, shown_text, (text_x+1,text_y+1),
                            font, 0.46, glow_col, 3, cv2.LINE_AA)
                # main white text
                cv2.putText(cap_ov, shown_text, (text_x, text_y),
                            font, 0.46, (240,240,240), 1, cv2.LINE_AA)
                # blinking cursor while typing
                if n_chars < len(subtitle):
                    (cw2, _), _ = cv2.getTextSize(shown_text, font, 0.46, 1)
                    cur_x = text_x + cw2 + 3
                    cur_blink = abs(math.sin(t*8))
                    cur_col   = tuple(int(c*cur_blink) for c in glow_col)
                    cv2.line(cap_ov, (cur_x, text_y-th), (cur_x, text_y+2), cur_col, 2, cv2.LINE_AA)
                cv2.addWeighted(cap_ov, cap_alpha, canvas, 1-cap_alpha, 0, canvas)

            # ── ENTER prompt fades in after caption finishes ──
            enter_alpha = min(max(p2_t - len(subtitle)/TYPEWR_SPEED - 0.3, 0) / 0.5, 1.0)
            if enter_alpha > 0:
                p3  = 0.4+0.6*abs(math.sin(t*2.2))
                ec  = (int(110*p3*enter_alpha), int(110*p3*enter_alpha), int(125*p3*enter_alpha))
                ep_ov = canvas.copy()
                put_centered(ep_ov, "PRESS  ENTER  TO  SCAN  AGAIN",
                             FACE_H+148, font, 0.34, ec, 1, shadow=False)
                cv2.addWeighted(ep_ov, enter_alpha, canvas, 1-enter_alpha, 0, canvas)

    add_scanlines(canvas)
    return canvas


# ──────────────────────────── STATES ───────────────────────────
S_WAIT    = 0
S_ALIGN   = 1
S_FLASH   = 2
S_LOAD    = 3
S_CONFIRM = 4
S_RESULT  = 5

state              = S_WAIT
history            = deque(maxlen=SMOOTH_N)
phase_start        = 0.0
result_phase_start = 0.0    # tracks time since S_RESULT began
final_label        = "Neutral"
final_color        = (200,200,200)
final_probs        = None
last_face          = np.zeros((200,200,3), dtype=np.uint8)
snapshot_face      = None
snapshot_crop      = None
snapshot_rect      = None
last_crop          = None
last_rect          = None
t                  = 0.0
prev_time          = time.time()
enter_pressed      = False
face_miss          = 0
FACE_MISS_MAX      = 12

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam."); exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    now       = time.time()
    dt        = now - prev_time
    prev_time = now
    t        += dt

    face_result = get_face(frame)
    if face_result is not None:
        display_frame, model_crop, face_rect = face_result
    else:
        display_frame = model_crop = face_rect = None

    face_found = face_result is not None
    elapsed    = now - phase_start

    # ─── State transitions ───────────────────────────────────
    if state == S_WAIT:
        if face_found:
            state       = S_ALIGN
            phase_start = now
            last_face   = display_frame

    elif state == S_ALIGN:
        if not face_found:
            face_miss += 1
            if face_miss > FACE_MISS_MAX:
                state     = S_WAIT
                face_miss = 0
        else:
            face_miss = 0
            last_face = display_frame
            last_crop = model_crop
            last_rect = face_rect
            if elapsed >= ALIGN_SECS and last_face is not None and last_crop is not None:
                snapshot_face = last_face.copy()
                snapshot_crop = last_crop.copy()
                snapshot_rect = last_rect
                state         = S_FLASH
                phase_start   = now

    elif state == S_FLASH:
        if elapsed >= FLASH_SECS:
            state       = S_LOAD
            phase_start = now
            history.clear()

    elif state == S_LOAD:
        
        if snapshot_crop is not None and len(history) < SMOOTH_N:
            probs = predict_single(snapshot_crop)
            history.append(probs)

        progress = min(elapsed / LOAD_SECS, 1.0)

        if elapsed >= LOAD_SECS and len(history) >= 1:
            avg         = np.mean(history, axis=0)
            final_label = LABELS[np.argmax(avg)]
            final_color = COLORS[final_label]
            final_probs = avg
            top3        = np.argsort(avg)[::-1][:3]
            print(f"[RESULT] {final_label} ({avg[np.argmax(avg)]*100:.1f}%)  "
                  + " | ".join(f"{LABELS[i]} {avg[i]*100:.1f}%" for i in top3)
                  + f"  ({len(history)} real passes)")
            state       = S_CONFIRM
            phase_start = now

    elif state == S_CONFIRM:
        if elapsed >= CONFIRM_SECS:
            state              = S_RESULT
            phase_start        = now
            result_phase_start = now
            enter_pressed      = False

    elif state == S_RESULT:
        if face_found:
            last_face = display_frame
        if enter_pressed:
            state         = S_WAIT
            history.clear()
            enter_pressed = False
            final_probs   = None

    # ─── Render ─────────────────────────────────────────────
    if   state == S_WAIT:
        canvas = screen_waiting(t)
    elif state == S_ALIGN:
        canvas = screen_align(last_face, t, elapsed, last_rect if face_found else None)
    elif state == S_FLASH:
        flash_alpha = max(0.0, 1.0-elapsed/FLASH_SECS)
        canvas = screen_snapshot_flash(snapshot_face, flash_alpha, snapshot_rect)
    elif state == S_LOAD:
        progress = min(elapsed/LOAD_SECS, 1.0)
        canvas   = screen_loading(snapshot_face, t, progress, len(history))
    elif state == S_CONFIRM:
        alpha  = min(elapsed/0.28, 1.0)
        canvas = screen_confirm(snapshot_face, final_label, final_color, alpha, t)
    elif state == S_RESULT:
        result_elapsed = now - result_phase_start
        canvas = screen_result(snapshot_face, final_label, final_color, t, final_probs, result_elapsed)

    cv2.imshow("Emotion Detector  |  ENTER=scan again  Q=quit", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    if key == 13:
        if state == S_RESULT:
            enter_pressed = True

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.") 