import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# ───────── CONFIG ─────────
MODEL_PATH = r"D:\Facial Emotion Detection\Backend\Models\phase2_best_model.keras"
IMG_SIZE = (224, 224)
NUM_PASSES = 5

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise"
]

# ───────── LOAD MODEL (ONLY ONCE) ─────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded")

# ───────── LOAD FACE DETECTOR ─────────
detector = MTCNN()


# ───────── MAIN FUNCTION ─────────
def predict_emotion_from_image(img):
    """
    Input: img (numpy array - BGR or RGB)
    Output: idx, confidence, probabilities
    """

    # If image comes from PIL → already RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = img
    else:
        raise ValueError("Invalid image format")

    # ───────── FACE DETECTION ─────────
    faces = detector.detect_faces(img_rgb)

    if not faces:
        face = img_rgb
    else:
        x, y, w, h = faces[0]["box"]

        x = max(0, x)
        y = max(0, y)

        face = img_rgb[y:y+h, x:x+w]

    # ───────── PREPROCESS ─────────
    face = cv2.resize(face, IMG_SIZE)
    face = face.astype("float32")

    # MobileNetV2 normalization
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

    face = np.expand_dims(face, axis=0)

    # ───────── MULTI-PASS PREDICTION ─────────
    predictions = []

    for _ in range(NUM_PASSES):
        p = model.predict(face, verbose=0)[0]
        predictions.append(p)

    predictions = np.array(predictions)
    avg_pred = np.mean(predictions, axis=0)

    # ───────── RESULT ─────────
    idx = int(np.argmax(avg_pred))
    confidence = float(avg_pred[idx] * 100)

    return idx, confidence, avg_pred.tolist()