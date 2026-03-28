import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Suppress noisy TF warnings
tf.get_logger().setLevel("ERROR")

# CONFIG 
MODEL_PATH = r"D:\EMOTION-ANALYSIS\Backend\Models\phase2_best_model.keras"
IMG_SIZE   = (224, 224)

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

# LAZY SINGLETONS 
_model    = None
_detector = None


def get_model():
    global _model
    if _model is None:
        print("[EmotionAI] Loading model...")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("[EmotionAI] Model loaded OK")
    return _model


def get_detector():
    global _detector
    if _detector is None:
        print("[EmotionAI] Initialising MTCNN detector...")
        _detector = MTCNN()
        print("[EmotionAI] MTCNN ready")
    return _detector


# MAIN FUNCTION 

def predict_emotion_from_image(img: np.ndarray):
    """
    Parameters
    ----------
    img : np.ndarray  — RGB image (H x W x 3)

    Returns
    -------
    idx        : int   — index into CLASS_NAMES
    confidence : float — percentage value (0–100)
    avg_pred   : list  — probability list, one value per class
    """

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    # Face detection
    detector = get_detector()
    faces    = detector.detect_faces(img)

    if faces:
        x, y, w, h = faces[0]["box"]
        x, y = max(0, x), max(0, y)
        face = img[y : y + h, x : x + w]
        if face.size == 0:
            face = img
    else:
        face = img  # No face found — use full frame

    # Preprocessing 
    face = cv2.resize(face, IMG_SIZE)
    face = face.astype("float32")
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    face = np.expand_dims(face, axis=0)  # (1, 224, 224, 3)

    # Inference 
    model    = get_model()
    avg_pred = model.predict(face, verbose=0)[0]

    # Result 
    idx        = int(np.argmax(avg_pred))
    confidence = float(avg_pred[idx] * 100)  # 0–100 percentage

    return idx, confidence, avg_pred.tolist()