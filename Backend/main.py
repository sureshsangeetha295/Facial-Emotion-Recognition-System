from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

from testing import predict_emotion_from_image

app = FastAPI()

EMOTION_LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise"
]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img = Image.open(file.file).convert("RGB")
        img = np.array(img)

        # Prediction
        idx, confidence, probs = predict_emotion_from_image(img)

        emotion = EMOTION_LABELS[idx]

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "all_probabilities": probs
        }

    except Exception as e:
        return {"error": str(e)}