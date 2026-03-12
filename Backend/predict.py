import cv2
import numpy as np
import tensorflow as tf
import os

#Config 
MODEL_PATH  = "models/phase2_best_model.keras"
IMG_SIZE    = (224, 224)

#Change this to any image path you want to test
IMAGE_PATH  = r"C:\Users\sures\Downloads\images (3).jpeg"

# Must match sorted() folder names from your training dataset exactly
CLASS_NAMES    = sorted(["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"])
INDEX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
print(f"[INFO] Classes : {CLASS_NAMES}")

#Load model 
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Ready.\n")

#Load & preprocess (matches phase6 exactly) 
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE).astype(np.float32)
img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
img = np.expand_dims(img, axis=0)

#Predict 
probs      = model.predict(img, verbose=0)[0]
pred_idx   = int(np.argmax(probs))
pred_label = INDEX_TO_CLASS[pred_idx]
confidence = probs[pred_idx] * 100

#Print results 
print(f"  Emotion    : {pred_label}")
print(f"  Confidence : {confidence:.2f}%")
print("\n  All scores :")
for i, prob in enumerate(probs):
    bar  = "█" * int(prob * 40)
    mark = " ← predicted" if i == pred_idx else ""
    print(f"    {INDEX_TO_CLASS[i]:<12} {prob*100:5.1f}%  {bar}{mark}")

#Draw & show 
label_text = f"{pred_label}  {confidence:.1f}%"
cv2.putText(frame, label_text, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.imshow("Prediction", frame)
cv2.imwrite("result.jpg", frame)
print("\n  [Saved] result.jpg")
print("  [Press any key to close]")
cv2.waitKey(0)
cv2.destroyAllWindows()