import os
import numpy as np
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/phase6_best_model.keras"
TEST_PATH  = r"D:\Facial Emotion Detection\Real_Dataset\test"
IMG_SIZE   = (224, 224)

# ─── Load model ───────────────────────────────────────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded.\n")

# ─── Auto-read class names from test folder (same sorted() as training) ───────
CLASS_NAMES    = sorted([c for c in os.listdir(TEST_PATH)
                         if os.path.isdir(os.path.join(TEST_PATH, c))])
INDEX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
print(f"[INFO] Classes found : {CLASS_NAMES}\n")

# ─── Preprocess (matches phase6 exactly) ──────────────────────────────────────
def preprocess(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

# ─── Batch test ───────────────────────────────────────────────────────────────
correct   = 0
total     = 0
per_class = {name: {"correct": 0, "total": 0} for name in CLASS_NAMES}

print(f"[INFO] Testing on : {TEST_PATH}\n")
print(f"  {'Image':<40} {'True':<14} {'Predicted':<14} {'Conf':>6}  {'✓/✗'}")
print("  " + "-" * 82)

for true_label in CLASS_NAMES:
    class_dir = os.path.join(TEST_PATH, true_label)

    for fname in os.listdir(class_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_dir, fname)
        try:
            inp        = preprocess(img_path)
            probs      = model.predict(inp, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            pred_label = INDEX_TO_CLASS[pred_idx]
            confidence = probs[pred_idx] * 100

            is_correct = (pred_label.lower() == true_label.lower())
            if is_correct:
                correct += 1
            total += 1

            per_class[true_label]["total"]   += 1
            per_class[true_label]["correct"] += int(is_correct)

            mark = "✓" if is_correct else "✗"
            print(f"  {fname:<40} {true_label:<14} {pred_label:<14} {confidence:5.1f}%  {mark}")

        except Exception as e:
            print(f"  [ERROR] {fname} → {e}")

# ─── Summary ──────────────────────────────────────────────────────────────────
overall_acc = (correct / total * 100) if total > 0 else 0

print("\n" + "=" * 52)
print("  RESULTS PER CLASS")
print("=" * 52)
for name in CLASS_NAMES:
    c   = per_class[name]["correct"]
    t   = per_class[name]["total"]
    acc = (c / t * 100) if t > 0 else 0
    bar = "█" * int(acc / 5)
    print(f"  {name:<12}  {c:>3}/{t:<4}  {acc:5.1f}%  {bar}")

print("=" * 52)
print(f"  Overall Accuracy : {correct}/{total}  →  {overall_acc:.2f}%")
print("=" * 52)