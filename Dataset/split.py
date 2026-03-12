import os
import shutil
import random
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"D:\Facial Emotion Detection\Real_Dataset"
VAL_SPLIT    = 0.20
SEED         = 42

EMOTION_MAP = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral"
}

random.seed(SEED)

# ── Helper ────────────────────────────────────────────────────────────────────
def create_folder(path):
    os.makedirs(path, exist_ok=True)

# ── Step 1: Rename train/ numbered folders → emotion-named folders ────────────
def rename_train():
    train_path = os.path.join(DATASET_PATH, "train")

    for folder in sorted(os.listdir(train_path)):
        if folder not in EMOTION_MAP:
            continue
        src = os.path.join(train_path, folder)
        dst = os.path.join(train_path, EMOTION_MAP[folder])
        os.rename(src, dst)
        print(f"  Renamed train/{folder}/ → train/{EMOTION_MAP[folder]}/")

# ── Step 2: Split train/ → train/ (80%) + val/ (20%) ─────────────────────────
def split_train():
    train_path = os.path.join(DATASET_PATH, "train")
    val_path   = os.path.join(DATASET_PATH, "val")

    stats = defaultdict(lambda: {"train": 0, "val": 0})

    for emotion in sorted(os.listdir(train_path)):
        emotion_path = os.path.join(train_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        images = [f for f in os.listdir(emotion_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(images)

        val_count    = int(len(images) * VAL_SPLIT)
        val_images   = images[:val_count]
        train_images = images[val_count:]

        # create val subfolder
        val_emotion_dir = os.path.join(val_path, emotion)
        create_folder(val_emotion_dir)

        # move val images out of train
        for fname in val_images:
            shutil.move(os.path.join(emotion_path, fname),
                        os.path.join(val_emotion_dir, fname))

        stats[emotion]["train"] = len(train_images)
        stats[emotion]["val"]   = len(val_images)

    return stats

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Step 1
    print(f"\n{'='*52}")
    print(f"  STEP 1: Renaming train/ numbered → emotion-named")
    print(f"{'='*52}")
    rename_train()

    # Step 2
    print(f"\n{'='*52}")
    print(f"  STEP 2: Splitting train/ → train/ (80%) + val/ (20%)")
    print(f"{'='*52}")
    stats = split_train()

    # Summary
    print(f"\n{'='*52}")
    print(f"  SPLIT SUMMARY")
    print(f"{'='*52}")
    print(f"  {'Emotion':<12} {'Train (80%)':>12} {'Val (20%)':>11}")
    print(f"  {'-'*38}")
    total_train = total_val = 0
    for emotion, counts in stats.items():
        print(f"  {emotion:<12} {counts['train']:>12} {counts['val']:>11}")
        total_train += counts["train"]
        total_val   += counts["val"]
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':<12} {total_train:>12} {total_val:>11}")
    print(f"{'='*52}")

    print(f"\n  Final structure:")
    print(f"  Real_Dataset/")
    print(f"  ├── train/   → emotion-named subfolders (80%)")
    print(f"  ├── val/     → emotion-named subfolders (20%)")
    print(f"  └── test/    → untouched")
    print()

if __name__ == "__main__":
    main()