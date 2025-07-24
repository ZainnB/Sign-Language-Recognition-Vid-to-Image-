import os
import cv2
import numpy as np

# === Config ===
IMG_SIZE = 128
TOP5_FILE = "top5_landmarks.txt"
TRAIN_ROOT = "data/landmark_images/train"
TEST_ROOT = "data/landmark_images/test"

OUTPUT_TRAIN = "data/aggregated/train"
OUTPUT_TEST = "data/aggregated/test"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)

# === Load Top 5 Landmark Names ===
with open(TOP5_FILE, 'r') as f:
    top5_landmarks = [line.strip() for line in f.readlines()]

print(f"Top-5 Landmarks: {top5_landmarks}")

# === Combine Images ===
def combine_images(sample_name, landmark_paths):
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    for lm_path in landmark_paths:
        if not os.path.exists(lm_path):
            continue
        img = cv2.imread(lm_path, cv2.IMREAD_GRAYSCALE)
        canvas = cv2.bitwise_or(canvas, img)

    return canvas

# === Process One Set ===
def process_set(landmark_root, output_dir):
    samples = sorted(os.listdir(os.path.join(landmark_root, top5_landmarks[0])))

    for filename in samples:
        landmark_imgs = []
        for lm in top5_landmarks:
            landmark_imgs.append(os.path.join(landmark_root, lm, filename))

        combined = combine_images(filename, landmark_imgs)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, combined)

# === Run for both train & test ===
print("Combining training set...")
process_set(TRAIN_ROOT, OUTPUT_TRAIN)

print("Combining testing set...")
process_set(TEST_ROOT, OUTPUT_TEST)

print("Done. Combined images saved in 'data/aggregated/train' and 'data/aggregated/test'")
