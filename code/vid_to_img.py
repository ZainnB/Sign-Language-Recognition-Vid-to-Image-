import cv2
import mediapipe as mp
import numpy as np
import os
import string

# === Constants ===
IMG_WIDTH = 128
IMG_HEIGHT = 128
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# === MediaPipe Hands Init ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# === Extraction Function ===
def extract_landmark_images(video_path, sign_name, mode, output_base):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {video_path}")
        return

    canvases = [np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) for _ in range(21)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            for i, landmark in enumerate(lm):
                xi = int(landmark.x * IMG_WIDTH)
                yi = int(landmark.y * IMG_HEIGHT)
                if 0 <= xi < IMG_WIDTH and 0 <= yi < IMG_HEIGHT:
                    canvases[i][yi, xi] = 255

    cap.release()

    # Save 21 images in corresponding landmark folder
    for i in range(21):
        lm_name = LANDMARK_NAMES[i]
        save_dir = os.path.join(output_base, mode, lm_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{sign_name}.png")
        cv2.imwrite(save_path, canvases[i])

    print(f"✅ Saved 21 {mode} images for {sign_name}")

# === Batch Runner ===
def run_extraction_pipeline():
    output_base = 'data/landmark_images'
    for letter in string.ascii_uppercase:
        # First attempt → training
        video_train = os.path.join('data', 'vid1', f"{letter}_1.mp4")
        extract_landmark_images(video_train, letter, mode='train', output_base=output_base)

        # Second attempt → testing
        video_test = os.path.join('data', 'vid2', f"{letter}_2.mp4")
        extract_landmark_images(video_test, letter, mode='test', output_base=output_base)

# === Run it ===
run_extraction_pipeline()
hands.close()
