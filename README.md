# Pakistan Sign Language (PSL) Landmark Recognition

This project implements a lightweight, efficient approach to recognize **Pakistan Sign Language (PSL)** signs using **MediaPipe Hand Landmarks** and **CNN-based image classification**, based on the research idea to convert videos into static spatial trajectory images.

---

## 🧠 Core Idea

Traditional models (like I3D, C3D, TSM) are powerful but **resource-heavy**, requiring GPUs and large video datasets.

Our method:
- 🎥 Converts gesture **videos into single image trajectories** using MediaPipe.
- 🖼️ Extracts **21 hand landmarks per frame** → builds 21 binary images representing motion trajectory.
- 🧠 Trains a **CNN** on these images instead of video sequences → significantly faster & deployable.

---

## 📁 Directory Structure

```text
PSL_Landmark_Recognition
├── data
│   ├── alphabets              # Original full gesture videos (A.mp4, B.mp4, ...)
│   ├── vid1                   # First half (trimmed) of each letter video
│   ├── vid1_old               # First half (no trimming) – previously used
│   ├── vid2                   # Second half (trimmed)
│   ├── vid2_old               # Second half (no trimming)
│   ├── landmark_images        # PNG images for each landmark trajectory (trimmed)
│   │   ├── WRIST
│   │   ├── INDEX_FINGER_TIP
│   │   └── ...
│   ├── landmark_images_old    # Same as above, but from untrimmed videos
│   └── aggregated             # Composite images of top-5 landmarks
│       ├── train
│       └── test
├── code
│   ├── cnn_model.py           # CNN training on images
│   ├── combine_top5.py        # Combines top-5 landmark images into one
│   ├── split.py               # Splits each A.mp4 into A_1 and A_2
│   └── vid_to_img.py          # Converts video frames to 21 landmark images
├── my_model.keras             # Saved trained model
├── structure.txt              # Directory structure saved as text
└── top5_landmarks.txt         # Landmarks with highest accuracy
```
---

## ⚙️ How It Works

### 1. **Preprocessing**
- All `A.mp4 → Z.mp4` videos are split into two using `split.py`.
- First & second attempts per sign are stored in `vid1/` and `vid2/`.

### 2. **Optional Trimming**
To improve sign focus:
- We experimented with trimming **20% from start and end**.
- Results suggest trimmed versions improve focus on the actual sign.
- Non-trimmed (`*_old/`) videos may show higher accuracy but possibly learned irrelevant hand-raising/lowering patterns.

### 3. **Landmark Extraction**
- `vid_to_img.py` uses MediaPipe to extract 21 landmark trajectories.
- Each landmark becomes one static 128x128 image per video.

### 4. **Model Training**
- `cnn_model.py` trains a CNN per landmark.
- `top5_landmarks.txt` logs the best performing landmarks.
- `combine_top5.py` aggregates top-5 landmark images into composite samples for better accuracy (~77%).

---

## 🧪 Observations & Learnings

- **Trimming video ends** helps focus on actual sign, avoiding noisy background motion.
- **Old dataset** (no trimming) showed better accuracy, but likely learned repetitive patterns (e.g., hand raise/drop), not the signs.
- **Aggregation of top landmarks** boosts accuracy by combining richer sign-specific features.

---

## 🚀 Future Directions

1. 🔍 **Frame-level Cropping**:
   - Identify and retain only frames where sign is actively performed.
   - Discard raising/lowering frames for purer signal learning.

2. 📈 **Dataset Expansion**:
   - Add full words from PSL dictionary.
   - Retain same landmark and preprocessing pipeline for consistency.

3. 📱 **Mobile Deployment**:
   - Convert CNN model to TFLite for edge deployment on phones or tablets.

---

## ✅ Current Accuracy (Trimmed + Aggregated)
- CNN Accuracy: **~77%**
- Based on top-5 landmark combination from 26 signs (A–Z), 2 samples each.

---

## ✍️ Credits & Team Note

This work is part of our ongoing Final Year Project (FYP). The current repo reflects:
- A complete working pipeline
- Preprocessing logic & experiments
- Model performance insights

Special focus is on reproducibility, lightweight deployment, and improved understanding of sign language via efficient image-based modeling.

---
