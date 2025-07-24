import cv2
import os
import string

# === Input & Output Paths ===
input_folder = os.path.join('data', 'alphabets')
output_folder_1 = os.path.join('data', 'vid1')
output_folder_2 = os.path.join('data', 'vid2')

# Ensure output folders exist
os.makedirs(output_folder_1, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

def trim_and_write(cap, start_idx, end_idx, writer):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for i in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

def split_and_trim_video(video_path, output_path1, output_path2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Split point
    mid = total_frames // 2
    half1_start, half1_end = 0, mid
    half2_start, half2_end = mid, total_frames

    # Trim 30% from start of half 1 and end of half 2
    # and trim 20% from start of half 2 and end of half 1
    trim1 = int((half1_end - half1_start) * 0.4)
    trim2 = int((half1_end - half1_start) * 0.2)
    trim3 = int((half2_end - half2_start) * 0.3)

    h1_start = half1_start + trim1
    h1_end = half1_end - trim2

    h2_start = half2_start + trim3
    h2_end = half2_end - trim3

    out1 = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))

    trim_and_write(cap, h1_start, h1_end, out1)
    trim_and_write(cap, h2_start, h2_end, out2)

    cap.release()
    out1.release()
    out2.release()
    print(f"✅ Trimmed Split Done: {os.path.basename(video_path)}")

# === Loop over A–Z videos ===
for letter in string.ascii_uppercase:
    file_name = f"{letter}.mp4"
    video_path = os.path.join(input_folder, file_name)
    output1 = os.path.join(output_folder_1, f"{letter}_1.mp4")
    output2 = os.path.join(output_folder_2, f"{letter}_2.mp4")
    split_and_trim_video(video_path, output1, output2)
