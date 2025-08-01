import cv2
import os

# === Input & Output Paths ===
input_folder = os.path.join('downloads')
output_folder_1 = os.path.join('PSL_Dictionary', 'vid1')
output_folder_2 = os.path.join('PSL_Dictionary', 'vid2')

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
        print(f"❌ Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Split point
    mid = total_frames // 2
    trim = int(mid * 0.2)

    h1_start, h1_end = 0 + trim, mid - trim
    h2_start, h2_end = mid + trim, total_frames - trim

    out1 = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))

    trim_and_write(cap, h1_start, h1_end, out1)
    trim_and_write(cap, h2_start, h2_end, out2)

    cap.release()
    out1.release()
    out2.release()
    print(f"✅ Trimmed Split Done: {os.path.basename(video_path)}")

# === Automatically loop over all .mp4 files ===
for file in os.listdir(input_folder):
    if file.endswith(".mp4"):
        file_name = os.path.splitext(file)[0]  # Remove extension
        video_path = os.path.join(input_folder, file)
        output1 = os.path.join(output_folder_1, f"{file_name}_1.mp4")
        output2 = os.path.join(output_folder_2, f"{file_name}_2.mp4")
        split_and_trim_video(video_path, output1, output2)
