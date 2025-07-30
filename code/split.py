import cv2
import os
import string

# === Input & Output Paths ===
input_folder = os.path.join('data', 'PSL_Dictionary')
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

    # Trim 20% from start and ends of both halves
    trim = int((half1_end - half1_start) * 0.2)

    h1_start = half1_start + trim
    h1_end = half1_end - trim

    h2_start = half2_start + trim
    h2_end = half2_end - trim

    out1 = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))

    trim_and_write(cap, h1_start, h1_end, out1)
    trim_and_write(cap, h2_start, h2_end, out2)

    cap.release()
    out1.release()
    out2.release()
    print(f"✅ Trimmed Split Done: {os.path.basename(video_path)}")

# === Loop over A–Z videos ===

wordList=[
    "10", "100", "50", "Abnormal", "Absolutely", "Afraid", "Almost", "Ancient", "Another", "Any", "Because", "Both", 
    "Brain", "Children", "Come", "Continuously", "Do", "Dry", "Elevator", "Empty", "Eye", "Father", "Few", "From", 
    "Go", "He", "Heart", "Hear", "However", "I", "Jeep", "Knock", "Lakh", "Literally", "Mehr", "Mine", "Mother", 
    "Mouth", "Move", "Near", "Off", "Often", "One", "Our", "Outdoors", "Outside", "Parents", "Request", "Roundabout", 
    "Say", "She", "Sister", "So_(Accentuator)", "So_(In_Order_To)", "Some", "Soon", "Subsequent", "Subway", 
    "Sufficient", "There", "This", "Thoughtful", "Tongue", "Trust", "Truthful", "Universal", "Up", "Upward", 
    "Usually", "Walk", "Warm", "We", "Weak", "Without", "Woman", "Work", "Worthy", "Yellow_Light", "You"
]

for words in wordList:
    file_name = f"{words}.mp4"
    video_path = os.path.join(input_folder, file_name)
    output1 = os.path.join(output_folder_1, f"{words}_1.mp4")
    output2 = os.path.join(output_folder_2, f"{words}_2.mp4")
    split_and_trim_video(video_path, output1, output2)
