# extract thumbnails from a video

import os
import cv2

def extract_frames(video_file, out_dir, interval=100):
    """
    Extract frames from a video at regular intervals.
    interval=100 means save every 100th frame (~4s at 25fps).
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)

    frame_count, saved_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            out_path = os.path.join(out_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} thumbnails to {out_dir}")
