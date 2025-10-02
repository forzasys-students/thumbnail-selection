import os
import cv2
from typing import List

def safe_label(label: str) -> str:
    """
    Convert a raw label (e.g. "Main camera center") into a safe folder name.
    This way, labels can be used as folder names on all systems.
    """
    return label.replace(" ", "_").replace("/", "_")

def is_blurry(frame, threshold: float = 200.0) -> bool:
    """
    Check if a frame is blurry using the Laplacian variance method.
    If variance < threshold → frame considered blurry.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def choose_sample_times(start_seconds: float, end_seconds: float) -> List[float]:
    """
    Decide which timestamps (in seconds) to sample frames from a shot window.

    Adaptive strategy:
    - Very short shot (< 3s): 1 frame at midpoint
    - Short shot (3–10s): 2 frames, at 1/3 and 2/3
    - Medium shot (10–30s): 3 frames, at 1/4, 1/2, 3/4
    - Long shot (> 30s): sample every 5s, maximum 10 frames
    """
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        return []

    if duration < 3.0:
        return [start_seconds + 0.5 * duration]
    if duration < 10.0:
        return [start_seconds + duration / 3.0, start_seconds + 2.0 * duration / 3.0]
    if duration < 30.0:
        return [
            start_seconds + 0.25 * duration,
            start_seconds + 0.5 * duration,
            start_seconds + 0.75 * duration,
        ]

    # Long segment: every 5 seconds, capped at 10 samples
    step = 5.0
    times = [start_seconds + i * step for i in range(int(duration // step) + 1)]
    return times[:10]

def extract_adaptive_frames(
    video_file: str,
    out_root: str,
    game_name: str,
    shot_id: str,
    start_seconds: float,
    end_seconds: float,
    label: str,
    blur_threshold: float = 200.0,
    ensure_one: bool = True
) -> List[str]:
    """
    Extract frames from a video for one shot.

    Steps:
    1. Open the video file with OpenCV.
    2. Clamp start/end times to video duration (avoid going out of range).
    3. Pick sample times adaptively based on shot length.
    4. Convert times to frame indices (fps * seconds).
    5. For each candidate frame:
       - Seek to frame
       - Read it
       - Skip if unreadable or blurry
       - Save to folder named by label
    6. If no frame was saved and ensure_one=True:
       - Always save one frame at the midpoint of the shot (fallback).

    Returns:
        List of file paths for the saved frames.
    """

    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback to 25 if fps not found
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    # Clamp times to actual video length 
    start = max(0.0, min(start_seconds, duration))
    end   = max(0.0, min(end_seconds, duration))
    if end <= start:
        cap.release()
        return []

    # Pick sample times and convert to frame indices
    sample_seconds = choose_sample_times(start, end)
    candidate_frames = sorted({int(sec * fps) for sec in sample_seconds if 0 <= sec < duration})

    # Prepare output folder 
    label_folder = os.path.join(out_root, game_name, safe_label(label))
    os.makedirs(label_folder, exist_ok=True)

    saved_paths: List[str] = []

    # Loop through candidate frame indices
    for frame_index in candidate_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue  # skip if frame not readable

        if is_blurry(frame, blur_threshold):
            continue  # skip blurry frames

        # Save frame as image file
        out_path = os.path.join(label_folder, f"{shot_id}_{frame_index}.jpg")
        cv2.imwrite(out_path, frame)
        saved_paths.append(out_path)

    # Ensure at least one frame is saved 
    if ensure_one and not saved_paths:
        midpoint = int(((start + end) / 2.0) * fps)
        if 0 <= midpoint < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
            ok, frame = cap.read()
            if ok and frame is not None:
                out_path = os.path.join(label_folder, f"{shot_id}_{midpoint}.jpg")
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)

    cap.release()
    return saved_paths
