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


def choose_sample_times_weighted(start_seconds: float, end_seconds: float, sample_factor: float) -> List[float]:
    """
    Decide which timestamps to sample frames from inside a shot.

    - sample_factor > 1.0 = denser sampling (take more frames).
    - sample_factor < 1.0 = sparser sampling (take fewer frames).

    Rules:
      - Shots <3s: normally 1 frame → round(1*factor).
      - Shots 3–10s: normally 2 frames → round(2*factor).
      - Shots 10–30s: normally 3 frames → round(3*factor).
      - Shots >30s: take frames every ~5s, but shrink/grow step size by factor.
    """
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        return []

    factor = max(0.1, float(sample_factor))  # safety: no zero or negative

    # Very short shots (<3s): usually just 1 frame
    if duration < 3.0:
        count = max(1, round(1 * factor))
        if count == 1:
            return [start_seconds + 0.5 * duration]  # middle of shot
        # evenly spread multiple frames across the interval
        return [start_seconds + (i+1) * (duration / (count + 1)) for i in range(count)]

    # Short shots (3–10s): normally 2 frames → scale by factor
    if duration < 10.0:
        count = max(1, round(2 * factor))
        return [start_seconds + (i+1) * (duration / (count + 1)) for i in range(count)]

    # Medium shots (10–30s): normally 3 frames → scale by factor
    if duration < 30.0:
        count = max(1, round(3 * factor))
        return [start_seconds + (i+1) * (duration / (count + 1)) for i in range(count)]

    # Long shots (>30s): sample every ~5s, but adjust by factor
    step = 5.0 / factor
    step = max(0.5, step)  # don’t make step ridiculously small
    cap = max(1, int(round(10 * factor)))  # at most ~10*factor frames
    times = [start_seconds + i * step for i in range(int(duration // step) + 1)]
    return times[:cap]    


def extract_adaptive_frames(
    video_file: str,
    out_root: str,
    game_name: str,
    shot_id: str,
    start_seconds: float,
    end_seconds: float,
    label: str,
    blur_threshold: float = 200.0,
    ensure_one: bool = True,
    sample_factor: float = 1.0
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
    
    # Pick sample times (weighted by label factor) and convert to frame indices
    sample_seconds = choose_sample_times_weighted(start, end, sample_factor)
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


def extract_frames_from_clip(video_path, output_dir, fps_target=5):
    """
    Extract frames from a single video clip at a fixed temporal rate.
    Args:
        video_path (str): path to the video (mp4, mkv, etc.)
        output_dir (str): directory to save frames
        fps_target (int): frames per second to save (1 = one frame per second)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 25  # fallback if FPS not detected
    frame_interval = int(round(fps_video / fps_target))

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames (~{fps_target} FPS) from {video_path} -> {output_dir}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Extract frames from single or multiple videos.")
    parser.add_argument("--input_path", required=True,
                        help="Path to a single video file or folder containing multiple game clips.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory where frames are saved (subfolders per game).")
    parser.add_argument("--fps", type=int, default=1,
                        help="Target FPS for frame extraction (default: 1).")
    args = parser.parse_args()

    input_path = args.input_path
    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    # Gather video files (both MKV and MP4)
    video_files = []
    if os.path.isfile(input_path):
        video_files = [input_path]
    else:
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith((".mkv", ".mp4")):
                    video_files.append(os.path.join(root, f))

    print(f"[INFO] Found {len(video_files)} video files under {input_path}")

    for video_path in video_files:
        # Match folder name (e.g. "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley")
        parent_folder = os.path.basename(os.path.dirname(video_path))
        clip_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create output directory for each game and clip
        output_dir = os.path.join(output_root, parent_folder, clip_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[INFO] Extracting {video_path} -> {output_dir}")
        try:
            extract_frames_from_clip(video_path, output_dir, fps_target=args.fps)
        except Exception as e:
            print(f"[WARN] Failed to process {video_path}: {e}")

