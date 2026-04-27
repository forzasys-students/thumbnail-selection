import os
import re
import cv2
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor


def safe_label(label: str) -> str:
    """
    Convert a raw label (e.g. "Main camera center") into a safe folder name.
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


def extract_video_id_from_path(path: str) -> Optional[str]:
    """
    Try to extract a Forzasys video_id from a URL or filename.

    Handles:
      - m3u8 URL:  .../playlist.m3u8/17534:6666000:6718000/Manifest.m3u8
      - saved file: video_17534.mp4  or  video_17534_frame_00000.jpg
    """
    # Pattern 1: Forzasys m3u8 URL  /17534:start:end/
    m = re.search(r'/(\d+):\d+:\d+/', path)
    if m:
        return m.group(1)

    # Pattern 2: already-renamed file  video_17534...
    m = re.search(r'video_(\d+)', os.path.basename(path))
    if m:
        return m.group(1)

    return None


def choose_sample_times_weighted(
    start_seconds: float, end_seconds: float, sample_factor: float
) -> List[float]:
    """
    Decide which timestamps to sample frames from inside a shot.

    - sample_factor > 1.0 = denser sampling (take more frames).
    - sample_factor < 1.0 = sparser sampling (take fewer frames).
    """
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        return []

    factor = max(0.1, float(sample_factor))

    if duration < 3.0:
        count = max(1, round(1 * factor))
        if count == 1:
            return [start_seconds + 0.5 * duration]
        return [start_seconds + (i + 1) * (duration / (count + 1)) for i in range(count)]

    if duration < 10.0:
        count = max(1, round(2 * factor))
        return [start_seconds + (i + 1) * (duration / (count + 1)) for i in range(count)]

    if duration < 30.0:
        count = max(1, round(3 * factor))
        return [start_seconds + (i + 1) * (duration / (count + 1)) for i in range(count)]

    step = max(0.5, 5.0 / factor)
    cap = max(1, int(round(10 * factor)))
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
    sample_factor: float = 1.0,
    video_id: Optional[str] = None,
) -> List[str]:
    """
    Extract frames from a video for one shot.

    video_id is embedded in output filenames so the full pipeline can
    recover it without relying on the original URL being present.
    """
    # Resolve video_id if not explicitly supplied
    if not video_id:
        video_id = extract_video_id_from_path(video_file) or "unknown"

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    start = max(0.0, min(start_seconds, duration))
    end = max(0.0, min(end_seconds, duration))
    if end <= start:
        cap.release()
        return []

    sample_seconds = choose_sample_times_weighted(start, end, sample_factor)
    candidate_frames = sorted(
        {int(sec * fps) for sec in sample_seconds if 0 <= sec < duration}
    )

    label_folder = os.path.join(out_root, game_name, safe_label(label))
    os.makedirs(label_folder, exist_ok=True)

    saved_paths: List[str] = []

    for frame_index in candidate_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if is_blurry(frame, blur_threshold):
            continue

        # video_id always embedded in filename
        out_path = os.path.join(
            label_folder, f"video_{video_id}_{shot_id}_{frame_index}.jpg"
        )
        cv2.imwrite(out_path, frame)
        saved_paths.append(out_path)

    if ensure_one and not saved_paths:
        midpoint = int(((start + end) / 2.0) * fps)
        if 0 <= midpoint < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
            ok, frame = cap.read()
            if ok and frame is not None:
                out_path = os.path.join(
                    label_folder, f"video_{video_id}_{shot_id}_{midpoint}.jpg"
                )
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)

    cap.release()
    return saved_paths


def _write_frame(args):
    path, frame = args
    cv2.imwrite(path, frame)

def extract_frames_from_clip(
    video_path: str,
    output_dir: str,
    fps_target: int = 5,
    video_id: Optional[str] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if not video_id:
        video_id = extract_video_id_from_path(video_path)
    if video_id:
        print(f"[INFO] video_id={video_id} (source: arg)")
    else:
        video_id = "unknown"
        print(f"[WARN] Could not determine video_id for {video_path}. Frames will use 'unknown'.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(round(fps_video / fps_target)))

    idx = 0
    saved = 0
    write_queue = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_name = f"video_{video_id}_frame_{idx:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            write_queue.append((frame_path, frame.copy()))
            saved += 1
        idx += 1

    cap.release()

    # Write all frames in parallel — cv2.imwrite releases the GIL
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(_write_frame, write_queue)

    print(f"[INFO] Extracted {saved} frames → {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from a single video or a folder of video clips."
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to a video file OR a folder of .mp4/.mkv files.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root output directory. Subfolders are created per clip.",
    )
    parser.add_argument(
        "--fps", type=int, default=1,
        help="Target frames-per-second to extract (default: 1).",
    )
    parser.add_argument(
        "--video_id", default=None,
        help=(
            "Forzasys video asset ID to embed in frame filenames. "
            "If omitted the extractor tries to parse it from the path."
        ),
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    # Gather video files
    video_files: List[str] = []
    if os.path.isfile(input_path):
        video_files = [input_path]
    else:
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith((".mkv", ".mp4")):
                    video_files.append(os.path.join(root, f))

    print(f"[INFO] Found {len(video_files)} video file(s) under {input_path}")

    for video_path in video_files:
        clip_name = os.path.splitext(os.path.basename(video_path))[0]

        # Single-file input: frames go directly into output_root/clip_name.
        # Folder input: preserve the parent subfolder for organisation.
        if os.path.isfile(input_path):
            output_dir = os.path.join(output_root, clip_name)
        else:
            parent_folder = os.path.basename(os.path.dirname(video_path))
            output_dir = os.path.join(output_root, parent_folder, clip_name)

        os.makedirs(output_dir, exist_ok=True)

        # Resolve video_id: CLI arg wins, then try to parse from the filename
        resolved_video_id = args.video_id or extract_video_id_from_path(video_path)
        if resolved_video_id:
            print(f"[INFO] video_id={resolved_video_id} (source: {'arg' if args.video_id else 'filename'})")
        else:
            resolved_video_id = None  # extract_frames_from_clip will use 'unknown'

        print(f"[INFO] Extracting {video_path} → {output_dir}")
        try:
            extract_frames_from_clip(
                video_path,
                output_dir,
                fps_target=args.fps,
                video_id=resolved_video_id,
            )
        except Exception as e:
            print(f"[WARN] Failed to process {video_path}: {e}")