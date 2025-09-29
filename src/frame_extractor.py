import os
import cv2
from ultralytics import YOLO
import subprocess

# ---------- helpers ----------

def is_blurry(frame, thresh=200.0) -> bool:
    """Return True if the frame is blurry (variance of Laplacian below threshold)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < thresh

def is_logo_label(label: str) -> bool:
    """Return True if this shot label is a 'Logo' shot (skip entirely)."""
    return label.strip().lower() == "logo"

def safe_label(label: str) -> str:
    """Filesystem-safe label folder name (raw JSON label, not grouped)."""
    return label.replace(" ", "_").replace("/", "_")



# Mapping of labels to broader categories (not applied in current pipeline,
# but useful if you want to collapse classes later).
LABEL_MAP = {
    "Main camera left": "Wide",
    "Main camera center": "Wide",
    "Main camera right": "Wide",
    "Close-up player or field referee": "CloseUp",
    "Close-up side staff": "CloseUp",
    "Close-up behind the goal": "CloseUp",
    "Bench": "Outer",
    "Coach": "Outer",
    "Public": "Outer",
    "Replay": "Replay",
    "Logo": "Replay",
}


# ---------- core ----------

def extract_frames_from_shot_seconds(
    video_file: str,
    out_root: str,
    game_name: str,
    shot_id: str,
    start_sec: float,
    end_sec: float,
    label: str,
    step_sec: float = 3.0,
    edge_trim_sec: float = 1.0,
    blur_thresh: float = 200.0,
) -> list[str]:
    """
    Extract multiple frames for one shot using second-based boundaries.

    - Uses start_sec/end_sec (seconds from the half start)
    - Skips first and last `edge_trim_sec`
    - Samples every `step_sec` seconds
    - Skips 'Logo' shots
    - Drops blurry frames via Laplacian variance

    Returns a list of saved file paths.
    """
    if is_logo_label(label):
        return []

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # SoccerNet halves are 25fps; fallback if reader lies

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Trim edges to avoid cut artifacts
    start_f = int((start_sec + edge_trim_sec) * fps)
    end_f   = int((end_sec   - edge_trim_sec) * fps)

    # Clamp to video bounds
    start_f = max(0, min(start_f, total_frames - 1))
    end_f   = max(0, min(end_f,   total_frames - 1))

    if end_f <= start_f:
        cap.release()
        return []

    step = max(1, int(step_sec * fps))

    # Output dir: raw JSON label (no grouping)
    label_dir = os.path.join(out_root, game_name, safe_label(label))
    os.makedirs(label_dir, exist_ok=True)

    saved = []
    for fidx in range(start_f, end_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if is_blurry(frame, thresh=blur_thresh):
            continue

        out_path = os.path.join(label_dir, f"{shot_id}_{fidx}.jpg")
        cv2.imwrite(out_path, frame)
        saved.append(out_path)

    cap.release()
    return saved