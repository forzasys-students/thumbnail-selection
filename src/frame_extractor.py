import os
import cv2
from ultralytics import YOLO
import subprocess


# Load YOLO once (global)
yolo_model = YOLO("yolov8n.pt")  # pretrained on COCO, class 0 = "person"

def has_player(frame):
    """Return True if YOLO detects at least one person in the frame."""
    results = yolo_model(frame, verbose=False)
    for r in results:
        for c in r.boxes.cls:  # class IDs
            if int(c) == 0:  # 0 = "person"
                return True
    return False


def is_sharp(frame, thresh=200.0):
    """
    Return True if the frame is sharp enough, False if blurry.
    
    Args:
        thresh: higher = stricter (default 100.0 is reasonable).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()  # variance of Laplacian
    return fm > thresh


def is_closeup(frame, min_area_ratio=0.1, max_players=3):
    """
    Return True if frame looks like a close-up shot of players.
    
    Args:
        min_area_ratio: minimum fraction of the frame a bounding box should cover
        max_players: maximum number of detected players allowed
    """
    results = yolo_model(frame, verbose=False)
    h, w, _ = frame.shape
    frame_area = h * w

    player_boxes = []
    for r in results:
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls_id) == 0:  # class 0 = person
                x1, y1, x2, y2 = box.tolist()
                box_area = (x2 - x1) * (y2 - y1)
                player_boxes.append(box_area / frame_area)

    if not player_boxes:
        return False

    # Close-up = at least one big player, and not too many total
    return (max(player_boxes) > min_area_ratio) and (len(player_boxes) <= max_players)


def extract_frames(video_file, out_dir, interval=100):
    """
    Extract frames from a video at regular intervals, keeping only frames with players.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)

    frame_count, saved_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            if has_player(frame) and is_closeup(frame) and is_sharp(frame): # <- filter step
                out_path = os.path.join(out_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1

            
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} player-containing thumbnails to {out_dir}")



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


def extract_frame(video_file, out_root, game_name, shot_id, frame_idx, label):
    """
    Extract a single frame from `video_file` at index `frame_idx`
    and save it into:
        out_root / game_name / <label> / <shot_id>_<frame_idx>.jpg

    Args:
        video_file (str): path to video (half .mkv)
        out_root (str): base output directory
        game_name (str): folder name for the current game
        shot_id (str): identifier for the shot (e.g. "23_H1")
        frame_idx (int): frame index to extract
        label (str): annotation label, used as subfolder name

    Returns:
        out_path (str): path to the saved .jpg
    """
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_file}")

    # Clean label for folder naming
    safe_label = label.replace(" ", "_").replace("/", "_")

    # Output directory structure: shot_frames/game/label/
    out_dir = os.path.join(out_root, game_name, safe_label)
    os.makedirs(out_dir, exist_ok=True)

    # Write frame to disk
    out_path = os.path.join(out_dir, f"{shot_id}_{frame_idx}.jpg")
    cv2.imwrite(out_path, frame)
    return out_path

