import os
import cv2
from ultralytics import YOLO

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
