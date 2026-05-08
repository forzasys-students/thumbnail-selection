import json
import math
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DATASET_ROOT = r"C:\Users\aliaa\Desktop\3D-Shot-Posture-Dataset\3dsp\3dsp\train"
MODEL_PATH = "models/yolo/yolo26m-pose.pt"

PDJ_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ------------------------------------------------
# COCO 17 KEYPOINT MAPPING
# ------------------------------------------------
JOINT_MAP = {
    "Head": 0,
    "Left Shoulder": 5,
    "Right Shoulder": 6,
    "Left Elbow": 7,
    "Right Elbow": 8,
    "Left Wrist": 9,
    "Right Wrist": 10,
    "Left Hip": 11,
    "Right Hip": 12,
    "Left Knee": 13,
    "Right Knee": 14,
    "Left Ankle": 15,
    "Right Ankle": 16,
}


print("Loading YOLO pose model...")
model = YOLO(MODEL_PATH)


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def load_gt(json_path):
    with open(json_path) as f:
        data = json.load(f)

    kp_dict = data["keypoint_2d"]

    gt = {}

    for v in kp_dict.values():
        name = v["name"]
        x = v["x"]
        y = v["y"]

        if name in JOINT_MAP:
            gt[name] = (x, y)

    return gt


def compute_normalization(gt):
    required = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"]

    for joint in required:
        if joint not in gt:
            return None

    l_sh = gt["Left Shoulder"]
    r_sh = gt["Right Shoulder"]
    l_hip = gt["Left Hip"]
    r_hip = gt["Right Hip"]

    shoulder_center = (
        (l_sh[0] + r_sh[0]) / 2,
        (l_sh[1] + r_sh[1]) / 2,
    )

    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2,
    )

    return euclidean(shoulder_center, hip_center)


def select_person(result, img_w, img_h):
    if result.keypoints is None:
        return None

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

    center_x = img_w / 2
    center_y = img_h / 2

    best_idx = None
    best_dist = 1e9

    boxes = result.boxes.xyxy.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        d = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


# ------------------------------------------------
# Counters
# ------------------------------------------------
total = 0

correct_by_threshold = {t: 0 for t in PDJ_THRESHOLDS}

joint_total = {k: 0 for k in JOINT_MAP}
joint_correct = {k: 0 for k in JOINT_MAP}


# ------------------------------------------------
# Iterate dataset
# ------------------------------------------------
root = Path(DATASET_ROOT)
clips = sorted(root.glob("*"))

print("Clips found:", len(clips))


for clip in clips:
    img_dir = clip / "img"
    pose_dir = clip / "posture"

    if not img_dir.exists() or not pose_dir.exists():
        continue

    images = sorted(img_dir.glob("*.jpg"))

    for img_path in images:
        json_path = pose_dir / (img_path.stem + ".json")

        if not json_path.exists():
            continue

        gt = load_gt(json_path)

        # Need torso joints for normalization.
        norm = compute_normalization(gt)

        if norm is None:
            continue

        if norm < 1e-6:
            continue

        result = model.predict(
            str(img_path),
            imgsz=640,
            conf=0.25,
            verbose=False,
        )[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            continue

        img_h, img_w = result.orig_shape

        idx = select_person(result, img_w, img_h)

        if idx is None:
            continue

        pred = result.keypoints.xy[idx].cpu().numpy()

        for joint_name, coco_id in JOINT_MAP.items():
            if joint_name not in gt:
                continue

            gt_xy = gt[joint_name]
            pred_xy = pred[coco_id]

            # YOLO sometimes returns (0, 0) for missing keypoints.
            # Skip those, otherwise they will unfairly count as bad detections.
            if pred_xy[0] == 0 and pred_xy[1] == 0:
                continue

            dist = euclidean(gt_xy, pred_xy)
            norm_dist = dist / norm

            total += 1
            joint_total[joint_name] += 1

            for t in PDJ_THRESHOLDS:
                if norm_dist < t:
                    correct_by_threshold[t] += 1

            if norm_dist < 0.5:
                joint_correct[joint_name] += 1


# ------------------------------------------------
# Results
# ------------------------------------------------
print("\n==============================")
print("POSE ESTIMATION EVALUATION")
print("==============================")

if total == 0:
    print("No valid keypoints were evaluated.")
    print("Check dataset path, JSON names, keypoint names, and YOLO detections.")
    exit()

pdj_scores = []

for t in PDJ_THRESHOLDS:
    pdj = correct_by_threshold[t] / total
    pdj_scores.append(pdj)

    print(f"PDJ@{t:.1f}: {pdj:.4f}")

mean_pdj = np.mean(pdj_scores)

auc = np.trapz(pdj_scores, PDJ_THRESHOLDS) / (
    PDJ_THRESHOLDS[-1] - PDJ_THRESHOLDS[0]
)

print("\nMean PDJ:", round(mean_pdj, 4))
print("AUC:", round(auc, 4))


print("\nPer joint PDJ@0.5")

for j in JOINT_MAP:
    if joint_total[j] == 0:
        print(f"{j:15s} : no GT / no valid predictions")
        continue

    score = joint_correct[j] / joint_total[j]

    print(f"{j:15s} : {score:.3f}")


print("\nTotal evaluated keypoints:", total)
print("Done.")