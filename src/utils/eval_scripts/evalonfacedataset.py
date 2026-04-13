import os
import cv2
import json
import time
import argparse
from insightface.app import FaceAnalysis


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def coco_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def run_evaluation(dataset_folder, output_folder, device="cuda", iou_thresh=0.5):

    os.makedirs(output_folder, exist_ok=True)

    # Load COCO GT
    coco_path = os.path.join(dataset_folder, "result.json")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images_info = coco["images"]
    annotations = coco["annotations"]

    # Map image_id → GT boxes
    gt_dict = {}
    for ann in annotations:
        img_id = ann["image_id"]
        bbox = coco_to_xyxy(ann["bbox"])
        gt_dict.setdefault(img_id, []).append(bbox)

    print("Initializing SCRFD...")
    ctx_id = 0 if device == "cuda" else -1
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=ctx_id, det_size=(256, 256))

    TP = 0
    FP = 0
    FN = 0

    total_time = 0

    for img_info in images_info:

        img_id = img_info["id"]
        file_name = img_info["file_name"].replace("\\", "/")

        img_path = os.path.join(dataset_folder, file_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}")
            continue

        start = time.time()
        faces = face_app.get(img)
        total_time += (time.time() - start)

        preds = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            preds.append([x1, y1, x2, y2])

        gts = gt_dict.get(img_id, [])

        matched_gt = set()

        # Match predictions
        for p in preds:
            match_found = False
            for i, g in enumerate(gts):
                if i in matched_gt:
                    continue
                if iou(p, g) >= iou_thresh:
                    TP += 1
                    matched_gt.add(i)
                    match_found = True
                    break
            if not match_found:
                FP += 1

        # Missed GTs
        FN += (len(gts) - len(matched_gt))

        # Optional: save visualization
        for g in gts:
            x1, y1, x2, y2 = map(int, g)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # GT = blue

        for p in preds:
            x1, y1, x2, y2 = map(int, p)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Pred = green

        out_path = os.path.join(output_folder, os.path.basename(file_name))
        cv2.imwrite(out_path, img)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    print("\n===== RESULTS =====")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Avg inference time: {(total_time / len(images_info)) * 1000:.2f} ms")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    run_evaluation(
        args.dataset_folder,
        args.output_folder,
        args.device
    )