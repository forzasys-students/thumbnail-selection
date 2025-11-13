import os
import cv2
import pandas as pd
from tqdm import tqdm

def laplacian_variance(img_path: str) -> float:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def detect_faces(img_path: str) -> int:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(img_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces)

def extract_frame_index(path: str) -> int:
    """Extract integer index from filenames like frame_00250.jpg."""
    name = os.path.basename(path)
    return int("".join(c for c in name if c.isdigit()))

def select_keyframes(pred_csv, seg_csv, output_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_preds = pd.read_csv(pred_csv)
    df_segs = pd.read_csv(seg_csv)

    df_preds["frame_index"] = df_preds["frame_path"].apply(extract_frame_index)

    keyframes = []

    for _, seg in tqdm(df_segs.iterrows(), total=len(df_segs), desc="Selecting keyframes"):
        seg_id = seg["segment_id"]
        start_f = extract_frame_index(seg["start_frame"])
        end_f = extract_frame_index(seg["end_frame"])

        # Frames belonging to this segment
        seg_frames = df_preds[
            (df_preds["frame_index"] >= start_f) &
            (df_preds["frame_index"] <= end_f)
        ]

        if len(seg_frames) == 0:
            continue

        best_score = -1
        best_path = None
        best_sharp = 0
        best_faces = 0

        for _, row in seg_frames.iterrows():
            path = row["frame_path"]

            sharp = laplacian_variance(path)
            faces = detect_faces(path)

            score = sharp + faces * 50  # reward visible faces

            if score > best_score:
                best_score = score
                best_path = path
                best_sharp = sharp
                best_faces = faces

        # Save the keyframe image to output_dir
        out_path = os.path.join(output_dir, f"segment_{seg_id:03d}.jpg")
        img = cv2.imread(best_path)
        if img is not None:
            cv2.imwrite(out_path, img)

        keyframes.append({
            "segment_id": seg_id,
            "keyframe_path": out_path,
            "sharpness": round(best_sharp, 2),
            "faces": best_faces
        })

    df_out = pd.DataFrame(keyframes)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Saved {len(df_out)} keyframes → {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--seg_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    select_keyframes(
        args.pred_csv,
        args.seg_csv,
        args.output_csv,
        args.output_dir
    )
