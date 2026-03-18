import os
import cv2
import argparse
import time
from insightface.app import FaceAnalysis


def run_face_detection(input_folder, output_folder, device="cuda"):

    os.makedirs(output_folder, exist_ok=True)

    print("Initializing SCRFD face detector (buffalo_s)...")

    ctx_id = 0 if device == "cuda" else -1

    # smaller model
    face_app = FaceAnalysis(name="buffalo_s")

    # smaller detection resolution
    face_app.prepare(ctx_id=ctx_id, det_size=(256,256))
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Found {len(image_files)} images")

    total_time = 0
    processed = 0

    for img_name in image_files:

        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_name} (could not read image)")
            continue

        start_time = time.time()

        faces = face_app.get(img)

        inference_time = (time.time() - start_time) * 1000
        total_time += inference_time
        processed += 1

        print(f"\n{img_name}")
        print(f"Detected faces: {len(faces)}")
        print(f"Inference time: {inference_time:.2f} ms")

        for i, face in enumerate(faces):

            bbox = face.bbox.astype(int)
            conf = face.det_score

            print(f"Face {i+1} confidence: {conf:.3f}")

            x1, y1, x2, y2 = bbox

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{conf:.2f}"

            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            img,
            f"Inference: {inference_time:.2f} ms",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)

    if processed > 0:
        avg_time = total_time / processed
        print("\nEvaluation complete.")
        print(f"Average inference time: {avg_time:.2f} ms")
    else:
        print("No valid images processed.")

    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    run_face_detection(
        args.input_folder,
        args.output_folder,
        args.device
    )