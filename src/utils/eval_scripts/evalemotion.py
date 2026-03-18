import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd

from hsemotion.facial_emotions import HSEmotionRecognizer
from insightface.app import FaceAnalysis


def run_emotion_detection(input_folder, output_folder, save_csv=None):

    os.makedirs(output_folder, exist_ok=True)

    print("Initializing models...")

    # ------------------------
    # Face detector (same as pipeline)
    # ------------------------
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640,640))

    # ------------------------
    # Emotion model
    # ------------------------
    emotion_model = HSEmotionRecognizer(
        model_name="enet_b0_8_best_afew",
        device="cuda"
    )

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg",".png",".jpeg"))
    ]

    print(f"Found {len(image_files)} images")

    results = []
    total_time = 0

    emotion_index = {
        "Anger":0,
        "Contempt":1,
        "Disgust":2,
        "Fear":3,
        "Happiness":4,
        "Neutral":5,
        "Sadness":6,
        "Surprise":7
    }

    expressive = ["Happiness","Surprise","Anger","Fear","Disgust","Contempt","Sadness"]

    for img_name in image_files:

        img_path = os.path.join(input_folder,img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        start = time.time()

        # ------------------------
        # Face detection
        # ------------------------
        faces = face_app.get(img)

        if not faces:
            continue

        h,w = img.shape[:2]

        # sort by area (same as pipeline)
        def area(f):
            x1,y1,x2,y2 = map(float,f.bbox.tolist())
            return (x2-x1)*(y2-y1)

        faces = sorted(faces,key=area,reverse=True)[:5]

        best_emotion = "none"
        best_intensity = 0

        for face in faces:

            x1,y1,x2,y2 = map(int,face.bbox.tolist())

            x1=max(0,x1)
            y1=max(0,y1)
            x2=min(w-1,x2)
            y2=min(h-1,y2)

            crop = img[y1:y2,x1:x2]

            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)

            emotion, scores = emotion_model.predict_emotions(crop_rgb)

            # expressive emotion intensity
            intensity = max(float(scores[emotion_index[e]]) for e in expressive)

            face_area = float((x2-x1)*(y2-y1))/(w*h)
            area_weight = min(face_area/0.25,1.0)

            intensity *= area_weight

            if intensity > best_intensity:
                best_intensity = intensity
                best_emotion = emotion

            # draw bbox
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        inference_time = (time.time()-start)*1000
        total_time += inference_time

        label = f"{best_emotion} ({best_intensity:.2f})"

        cv2.putText(
            img,
            label,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        print(f"{img_name} → {best_emotion} | intensity {best_intensity:.3f} | {inference_time:.2f} ms")

        cv2.imwrite(os.path.join(output_folder,img_name),img)

        results.append({
            "image":img_name,
            "emotion":best_emotion,
            "intensity":float(best_intensity),
            "inference_ms":inference_time
        })

    avg_time = total_time/len(results)

    print("\nEvaluation complete")
    print(f"Average inference time: {avg_time:.2f} ms")

    if save_csv:
        df = pd.DataFrame(results)
        df.to_csv(save_csv,index=False)
        print(f"Saved CSV → {save_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder",required=True)
    parser.add_argument("--output_folder",required=True)
    parser.add_argument("--save_csv",default=None)

    args = parser.parse_args()

    run_emotion_detection(
        args.input_folder,
        args.output_folder,
        args.save_csv
    )