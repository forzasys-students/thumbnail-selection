"""
Inference script for shot-type classification using trained model.

Runs inference on extracted frames and saves predictions to a CSV file.
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.training.train import get_model  # already supports multiple models

LABELS = [
    "Close-up_behind_the_goal",
    "Close-up_corner",
    "Close-up_player_or_field_referee",
    "Close-up_side_staff",
    "Main_camera_center",
    "Main_camera_left",
    "Main_camera_right",
    "Public"
]


def load_model(model_name: str, weights_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build correct architecture
    model = get_model(model_name, num_classes=num_classes)

    # ensure models exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[ERROR] Model weights not found at: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()
    model.to(device)

    print(f"[INFO] Loaded model '{model_name}' from {weights_path} on {device}")
    return model, device


def classify_all_frames(model, root_dir: str, device, output_csv="predictions.csv"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    results = []
    with torch.no_grad():
        for root, _, files in os.walk(root_dir):
            frame_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not frame_files:
                continue

            for fname in sorted(frame_files):
                frame_path = os.path.join(root, fname)
                img = Image.open(frame_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)

                logits = model(x)
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred_idx].item()

                parts = os.path.normpath(frame_path).split(os.sep)
                game_name = parts[-3] if len(parts) >= 3 else "unknown_game"
                clip_name = parts[-2] if len(parts) >= 2 else "unknown_clip"

                results.append((game_name, clip_name, frame_path, LABELS[pred_idx], conf))

    df = pd.DataFrame(results, columns=["game_name", "clip_name", "frame_path", "pred_label", "confidence"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved predictions for {len(df)} frames -> {output_csv}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference using trained model on extracted frames.")
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture: resnet18, resnet50, efficientnet, convnext, vit")
    parser.add_argument("--output_csv", type=str, default="../data/predictions/predictions_model.csv")

    args = parser.parse_args()

    model, device = load_model(args.model, args.weights, num_classes=len(LABELS))
    classify_all_frames(model, args.frames_root, device, args.output_csv)
