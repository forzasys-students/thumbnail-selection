"""
Inference script for shot-type classification using trained ResNet-50.

This version:
 - Recursively scans `frames_inference/` (organized by game/clip)
 - Runs inference on all frames
 - Saves results to a structured CSV for later processing (close-up segment detection)
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.training.train import get_model

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


def load_model(weights_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("resnet50", num_classes=num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    print(f"[INFO] Loaded ResNet-50 from {weights_path} on {device}")
    return model, device


def classify_all_frames(model, root_dir: str, device, output_csv="predictions.csv"):
    """
    Recursively classify frames under root_dir.
    Each prediction row includes: game_name, clip_name, frame_path, label, confidence.
    """
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

                # Parse game + clip names
                parts = os.path.normpath(frame_path).split(os.sep)
                game_name = parts[-3] if len(parts) >= 3 else "unknown_game"
                clip_name = parts[-2] if len(parts) >= 2 else "unknown_clip"

                results.append((game_name, clip_name, frame_path, LABELS[pred_idx], conf))

    df = pd.DataFrame(results, columns=["game_name", "clip_name", "frame_path", "pred_label", "confidence"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved predictions for {len(df)} frames → {output_csv}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference using trained ResNet-50 on extracted frames.")
    parser.add_argument("--frames_root", type=str, required=True,
                        help="Root folder containing per-game frame folders (e.g. data/frames_inference/)")
    parser.add_argument("--weights", type=str, default="../checkpoints/resnet50_best.pt",
                        help="Path to trained ResNet-50 checkpoint.")
    parser.add_argument("--output_csv", type=str, default="../data/predictions/predictions_resnet50.csv",
                        help="Path to save the prediction CSV.")
    args = parser.parse_args()

    model, device = load_model(args.weights, num_classes=len(LABELS))
    classify_all_frames(model, args.frames_root, device, args.output_csv)
