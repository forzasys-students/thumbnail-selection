"""
Evaluation script for manually collected test dataset.

This script loads a trained model (ResNet18, ResNet50, ViT, EfficientNet, or ConvNeXt),
runs it on a manually created test set (one folder per class),
and computes detailed evaluation metrics:
- Precision, Recall, F1-score (per-class and macro)
- Overall accuracy
- Confusion Matrix
- Average inference speed (ms per frame) — measured correctly using CUDA events
- Precision-Recall and F1-Recall curves

Inference time methodology:
  - GPU warmup (10 batches) is performed before timing begins, to exclude
    CUDA initialization overhead from the measurement.
  - Timing is measured using torch.cuda.Event (GPU) or time.perf_counter (CPU),
    isolating only the forward pass. DataLoader I/O and .to(device) transfers
    are explicitly excluded.
  - torch.cuda.synchronize() is called before and after each timed forward pass
    to ensure the GPU has finished before the timer stops.
  - The reported value is: total_forward_pass_time_ms / total_images_timed
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../evaluation")
CONF_MATRIX_DIR = os.path.join(EVAL_DIR, "confusion_matrix")
GRAPH_DIR = os.path.join(EVAL_DIR, "graph_curve")
METRICS_DIR = os.path.join(EVAL_DIR, "training_metrics")

os.makedirs(CONF_MATRIX_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

WARMUP_BATCHES = 10  # Number of batches to run before timing starts


def get_model(model_name, num_classes, weights_path):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "convnext":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([
            cls for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls)) and any(
                fname.lower().endswith((".jpg", ".jpeg", ".png"))
                for fname in os.listdir(os.path.join(root_dir, cls))
            )
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_path, file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def measure_inference_time(model, loader, device, warmup_batches=WARMUP_BATCHES):
    """
    Measure average per-frame inference time (forward pass only).

    Strategy:
      1. Run `warmup_batches` batches through the model without timing.
         This ensures CUDA kernels are compiled and GPU memory is allocated,
         so the first real batch is not penalised by initialisation overhead.
      2. For each remaining batch, move data to device (not timed), then
         time only the forward pass using CUDA events (GPU) or
         time.perf_counter (CPU).
      3. On GPU, torch.cuda.synchronize() is called before starting and
         after stopping each event pair, ensuring the timer only captures
         completed GPU work.

    Returns:
        avg_ms_per_frame (float): mean forward-pass time in milliseconds per image.
    """
    model.eval()
    use_cuda = (device.type == "cuda")

    # --- Warmup pass (not timed) ---
    print(f"  Running {warmup_batches} warmup batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            images = images.to(device)
            _ = model(images)
    if use_cuda:
        torch.cuda.synchronize()

    # --- Timed pass ---
    total_time_ms = 0.0
    total_images = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)           # data transfer NOT timed
            batch_size = images.size(0)

            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()         # wait for any pending GPU work
                start_event.record()
                _ = model(images)
                end_event.record()
                torch.cuda.synchronize()         # wait for forward pass to finish
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                # CPU fallback: perf_counter is higher resolution than time.time()
                t0 = time.perf_counter()
                _ = model(images)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

            total_time_ms += elapsed_ms
            total_images += batch_size

    avg_ms_per_frame = total_time_ms / total_images
    return avg_ms_per_frame


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on unique test set.")
    parser.add_argument("--data_root", type=str,
                        default="/fp/homes01/u01/ec-aliaana/data/unique_test_set_v2",
                        help="Path to manually collected test dataset.")
    parser.add_argument("--model_name", type=str, default="resnet18",
                        help="Model architecture: resnet18, resnet50, vit, convnext, efficientnet.")
    parser.add_argument("--weights", type=str, default="models/resnet18_best.pt",
                        help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolderDataset(args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Evaluation label map:", dataset.class_to_idx)

    model = get_model(args.model_name, len(dataset.classes), args.weights).to(device)
    model.eval()

    print(f"\nEvaluating {args.model_name.upper()} on {len(dataset)} images "
          f"({len(dataset.classes)} classes)")
    print(f"Batch size: {args.batch_size}, Warmup batches: {WARMUP_BATCHES}\n")

    # ------------------------------------------------------------------ #
    #  Pass 1: Collect predictions for metrics (not timed)               #
    # ------------------------------------------------------------------ #
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ------------------------------------------------------------------ #
    #  Pass 2: Measure inference time correctly                           #
    # ------------------------------------------------------------------ #
    print("Measuring inference time (forward pass only, with GPU warmup)...")
    avg_inference_time_ms = measure_inference_time(model, loader, device)

    # ------------------------------------------------------------------ #
    #  Metrics                                                            #
    # ------------------------------------------------------------------ #
    all_labels_np = np.array(all_labels)
    all_preds_np  = np.array(all_preds)
    all_probs_np  = np.array(all_probs)

    overall_accuracy = float((all_labels_np == all_preds_np).mean()) * 100.0

    report = classification_report(
        all_labels_np, all_preds_np,
        target_names=dataset.classes,
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    conf_mat  = confusion_matrix(all_labels_np, all_preds_np)

    macro_f1      = report["macro avg"]["f1-score"]
    macro_prec    = report["macro avg"]["precision"]
    macro_recall  = report["macro avg"]["recall"]

    # ------------------------------------------------------------------ #
    #  Console output                                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"  Model:              {args.model_name.upper()}")
    print(f"  Device:             {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Test images:        {len(dataset)}")
    print(f"  Overall Accuracy:   {overall_accuracy:.2f}%")
    print(f"  Macro Precision:    {macro_prec:.4f}")
    print(f"  Macro Recall:       {macro_recall:.4f}")
    print(f"  Macro F1-Score:     {macro_f1:.4f}")
    print(f"  Avg Inference Time: {avg_inference_time_ms:.2f} ms/frame")
    print("  (forward pass only, GPU warmed up, CUDA-synchronized)")
    print("=" * 60)

    print("\nPer-class Classification Report:")
    print(df_report.to_string())

    # ------------------------------------------------------------------ #
    #  Save summary JSON (easy to copy into thesis tables)               #
    # ------------------------------------------------------------------ #
    summary = {
        "model": args.model_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "test_images": len(dataset),
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_inference_time_ms_per_frame": round(avg_inference_time_ms, 2),
        "timing_method": (
            "CUDA events (forward pass only, GPU warmed up, CUDA-synchronized)"
            if device.type == "cuda"
            else "time.perf_counter (forward pass only, CPU)"
        ),
        "warmup_batches": WARMUP_BATCHES,
        "batch_size": args.batch_size,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1-score":  round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in dataset.classes
        }
    }
    summary_path = os.path.join(METRICS_DIR, f"summary_{args.model_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Save CSV metrics                                                   #
    # ------------------------------------------------------------------ #
    df_report.to_csv(os.path.join(METRICS_DIR, f"metrics_{args.model_name}.csv"))

    # ------------------------------------------------------------------ #
    #  Confusion Matrix                                                   #
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {args.model_name.upper()}")
    plt.tight_layout()
    cm_path = os.path.join(CONF_MATRIX_DIR, f"confusion_matrix_{args.model_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------------ #
    #  Precision-Recall and F1-Recall curves                             #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for class_idx, class_name in enumerate(dataset.classes):
        precision, recall, _ = precision_recall_curve(
            all_labels_np == class_idx, all_probs_np[:, class_idx]
        )
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        axes[0].plot(recall, precision, label=class_name)
        axes[1].plot(recall, f1, label=class_name)

    axes[0].set_title("Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].set_title("F1-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(fontsize=7)
    axes[1].grid(True)

    fig.suptitle(f"{args.model_name.upper()} — Precision-Recall & F1 Curves", fontsize=12)
    plt.tight_layout()
    curve_path = os.path.join(GRAPH_DIR, f"f1_precision_recall_{args.model_name}.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------------ #
    #  Final output paths                                                 #
    # ------------------------------------------------------------------ #
    print(f"\nSaved:")
    print(f"  Confusion matrix : {cm_path}")
    print(f"  CSV metrics      : {os.path.join(METRICS_DIR, f'metrics_{args.model_name}.csv')}")
    print(f"  PR/F1 curves     : {curve_path}")
    print(f"  Summary JSON     : {summary_path}")
    print(f"\n  >>> Copy these numbers into your thesis table:")
    print(f"      Accuracy  : {overall_accuracy:.2f}%")
    print(f"      Macro F1  : {macro_f1:.4f}")
    print(f"      Inf. time : {avg_inference_time_ms:.2f} ms/frame\n")


if __name__ == "__main__":
    main()