"""
Evaluation script for manually collected test dataset.

This script loads a trained model (ResNet18, ResNet50, or ViT),
runs it on a manually created test set (one folder per class),
and computes detailed evaluation metrics:
- Precision, Recall, F1-score
- Confusion Matrix
- Average inference speed (ms per frame)
- Precision-Recall and F1-Recall curves
"""

import os
import time
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on unique test set.")
    parser.add_argument("--data_root", type=str, default="/fp/homes01/u01/ec-aliaana/data/unique_test_set_v2",
                        help="Path to manually collected test dataset.")
    parser.add_argument("--model_name", type=str, default="resnet18",
                        help="Model architecture: resnet18, resnet50, or vit.")
    parser.add_argument("--weights", type=str, default="models/resnet18_best.pt",
                        help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolderDataset(args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Evaluation label map:", dataset.class_to_idx)
    model = get_model(args.model_name, len(dataset.classes), args.weights).to(device)
    model.eval()

    print(f"\nEvaluating {args.model_name.upper()} on {len(dataset)} images ({len(dataset.classes)} classes)\n")

    all_preds, all_labels, all_probs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    elapsed = time.time() - start_time
    avg_inference_time = (elapsed / len(dataset)) * 1000

    report = classification_report(all_labels, all_preds, target_names=dataset.classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    conf_mat = confusion_matrix(all_labels, all_preds)

    print("Classification Report:")
    print(df_report)
    print(f"\nAverage inference speed: {avg_inference_time:.2f} ms/frame")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {args.model_name.upper()}")
    plt.tight_layout()

    # Additional: Precision-Recall and F1-Recall curves
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for class_idx, class_name in enumerate(dataset.classes):
        precision, recall, _ = precision_recall_curve(y_true == class_idx, y_probs[:, class_idx])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        axes[0].plot(recall, precision, label=class_name)
        axes[1].plot(recall, f1, label=class_name)

    axes[0].set_title("Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("F1-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"f1_precision_recall_{args.model_name}.png"))

    # Save metrics and confusion matrix to proper folders
    df_report.to_csv(os.path.join(METRICS_DIR, f"metrics_{args.model_name}.csv"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {args.model_name.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(CONF_MATRIX_DIR, f"confusion_matrix_{args.model_name}.png"))

    print(f"\nConfusion matrix saved to: {os.path.join(CONF_MATRIX_DIR, f'confusion_matrix_{args.model_name}.png')}")
    print(f"Metrics saved to: {os.path.join(METRICS_DIR, f'metrics_{args.model_name}.csv')}")
    print(f"PR/F1 curves saved to: {os.path.join(GRAPH_DIR, f'f1_precision_recall_{args.model_name}.png')}\n")


if __name__ == "__main__":
    main()
