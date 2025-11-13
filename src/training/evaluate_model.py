"""
Evaluation script for manually collected test dataset.

This script loads a trained model (ResNet18, ResNet50, or ViT),
runs it on a manually created test set (one folder per class),
and computes detailed evaluation metrics:
- Precision, Recall, F1-score
- Confusion Matrix
- Average inference speed (ms per frame)
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

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
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "yoloclass":
        from ultralytics import YOLO
        model = YOLO("yolov8n-cls.pt")
        model.model[-1] = nn.Linear(model.model[-1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model

# CUSTOM DATASET
class ImageFolderDataset(torch.utils.data.Dataset):
    """
    Dataset for evaluating manually collected images.

    Assumes folder structure like:
        root/
        ├── Main_camera_center/
        ├── Main_camera_left/
        ├── Main_camera_right/
        └── Public/

    Each folder name acts as a class label.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # Each subfolder = one class
        # Only include folders that actually contain images
        self.classes = sorted([
            cls for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls)) and any(
                fname.lower().endswith((".jpg", ".jpeg", ".png"))
                for fname in os.listdir(os.path.join(root_dir, cls))
            )
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}


        # Collect image paths + corresponding class index
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_path, file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image and label
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# MAIN EVALUATION PIPELINE
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on unique test set.")
    parser.add_argument("--data_root", type=str, default="/fp/homes01/u01/ec-aliaana/data/unique_test_set",
                        help="Path to manually collected test dataset.")
    parser.add_argument("--model_name", type=str, default="resnet18",
                        help="Model architecture: resnet18, resnet50, or vit.")
    parser.add_argument("--weights", type=str, default="checkpoints/resnet18_best.pt",
                        help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()

    # Setup & Preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Standard preprocessing used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize (ImageNet mean/std)
    ])

    # Load dataset and model
    dataset = ImageFolderDataset(args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Evaluation label map:", dataset.class_to_idx)
    model = get_model(args.model_name, len(dataset.classes), args.weights).to(device)
    model.eval()

    print(f"\nEvaluating {args.model_name.upper()} on {len(dataset)} images ({len(dataset.classes)} classes)\n")

    # Inference (Classification)
    """
    The core classification happens here.
    - Each image is passed through the model.
    - The model outputs a vector of probabilities (one per class).
    - The class with the highest probability is the predicted label.
    """
    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():  # Disable gradients for faster inference
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)            # Raw model predictions (logits)
            preds = torch.argmax(outputs, 1)   # Pick class with highest score
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Inference Time Measurement
    elapsed = time.time() - start_time
    avg_inference_time = (elapsed / len(dataset)) * 1000  # milliseconds per frame

    # Evaluation Metrics
    """
    Here we compare the predicted labels against the true labels (from folder names)
    and compute metrics such as:
      - Precision: How many predicted positives were correct?
      - Recall: How many true positives were detected?
      - F1-score: Harmonic mean of Precision and Recall.
      - Support: Number of samples per class.
    """
    report = classification_report(all_labels, all_preds, target_names=dataset.classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Print classification summary
    print("Classification Report:")
    print(df_report)
    print(f"\nAverage inference speed: {avg_inference_time:.2f} ms/frame")

    # Visualization: Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {args.model_name.upper()}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{args.model_name}.png")

    # Save Results
    df_report.to_csv(f"metrics_{args.model_name}.csv")
    print(f"\n Confusion matrix saved as confusion_matrix_{args.model_name}.png")
    print(f" Metrics saved as metrics_{args.model_name}.csv\n")


if __name__ == "__main__":
    main()
