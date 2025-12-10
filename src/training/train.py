import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from src.training.dataset import FrameDataset


def get_model(model_name: str, num_classes: int):
    """Return a model architecture based on name."""
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vit":
        model = models.vit_b_16(weights="IMAGENET1K_V1")
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "r3d":
        model = models.video.r3d_18(weights="KINETICS400_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "convnext":
        model = models.convnext_base(weights="IMAGENET1K_V1")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "yoloclass":
        from ultralytics import YOLO
        model = YOLO("yolov8n-cls.pt")
        model.model[-1] = nn.Linear(model.model[-1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit", "r3d", "convnext", "efficientnet", "yoloclass"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str,
                        default="/fp/homes01/u01/ec-aliaana/data/balanced_dataset_v1",)
    parser.add_argument("--csv_file", type=str, default="metadata.csv")
    args = parser.parse_args()

    # Setup
    data_root = args.data_root
    csv_file = os.path.join(data_root, args.csv_file)
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    model_name = args.model

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    dataset = FrameDataset(csv_file=csv_file, root_dir=data_root, transform=transform)
    print("Training label map:", dataset.label_map)
    num_classes = len(dataset.label_map)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = get_model(model_name, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training {model_name.upper()} on {device} for {num_epochs} epochs "
          f"| {num_classes} classes | {len(dataset)} samples")

    best_val_acc = 0.0
    best_model_path = f"checkpoints/{model_name}_best.pt"

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {i} Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs} done in {time.time() - start_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model: {best_model_path}")

    print("Training complete.")

    # Final test evaluation 
    print("\nRunning final test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"[TEST] Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    print(f"Best validation accuracy was {best_val_acc:.2f}% (saved at {best_model_path})")


if __name__ == "__main__":
    main()
