import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from src.training.dataset import FrameDataset


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    elif model_name == "convnext":
        model = models.convnext_base(weights="IMAGENET1K_V1")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit", "convnext", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_root", type=str,
                        default="/fp/homes01/u01/ec-aliaana/data/balanced_dataset_v2")
    parser.add_argument("--csv_file", type=str, default="metadata.csv")
    parser.add_argument("--augment", action="store_true",
                        help="Use data augmentation during training")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    data_root = args.data_root
    csv_file = os.path.join(data_root, args.csv_file)

    # Define transforms
    if args.augment:
        print("Using data augmentation")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomRotation(5), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load dataset
    dataset = FrameDataset(csv_file=csv_file, root_dir=data_root, transform=transform)
    print("Training label map:", dataset.label_map)
    num_classes = len(dataset.label_map)

    # ============================
    # 85 / 15 GAME-LEVEL SPLIT
    # ============================
    df = pd.read_csv(csv_file)
    if "game" not in df.columns:
        raise ValueError("metadata.csv must contain a 'game' column")

    groups = df["game"].values
    indices = df.index.values

    print(f"\nDataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique games: {df['game'].nunique()}")
    print(f"  Classes: {num_classes}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_idx, val_idx = next(gss.split(indices, groups=groups))

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ds)} samples from {len(set(groups[train_idx]))} games")
    print(f"  Val:   {len(val_ds)} samples from {len(set(groups[val_idx]))} games")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model setup
    model = get_model(args.model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    os.makedirs("models", exist_ok=True)

    print(f"\nTraining {args.model.upper()} on {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}\n")

    # Training state
    best_val_acc = 0.0
    best_model_path = f"models/{args.model}_best.pt"
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'epochs': []
    }

    # Training loop
    for epoch in range(args.epochs):
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

            # Less verbose logging
            if (i + 1) % 50 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Batch {i+1}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}")

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

        # Update learning rate + detect LR change
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.6f} -> {current_lr:.6f}")

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['epochs'].append(epoch + 1)

        print(f"\nEpoch {epoch+1}/{args.epochs} completed in {time.time() - start_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.2f}%")
        print(f"  LR:         {current_lr:.6f}")

        # Save best model (no early stopping)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved: {best_model_path}")

        print()  # Empty line for readability

    # Save training history
    history_path = f"models/{args.model}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {history_path}")
    print("\nFor final evaluation, use evaluate_model.py with your manual test set.")
    print("=" * 60)


if __name__ == "__main__":
    main()
