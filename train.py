import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dataset import FrameDataset

def main():
    # Paths
    data_root = "/fp/homes01/u01/ec-aliaana/data/frames_global"
    csv_file = os.path.join(data_root, "metadata.csv") 

    # Training config
    num_epochs = 5
    batch_size = 32
    learning_rate = 1e-4

    # Transforms
    transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])


    # Dataset + splits
    dataset = FrameDataset(csv_file=csv_file, root_dir=data_root, transform=transform)
    num_classes = len(dataset.label_map)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Output dirs
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training on {device} for {num_epochs} epochs | {num_classes} classes | {len(dataset)} samples")

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

        # Save checkpoint 
        checkpoint_path = f"checkpoints/resnet18_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete.")

    # Final test evaluation
    print("\nRunning final test evaluation...")
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


if __name__ == "__main__":
    main()
