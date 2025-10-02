import os
import random
import cv2
import numpy as np
import shutil
import csv
from collections import Counter

def augment_image(img):
    """Apply a random augmentation: flip, rotate, brightness, or noise."""
    aug = img.copy()
    choice = random.choice(["flip", "rotate", "brightness", "noise"])

    if choice == "flip":
        aug = cv2.flip(aug, 1)

    elif choice == "rotate":
        angle = random.uniform(-10, 10)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    elif choice == "brightness":
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    elif choice == "noise":
        noise = np.random.normal(0, 10, aug.shape).astype(np.uint8)
        aug = cv2.add(aug, noise)

    return aug


def balance_game_frames(frames_root: str, balanced_root: str, total_target: int = 3000):
    """
    Balance dataset per game:
    - Reads raw frames from frames_root
    - Creates balanced copy in balanced_root
    - Ensures each game has ~total_target frames evenly split across labels
    - Writes per-game counts, global counts, and metadata
    """
    os.makedirs(balanced_root, exist_ok=True)
    global_counts = Counter()
    all_metadata = []  # for metadata.csv

    for game_name in os.listdir(frames_root):
        game_path = os.path.join(frames_root, game_name)
        if not os.path.isdir(game_path):
            continue

        print(f"\n=== Balancing game: {game_name} ===")
        game_balanced_path = os.path.join(balanced_root, game_name)
        os.makedirs(game_balanced_path, exist_ok=True)

        label_folders = [
            os.path.join(game_path, d)
            for d in os.listdir(game_path)
            if os.path.isdir(os.path.join(game_path, d))
        ]
        if not label_folders:
            continue

        num_labels = len(label_folders)
        target_per_label = total_target // num_labels
        print(f"Labels: {num_labels}, target per label: {target_per_label}")

        game_counts = {}

        for label_folder in label_folders:
            label_name = os.path.basename(label_folder)
            balanced_label_path = os.path.join(game_balanced_path, label_name)
            os.makedirs(balanced_label_path, exist_ok=True)

            images = [
                f for f in os.listdir(label_folder)
                if f.lower().endswith((".jpg", ".png"))
            ]
            if not images:
                continue

            paths = [os.path.join(label_folder, f) for f in images]

            if len(paths) >= target_per_label:
                keep = random.sample(paths, target_per_label)
                for src in keep:
                    dst = os.path.join(balanced_label_path, os.path.basename(src))
                    shutil.copy(src, dst)
                    all_metadata.append([game_name, label_name, os.path.basename(src), dst, False])
                game_counts[label_name] = target_per_label

            else:
                # Copy existing
                for src in paths:
                    dst = os.path.join(balanced_label_path, os.path.basename(src))
                    shutil.copy(src, dst)
                    all_metadata.append([game_name, label_name, os.path.basename(src), dst, False])

                # Augment until target reached
                needed = target_per_label - len(paths)
                for i in range(needed):
                    src = random.choice(paths)
                    img = cv2.imread(src)
                    if img is None:
                        continue
                    aug_img = augment_image(img)
                    base, ext = os.path.splitext(os.path.basename(src))
                    new_name = f"{base}_aug{i}{ext}"
                    new_path = os.path.join(balanced_label_path, new_name)
                    cv2.imwrite(new_path, aug_img)
                    all_metadata.append([game_name, label_name, new_name, new_path, True])

                game_counts[label_name] = target_per_label

            global_counts[label_name] += target_per_label

        # Save per-game counts inside that game’s balanced folder
        game_counts_csv = os.path.join(game_balanced_path, f"label_counts_{game_name}.csv")
        with open(game_counts_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "count"])
            for label, count in sorted(game_counts.items()):
                writer.writerow([label, count])

        print(f"\n=== Label counts for {game_name} ===")
        for label, count in sorted(game_counts.items()):
            print(f"{label:30s} {count}")
        print(f"(Saved per-game counts to {game_counts_csv})")

    # Save global counts
    global_counts_csv = os.path.join(balanced_root, "label_counts.csv")
    with open(global_counts_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count"])
        for label, count in sorted(global_counts.items()):
            writer.writerow([label, count])

    print("\n=== Global Label counts (all games) ===")
    for label, count in sorted(global_counts.items()):
        print(f"{label:30s} {count}")
    print(f"(Saved global counts to {global_counts_csv})")

    # Save metadata.csv
    metadata_csv = os.path.join(balanced_root, "metadata.csv")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "label", "filename", "filepath", "augmented"])
        writer.writerows(all_metadata)

    print(f"Metadata written to: {metadata_csv}")


if __name__ == "__main__":
    frames_root = "C:/Users/roshi/Desktop/MasterOppgave/data/frames"
    balanced_root = "C:/Users/roshi/Desktop/MasterOppgave/data/frames_balanced"
    balance_game_frames(frames_root, balanced_root, total_target=3000)
