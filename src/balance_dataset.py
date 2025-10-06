import os
import random
import cv2
import numpy as np
import shutil
import csv
from collections import Counter

# -------- config: classes we want to keep --------
CORE_CLASSES = {
    "Main camera center",
    "Main camera left",
    "Main camera right",
    "Main behind the goal",
    "Close-up player or field referee",
    "Close-up side staff",
    "Close-up behind the goal",
    "Close-up corner",
    "Public",
}

# Dropped implicitly: "Other" (or any label not in CORE_CLASSES)

MAX_AUG_RATIO = 0.25             # <= 25% of a class may be augmented
TARGET_TOTAL_PER_GAME = 3000     # ~balanced per game

# -------- small helpers --------
def safe_label(label: str) -> str:
    """Make label names safe for folder names."""
    return label.replace(" ", "_").replace("/", "_")

def unsafe_label(folder_name: str) -> str:
    """Convert saved folder name back to readable label (best-effort)."""
    return folder_name.replace("_", " ")

def map_label(raw_folder_name: str) -> str | None:
    """
    Map a frames/ folder name directly to a core label if it matches.
    Drop anything else (no merging).
    """
    label_guess = unsafe_label(raw_folder_name).strip()
    if label_guess in CORE_CLASSES:
        return label_guess
    return None

# -------- augmentation: orientation-safe only --------
def augment_image(img):
    """
    Orientation-safe augmentation ONLY:
      - brightness/contrast shift
      - gaussian noise
    (No flips, no rotations, to preserve left/right semantics.)
    """
    aug = img.copy()
    choice = random.choice(["brightness", "noise"])

    if choice == "brightness":
        alpha = random.uniform(0.85, 1.15)  # mild contrast
        beta = random.randint(-15, 15)      # mild brightness
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    elif choice == "noise":
        noise = np.random.normal(0, 8, aug.shape).astype(np.int16)
        aug16 = aug.astype(np.int16) + noise
        aug = np.clip(aug16, 0, 255).astype(np.uint8)

    return aug

# -------- main balancing --------
def balance_game_frames(frames_root: str, balanced_root: str, target_total: int = TARGET_TOTAL_PER_GAME):
    """
    Build a balanced copy of each game under `balanced_root`:
      - keeps only CORE_CLASSES
      - caps augmentation per class to MAX_AUG_RATIO
      - aims for ~target_total frames per game (even split across kept classes)
      - writes per-game counts in each game folder + global counts + metadata
    """
    os.makedirs(balanced_root, exist_ok=True)
    global_counts = Counter()
    all_metadata = []  # [game, label, filename, filepath, augmented(bool)]

    for game_name in os.listdir(frames_root):
        game_src = os.path.join(frames_root, game_name)
        if not os.path.isdir(game_src):
            continue

        print(f"\n=== Balancing game: {game_name} ===")
        game_dst = os.path.join(balanced_root, game_name)
        os.makedirs(game_dst, exist_ok=True)

        # Gather folders per canonical label
        folders_by_label: dict[str, list[str]] = {}
        for folder in os.listdir(game_src):
            folder_path = os.path.join(game_src, folder)
            if not os.path.isdir(folder_path):
                continue
            canonical = map_label(folder)
            if canonical is None:
                continue  # dropped class
            folders_by_label.setdefault(canonical, []).append(folder_path)

        if not folders_by_label:
            print("No kept labels found — skipping.")
            continue

        kept_labels = sorted(folders_by_label.keys())
        target_per_label = max(1, target_total // len(kept_labels))
        print(f"Labels kept: {len(kept_labels)} → target per label: {target_per_label}")

        game_counts = {}

        for canonical_label in kept_labels:
            # collect all images across source folders
            src_paths = []
            for src_folder in folders_by_label[canonical_label]:
                for f in os.listdir(src_folder):
                    if f.lower().endswith((".jpg", ".png")):
                        src_paths.append(os.path.join(src_folder, f))

            if not src_paths:
                continue

            # destination label folder (sanitized)
            dst_label_folder = os.path.join(game_dst, safe_label(canonical_label))
            os.makedirs(dst_label_folder, exist_ok=True)

            # If we have enough originals, downsample to target_per_label
            if len(src_paths) >= target_per_label:
                keep = random.sample(src_paths, target_per_label)
                for src in keep:
                    dst = os.path.join(dst_label_folder, os.path.basename(src))
                    shutil.copy(src, dst)
                    all_metadata.append([game_name, canonical_label, os.path.basename(src), dst, False])
                final_count = target_per_label

            else:
                # copy all originals
                for src in src_paths:
                    dst = os.path.join(dst_label_folder, os.path.basename(src))
                    shutil.copy(src, dst)
                    all_metadata.append([game_name, canonical_label, os.path.basename(src), dst, False])

                # augment up to cap
                max_aug = int(target_per_label * MAX_AUG_RATIO)
                deficit = target_per_label - len(src_paths)
                aug_n = max(0, min(deficit, max_aug))
                for i in range(aug_n):
                    src = random.choice(src_paths)
                    img = cv2.imread(src)
                    if img is None:
                        continue
                    aug_img = augment_image(img)
                    base, ext = os.path.splitext(os.path.basename(src))
                    new_name = f"{base}_aug{i}{ext}"
                    new_path = os.path.join(dst_label_folder, new_name)
                    cv2.imwrite(new_path, aug_img)
                    all_metadata.append([game_name, canonical_label, new_name, new_path, True])
                final_count = len(src_paths) + aug_n

            game_counts[canonical_label] = final_count
            global_counts[canonical_label] += final_count

        # write per-game counts under the game folder
        counts_csv = os.path.join(game_dst, f"label_counts_{game_name}.csv")
        with open(counts_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "count"])
            for label in sorted(game_counts.keys()):
                w.writerow([label, game_counts[label]])

        print(f"\n=== Label counts for {game_name} ===")
        for label in sorted(game_counts.keys()):
            print(f"{label:30s} {game_counts[label]}")
        print(f"(Saved per-game counts to {counts_csv})")

    # global counts under frames_balanced/
    global_counts_csv = os.path.join(balanced_root, "label_counts.csv")
    with open(global_counts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for label in sorted(global_counts.keys()):
            w.writerow([label, global_counts[label]])

    print("\n=== Global Label counts (all games) ===")
    for label in sorted(global_counts.keys()):
        print(f"{label:30s} {global_counts[label]}")
    print(f"(Saved global counts to {global_counts_csv})")

    # metadata under frames_balanced/
    metadata_csv = os.path.join(balanced_root, "metadata.csv")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game", "label", "filename", "filepath", "augmented"])
        w.writerows(all_metadata)
    print(f"Metadata written to: {metadata_csv}")


if __name__ == "__main__":
    frames_root   = "C:/Users/roshi/Desktop/MasterOppgave/data/frames"
    balanced_root = "C:/Users/roshi/Desktop/MasterOppgave/data/frames_balanced"
    balance_game_frames(frames_root, balanced_root, target_total=TARGET_TOTAL_PER_GAME)
