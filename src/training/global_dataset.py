import os
import shutil
import csv
from collections import Counter, defaultdict
from PIL import Image
import imagehash

# Paths
DATA_ROOT   = "C:/Users/roshi/Desktop/MasterOppgave/data"
FRAMES_ROOT = os.path.join(DATA_ROOT, "frames")          # raw per-game frames
GLOBAL_ROOT = os.path.join(DATA_ROOT, "frames_global")   # balanced + deduped global dataset

# Deduplication settings
HASH_SIZE = 16         # bigger = stricter
HASH_THRESHOLD = 5     # smaller = stricter, higher = looser
TARGET_PER_CLASS = 1000  # balance cap: max frames per class

def safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_")

def global_dataset():
    os.makedirs(GLOBAL_ROOT, exist_ok=True)

    # Collect all frames by class across all games
    class_to_images = defaultdict(list)
    for game in os.listdir(FRAMES_ROOT):
        game_path = os.path.join(FRAMES_ROOT, game)
        if not os.path.isdir(game_path):
            continue
        for label in os.listdir(game_path):
            label_path = os.path.join(game_path, label)
            if not os.path.isdir(label_path):
                continue
            for img_name in os.listdir(label_path):
                if not img_name.lower().endswith((".jpg", ".png")):
                    continue
                src_path = os.path.join(label_path, img_name)
                class_to_images[label].append((game, img_name, src_path))

    global_counts = Counter()
    all_metadata = []

    # Process each class into global balanced dataset
    for label, images in class_to_images.items():
        print(f"\nProcessing class: {label} ({len(images)} candidates)")
        dst_folder = os.path.join(GLOBAL_ROOT, safe_label(label))
        os.makedirs(dst_folder, exist_ok=True)

        seen_hashes = []
        final_count = 0

        for game, fname, src in images:
            if final_count >= TARGET_PER_CLASS:
                break
            try:
                # Compute perceptual hash
                img = Image.open(src).convert("L").resize((128, 128))
                h = imagehash.average_hash(img, hash_size=HASH_SIZE)

                # Deduplicate: skip if too similar
                if any(abs(h - prev) <= HASH_THRESHOLD for prev in seen_hashes):
                    continue

                seen_hashes.append(h)

                # Save unique frame with traceability
                new_name = f"{game}_{fname}"
                dst_path = os.path.join(dst_folder, new_name)
                shutil.copy(src, dst_path)

                all_metadata.append([label, game, new_name, dst_path, False])
                final_count += 1

            except Exception as e:
                print(f"[ERROR] {src}: {e}")

        global_counts[label] = final_count
        print(f"Kept {final_count} unique frames for {label}")

    # Save label counts
    counts_csv = os.path.join(GLOBAL_ROOT, "label_counts.csv")
    with open(counts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for label, count in sorted(global_counts.items()):
            w.writerow([label, count])
    print(f"\nSaved global label counts to {counts_csv}")

    # Save metadata
    metadata_csv = os.path.join(GLOBAL_ROOT, "metadata.csv")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "game", "filename", "filepath", "augmented"])
        w.writerows(all_metadata)
    print(f"Saved metadata to {metadata_csv}")


if __name__ == "__main__":
    global_dataset()
