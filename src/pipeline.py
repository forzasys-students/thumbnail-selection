import os
import csv
import cv2
from collections import Counter
from typing import List

from parse_labels import parse_camera_labels
from frame_extractor import extract_adaptive_frames


def find_all_games(data_path: str) -> List[str]:
    """
    Recursively search for valid game directories inside SoccerNet.

    A directory is considered valid if it contains:
      - Labels-cameras.json (camera annotations)
      - 1_224p.mkv (first half video)
      - 2_224p.mkv (second half video)
    """
    game_directories = []
    for root, _, files in os.walk(data_path):
        if {"Labels-cameras.json", "1_224p.mkv", "2_224p.mkv"}.issubset(set(files)):
            game_directories.append(root)
    return sorted(game_directories)


def run_pipeline():
    """
    End-to-end frame extraction pipeline:

    1. Find all valid game directories.
    2. For each game:
        - Parse camera shot segments using parse_camera_labels().
        - For each half:
            - Open video file.
            - Loop over all shot segments.
            - Extract representative frames using extract_adaptive_frames().
            - Save frames to per-label folders.
            - Collect metadata (game, half, label, frame_idx, filepath).
    3. Write all metadata to metadata.csv.
    4. Count how many frames per label and save to label_counts.csv.
    5. Print label counts summary to console.
    """

    # Paths configuration
    data_root = "C:/Users/roshi/Desktop/MasterOppgave/data"
    data_path = os.path.join(data_root, "SoccerNet")   # raw game data
    out_root  = os.path.join(data_root, "frames")      # extracted frames
    os.makedirs(out_root, exist_ok=True)

    metadata_csv   = os.path.join(data_root, "metadata.csv")
    label_counts_csv = os.path.join(data_root, "label_counts.csv")

    # Find all games
    game_directories = find_all_games(data_path)
    if not game_directories:
        raise FileNotFoundError("No valid game directories found.")

    label_counter = Counter()
    all_metadata = []

    # Process each game 
    for game_dir in game_directories:
        game_name = os.path.basename(game_dir)
        print(f"\n=== Processing {game_name} ===")

        # Parse shots from Labels-cameras.json
        labels_file = os.path.join(game_dir, "Labels-cameras.json")
        shots_by_half = parse_camera_labels(labels_file)

        # Loop over both halves
        for half in (1, 2):
            video_file = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_file):
                continue

            # Open video to confirm it's readable
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames <= 0:
                print(f"Skipping unreadable video: {video_file}")
                continue

            # Loop over all shot segments
            for shot_index, (start_seconds, end_seconds, label) in enumerate(shots_by_half[half]):
                try:
                    saved_paths = extract_adaptive_frames(
                        video_file=video_file,
                        out_root=out_root,
                        game_name=game_name,
                        shot_id=f"{shot_index}_H{half}",
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        label=label
                    )
                    # Add metadata rows and update label counts
                    for path in saved_paths:
                        frame_idx = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
                        all_metadata.append([game_name, half, label, frame_idx, path])
                        label_counter[label] += 1

                except Exception as e:
                    print(f"[ERROR] {game_name} H{half} shot {shot_index} ({label}): {e}")

    # Save metadata.csv
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "half", "label", "frame_idx", "filepath"])
        writer.writerows(all_metadata)

    # Save label_counts.csv
    with open(label_counts_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count"])
        for label, count in label_counter.most_common():
            writer.writerow([label, count])

    # Print summary
    print("\n=== Label counts ===")
    for label, count in label_counter.most_common():
        print(f"{label:30s} {count}")

    print(f"\nMetadata written to: {metadata_csv}")
    print(f"Label counts written to: {label_counts_csv}")


if __name__ == "__main__":
    run_pipeline()
