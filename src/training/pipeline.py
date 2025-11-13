import os, sys
import csv
import cv2
from collections import Counter
from typing import List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.parse_labels import parse_camera_labels
from src.utils.frame_extractor import extract_adaptive_frames


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
    4. Count how many frames per label globally and save to label_counts.csv.
    5. Print per-game and global summaries.
    """

    # Paths configuration
    data_root = "C:/Users/roshi/Desktop/MasterOppgave/data"
    data_path = os.path.join(data_root, "SoccerNet")   # raw game data
    out_root  = os.path.join(data_root, "frames")      # extracted frames
    os.makedirs(out_root, exist_ok=True)

    metadata_csv   = os.path.join(out_root, "metadata.csv")
    global_label_counts_csv = os.path.join(out_root, "label_counts.csv")


    # Toggle: True = weighted factors, False = raw extraction
    USE_SAMPLE_FACTORS = True

    # Per-label sampling weights
    label_sample_factor = {
        "Main camera center": 0.2,
        "Close-up player or field referee": 0.1,
        "Close-up side staff": 3.0,
        "Close-up behind the goal": 5.0,
        "Main camera left": 5.0,
        "Main camera right": 5.0,
        "Close-up corner": 12.0,
        "Public": 15.0,
        "Main behind the goal": 4.0,
        "Goal line technology camera": 25.0,
        "Spider camera": 8.0,
        "Other": 10.0,
    }

    # Find all games
    game_directories = find_all_games(data_path)
    if not game_directories:
        raise FileNotFoundError("No valid game directories found.")

    global_label_counter = Counter()
    all_metadata = []

    # Process each game 
    for game_dir in game_directories:
        game_name = os.path.basename(game_dir)
        print(f"\n=== Processing {game_name} ===")

        # Parse shots
        labels_file = os.path.join(game_dir, "Labels-cameras.json")
        shots_by_half = parse_camera_labels(labels_file)

        game_label_counter = Counter()

        # Loop over halves
        for half in (1, 2):
            video_file = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_file):
                continue

            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames <= 0:
                print(f"Skipping unreadable video: {video_file}")
                continue

            for shot_index, (start_seconds, end_seconds, label) in enumerate(shots_by_half[half]):
                try:
                    
                    # Choose factor based on mode
                    factor = 1.0
                    if USE_SAMPLE_FACTORS:
                        factor = label_sample_factor.get(label, 1.0)

                    # Extract frames
                    saved_paths = extract_adaptive_frames(
                        video_file=video_file,
                        out_root=out_root,
                        game_name=game_name,
                        shot_id=f"{shot_index}_H{half}",
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        label=label,
                        sample_factor=factor
                    )

                    # Collect metadata + update counters
                    for path in saved_paths:
                        frame_idx = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
                        all_metadata.append([game_name, half, label, frame_idx, path])
                        global_label_counter[label] += 1
                        game_label_counter[label] += 1

                except Exception as e:
                    print(f"[ERROR] {game_name} H{half} shot {shot_index} ({label}): {e}")

        # Save per-game label counts inside that game's frames folder
        game_frames_path = os.path.join(out_root, game_name)
        os.makedirs(game_frames_path, exist_ok=True)
        game_counts_csv = os.path.join(game_frames_path, f"label_counts_{game_name}.csv")
        with open(game_counts_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "count"])
            for label, count in game_label_counter.most_common():
                writer.writerow([label, count])

        # Print per-game summary
        print(f"\n=== Label counts for {game_name} ===")
        for label, count in game_label_counter.most_common():
            print(f"{label:30s} {count}")
        print(f"(Saved per-game counts to {game_counts_csv})")

    # Save metadata.csv
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "half", "label", "frame_idx", "filepath"])
        writer.writerows(all_metadata)

    # Save global label counts
    with open(global_label_counts_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count"])
        for label, count in global_label_counter.most_common():
            writer.writerow([label, count])

    print("\n=== Global Label counts (all games) ===")
    for label, count in global_label_counter.most_common():
        print(f"{label:30s} {count}")

    print(f"\nMetadata written to: {metadata_csv}")
    print(f"Global label counts written to: {global_label_counts_csv}")


if __name__ == "__main__":
    run_pipeline()
