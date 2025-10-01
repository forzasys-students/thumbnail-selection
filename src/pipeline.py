import os
import cv2
import csv
from collections import Counter
from parse_labels import parse_camera_labels
from frame_extractor import extract_frames_from_shot_seconds

def find_all_games(data_path: str) -> list[str]:
    """
    Recursively search under `data_path` to find all valid game directories.

    A directory is considered a "valid game" if it contains:
        - Labels-cameras.json (the annotation file)
        - 1_224p.mkv (first half video)
        - 2_224p.mkv (second half video)
    """
    game_dirs = []
    for root, _, files in os.walk(data_path):
        if "Labels-cameras.json" in files and "1_224p.mkv" in files and "2_224p.mkv" in files:
            game_dirs.append(root)
    return game_dirs

def run_pipeline():
    """
    End-to-end pipeline for frame extraction and metadata logging.

    Steps:
    1. Find all valid game directories.
    2. For each game:
        - Parse shot boundaries and labels from Labels-cameras.json.
        - For each half (1st, 2nd):
            - Load the corresponding video (1_224p.mkv or 2_224p.mkv).
            - For each shot:
                - Extract multiple representative frames using `frame_extractor`.
                - Save frames in organized folders (by label).
                - Track metadata: (game, half, label, frame_idx, file_path).
                - Count label occurrences for later distribution analysis.
    3. Write all metadata to metadata.csv.
    4. Write label distribution to label_counts.csv.
    """
    
    # --- Paths configuration ---
    data_root = "C:/Users/roshi/Desktop/MasterOppgave/data"
    data_path = os.path.join(data_root, "SoccerNet")   # Where raw games/videos are stored
    out_root  = os.path.join(data_root, "frames")      # Where extracted frames are stored
    
    metadata_path = os.path.join(data_root, "metadata.csv")  # CSV for frame metadata
    label_counts = Counter()      # Counter to keep track of how many frames per label
    all_metadata = []             # Stores rows for metadata.csv


    # --- Find games ---
    games = find_all_games(data_path)
    if not games:
        raise FileNotFoundError("No games with videos + labels. Run download_data.py first.")

    # --- Loop through games ---
    for game_dir in games:
        game_name = os.path.basename(game_dir)
        print(f"Processing game: {game_name}")

        # Parse shot segments (start_sec, end_sec, label) for each half
        label_file = os.path.join(game_dir, "Labels-cameras.json")
        shots_by_half = parse_camera_labels(label_file)

        # Each game has 2 halves
        for half in [1, 2]:
            video_file = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_file):
                continue

            # --- Sanity check: open video to confirm it's readable ---
            cap = cv2.VideoCapture(video_file)
            #cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback if FPS is missing
            cap.release()

            if total_frames <= 0:
                print(f"Skipping unreadable video: {video_file}")
                continue

            # --- Loop through shot annotations for this half ---
            for shot_id, (start_sec, end_sec, label) in enumerate(shots_by_half[half]):
                try:
                    # Extract frames for this shot (handles skipping logos, trimming edges, skipping blurry frames)
                    saved = extract_frames_from_shot_seconds(
                        video_file=video_file,
                        out_root=out_root,
                        game_name=game_name,
                        shot_id=f"{shot_id}_H{half}",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label=label,
                        step_sec=3.0,        # sample every 3 seconds
                        edge_trim_sec=1.0,   # skip 1.0 s at each shot edge
                        blur_thresh=200.0,   # Laplacian threshold for blurry frame detection
                    )
                    # For each saved frame, add metadata + update label counts
                    for path in saved:
                        # Extract numeric frame index from filename
                        frame_idx = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
                        all_metadata.append([game_name, half, label, frame_idx, path])
                        label_counts[label] += 1
        
                    if saved:
                        print(f"Saved {len(saved):3d} frames for shot {shot_id} ({label}) in H{half}")
                
                except Exception as e:
                    print(f"Failed on shot {shot_id} ({label}) in {game_name} H{half}: {e}")


    # ---- Save metadata.csv ----
    print(f"\nWriting metadata to {metadata_path}")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "half", "label", "frame_idx", "filepath"])
        writer.writerows(all_metadata)

    # ---- Save and print label counts ----
    print("\n=== Label counts ===")
    for label, count in label_counts.most_common():
        print(f"{label:30s} {count}")

    counts_path = os.path.join(data_root, "label_counts.csv")
    with open(counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count"])
        for label, count in label_counts.most_common():
            writer.writerow([label, count])

    print(f"\nMetadata: {metadata_path}, Label counts: {counts_path}")


if __name__ == "__main__":
    run_pipeline()
