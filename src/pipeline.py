import os
import cv2
from parse_labels import parse_camera_labels
from frame_extractor import extract_frame



def find_all_games(data_path):
    """
        Walk through `data_path` and return all game directories
        that contain the required files:
        - Labels-cameras.json
        - 1_224p.mkv (first half video)
        - 2_224p.mkv (second half video)
    """
    game_dirs = []
    for root, dirs, files in os.walk(data_path):
        if "Labels-cameras.json" in files and "1_224p.mkv" in files and "2_224p.mkv" in files:
            game_dirs.append(root)
    return game_dirs



def run_pipeline():
    # Input root: where SoccerNet data is stored
    data_path = "C:/Users/roshi/Desktop/MasterOppgave/data/SoccerNet"
    # Output root: where extracted frames will be saved
    out_root = "C:/Users/roshi/Desktop/MasterOppgave/data/shot_frames"


    # Find all games with both halves + labels
    game_dirs = find_all_games(data_path)
    if not game_dirs:
        raise FileNotFoundError("No games found with videos + labels. Did you run download_data.py?")

    # Loop through each game
    for game_dir in game_dirs:
        game_name = os.path.basename(game_dir)
        print(f"Processing game: {game_name}")

        # Parse annotations into per-half lists of (frame_idx, label)
        label_file = os.path.join(game_dir, "Labels-cameras.json")
        shots_by_half = parse_camera_labels(label_file)

        # Process each half separately
        for half in [1, 2]:
            video_file = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_file):
                continue  # Skip if video half missing

            # Get total frame count (to catch invalid annotations)
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Iterate through annotations for this half
            for shot_id, (frame_idx, label) in enumerate(shots_by_half[half]):
                if frame_idx >= total_frames:
                    # Some annotations overshoot the actual video length
                    print(f"Skipping shot {shot_id} ({label}) in half {half} of {game_name}: frame {frame_idx} >= total_frames")
                    continue

                try:
                    # Extract and save frame to the correct folder
                    extract_frame(video_file, out_root, game_name, f"{shot_id}_H{half}", frame_idx, label)
                except Exception as e:
                    print(f"Failed on shot {shot_id} ({label}) in half {half} of {game_name}: {e}")

    print("Finished extracting frames for all games.")

if __name__ == "__main__":
    run_pipeline()
