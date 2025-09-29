import os
import cv2
from parse_labels import parse_camera_labels
from frame_extractor import extract_frames_from_shot_seconds

def find_all_games(data_path: str) -> list[str]:
    """Return all game dirs that have both halves + Labels-cameras.json."""
    game_dirs = []
    for root, _, files in os.walk(data_path):
        if "Labels-cameras.json" in files and "1_224p.mkv" in files and "2_224p.mkv" in files:
            game_dirs.append(root)
    return game_dirs

def run_pipeline():
    data_root = "C:/Users/roshi/Desktop/MasterOppgave/data"
    data_path = os.path.join(data_root, "SoccerNet")
    out_root  = os.path.join(data_root, "frames")

    games = find_all_games(data_path)
    if not games:
        raise FileNotFoundError("No games with videos + labels. Run download_data.py first.")

    for game_dir in games:
        game_name = os.path.basename(game_dir)
        print(f"Processing game: {game_name}")

        # Parse shot windows in seconds per half
        label_file = os.path.join(game_dir, "Labels-cameras.json")
        shots_by_half = parse_camera_labels(label_file)

        for half in [1, 2]:
            video_file = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_file):
                continue

            # Open once to know bounds (also sanity-check file is readable)
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if total_frames <= 0:
                print(f"Skipping unreadable video: {video_file}")
                continue

            for shot_id, (start_sec, end_sec, label) in enumerate(shots_by_half[half]):
                # Call extractor (it will handle edge trim, logos, blur, bounds)
                try:
                    saved = extract_frames_from_shot_seconds(
                        video_file=video_file,
                        out_root=out_root,
                        game_name=game_name,
                        shot_id=f"{shot_id}_H{half}",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label=label,
                        step_sec=1.0,        # 1 frame per second
                        edge_trim_sec=0.5,   # skip 0.5s at each shot edge
                        blur_thresh=200.0,   # Laplacian var threshold
                    )
                    if saved:
                        print(f"Saved {len(saved):3d} frames for shot {shot_id} ({label}) in H{half}")
                except Exception as e:
                    print(f"Failed on shot {shot_id} ({label}) in {game_name} H{half}: {e}")

    print("Done.")

if __name__ == "__main__":
    run_pipeline()
