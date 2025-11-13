"""
This script does the following:
  1) Sorts frames correctly by game → clip → frame index
  2) Groups consecutive frames predicted as close-ups
  3) Writes a CSV: closeup_segments.csv
  4) (Optional) copies each segment into a folder for debugging

"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
from pathlib import Path
import shutil

# Which labels count as close-up
CLOSEUP_LABELS = {
    "Close-up_behind_the_goal",
    "Close-up_corner",
    "Close-up_player_or_field_referee",
    "Close-up_side_staff"
}


def natural_frame_index(path: str):
    """
    Extract the numeric frame index from paths such as:
        ".../frame_00450.jpg" → 450
    """
    base = os.path.basename(path)
    num = ''.join([c for c in base if c.isdigit()])
    return int(num) if num else -1


def find_closeup_segments(df: pd.DataFrame, min_length: int = 3):
    """
    Identifies consecutive close-up frame sequences **per game and per clip**.
    Ensures proper sorting by actual frame index.
    """
    segments = []

    # Sort correctly
    df = df.sort_values(
        by=["game_name", "clip_name"],
        kind="stable"
    ).assign(frame_idx=df["frame_path"].apply(natural_frame_index))

    df = df.sort_values(
        by=["game_name", "clip_name", "frame_idx"],
        kind="stable"
    )

    current_segment = []
    current_game = None
    current_clip = None

    for _, row in df.iterrows():
        game = row["game_name"]
        clip = row["clip_name"]
        label = row["pred_label"]
        path = row["frame_path"]

        # Reset when moving to a new game/clip
        if game != current_game or clip != current_clip:
            if len(current_segment) >= min_length:
                segments.append((current_game, current_clip, current_segment))
            current_segment = []

        # Add to segment
        if label in CLOSEUP_LABELS:
            current_segment.append(path)
        else:
            if len(current_segment) >= min_length:
                segments.append((game, clip, current_segment))
            current_segment = []

        current_game = game
        current_clip = clip

    # Final segment
    if len(current_segment) >= min_length:
        segments.append((current_game, current_clip, current_segment))

    return segments


def save_segments(segments, output_csv="closeup_segments.csv", copy_dir=None):
    """
    Save segment metadata to CSV + optionally copy frames.
    """
    rows = []

    for idx, (game, clip, seg_frames) in enumerate(segments):
        rows.append({
            "segment_id": idx,
            "game_name": game,
            "clip_name": clip,
            "num_frames": len(seg_frames),
            "start_frame": os.path.basename(seg_frames[0]),
            "end_frame": os.path.basename(seg_frames[-1])
        })

        if copy_dir:
            seg_folder = Path(copy_dir) / f"segment_{idx:03d}"
            seg_folder.mkdir(parents=True, exist_ok=True)
            for f in seg_frames:
                shutil.copy(f, seg_folder)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved {len(segments)} close-up segments → {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract close-up segments.")
    parser.add_argument("--pred_csv", type=str, required=True,
                        help="CSV file from inference_model.py")
    parser.add_argument("--min_length", type=int, default=3,
                        help="Minimum segment length")
    parser.add_argument("--copy_dir", type=str, default=None,
                        help="Optional folder to store segment frames")
    parser.add_argument("--output_csv", type=str, default=None,
                    help="Optional custom output CSV path. Default = copy_dir/closeup_segments.csv")
    args = parser.parse_args()
    

    df = pd.read_csv(args.pred_csv)
    segments = find_closeup_segments(df, min_length=args.min_length)
    
    # Determine output CSV location
    if args.output_csv is None:
        # Default → drop inside the copy_dir folder
        output_csv = os.path.join(args.copy_dir, "closeup_segments.csv")
    else:
        output_csv = args.output_csv
    
    save_segments(
        segments,
        output_csv=output_csv,
        copy_dir=args.copy_dir
    )

    # Ensure copy directory exists
    if args.copy_dir:
        os.makedirs(args.copy_dir, exist_ok=True)

