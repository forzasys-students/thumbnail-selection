"""
SEGMENT DETECTION WITH PRIORITY FALLBACK

OVERVIEW:
This script identifies frame segments suitable for thumbnail selection.
Uses a priority-based approach to ensure sufficient candidates.

PRIORITY SYSTEM:
1. P1 (Closeup shots) - Best for thumbnails
2. P2 (Main camera angles) - Good fallback
3. P3 (Public/crowd) - Last resort

WORKFLOW:
1. Extract P1 segments (closeups)
2. If insufficient, add P2 segments (main camera)
3. If still insufficient, add P3 segments (public)
4. Ensures minimum segment coverage per clip
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
from pathlib import Path
import shutil


# SHOT TYPE PRIORITIES

# P1: Close-up shots (highest priority for thumbnails)
CLOSEUP_LABELS = {
    "Close-up_behind_the_goal",
    "Close-up_corner",
    "Close-up_player_or_field_referee",
    "Close-up_side_staff"
}

# P2: Main camera angles (good framing, fallback option)
MAIN_CAMERA_LABELS = {
    "Main_camera_center",
    "Main_camera_left",
    "Main_camera_right"
}

# P3: Public/crowd shots (last resort)
PUBLIC_LABELS = {
    "Public"
}


def natural_frame_index(path: str):
    """
    Extract numeric frame index from filename.
    Example: ".../frame_00450.jpg" → 450
    
    Returns:
        int: Frame number, or -1 if no digits found
    """
    base = os.path.basename(path)
    num = ''.join([c for c in base if c.isdigit()])
    return int(num) if num else -1


def find_segments_by_priority(df: pd.DataFrame, label_set: set, min_length: int = 3):
    """
    Find consecutive frame segments matching a specific label set.
    
    Args:
        df: DataFrame with columns [game_name, clip_name, frame_path, pred_label]
        label_set: Set of labels to match (e.g., CLOSEUP_LABELS)
        min_length: Minimum consecutive frames to form a segment
        
    Returns:
        list: Segments as (game, clip, priority_name, frames_list)
    """
    segments = []

    # Sort by game → clip → frame index
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

        # Reset when switching game/clip
        if game != current_game or clip != current_clip:
            if len(current_segment) >= min_length:
                segments.append((current_game, current_clip, current_segment))
            current_segment = []

        # Check if frame matches priority labels
        if label in label_set:
            current_segment.append(path)
        else:
            # Break in sequence - save if long enough
            if len(current_segment) >= min_length:
                segments.append((game, clip, current_segment))
            current_segment = []

        current_game = game
        current_clip = clip

    # Save final segment
    if len(current_segment) >= min_length:
        segments.append((current_game, current_clip, current_segment))

    return segments


def find_all_segments_with_fallback(
    df: pd.DataFrame,
    min_length: int = 3,
    min_segments_per_clip: int = 2
):
    """
    Hierarchical segment extraction with priority fallback.
    
    Strategy:
    1. Extract P1 (closeup) segments
    2. Per clip, if < min_segments, add P2 (main camera) segments
    3. If still insufficient, add P3 (public) segments
    
    This ensures every clip contributes candidates, even short clips.
    
    Args:
        df: Predictions DataFrame
        min_length: Minimum consecutive frames for a segment
        min_segments_per_clip: Target segments per clip (fallback trigger)
        
    Returns:
        list: All segments with priority labels
    """
    
    # Extract segments for each priority level
    p1_segments = find_segments_by_priority(df, CLOSEUP_LABELS, min_length)
    p2_segments = find_segments_by_priority(df, MAIN_CAMERA_LABELS, min_length)
    p3_segments = find_segments_by_priority(df, PUBLIC_LABELS, min_length)
    
    # Tag each segment with its priority
    p1_tagged = [(g, c, "P1_closeup", frames) for g, c, frames in p1_segments]
    p2_tagged = [(g, c, "P2_main_camera", frames) for g, c, frames in p2_segments]
    p3_tagged = [(g, c, "P3_public", frames) for g, c, frames in p3_segments]
    
    # Group segments by (game, clip)
    from collections import defaultdict
    segments_by_clip = defaultdict(list)
    
    for seg in p1_tagged:
        segments_by_clip[(seg[0], seg[1])].append(seg)
    
    # Apply fallback logic per clip
    all_clips = df[["game_name", "clip_name"]].drop_duplicates().values
    
    for game, clip in all_clips:
        current_count = len(segments_by_clip[(game, clip)])
        
        # Fallback to P2 if needed
        if current_count < min_segments_per_clip:
            p2_for_clip = [s for s in p2_tagged if s[0] == game and s[1] == clip]
            needed = min_segments_per_clip - current_count
            segments_by_clip[(game, clip)].extend(p2_for_clip[:needed])
            current_count = len(segments_by_clip[(game, clip)])
        
        # Fallback to P3 if still needed
        if current_count < min_segments_per_clip:
            p3_for_clip = [s for s in p3_tagged if s[0] == game and s[1] == clip]
            needed = min_segments_per_clip - current_count
            segments_by_clip[(game, clip)].extend(p3_for_clip[:needed])
    
    # Flatten back to list
    all_segments = []
    for segments in segments_by_clip.values():
        all_segments.extend(segments)
    
    return all_segments


def save_segments(segments, output_csv="segments.csv", copy_dir=None):
    """
    Save segment metadata to CSV and optionally copy frames.
    
    Args:
        segments: List of (game, clip, priority, frames) tuples
        output_csv: Output CSV path
        copy_dir: Optional directory to copy segment frames
    """
    rows = []

    for idx, (game, clip, priority, seg_frames) in enumerate(segments):
        rows.append({
            "segment_id": idx,
            "game_name": game,
            "clip_name": clip,
            "priority": priority,
            "num_frames": len(seg_frames),
            "start_frame": os.path.basename(seg_frames[0]),
            "end_frame": os.path.basename(seg_frames[-1])
        })

        # Optional: Copy frames for debugging
        if copy_dir:
            seg_folder = Path(copy_dir) / f"segment_{idx:03d}_{priority}"
            seg_folder.mkdir(parents=True, exist_ok=True)
            for f in seg_frames:
                shutil.copy(f, seg_folder)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"[INFO] Saved {len(segments)} segments -> {output_csv}")
    print(f"[INFO] Priority breakdown:")
    print(df["priority"].value_counts().to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract segments with priority fallback for thumbnail selection"
    )
    parser.add_argument(
        "--pred_csv", 
        type=str, 
        required=True,
        help="CSV file from inference (must have: game_name, clip_name, frame_path, pred_label)"
    )
    parser.add_argument(
        "--min_length", 
        type=int, 
        default=3,
        help="Minimum consecutive frames to form a segment"
    )
    parser.add_argument(
        "--min_segments_per_clip",
        type=int,
        default=2,
        help="Minimum segments per clip (triggers fallback if not met)"
    )
    parser.add_argument(
        "--copy_dir", 
        type=str, 
        default=None,
        help="Optional folder to copy segment frames for inspection"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None,
        help="Output CSV path (default: copy_dir/segments.csv)"
    )
    
    args = parser.parse_args()

    # Load predictions
    df = pd.read_csv(args.pred_csv)
    
    # Extract segments with fallback
    segments = find_all_segments_with_fallback(
        df,
        min_length=args.min_length,
        min_segments_per_clip=args.min_segments_per_clip
    )
    
    # Determine output path
    if args.output_csv is None:
        if args.copy_dir:
            output_csv = os.path.join(args.copy_dir, "segments.csv")
        else:
            output_csv = "segments.csv"
    else:
        output_csv = args.output_csv
    
    # Ensure copy directory exists
    if args.copy_dir:
        os.makedirs(args.copy_dir, exist_ok=True)
    
    # Save results
    save_segments(
        segments,
        output_csv=output_csv,
        copy_dir=args.copy_dir
    )