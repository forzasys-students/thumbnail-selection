"""
CLOSEUP-ONLY SEGMENT EXTRACTION FOR THUMBNAIL SELECTION

Strategy: Only process closeup shots, ignore main camera and public.
This saves processing time and focuses on the best thumbnail candidates.

Priorities:
1. P1_player_referee (Close-up_player_or_field_referee) - BEST thumbnails
2. P2_corner (Close-up_corner) - Set pieces
3. P3_side_staff (Close-up_side_staff) - Coach reactions
4. P4_behind_goal (Close-up_behind_the_goal) - Goalkeeper shots

IGNORED for thumbnails:
- Main_camera_* (too wide for thumbnails)
- Public (not engaging enough)
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from pathlib import Path
import shutil
from collections import defaultdict


# ===== CLOSEUP-ONLY LABEL MAPPING =====

CLOSEUP_P1_PLAYER_REFEREE = {
    "Close-up_player_or_field_referee",
}

CLOSEUP_P2_CORNER = {
    "Close-up_corner",
}

CLOSEUP_P3_SIDE_STAFF = {
    "Close-up_side_staff",
}

CLOSEUP_P4_BEHIND_GOAL = {
    "Close-up_behind_the_goal",
}

# These are IGNORED - not processed at all
IGNORED_LABELS = {
    "Main_camera_center",
    "Main_camera_left",
    "Main_camera_right",
    "Public",
}


def natural_frame_index(path: str) -> int:
    """Extract frame number from path like 'frame_00123.jpg' -> 123"""
    base = os.path.basename(path)
    num = "".join([c for c in base if c.isdigit()])
    return int(num) if num else -1


def find_segments_by_label_set(df: pd.DataFrame, label_set: set, priority_name: str, min_length: int = 5):
    """
    Extract continuous segments of frames matching any label in label_set.
    
    Args:
        df: predictions dataframe
        label_set: set of labels to match (e.g., {"Close-up_corner"})
        priority_name: name for this segment type (e.g., "P2_corner")
        min_length: minimum consecutive frames to form a segment
    
    Returns:
        List of (game_name, clip_name, priority_name, frames_list) tuples
    """
    segments = []

    # Sort by game/clip/frame for proper ordering
    df = df.sort_values(by=["game_name", "clip_name"], kind="stable").assign(
        frame_idx=df["frame_path"].apply(natural_frame_index)
    )
    df = df.sort_values(by=["game_name", "clip_name", "frame_idx"], kind="stable")

    current_segment = []
    current_game = None
    current_clip = None

    for _, row in df.iterrows():
        game = row["game_name"]
        clip = row["clip_name"]
        label = row["pred_label"]
        path = row["frame_path"]

        # New game/clip -> save previous segment if valid
        if game != current_game or clip != current_clip:
            if len(current_segment) >= min_length:
                segments.append((current_game, current_clip, priority_name, current_segment))
            current_segment = []

        # Check if this frame matches our label set
        if label in label_set:
            current_segment.append(path)
        else:
            # Label doesn't match -> end current segment
            if len(current_segment) >= min_length:
                segments.append((game, clip, priority_name, current_segment))
            current_segment = []

        current_game = game
        current_clip = clip

    # Don't forget last segment
    if len(current_segment) >= min_length:
        segments.append((current_game, current_clip, priority_name, current_segment))

    return segments


def extract_closeup_segments(
    df: pd.DataFrame,
    min_length: int = 5,
    min_segments_per_clip: int = 10,
    verbose: bool = True
):
    """
    Extract segments from ONLY closeup shots, ignoring main camera and public.
    
    Returns all closeup segments without hierarchical fallback since we're
    not trying to hit a minimum count anymore - we just want good closeups.
    """
    
    if verbose:
        print("\n[INFO] Extracting closeup-only segments...")
        print(f"[INFO] Minimum segment length: {min_length} frames")
        print(f"[INFO] Ignoring: {', '.join(IGNORED_LABELS)}\n")
    
    # Extract segments per closeup type
    p1_segs = find_segments_by_label_set(df, CLOSEUP_P1_PLAYER_REFEREE, "P1_player_referee", min_length)
    p2_segs = find_segments_by_label_set(df, CLOSEUP_P2_CORNER, "P2_corner", min_length)
    p3_segs = find_segments_by_label_set(df, CLOSEUP_P3_SIDE_STAFF, "P3_side_staff", min_length)
    p4_segs = find_segments_by_label_set(df, CLOSEUP_P4_BEHIND_GOAL, "P4_behind_goal", min_length)
    
    # Combine all closeup segments
    all_segments = p1_segs + p2_segs + p3_segs + p4_segs
    
    if verbose:
        print(f"[INFO] Found segments by type:")
        print(f"  P1 (player/referee): {len(p1_segs)}")
        print(f"  P2 (corner): {len(p2_segs)}")
        print(f"  P3 (side staff): {len(p3_segs)}")
        print(f"  P4 (behind goal): {len(p4_segs)}")
        print(f"  TOTAL: {len(all_segments)}\n")
    
    # Optional: If you still want a minimum per clip, apply hierarchical fallback
    # But recommend removing this - just use all closeups you find
    if min_segments_per_clip > 0:
        segments_by_clip = defaultdict(list)
        for seg in all_segments:
            segments_by_clip[(seg[0], seg[1])].append(seg)
        
        # Check if any clip is under minimum
        all_clips = df[["game_name", "clip_name"]].drop_duplicates().values
        for game, clip in all_clips:
            current = len(segments_by_clip[(game, clip)])
            if current < min_segments_per_clip and verbose:
                print(f"[WARN] Clip '{clip}' only has {current} closeup segments (wanted {min_segments_per_clip})")
    
    return all_segments


def save_segments(segments, output_csv="segments.csv", copy_dir=None, verbose=True):
    """Save segments to CSV and optionally copy frames to folders."""
    rows = []

    for idx, (game, clip, priority, frames) in enumerate(segments):
        rows.append({
            "segment_id": idx,
            "game_name": game,
            "clip_name": clip,
            "priority": priority,
            "num_frames": len(frames),
            "start_frame": os.path.basename(frames[0]),
            "end_frame": os.path.basename(frames[-1]),
        })

        if copy_dir:
            seg_folder = Path(copy_dir) / f"segment_{idx:03d}_{priority}"
            seg_folder.mkdir(parents=True, exist_ok=True)
            for f in frames:
                shutil.copy(f, seg_folder)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    if verbose:
        print(f"[INFO] Saved {len(out)} closeup segments -> {output_csv}")
        print("\n[INFO] Breakdown by closeup type:")
        counts = out["priority"].value_counts()

        order = ["P1_player_referee", "P2_corner", "P3_side_staff", "P4_behind_goal"]
        for p in order:
            if p in counts:
                c = int(counts[p])
                pct = (c / len(out)) * 100
                print(f"  {p}: {c} ({pct:.1f}%)")
        
        total_frames = out["num_frames"].sum()
        print(f"\n[INFO] Total closeup frames: {total_frames}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ONLY closeup segments for thumbnail selection"
    )
    parser.add_argument("--pred_csv", required=True, 
                       help="Predictions CSV (game_name, clip_name, frame_path, pred_label)")
    parser.add_argument("--min_length", type=int, default=5,
                       help="Minimum consecutive frames to form a segment")
    parser.add_argument("--min_segments_per_clip", type=int, default=10,
                       help="Minimum segments per clip (0 = no minimum, use all closeups)")
    parser.add_argument("--copy_dir", type=str, default=None,
                       help="Directory to copy segment frames (optional)")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV path")
    args = parser.parse_args()

    # Load predictions
    df = pd.read_csv(args.pred_csv)
    
    # Filter out ignored labels BEFORE processing (saves memory)
    initial_count = len(df)
    df = df[~df["pred_label"].isin(IGNORED_LABELS)]
    filtered_count = initial_count - len(df)
    
    print(f"[INFO] Filtered out {filtered_count} non-closeup frames")
    print(f"[INFO] Processing {len(df)} closeup frames\n")

    # Extract closeup segments
    segments = extract_closeup_segments(
        df,
        min_length=args.min_length,
        min_segments_per_clip=args.min_segments_per_clip,
        verbose=True
    )

    # Determine output path
    if args.output_csv is None:
        output_csv = os.path.join(args.copy_dir, "segments.csv") if args.copy_dir else "segments.csv"
    else:
        output_csv = args.output_csv

    if args.copy_dir:
        os.makedirs(args.copy_dir, exist_ok=True)

    # Save
    save_segments(segments, output_csv=output_csv, copy_dir=args.copy_dir, verbose=True)