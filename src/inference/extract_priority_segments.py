"""
SEGMENT DETECTION WITH HIERARCHICAL CLOSEUP CLASSIFICATION

Priority:
1. P1A_closeup  (Close-up_corner, Close-up_player_or_field_referee)
2. P1B_closeup (Close-up_behind_the_goal, Close-up_side_staff)
3. P2_main_camera (Main_camera_*)
4. P3_public (Public)
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from pathlib import Path
import shutil
from collections import defaultdict


# ===== Model label -> hierarchy mapping =====

CLOSEUP_1A_LABELS = {
    "Close-up_player_or_field_referee",
}

CLOSEUP_1B_LABELS = {
    "Close-up_side_staff",
    "Close-up_behind_the_goal",
    "Close-up_corner",
}

MAIN_CAMERA_LABELS = {
    "Main_camera_center",
    "Main_camera_left",
    "Main_camera_right",
}

PUBLIC_LABELS = {
    "Public",
}


def natural_frame_index(path: str) -> int:
    base = os.path.basename(path)
    num = "".join([c for c in base if c.isdigit()])
    return int(num) if num else -1


def find_segments_by_priority(df: pd.DataFrame, label_set: set, min_length: int = 3):
    segments = []

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

        if game != current_game or clip != current_clip:
            if len(current_segment) >= min_length:
                segments.append((current_game, current_clip, current_segment))
            current_segment = []

        if label in label_set:
            current_segment.append(path)
        else:
            if len(current_segment) >= min_length:
                segments.append((game, clip, current_segment))
            current_segment = []

        current_game = game
        current_clip = clip

    if len(current_segment) >= min_length:
        segments.append((current_game, current_clip, current_segment))

    return segments


def find_all_segments_with_hierarchical_fallback(
    df: pd.DataFrame,
    min_length: int = 3,
    min_segments_per_clip: int = 2
):
    # Extract segments per tier
    p1a = find_segments_by_priority(df, CLOSEUP_1A_LABELS, min_length)
    p1b = find_segments_by_priority(df, CLOSEUP_1B_LABELS, min_length)
    p2  = find_segments_by_priority(df, MAIN_CAMERA_LABELS, min_length)
    p3  = find_segments_by_priority(df, PUBLIC_LABELS, min_length)

    # Tag
    p1a_tagged = [(g, c, "P1A_closeup", frames) for g, c, frames in p1a]
    p1b_tagged = [(g, c, "P1B_closeup", frames) for g, c, frames in p1b]
    p2_tagged  = [(g, c, "P2_main_camera", frames) for g, c, frames in p2]
    p3_tagged  = [(g, c, "P3_public", frames) for g, c, frames in p3]

    segments_by_clip = defaultdict(list)

    # Start with P1A
    for seg in p1a_tagged:
        segments_by_clip[(seg[0], seg[1])].append(seg)

    all_clips = df[["game_name", "clip_name"]].drop_duplicates().values

    for game, clip in all_clips:
        current = len(segments_by_clip[(game, clip)])

        if current < min_segments_per_clip:
            needed = min_segments_per_clip - current
            add = [s for s in p1b_tagged if s[0] == game and s[1] == clip][:needed]
            segments_by_clip[(game, clip)].extend(add)
            current = len(segments_by_clip[(game, clip)])

        if current < min_segments_per_clip:
            needed = min_segments_per_clip - current
            add = [s for s in p2_tagged if s[0] == game and s[1] == clip][:needed]
            segments_by_clip[(game, clip)].extend(add)
            current = len(segments_by_clip[(game, clip)])

        if current < min_segments_per_clip:
            needed = min_segments_per_clip - current
            add = [s for s in p3_tagged if s[0] == game and s[1] == clip][:needed]
            segments_by_clip[(game, clip)].extend(add)

    all_segments = []
    for segs in segments_by_clip.values():
        all_segments.extend(segs)

    return all_segments


def save_segments(segments, output_csv="segments.csv", copy_dir=None):
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

    print(f"[INFO] Saved {len(out)} segments -> {output_csv}")
    print("\n[INFO] Hierarchical priority breakdown:")
    counts = out["priority"].value_counts()

    order = ["P1A_closeup", "P1B_closeup", "P2_main_camera", "P3_public"]
    for p in order:
        if p in counts:
            c = int(counts[p])
            pct = (c / len(out)) * 100
            print(f"  {p}: {c} ({pct:.1f}%)")

    p1_total = int(counts.get("P1A_closeup", 0) + counts.get("P1B_closeup", 0))
    if p1_total:
        print(f"\n[INFO] Total closeup segments (P1A+P1B): {p1_total} ({(p1_total/len(out))*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract segments with hierarchical closeup classification")
    parser.add_argument("--pred_csv", required=True, help="Predictions CSV (game_name, clip_name, frame_path, pred_label)")
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--min_segments_per_clip", type=int, default=2)
    parser.add_argument("--copy_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)

    segments = find_all_segments_with_hierarchical_fallback(
        df,
        min_length=args.min_length,
        min_segments_per_clip=args.min_segments_per_clip
    )

    if args.output_csv is None:
        output_csv = os.path.join(args.copy_dir, "segments.csv") if args.copy_dir else "segments.csv"
    else:
        output_csv = args.output_csv

    if args.copy_dir:
        os.makedirs(args.copy_dir, exist_ok=True)

    save_segments(segments, output_csv=output_csv, copy_dir=args.copy_dir)
