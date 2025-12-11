"""
Regenerate metadata files after manually cleaning the balanced dataset.

Usage:
    1. Manually delete noisy/wrong frames from balanced_dataset/
    2. Run this script: python regenerate_metadata.py
    3. All CSV files and summaries will be updated to reflect current state

This script scans what's actually in the balanced_dataset folders and 
regenerates all metadata files based on the real current state.
"""

import os
import csv
from collections import defaultdict

BALANCED_ROOT = "../../../data/balanced_dataset_v2"

LABELS = [
    "Close-up_behind_the_goal",
    "Close-up_corner",
    "Close-up_player_or_field_referee",
    "Close-up_side_staff",
    "Main_camera_center",
    "Main_camera_left",
    "Main_camera_right",
    "Public"
]

def extract_game_from_filename(filename):
    """
    Extract game name from filename.
    Expected format: {index}_{game_name}_{original_filename}
    Example: 00123_2015-02-21 - 18-00 Chelsea 1 - 1 Burnley_0_H1_1234.jpg
    
    Returns game name or 'unknown' if can't parse.
    """
    parts = filename.split('_', 2)  # split on first 2 underscores
    if len(parts) >= 3:
        # parts[0] is index, parts[1] is start of game name
        # Need to find where game name ends (before the shot_id pattern like "0_H1")
        remainder = parts[1] + '_' + parts[2]
        
        # Game names typically end before patterns like "0_H1_", "1_H2_", etc.
        # Split and rejoin until we hit the shot ID pattern
        tokens = remainder.split('_')
        game_parts = []
        
        for i, token in enumerate(tokens):
            # Check if this looks like a shot ID (digit followed by H1 or H2)
            if i + 1 < len(tokens) and token.isdigit() and tokens[i + 1].startswith('H'):
                break
            game_parts.append(token)
        
        if game_parts:
            return '_'.join(game_parts)
    
    return 'unknown'


def scan_balanced_dataset():
    """
    Scan the balanced dataset folder and return current state.
    
    Returns:
        metadata_rows: list of [label, game, filename, filepath, augmented]
        label_counts: dict of {label: count}
        game_distribution: dict of {label: {game: count}}
    """
    print("\n Scanning cleaned balanced dataset...\n")
    
    metadata_rows = []
    label_counts = {}
    game_distribution = defaultdict(lambda: defaultdict(int))
    
    for label in LABELS:
        label_path = os.path.join(BALANCED_ROOT, label)
        
        if not os.path.exists(label_path):
            print(f"  WARNING: Folder not found: {label}")
            label_counts[label] = 0
            continue
        
        files = [f for f in os.listdir(label_path) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        label_counts[label] = len(files)
        
        for filename in files:
            #filepath = os.path.join(label_path, filename)
            rel_path = f"{label}/{filename}"         
            game = extract_game_from_filename(filename)
            
            # augmented flag: 0 for now (could be enhanced to detect duplicates)
            #metadata_rows.append([label, game, filename, filepath, 0])
            metadata_rows.append([label, game, filename, rel_path, 0])
            game_distribution[label][game] += 1
        
        print(f"  {label:40s}: {len(files):5d} frames")
    
    total = sum(label_counts.values())
    print(f"\n  Total frames: {total:,}")
    
    return metadata_rows, label_counts, game_distribution


def save_metadata_files(metadata_rows, label_counts, game_distribution):
    """
    Save all metadata files based on current dataset state.
    """
    print("\n Writing updated metadata files...\n")
    
    counts_csv = os.path.join(BALANCED_ROOT, "label_counts.csv")
    meta_csv = os.path.join(BALANCED_ROOT, "metadata.csv")
    games_csv = os.path.join(BALANCED_ROOT, "game_distribution.csv")
    summary_txt = os.path.join(BALANCED_ROOT, "dataset_summary.txt")
    
    # Save label counts
    with open(counts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for label in LABELS:
            w.writerow([label, label_counts.get(label, 0)])
    print(f"  ✓ {counts_csv}")
    
    # Save metadata
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
#       w.writerow(["label", "game", "filename", "filepath", "augmented"])
        w.writerow(["label", "game", "filename", "rel_path", "augmented"])
        w.writerows(metadata_rows)
    print(f"  ✓ {meta_csv}")
    
    # Save game distribution
    with open(games_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "game", "frame_count"])
        for label in LABELS:
            for game, count in sorted(game_distribution[label].items(), 
                                     key=lambda x: x[1], reverse=True):
                w.writerow([label, game, count])
    print(f"  ✓ {games_csv}")
    
    # Save detailed summary
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BALANCED DATASET SUMMARY (AFTER MANUAL CLEANING)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total frames: {len(metadata_rows):,}\n")
        f.write(f"Number of classes: {len(LABELS)}\n\n")
        
        f.write("FRAMES PER CLASS:\n")
        for label in LABELS:
            f.write(f"  {label:40s}: {label_counts.get(label, 0):,}\n")
        
        for label in LABELS:
            games_used = game_distribution[label]
            f.write(f"\n{'-' * 70}\n")
            f.write(f"CLASS: {label}\n")
            f.write(f"{'-' * 70}\n")
            f.write(f"Total frames: {sum(games_used.values()):,}\n")
            f.write(f"Unique games: {len(games_used)}\n")
            f.write(f"\nAll games (sorted by contribution):\n")
            for game, count in sorted(games_used.items(), 
                                     key=lambda x: x[1], reverse=True):
                f.write(f"  {game}: {count} frames\n")
    
    print(f"  ✓ {summary_txt}")


def main():
    print("=" * 70)
    print("REGENERATE METADATA AFTER MANUAL CLEANING")
    print("=" * 70)
    
    # Check if balanced dataset exists
    if not os.path.exists(BALANCED_ROOT):
        print(f"\n ERROR: Balanced dataset not found at: {BALANCED_ROOT}")
        print("   Please check the path and try again.")
        return
    
    # Scan current state
    metadata_rows, label_counts, game_distribution = scan_balanced_dataset()
    
    # Save updated files
    save_metadata_files(metadata_rows, label_counts, game_distribution)
    
    print("\n" + "=" * 70)
    print("Metadata regenerated successfully!")


if __name__ == "__main__":
    main()