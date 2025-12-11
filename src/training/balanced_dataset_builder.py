import os, random, shutil, csv
from tqdm import tqdm
from PIL import Image
import imagehash
from collections import defaultdict

RAW_ROOT = "../../../data/frames"                 # input
OUT_ROOT = "../../../data/balanced_dataset"       # output

NOISE_BUFFER = 1.15  # Expect to remove ~10% as noise (adjust based on your data)
TARGET_PER_CLASS = int(1250 * NOISE_BUFFER) 

# Deduplication settings
HASH_SIZE = 16         # bigger = stricter
HASH_THRESHOLD = 5     # smaller = stricter, higher = looser

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

os.makedirs(OUT_ROOT, exist_ok=True)
for lbl in LABELS:
    os.makedirs(os.path.join(OUT_ROOT, lbl), exist_ok=True)

# SCAN RAW DATA
print("\n Indexing raw dataset...")

all_samples = {lbl: [] for lbl in LABELS}

games = [g for g in os.listdir(RAW_ROOT) 
         if os.path.isdir(os.path.join(RAW_ROOT, g)) and not g.endswith('.csv')]

print(f"Found {len(games)} game folders to scan")

for g in tqdm(games, desc="Scanning games"):
    game_path = os.path.join(RAW_ROOT, g)
    
    for lbl in LABELS:
        label_dir = os.path.join(game_path, lbl)
        if not os.path.exists(label_dir):
            continue
        
        files = [f for f in os.listdir(label_dir) 
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        for f in files:
            all_samples[lbl].append((g, f, os.path.join(label_dir, f)))

# Print what was found
print("\n Raw frame counts per class:")
for lbl in LABELS:
    print(f"  {lbl:40s}: {len(all_samples[lbl]):5d} frames")

total_available = sum(len(all_samples[lbl]) for lbl in LABELS)
print(f"\nTotal available frames: {total_available}")

# DEDUPLICATE + BUILD BALANCED DATASET
print(f"\n Building balanced dataset with perceptual deduplication")
print(f"  Hash size: {HASH_SIZE}, Threshold: {HASH_THRESHOLD}\n")

metadata_rows = []
label_counts = {}
dedup_stats = {}

for lbl in LABELS:
    paths = all_samples[lbl]
    count = len(paths)
    
    print(f"\n{'='*70}")
    print(f"Processing: {lbl}")
    print(f"{'='*70}")
    print(f"  Raw candidates: {count}")
    
    # Shuffle to get random distribution across games
    random.shuffle(paths)
    
    # Deduplicate using perceptual hashing
    seen_hashes = []
    unique_frames = []
    duplicates_removed = 0
    errors = 0
    
    for game, fname, src in tqdm(paths, desc="  Deduplicating", leave=False):
        # Stop if we have enough unique frames
        if len(unique_frames) >= TARGET_PER_CLASS:
            break
        
        try:
            # Compute perceptual hash
            img = Image.open(src).convert("L").resize((128, 128))
            h = imagehash.average_hash(img, hash_size=HASH_SIZE)
            
            # Check if too similar to existing frames
            if any(abs(h - prev) <= HASH_THRESHOLD for prev in seen_hashes):
                duplicates_removed += 1
                continue
            
            # Keep this unique frame
            seen_hashes.append(h)
            unique_frames.append((game, fname, src))
            
        except Exception as e:
            errors += 1
            continue
    
    # Save the unique frames
    save_dir = os.path.join(OUT_ROOT, lbl)
    saved_count = 0
    
    for idx, (game, fname, src) in enumerate(tqdm(unique_frames, desc="  Saving", leave=False)):
        unique_fname = f"{idx:05d}_{game}_{fname}"
        out_path = os.path.join(save_dir, unique_fname)
        shutil.copy2(src, out_path)
        #metadata_rows.append([lbl, game, unique_fname, out_path, 0])
        rel_path = f"{lbl}/{unique_fname}"  # <-- this is all we store
        metadata_rows.append([lbl, game, unique_fname, rel_path, 0])

        saved_count += 1
    
    label_counts[lbl] = saved_count
    dedup_stats[lbl] = {
        'raw': count,
        'duplicates': duplicates_removed,
        'errors': errors,
        'saved': saved_count
    }
    
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Errors: {errors}")
    print(f"  Unique frames saved: {saved_count}")
    
    if saved_count < TARGET_PER_CLASS:
        shortfall = TARGET_PER_CLASS - saved_count
        print(f"    SHORT by {shortfall} frames (need more data or lower threshold)")
    elif saved_count >= TARGET_PER_CLASS:
        print(f" Target reached!")

# ANALYZE GAME DISTRIBUTION
print("\n Computing game distribution per class...\n")

game_distribution = {lbl: {} for lbl in LABELS}

for row in metadata_rows:
    lbl, game = row[0], row[1]
    game_distribution[lbl][game] = game_distribution[lbl].get(game, 0) + 1

# Print distribution to console
for lbl in LABELS:
    games_used = game_distribution[lbl]
    print(f"\n{lbl}:")
    print(f"  Total frames: {sum(games_used.values())}")
    print(f"  Games contributing: {len(games_used)}")
    print(f"  Top 5 games:")
    for game, count in sorted(games_used.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    • {game}: {count} frames")

# SAVE METADATA FILES
counts_csv = os.path.join(OUT_ROOT, "label_counts.csv")
meta_csv = os.path.join(OUT_ROOT, "metadata.csv")
games_csv = os.path.join(OUT_ROOT, "game_distribution.csv")
summary_txt = os.path.join(OUT_ROOT, "dataset_summary.txt")
#dedup_csv = os.path.join(OUT_ROOT, "deduplication_stats.csv")

# Save label counts
with open(counts_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["label", "count"])
    for lbl in LABELS:
        w.writerow([lbl, label_counts[lbl]])

# Save metadata (includes game column)
with open(meta_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
#   w.writerow(["label", "game", "filename", "filepath", "augmented"])
    w.writerow(["label", "game", "filename", "rel_path", "augmented"])
    w.writerows(metadata_rows)

# Save game distribution per class
with open(games_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["label", "game", "frame_count"])
    for lbl in LABELS:
        for game, count in sorted(game_distribution[lbl].items(), key=lambda x: x[1], reverse=True):
            w.writerow([lbl, game, count])

# Save deduplication statistics
#with open(dedup_csv, "w", newline="", encoding="utf-8") as f:
#    w = csv.writer(f)
#    w.writerow(["label", "raw_frames", "duplicates_removed", "errors", "final_count"])
#    for lbl in LABELS:
#        stats = dedup_stats[lbl]
#        w.writerow([lbl, stats['raw'], stats['duplicates'], stats['errors'], stats['saved']])

# Save detailed summary
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("BALANCED DATASET SUMMARY \n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total frames: {len(metadata_rows):,}\n")
    f.write(f"Target per class: {TARGET_PER_CLASS}\n")
    f.write(f"Number of classes: {len(LABELS)}\n")
    f.write(f"Dedup settings: hash_size={HASH_SIZE}, threshold={HASH_THRESHOLD}\n\n")
    
#    f.write("DEDUPLICATION RESULTS:\n")
#    for lbl in LABELS:
#        stats = dedup_stats[lbl]
#        dup_rate = (stats['duplicates'] / stats['raw'] * 100) if stats['raw'] > 0 else 0
#        f.write(f"\n{lbl}:\n")
#        f.write(f"  Raw frames:        {stats['raw']:,}\n")
#        f.write(f"  Duplicates removed: {stats['duplicates']:,} ({dup_rate:.1f}%)\n")
#        f.write(f"  Errors:            {stats['errors']:,}\n")
#        f.write(f"  Final count:       {stats['saved']:,}\n")
    
    f.write(f"\n{'=' * 70}\n")
    f.write("GAME DISTRIBUTION:\n")
    f.write(f"{'=' * 70}\n")
    
    for lbl in LABELS:
        games_used = game_distribution[lbl]
        f.write(f"\n{lbl}:\n")
        f.write(f"  Total frames: {sum(games_used.values())}\n")
        f.write(f"  Unique games: {len(games_used)}\n")
        f.write(f"\n  All games (sorted by contribution):\n")
        for game, count in sorted(games_used.items(), key=lambda x: x[1], reverse=True):
            f.write(f"    {game}: {count} frames\n")

print("\n===============================================================")
print("Balanced dataset!")
print(f"→ Output: {OUT_ROOT}")
print(f"→ Total images: {len(metadata_rows):,}")
print(f"\n→ Files saved:")
print(f"   • {counts_csv}")
print(f"   • {meta_csv}")
print(f"   • {games_csv}")
#print(f"   • {dedup_csv}  (NEW: deduplication statistics)")
print(f"   • {summary_txt}")
print("===============================================================\n")