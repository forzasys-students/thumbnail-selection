"""
KEYFRAME SELECTION WITH BRISQUE QUALITY SCORING
================================================

OVERVIEW:
This script selects the top 3 thumbnail-worthy keyframes from each video segment.
It uses a multi-stage filtering approach to balance speed and quality.

WORKFLOW:
1. Fast pre-filtering: Remove overlays, blurry frames, and distant shots
2. Priority ranking: Faces > Celebration poses > Closeup framing
3. BRISQUE scoring: Apply expensive quality model only to top candidates
4. Final selection: Combine all metrics into weighted score, pick top 3

PRIORITY RULES:
- P1: Contains face(s) → ranked by largest face bbox area
- P2: No face but celebration pose > 0.5 → ranked by pose score
- P3: No face/pose but strong closeup framing → ranked by bbox coverage
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from brisque import BRISQUE


# ========== LOAD AI MODELS (ONE-TIME INITIALIZATION) ==========
# BRISQUE: No-reference image quality assessment (lower score = better quality)
brisque_model = BRISQUE()

# YOLOv8 Pose: Detects human keypoints for celebration pose detection
pose_model = YOLO("yolov8n-pose.pt")

# YOLOv8 Object Detection: Detects objects for closeup framing analysis
obj_model = YOLO("yolov8n.pt")

# Haar Cascade: Fast face detection (less accurate but very fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ========== QUALITY METRIC FUNCTIONS ==========

def laplacian_variance(path):
    """
    Measures image sharpness using Laplacian variance.
    
    How it works:
    - Converts image to grayscale
    - Applies Laplacian filter (detects edges)
    - Computes variance of result (high variance = sharp edges = sharp image)
    
    Returns:
        float: Sharpness score (higher = sharper, typical range 0-2000+)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    return 0.0 if img is None else cv2.Laplacian(img, cv2.CV_64F).var()


def brisque_score(path):
    """
    Computes BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
    
    How it works:
    - Uses machine learning model trained on natural scene statistics
    - No reference image needed (blind quality assessment)
    - Lower score = better quality
    
    Returns:
        float: Quality score (0-100, where 0 = perfect, 100 = worst)
        
    Note: This is the slowest metric, so we only apply it to top candidates
    """
    img = cv2.imread(path)
    if img is None: 
        return 100.0  # Worst possible score if image can't be loaded
    try: 
        return float(brisque_model.score(img))
    except: 
        return 100.0  # Fail-safe: return worst score on error


def saturation_score(path):
    """
    Measures color saturation (how vibrant the colors are).
    
    How it works:
    - Converts BGR to HSV color space
    - Extracts S (Saturation) channel
    - Computes mean saturation
    
    Returns:
        float: Normalized saturation (0.0 = grayscale, 1.0 = fully saturated)
    """
    img = cv2.imread(path)
    if img is None: return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    return float(np.mean(hsv[:, :, 1])) / 255.0  # Normalize S channel to 0-1


def detect_faces_and_size(path):
    """
    Detects faces and measures the largest face area.
    
    How it works:
    - Uses Haar Cascade classifier (fast but somewhat inaccurate)
    - Detects all faces in the image
    - Returns count and largest face area (in pixels²)
    
    Returns:
        tuple: (num_faces: int, largest_face_area: float)
        
    Use case: Thumbnails with faces get higher priority
    """
    img = cv2.imread(path)
    if img is None: 
        return 0, 0.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Haar cascade works on grayscale
    
    # detectMultiScale parameters:
    # - scaleFactor=1.15: how much image size is reduced at each scale
    # - minNeighbors=5: how many neighbors each candidate rectangle should have
    faces = face_cascade.detectMultiScale(gray, 1.15, 5)
    
    if len(faces) == 0:
        return 0, 0.0
    
    # Calculate area (width × height) for each detected face
    areas = [(w * h) for (_, _, w, h) in faces]
    return len(faces), max(areas)  # Return count and largest face


def closeup_bbox_score(path):
    """
    Measures how "zoomed in" the frame is using object detection.
    
    How it works:
    - Uses YOLO to detect all objects in frame
    - Finds largest detected object
    - Computes what fraction of the frame it occupies
    
    Returns:
        float: Coverage ratio (0.0 = tiny objects, 1.0 = object fills frame)
        
    Use case: Closeup shots make better thumbnails than wide shots
    """
    try:
        r = obj_model(path)[0]  # Run YOLO inference
        if len(r.boxes) == 0: 
            return 0.0  # No objects detected
        
        h, w = r.orig_shape[:2]  # Get image dimensions
        img_area = h * w
        
        # r.boxes.xywh format: [center_x, center_y, width, height]
        # Calculate area of each bbox and find max coverage
        return max([float(b[2]*b[3]) / float(img_area) for b in r.boxes.xywh])
    except:
        return 0.0  # Fail-safe on any error


def transition_overlay_score(path):
    """
    Detects transition frames and overlays using contrast measurement.
    
    How it works:
    - Computes standard deviation of grayscale pixel values
    - Low stddev = flat/uniform = likely a transition or overlay
    - High stddev = varied = normal content
    
    Returns:
        float: Contrast score (higher = more varied/normal content)
        
    Use case: Filter out transition frames (fades, wipes, title cards)
    """
    img = cv2.imread(path)
    if img is None: 
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # meanStdDev returns (mean, stddev) - we only need stddev
    return float(cv2.meanStdDev(gray)[1][0][0])


def celebration_score(path):
    """
    Detects celebration poses (arms raised) using pose estimation.
    
    How it works:
    - Uses YOLOv8-pose to detect human keypoints
    - Checks if wrists are above shoulders (raised arms)
    - Checks if arms are spread wide
    - Combines criteria to detect celebration
    
    Returns:
        float: Celebration score (0.0 = no pose, 0.5 = partial, 1.0 = full celebration)
        
    Keypoint indices:
    - 5, 6: Left/Right shoulders
    - 9, 10: Left/Right wrists
    - 11, 12: Left/Right hips
    """
    try:
        results = pose_model(path)
    except:
        return 0.0

    if len(results[0].keypoints) == 0:
        return 0.0  # No person detected

    best = 0.0  # Track best score across all detected people
    
    for det in results[0].keypoints:
        kps = det.xy[0]  # Get (x, y) coordinates for all keypoints
        
        # Extract relevant keypoints
        ls, rs = kps[5], kps[6]    # Left/Right shoulders
        lw, rw = kps[9], kps[10]   # Left/Right wrists
        lh, rh = kps[11], kps[12]  # Left/Right hips

        # Calculate torso height (shoulder to hip)
        torso_h = abs(lh[1] - ls[1])
        if torso_h <= 0:
            continue  # Invalid pose, skip

        # Calculate how high wrists are relative to torso
        # Positive ratio = wrists above shoulders
        left_ratio = (ls[1] - lw[1]) / torso_h
        right_ratio = (rs[1] - rw[1]) / torso_h
        
        # Calculate arm spread (wrist-to-wrist vs shoulder-to-shoulder)
        shoulder_w = abs(rs[0] - ls[0])
        wrist_w = abs(rw[0] - lw[0])
        spread_ratio = wrist_w / shoulder_w if shoulder_w > 0 else 0

        # Score based on celebration criteria
        if left_ratio > 0.4 and right_ratio > 0.4 and spread_ratio > 1.2:
            score = 1.0  # Full celebration: both arms high and spread
        elif left_ratio > 0.3 or right_ratio > 0.3:
            score = 0.5  # Partial: one arm raised
        else:
            score = 0.0  # No celebration

        best = max(best, score)  # Keep best score from all people
    return best


def extract_frame_index(path):
    """
    Extracts numeric frame index from filename.
    
    Example: "/path/to/frame_001234.jpg" -> 1234
    
    Returns:
        int: Frame number extracted from filename
    """
    return int("".join(c for c in os.path.basename(path) if c.isdigit()))


# ========== MAIN KEYFRAME SELECTION PIPELINE ==========
def select_keyframes(pred_csv, seg_csv, output_csv, output_dir):
    """
    Main function: Selects top 3 keyframes per video segment.
    
    Pipeline:
    1. Load segment boundaries and candidate frames
    2. For each segment:
       a. Fast pre-filter: Remove bad frames (overlays, blurry, distant)
       b. Priority ranking: Assign P1 (faces), P2 (poses), P3 (closeup)
       c. Take top 5 candidates
       d. Apply expensive BRISQUE quality scoring
       e. Compute final weighted score
       f. Select top 3 and save to disk
    3. Export results to CSV
    
    Args:
        pred_csv: Path to CSV with candidate frames (must have 'frame_path' column)
        seg_csv: Path to CSV with segment boundaries (must have 'segment_id', 'start_frame', 'end_frame')
        output_csv: Path to save results CSV
        output_dir: Directory to save selected keyframe images
    """
    
    # ===== LOAD DATA =====
    df_preds = pd.read_csv(pred_csv)    # Load all candidate frames
    df_segs = pd.read_csv(seg_csv)      # Load segment boundaries
    
    # Extract frame index from filename for range filtering
    df_preds["frame_index"] = df_preds["frame_path"].apply(extract_frame_index)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []  # Will store metadata for all selected keyframes

    # ===== PROCESS EACH SEGMENT =====
    for _, seg in tqdm(df_segs.iterrows(), total=len(df_segs), desc="Processing segments"):
        seg_id = seg["segment_id"]  # Unique segment identifier
        
        # Extract frame range for this segment
        start = extract_frame_index(seg["start_frame"])
        end = extract_frame_index(seg["end_frame"])

        # Filter candidate frames that fall within this segment
        segment_frames = df_preds[(df_preds.frame_index >= start) & (df_preds.frame_index <= end)]

        candidates = []  # Will store frames that pass all filters

        # ===== STAGE 1: FAST PRE-FILTERING =====
        # Loop through all candidate frames in this segment
        for _, row in segment_frames.iterrows():
            path = row["frame_path"]

            # FILTER 1: Remove transition/overlay frames
            # Threshold: 18 (empirically determined - frames with low contrast are likely transitions)
            overlay = transition_overlay_score(path)
            if overlay < 18:
                continue  # Skip this frame

            # FILTER 2: Remove blurry frames
            # Threshold: 25 (frames below this have poor edge definition)
            sharp = laplacian_variance(path)
            if sharp < 25:
                continue  # Skip this frame

            # FILTER 3: Remove distant shots
            # Threshold: 0.10 (object must occupy at least 10% of frame)
            close = closeup_bbox_score(path)
            if close < 0.12:
                continue  # Skip this frame

            # ===== FRAME PASSED ALL FILTERS - COMPUTE METRICS =====
            
            # Detect faces (for priority assignment)
            faces, face_area = detect_faces_and_size(path)
            
            # Detect celebration poses (for priority assignment)
            pose = celebration_score(path)
            
            # ===== ASSIGN PRIORITY LEVEL =====
            # Priority determines initial ranking before BRISQUE scoring
            
            if faces > 0:
                # P1: Face detected (highest priority)
                priority = 1
                rank_value = face_area  # Rank by face size (larger = better)
                
            elif pose > 0.5:
                # P2: Celebration pose detected (medium priority)
                priority = 2
                rank_value = pose  # Rank by pose score
                
            else:
                # P3: Good closeup framing (lower priority)
                priority = 3
                rank_value = close  # Rank by closeup coverage

            # Store candidate with all computed metrics
            candidates.append({
                'path': path,
                'priority': priority,
                'rank_value': rank_value,
                'faces': faces,
                'face_area': face_area,
                'pose': pose,
                'close': close,
                'sharp': sharp
            })

        # ===== STAGE 2: PRIORITY SORTING =====
        # Sort by priority first, then by rank_value (descending) within each priority
        # Take top 5 candidates to limit expensive BRISQUE computation
        candidates_sorted = sorted(candidates, key=lambda x: (x['priority'], -x['rank_value']))[:5]

        # ===== STAGE 3: APPLY BRISQUE (EXPENSIVE!) =====
        # Only compute BRISQUE on top 5 candidates per segment (not all frames)
        # This is the key optimization that makes the pipeline fast
        for c in candidates_sorted:
            c['brisque'] = brisque_score(c['path'])
            # Normalize BRISQUE to 0-1 range (higher = better)
            # Original: 0 = best, 100 = worst
            # Normalized: 1 = best, 0 = worst
            c['brisque_norm'] = max(0, 1 - c['brisque']/100)
        
        # ===== STAGE 4: FINAL WEIGHTED SCORING =====
        for c in candidates_sorted:
            # Compute additional metrics
            sat = saturation_score(c['path'])
            
            # WEIGHTED SCORE FORMULA
            # Weights chosen based on importance for thumbnail quality:
            # - 30% BRISQUE (overall quality)
            # - 25% Pose (engagement/action)
            # - 15% Closeup (framing)
            # - 10% Faces (presence)
            # - 10% Sharpness (edge clarity, capped at 1200)
            # - 10% Saturation (color vibrancy)
            score = (
                0.30 * c['brisque_norm'] +           # Quality assessment
                0.25 * c['pose'] +                    # Celebration detection
                0.15 * c['close'] +                   # Closeup framing
                0.10 * (c['faces'] > 0) +            # Face presence (binary)
                0.10 * min(c['sharp'] / 1200, 1.0) + # Sharpness (normalized)
                0.10 * sat                            # Color saturation
            )
            c['final_score'] = score

        # ===== STAGE 5: SELECT TOP 3 KEYFRAMES =====
        # Sort by final score (highest first) and take top 3
        top3 = sorted(candidates_sorted, key=lambda x: x['final_score'], reverse=True)[:3]

        # ===== STAGE 6: SAVE SELECTED KEYFRAMES =====
        for rank, c in enumerate(top3):
            # Generate output filename: segment_001_top1.jpg, segment_001_top2.jpg, etc.
            out = os.path.join(output_dir, f"segment_{seg_id:03d}_top{rank+1}.jpg")
            
            # Copy frame to output directory
            img = cv2.imread(c['path'])
            if img is not None:
                cv2.imwrite(out, img)

            # Record metadata for this selection
            results.append({
                "segment_id": seg_id,               # Which segment this is from
                "rank": rank + 1,                    # 1, 2, or 3
                "priority": c['priority'],           # P1, P2, or P3
                "final_score": round(c['final_score'], 3),
                "brisque": round(c['brisque'], 2),  # Quality score
                "faces": c['faces'],                 # Number of faces detected
                "celebration": round(c['pose'], 2),  # Celebration score
                "closeup": round(c['close'], 3),    # Closeup coverage ratio
                "sharpness": round(c['sharp'], 2),  # Laplacian variance
                "input_path": c['path'],            # Original frame path
                "saved_path": out                    # Where we saved it
            })

    # ===== GLOBAL TOP-N SELECTION =====
    # Sort ALL selected keyframes by final_score and keep only the best N
    # This ensures we get the absolute best frames across the entire video
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_score', ascending=False)
    
    # CONFIGURABLE: Change this number to control total output frames
    TOP_N_GLOBAL = 30  # Keep only top 15 frames across all segments
    
    # Get indices of frames to keep
    top_n_indices = set(results_df.head(TOP_N_GLOBAL).index)
    
    # Remove files and entries that didn't make the cut
    for idx, row in results_df.iterrows():
        if idx not in top_n_indices:
            # Delete the saved file since it's not in final top N
            if os.path.exists(row['saved_path']):
                os.remove(row['saved_path'])
    
    # Keep only top N in results
    final_results = results_df.head(TOP_N_GLOBAL)
    
    # ===== EXPORT RESULTS =====
    # Save all metadata to CSV for analysis/review
    final_results.to_csv(output_csv, index=False)
    print(f"[INFO] Selected top {TOP_N_GLOBAL} keyframes from {len(results)} candidates")
    print(f"[INFO] Keyframes saved to {output_csv}")
    print(f"[INFO] Score range: {final_results['final_score'].min():.3f} - {final_results['final_score'].max():.3f}")
    
    # Print priority distribution in final selection
    print("\n[INFO] Priority distribution in final selection:")
    #print(final_results['segment_priority'].value_counts().to_string())
    print(final_results['priority'].value_counts().to_string())


# ========== COMMAND-LINE INTERFACE ==========
if __name__ == "__main__":
    """
    Run from command line:
    
    python script.py \
        --pred_csv path/to/candidate_frames.csv \
        --seg_csv path/to/segments.csv \
        --output_csv path/to/selected_keyframes.csv \
        --output_dir path/to/output_thumbnails/
    """
    import argparse
    parser = argparse.ArgumentParser(description="Select top keyframes from video segments")
    parser.add_argument("--pred_csv", required=True, help="CSV with candidate frame paths")
    parser.add_argument("--seg_csv", required=True, help="CSV with segment boundaries")
    parser.add_argument("--output_csv", required=True, help="Output CSV for results")
    parser.add_argument("--output_dir", required=True, help="Directory to save keyframe images")
    args = parser.parse_args()

    select_keyframes(
        args.pred_csv,
        args.seg_csv,
        args.output_csv,
        args.output_dir
    )