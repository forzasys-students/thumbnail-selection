# keyframe_selector.py
"""
SOTA KEYFRAME SELECTION PIPELINE

What this file does (high level):
1) Load frame-level predictions (pred_csv) + segment ranges (seg_csv)
2) For each segment:
   A) Cheap preprocessing filters to remove obviously bad frames
   B) TEMPORAL diversity filter (saves YOLO calls on near-duplicates)
   C) Logo detection filter to remove branded frames
   D) Heavy SOTA signals (pose/face/celebration) on the reduced set
   E) MUSIQ (heavy IQA) only on the per-segment top-k (ACTUALLY batched now)
   F) Compute final score and keep only strong candidates
3) Global selection across all segments using quotas + time-gap constraint
4) Save selected keyframes and a CSV with score breakdown

Uses closeup-specific priorities from segments.csv:
- P1_player_referee - BEST thumbnails
- P2_corner - Set pieces
- P3_side_staff - Coach reactions
- P4_behind_goal - Goalkeeper shots
"""

from __future__ import annotations
from sota_models import SOTAModels, LogoDetector

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------
# Preprocessing utilities 
# -----------------------
from preprocess import (
    extract_frame_index,        # parse frame index from "frame_00123.jpg"
    transition_overlay_score,   # proxy: rejects "too flat/overlay-ish" frames via stddev
    blur_laplacian_var,         # blur metric: Laplacian variance (higher = sharper)
    texture_proxy,              # edge density proxy, rejects uniform/flat frames
    detect_and_mask_cuts,       # detects shot cuts using mean abs diff on downscaled gray
    motion_blur_score,          # motion blur detection using gradient variance ratio
)


# =============================================================================
# CONFIG (tuning knobs)
# =============================================================================

# Segment-specific multiplier: boosts or penalizes final score by segment type.
# Purpose: create clear ranking between closeup types for thumbnails.
# Spread out more than before (1.50 vs 1.25 vs 1.10 vs 1.00) for better differentiation.
SEGMENT_SCORE_MULT = {
    "P1_player_referee": 1.30,   # Highest - action/emotion shots (best thumbnails)
    "P2_corner": 1.20,           # High - set pieces, tactical moments
    "P3_side_staff": 1.10,       # Medium - coach reactions, bench celebrations
    "P4_behind_goal": 1.00,      # Baseline - goalkeeper shots, different angle
}

# Global quota fractions: ensures variety across closeup types.
# Example with top_n=10:
# - P1 gets ~5 frames, P2 ~3 frames, P3 ~1 frame, P4 ~1 frame (depending on rounding).
SEGMENT_QUOTA_FRAC = {
    "P1_player_referee": 0.60,   # 60% from player/referee closeups
    "P2_corner": 0.20,           # 20% from corner closeups
    "P3_side_staff": 0.10,       # 10% from staff/bench closeups
    "P4_behind_goal": 0.10,      # 10% from behind-goal closeups
}

# Segment-specific thresholds for filtering.
# These are used in:
# - cheap stage: min_sharpness / min_motion_blur / min_texture
# - heavy stage: min_closeup_ratio (pose-based)
# - IQA stage: min_musiq_norm (only used as a soft gate for P1)
QUALITY_THRESHOLDS = {
    "P1_player_referee": {
        "min_sharpness": 15,          
        "min_closeup_ratio": 0.10,    
        "min_musiq_norm": 0.40,       # Relaxed from 0.55 (soft gate only for P1)
        "min_texture": 6.0,         
    },
    "P2_corner": {
        "min_sharpness": 15,          
        "min_closeup_ratio": 0.00,    
        "min_musiq_norm": 0.0,        
        "min_texture": 5.0,
    },
    "P3_side_staff": {
        "min_sharpness": 15,
        "min_closeup_ratio": 0.10,   
        "min_musiq_norm": 0.0,
        "min_texture": 4.0,
    },
    "P4_behind_goal": {
        "min_sharpness": 15,
        "min_closeup_ratio": 0.00,    
        "min_musiq_norm": 0.0,
        "min_texture": 4.0,
    },
}

# Cut detection parameters:
# - CUT_DIFF_THRESHOLD: how large the mean abs diff spike must be to count as a cut
# - CUT_DROP_RADIUS: drop +/- N extracted frames around the cut boundary
CUT_DIFF_THRESHOLD = 80.0
CUT_DROP_RADIUS = 0

# Per-segment how many frames we want to keep AFTER heavy scoring but BEFORE MUSIQ.
# Note: MUSIQ runs only on this top-k (per segment), which saves runtime.
# Increased for all types compared to old version for better variety.
PER_SEGMENT_KEEP = {
    "P1_player_referee": 12,  
    "P2_corner": 10,          
    "P3_side_staff": 8,       
    "P4_behind_goal": 8,      
}

# Hard floors: frame must pass these to be eligible for final selection.
MIN_FINAL_CONF = 0.70
MIN_FINAL_SCORE = 0.30

# If quota selection can't fill top_n (common when filters are strict),
# allow a slightly lower score floor while keeping confidence unchanged.
ALLOW_SCORE_FLOOR_FALLBACK = True
FALLBACK_MIN_FINAL_SCORE = 0.20

# Temporal diversity gap of frames BEFORE heavy stage
TEMPORAL_DIVERSITY_GAP = 5  # 2 seconds - avoid running YOLO on same action

# Final selection (global, output quality)
MIN_FRAME_GAP = 15          # 5 seconds - viewers see distinct moments

# Logo detection parameters
LOGO_CKPT_PATH = "models/logo/logo_sef_2024_resnet50.pth"
LOGO_THRESHOLD = 0.50   # logo presence threshold
LOGO_BATCH_SIZE = 32

K_PER_WINDOW = 5   # try 2, then 3 if still starving

#YOLO_PATH = "models/yolo/yolo11m-pose.pt"

def normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

# Final scoring weights: these sum to ~1.0.
# They define how much each signal contributes to "score_pre_mult" before segment multiplier.
WEIGHTS = normalize_weights({
    "content": 0.40,       # face/celebration/closeup (depends on keyframe_priority)
    "musiq": 0.20,         # IQA
    "sharp": 0.15,         # blur-derived sharp_norm 
})

# =============================================================================
# Temporal diversity filter (runs BEFORE heavy stage)
# =============================================================================
def temporal_diversity_filter_sliding_window(
    items: List[dict],
    window_size: int,
    k_per_window: int = 2,
) -> List[dict]:
    """
    Non-overlapping windows. Keep top-K per window by a cheap quality proxy.
    """
    if not items:
        return []

    sorted_by_time = sorted(items, key=lambda x: extract_frame_index(x["path"]))
    selected = []
    i = 0

    while i < len(sorted_by_time):
        window_start_idx = extract_frame_index(sorted_by_time[i]["path"])

        window = []
        j = i
        while j < len(sorted_by_time):
            frame_idx = extract_frame_index(sorted_by_time[j]["path"])
            if frame_idx < window_start_idx + window_size:
                window.append(sorted_by_time[j])
                j += 1
            else:
                break

        if window:
            # Sort window by cheap quality (same keys you used before)
            window_sorted = sorted(
                window,
                key=lambda x: (
                    x["model_confidence"],
                    x["sharpness"],
                    x["texture"],
                    x["motion_blur"],
                ),
                reverse=True,
            )
            selected.extend(window_sorted[:max(1, int(k_per_window))])

        i = j

    # Optional: de-dup in case of weird overlaps (shouldn’t happen, but safe)
    seen = set()
    out = []
    for it in selected:
        if it["path"] not in seen:
            out.append(it)
            seen.add(it["path"])
    return out



# =============================================================================
# QUOTA SELECTION (global stage)
# =============================================================================

def _quota_select(all_candidates: List[dict], top_n: int) -> List[dict]:
    """
    Select final keyframes across all segments using:
    1) Bucket quotas based on SEGMENT_QUOTA_FRAC
    2) Within each bucket: pick highest final_score first

    Inputs
    ------
    all_candidates: list of candidates from all segments (already scored + filtered)
    top_n: total number of keyframes to output

    Output
    ------
    final list of candidates (<= top_n)
    """
    # Bucket candidates by segment priority.
    buckets = {k: [] for k in SEGMENT_QUOTA_FRAC.keys()}
    for c in all_candidates:
        sp = c.get("segment_priority", None)
        if sp in buckets:
            buckets[sp].append(c)

    # Sort each bucket by best score first.
    for sp in buckets:
        buckets[sp] = sorted(buckets[sp], key=lambda x: x["final_score"], reverse=True)

    # Compute integer quotas by floor, then distribute remaining slots round-robin.
    keys = list(SEGMENT_QUOTA_FRAC.keys())
    quotas: Dict[str, int] = {}
    remaining = top_n

    for sp in keys:
        q = int(np.floor(SEGMENT_QUOTA_FRAC[sp] * top_n))
        quotas[sp] = q
        remaining -= q

    i = 0
    while remaining > 0:
        quotas[keys[i % len(keys)]] += 1
        remaining -= 1
        i += 1

    final: List[dict] = []
    used = set()

    def add(sorted_list: List[dict], need: int) -> int:
        """
        Add up to `need` frames from sorted_list into final, respecting:
        - no duplicates by path
        - min_frame_gap constraint against already selected frames
        """
        nonlocal final
        added = 0
        for cand in sorted_list:
            if added >= need:
                break
            if cand["path"] in used:
                continue

            # Frame index parsed from filename; used for time-gap filtering.
            ci = extract_frame_index(cand["path"])

            # Check distance to already-selected frames.
            # Enforce min_frame_gap for temporal diversity.
            ok = True
            for s in final:
                si = extract_frame_index(s["path"])
                if abs(ci - si) < MIN_FRAME_GAP:
                    ok = False
                    break
            if not ok:
                continue

            final.append(cand)
            used.add(cand["path"])
            added += 1
        return added

    # First pass: fill quotas bucket by bucket.
    for sp in keys:
        add(buckets[sp], quotas[sp])

    # Second pass: if still short, fill from best overall (still respecting gap).
    if len(final) < top_n:
        overall = sorted(all_candidates, key=lambda x: x["final_score"], reverse=True)
        add(overall, top_n - len(final))
    
    return final


# =============================================================================
# MAIN PIPELINE ENTRYPOINT
# =============================================================================

def select_keyframes(
    pred_csv: str,
    seg_csv: str,
    output_csv: str,
    output_dir: str,
    top_n: int = 15,
    device: str = "cuda",
    yolo_pose_path: str = "models/yolo/yolo11m-pose.pt",
    debug: bool = True,
):
    """
    This is the main selection stage (STEP 4 in your pipeline).

    It expects:
    - pred_csv: per-frame classifier output (frame_path, predicted label, confidence, etc.)
    - seg_csv: segment table produced earlier (segment_id, start_frame, end_frame, priority)

    It produces:
    - output_dir: image files of selected keyframes
    - output_csv: CSV with keyframes + full score breakdown per selected frame

    """

    # ---- Load heavy models ONCE (expensive startup) ----
    print("Initializing heavy models")
    models = SOTAModels(device=device, yolo_pose_path=yolo_pose_path, debug=debug)

    # ---- Load logo detection model ----
    logo_det = LogoDetector(LOGO_CKPT_PATH, device=device)

    # ---- Load CSV inputs ----
    df_preds = pd.read_csv(pred_csv)
    df_segs = pd.read_csv(seg_csv)

    # Parse frame index from frame filename so we can do segment slicing + time-gap checks.
    df_preds["frame_index"] = df_preds["frame_path"].apply(extract_frame_index)

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # ---- Analytics: counters and timing so you can tell what is happening ----
    analytics = {
        "frames_processed": 0,            # after confidence gate (MIN_FINAL_CONF)
        "frames_dropped_cutmask": 0,      # dropped around detected cuts
        "frames_filtered_conf": 0,        # dropped by classifier confidence
        "frames_filtered_overlay": 0,     # dropped by overlay proxy threshold
        "frames_filtered_blur": 0,        # dropped by sharpness threshold
        "frames_filtered_texture": 0,     # dropped by texture (uniformity) threshold
        "frames_filtered_motion_blur": 0,  # dropped by motion blur threshold
        "frames_filtered_temporal": 0,    # dropped by temporal diversity BEFORE heavy stage
        "frames_filtered_closeup": 0,     # dropped by pose-derived closeup threshold
        "segments_empty_after_prefilter": 0,  # segments that become empty after early filtering
        "p1_failed_musiq_gate_fallbacks": 0, # how often P1 needed MUSIQ fallback
        "candidates_by_segment_type": {
            "P1_player_referee": 0,
            "P2_corner": 0,
            "P3_side_staff": 0,
            "P4_behind_goal": 0
        },
        "timing": {
            "preprocess_sec": 0.0,          # cheap stage
            "temporal_diversity_sec": 0.0,  # temporal filter before heavy stage
            "sota_heavy_sec": 0.0,          # pose/face/celebration
            "musiq_sec": 0.0,               # MUSIQ only 
            "scoring_sec": 0.0,             # final scoring + hard floors
        }
    }

    # Global pool of candidates (after all segment processing).
    all_candidates: List[dict] = []

    print(f"{'='*60}")
    print("CLOSEUP-ONLY KEYFRAME SELECTION")
    print(f"Processing {len(df_segs)} closeup segments")
    print(f"{'='*60}\n")

    # =============================================================================
    # Iterate over segments (this is where most work happens)
    # =============================================================================
    for _, seg in tqdm(df_segs.iterrows(), total=len(df_segs), desc="Processing segments"):
        # Segment id from segments.csv
        seg_id = int(seg["segment_id"])

        # Priority label from closeup hierarchy (P1/P2/P3/P4).
        seg_priority = str(seg.get("priority", "P1_player_referee"))

        # Segment multiplier to boost/penalize final score.
        seg_mult = float(SEGMENT_SCORE_MULT.get(seg_priority, 1.0))

        # Thresholds specific to that segment type.
        th = QUALITY_THRESHOLDS.get(seg_priority, QUALITY_THRESHOLDS["P1_player_referee"])

        # Per-segment limit for how many candidates go to MUSIQ/scoring.
        keep_k = int(PER_SEGMENT_KEEP.get(seg_priority, 6))

        # Segment frame range (paths contain frame indices in their filename).
        start = extract_frame_index(seg["start_frame"])
        end = extract_frame_index(seg["end_frame"])

        # Slice predictions to only frames within this segment.
        segment_frames = df_preds[(df_preds.frame_index >= start) & (df_preds.frame_index <= end)]

        if debug:
            print("\n" + "-" * 72)
            print(f"[SEGMENT] id={seg_id} priority={seg_priority} frames_in_range={len(segment_frames)}")
            print(f"[SEGMENT] start_idx={start} end_idx={end} keep_k={keep_k} mult={seg_mult}")
            print("-" * 72)

        # ---------------------------------------------------------------------
        # STAGE 1: PREPROCESSING
        # Goal: remove bad frames cheaply before you run any heavy networks.
        # ---------------------------------------------------------------------
        if debug:
            print("[PREPROCESS] >>> Stage 1: cut-mask + cheap filters (overlay/blur/texture/conf)")

        t_pre = time.time()

        # Sort paths by time so cut detection makes sense (diff between consecutive frames).
        seg_paths = sorted(segment_frames["frame_path"].tolist(), key=extract_frame_index)

        # Detect cut boundaries and mark +/- radius frames to drop.
        cut_drop = detect_and_mask_cuts(
            seg_paths,
            diff_thresh=CUT_DIFF_THRESHOLD,
            drop_radius=CUT_DROP_RADIUS
        )

        pre_items: List[dict] = []
        for _, row in segment_frames.iterrows():
            path = row["frame_path"]

            # 1) Drop frames near cuts (these are often blurred/transition frames).
            if path in cut_drop:
                analytics["frames_dropped_cutmask"] += 1
                continue

            # 2) Classifier confidence gate:
            # If the shot-type classifier isn't confident, don't waste compute on it.
            model_conf = float(row.get("confidence", 0.5))
            if model_conf < MIN_FINAL_CONF:
                analytics["frames_filtered_conf"] += 1
                continue

            # From here onward, we consider it "processed".
            analytics["frames_processed"] += 1

            # 3) Overlay / transition proxy:
            # Uses gray stddev heuristic to reject flat/overlay frames.
            overlay = transition_overlay_score(path)
            if overlay < 18:
                analytics["frames_filtered_overlay"] += 1
                continue

            # 4) Blur test: Laplacian variance.
            lap = blur_laplacian_var(path)
            if lap < th["min_sharpness"]:
                analytics["frames_filtered_blur"] += 1
                continue

            # 5) Uniformity test: texture proxy (edge density).
            tex = texture_proxy(path)
            if tex < float(th.get("min_texture", 0.0)):
                analytics["frames_filtered_texture"] += 1
                continue

            # 6) Motion blur test: motion blur score.
            motion_blur = motion_blur_score(path)
            if motion_blur < 0.70:  # Tune this (0.45 = moderate threshold)
                analytics["frames_filtered_motion_blur"] += 1
                continue

            # If it passes all cheap filters, keep it for the next stage.
            pre_items.append({
                "path": path,
                "segment_id": seg_id,
                "segment_priority": seg_priority,
                "segment_mult": seg_mult,
                "model_confidence": model_conf,
                "sharpness": float(lap),         # laplacian variance
                "texture": float(tex),
                "motion_blur": float(motion_blur),
            })

        analytics["timing"]["preprocess_sec"] += (time.time() - t_pre)

        if debug:
            print(f"[PREPROCESS] <<< Stage 1 done: kept={len(pre_items)} / processed={len(segment_frames)} "
                  f"(cut_drop={len(cut_drop)})")

        # If nothing survives cheap filters, segment contributes nothing.
        if not pre_items:
            analytics["segments_empty_after_prefilter"] += 1
            if debug:
                print("[PREPROCESS] !!! Segment empty after cheap filters. Skipping segment.")
            continue

        # ---------------------------------------------------------------------
        # STAGE 2A: TEMPORAL DIVERSITY
        # Goal: Remove temporally close frames BEFORE running heavy models.
        # ---------------------------------------------------------------------
        if debug:
            print(f"[TEMPORAL] >>> Stage 2A: temporal diversity (gap={TEMPORAL_DIVERSITY_GAP})")

        t_temp = time.time()
        before_temporal = len(pre_items)

        # Apply temporal filter - frames must be at least TEMPORAL_DIVERSITY_GAP apart
        pre_items = temporal_diversity_filter_sliding_window(pre_items, TEMPORAL_DIVERSITY_GAP, k_per_window=K_PER_WINDOW)
        analytics["frames_filtered_temporal"] += (before_temporal - len(pre_items))
        analytics["timing"]["temporal_diversity_sec"] += (time.time() - t_temp)

        if debug:
            print(f"[TEMPORAL] <<< {before_temporal} -> {len(pre_items)} (saved {before_temporal - len(pre_items)} YOLO calls!)")

        if not pre_items:
            continue


        # ---------------------------------------------------------------------
        # STAGE 2B : LOGO DETECTION FILTER
        # Goal: remove frames with prominent logos/branding.
        # ---------------------------------------------------------------------
        before_logo = len(pre_items)
        paths = [it["path"] for it in pre_items]

        try:
            logo_probs = logo_det.predict_logo_prob(paths, batch_size=LOGO_BATCH_SIZE)
        except Exception as e:
            if debug:
                print(f"[LOGO] ERROR: Logo detection failed: {e}")
                print(f"[LOGO] Skipping logo filter, keeping all {before_logo} frames")
            logo_probs = [0.0] * len(paths)

        kept = []
        for it, p in zip(pre_items, logo_probs):
            it["logo_prob"] = float(p)
                        
            if p >= LOGO_THRESHOLD:
                 continue
                        
            kept.append(it)

        pre_items = kept
        after_logo = len(pre_items)

        if debug:
            print(f"[LOGO] <<< Stage 2C logo: {before_logo} -> {after_logo} (thr={LOGO_THRESHOLD})")
            if before_logo > 0:
                avg_prob = sum(logo_probs) / len(logo_probs)
                print(f"[LOGO] Average logo prob: {avg_prob:.3f}")

        # ---------------------------------------------------------------------
        # STAGE 3: SOTA HEAVY SIGNALS
        # Goal: compute semantic signals:
        # - closeup ratio from pose boxes
        # - faces + face quality + emotion intensity
        # - celebration cue from pose
        # ---------------------------------------------------------------------
        if debug:
            print("[SOTA] >>> Stage 3: HEAVY scoring signals (pose closeup, face, celebration)")

        t_sota = time.time()

        candidates: List[dict] = []
        for it in pre_items:
            path = it["path"]
            
            closeup = models.closeup_ratio_from_pose(path)
            if closeup < th["min_closeup_ratio"]:
                analytics["frames_filtered_closeup"] += 1
                continue

            # Face quality
            num_faces, face_area, face_q, face_det, detected_faces, img = models.face_quality(path)
            # Emotion detecton
            emo_intensity, emo_label = models.emotion_intensity_from_faces(path, detected_faces, img=img, max_faces=2)
            # Pose detection
            pose_score, pose_label, pose_breakdown = models.pose_signals(path)

            # ===== CONFIG SECTION =====

            # Tier-specific signal weights (used in both STAGE 3 and STAGE 5)
            TIER_WEIGHTS = {
                1: {"face": 0.35, "emotion": 0.35, "pose": 0.30},  # Complete celebration
                2: {"face": 0.55, "emotion": 0.45},                # Expressive face
                3: {"pose": 0.65, "face": 0.35},                   # Pose-driven
                4: {"closeup": 0.60, "face": 0.40},                # Generic closeup
            }
            # =============================================================

            FACE_Q_THR = 0.50          # tune
            FACE_AREA_MIN = 0.03       # tune (if you normalized face_area)
            FACE_AREA_MAX = 0.40  
            EMO_THR = 0.20             # Expressive emotion threshold
            POSE_THR = 0.40            # Clear pose threshold

            has_good_face = (
                num_faces > 0 and 
                face_q >= FACE_Q_THR and 
                FACE_AREA_MIN <= face_area <= FACE_AREA_MAX
            )

            has_emotion = emo_intensity > EMO_THR
            has_pose = pose_score > POSE_THR

            # ===== STAGE 3 (using TIER_WEIGHTS) =====
            if has_good_face and has_emotion and has_pose:
                kf_priority = 1
                w = TIER_WEIGHTS[1]
                rank_value = (w["face"] * face_q + w["emotion"] * emo_intensity + w["pose"] * pose_score)
                
            elif has_good_face and has_emotion:
                kf_priority = 2
                w = TIER_WEIGHTS[2]
                rank_value = w["face"] * face_q + w["emotion"] * emo_intensity
                
            elif has_pose:
                kf_priority = 3
                w = TIER_WEIGHTS[3]
                rank_value = w["pose"] * pose_score + w["face"] * face_q
                
            else:
                kf_priority = 4
                w = TIER_WEIGHTS[4]
                rank_value = w["closeup"] * closeup + w["face"] * face_q


            candidates.append({
                "path": path,
                "segment_id": seg_id,
                "segment_priority": seg_priority,
                "segment_mult": seg_mult,
                "model_confidence": it["model_confidence"],

                "keyframe_priority": int(kf_priority),
                "rank_value": float(rank_value),

                "num_faces": int(num_faces),
                "face_area": float(face_area),
                "face_quality": float(face_q),
                "face_det_score": float(face_det),
                            
                "emotion_intensity": float(emo_intensity),
                "emotion_label": str(emo_label),

                "pose_score": float(pose_score),
                "pose_label": str(pose_label),
                "pose_breakdown": pose_breakdown,  
                    
                "closeup": float(closeup),
                "sharpness": float(it["sharpness"]),

            })

        analytics["timing"]["sota_heavy_sec"] += (time.time() - t_sota)

        if debug:
                print(f"[EMO] {os.path.basename(path)} faces={num_faces} fq={face_q:.2f} "
                    f"emo={emo_intensity:.2f} label={emo_label}")

        if debug:
            print(f"[SOTA] <<< Stage 3 done: candidates_after_pose/face/celebration={len(candidates)}")

        if not candidates:
            analytics["segments_empty_after_prefilter"] += 1
            if debug:
                print("[SOTA] !!! No candidates survived heavy stage. Skipping segment.")
            continue

        # Per-segment top-k BEFORE MUSIQ:
        # Sort by keyframe_priority (1 best) then rank_value (descending).
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (x["keyframe_priority"], -x["rank_value"])
        )[:keep_k]

        if debug:
            print(f"[SELECT] Segment top-k before MUSIQ: {len(candidates_sorted)} (keep_k={keep_k})")

        # ---------------------------------------------------------------------
        # STAGE 4: MUSIQ 
        # Goal: compute IQA only on the already-reduced per-segment top-k.
        # ---------------------------------------------------------------------
        if debug:
            print("[IQA] >>> Stage 4: BATCHED MUSIQ on segment top-k")

        t_m = time.time()
        
        musiq_paths = [c["path"] for c in candidates_sorted]
        musiq_scores = models.musiq_norm_batch(musiq_paths)
        
        for c, score in zip(candidates_sorted, musiq_scores):
            c["musiq_norm"] = score
            c["quality_fallback"] = 0

        analytics["timing"]["musiq_sec"] += (time.time() - t_m)

        if debug:
            print(f"[IQA] <<< Batched {len(musiq_scores)} in {time.time() - t_m:.2f}s")

        # P1 MUSIQ soft gate:
        # Only P1 uses a quality gate; BUT you never delete the entire segment.
        # If all fail, you keep the best 2 anyway (fallback) so P1 doesn't go empty.
        if seg_priority == "P1_player_referee":
            gated = [c for c in candidates_sorted if c["musiq_norm"] >= th["min_musiq_norm"]]
            if not gated:
                analytics["p1_failed_musiq_gate_fallbacks"] += 1
                if debug:
                    print("[IQA] !!! P1 MUSIQ gate failed for all. Using fallback: keep best 2 by priority/rank/sharp.")
                candidates_sorted = sorted(
                    candidates_sorted,
                    key=lambda x: (x["keyframe_priority"], -x["rank_value"], -x["sharpness"])
                )[:2]
                for c in candidates_sorted:
                    c["quality_fallback"] = 1
            else:
                candidates_sorted = gated
                if debug:
                    print(f"[IQA] P1 MUSIQ gate passed: kept {len(candidates_sorted)}")

        # ---------------------------------------------------------------------
        # STAGE 5: FINAL SCORE + HARD FLOORS
        # Goal: compute score_pre_mult (0..1-ish), multiply by segment_mult, and keep only strong frames.
        # ---------------------------------------------------------------------
        if debug:
            print("[SCORING] >>> Stage 5: final_score + hard floors")

        t_sc = time.time()

        # Sigmoid-compressed sharpness normalization
        sharp_vals = [c["sharpness"] for c in candidates_sorted]
        if len(sharp_vals) >= 2:
            sharp_p50 = float(np.percentile(sharp_vals, 50))
            sharp_p95 = float(np.percentile(sharp_vals, 95))
            sharp_range = max(sharp_p95 - sharp_p50, 1.0)  # Avoid div-by-zero
        else:
            sharp_p50 = 100.0
            sharp_range = 100.0

        scored: List[dict] = []
        for c in candidates_sorted:

            # ---------- BUILD 3 SEMANTIC SIGNALS (face / celebration / closeup) ----------
            
            # 1. Face signal: Quality is primary, extra faces add bonus.
            face_signal = 0.0
            if int(c.get("num_faces", 0)) > 0:
                face_q = float(c.get("face_quality", 0.0))
                nfaces = int(c.get("num_faces", 0))

                # Quality is the base
                face_signal = face_q
                
                # Multi-face bonus (small)
                if nfaces > 1:
                    multi_bonus = 0.05 * min((nfaces - 1) / 5.0, 1.0)
                    face_signal = min(face_signal + multi_bonus, 1.0)

            # 2. Emotion signal (separate)
            emo_signal = float(np.clip(c.get("emotion_intensity", 0.0), 0.0, 1.0))
            
            # 3. Celebration signal: from pose score only
            pose_signal = float(np.clip(c.get("pose_score", 0.0), 0.0, 1.0))
            
            # 4. Close signal: from closeup ratio only
            close_signal = float(np.clip(c.get("closeup", 0.0), 0.0, 1.0))


            # ---------- PRIORITY-BASED MIX (4-tier system) ----------             

            # Apply tier-specific weights (SAME as STAGE 3)
            kf_priority = c["keyframe_priority"]
            w = TIER_WEIGHTS[kf_priority]

            if kf_priority == 1:
                content_src = f"celebration:{c.get('pose_label', 'none')}"
                content_signal = (w["face"] * face_signal + w["emotion"] * emo_signal + w["pose"] * pose_signal)
                
            elif kf_priority == 2:
                content_src = f"expressive:{c.get('emotion_label', 'none')}"
                content_signal = w["face"] * face_signal + w["emotion"] * emo_signal
                
            elif kf_priority == 3:
                content_src = f"pose:{c.get('pose_label', 'none')}"
                content_signal = w["pose"] * pose_signal + w["face"] * face_signal
                
            else:
                content_src = "closeup"
                content_signal = w["closeup"] * close_signal + w["face"] * face_signal
                        
            content_signal = float(np.clip(content_signal, 0.0, 1.0))


            # ---------- NORMALIZE OTHER FEATURES ----------

            # Sigmoid compression: expands top region, no saturation
            z = (c["sharpness"] - sharp_p50) / sharp_range  # Center at median, scale by p50→p95
            sharp_norm = float(1.0 / (1.0 + np.exp(-2.5 * z)))  # Sigmoid with slope=2.5
           
            musiq_val = float(np.clip(c.get("musiq_norm", 0.0), 0.0, 1.0))

            # ---------- WEIGHTED CONTRIBUTIONS ----------
            w_content = WEIGHTS["content"] * content_signal
            w_musiq   = WEIGHTS["musiq"] * musiq_val
            w_sharp   = WEIGHTS["sharp"] * sharp_norm

            score_pre = w_content + w_musiq + w_sharp 

            # ---------- FINAL SCORE ----------
            c["final_score"] = float(score_pre * c["segment_mult"])

            # ---------- STORE BREAKDOWN FOR CSV ----------
            # Extra useful debug columns (optional but recommended)
            c["face_signal"] = float(face_signal)
            c["pose_signal"] = float(pose_signal)
            c["close_signal"] = float(close_signal)

            c["content_source"] = content_src
            c["content_signal"] = content_signal
            c["sharp_norm"] = sharp_norm

            c["emotion_intensity"] = float(c.get("emotion_intensity", 0.0))
            c["emotion_label"] = c.get("emotion_label", "none")

            c["w_content"] = w_content
            c["w_musiq"]   = w_musiq
            c["w_sharp"]   = w_sharp
            c["score_pre_mult"] = score_pre

            # ---------- HARD FLOORS ----------
            # A frame is only eligible if:
            # - confidence is above MIN_FINAL_CONF
            # - final_score is above MIN_FINAL_SCORE
            if c["model_confidence"] < MIN_FINAL_CONF:
                continue
            if c["final_score"] < MIN_FINAL_SCORE:
                continue

            scored.append(c)

        analytics["timing"]["scoring_sec"] += (time.time() - t_sc)

        if debug:
            print(f"[SCORING] <<< Stage 5 done: kept={len(scored)} after hard floors "
                  f"(MIN_CONF={MIN_FINAL_CONF}, MIN_SCORE={MIN_FINAL_SCORE})")

        if not scored:
            if debug:
                print("[SCORING] !!! No candidates survived scoring floors. Skipping segment.")
            continue

        # Track how many candidates per segment type survived into the global pool.
        if seg_priority in analytics["candidates_by_segment_type"]:
            analytics["candidates_by_segment_type"][seg_priority] += len(scored)

        all_candidates.extend(scored)

    # =============================================================================
    # Global finalize: quotas + time gap + fallback fill + save outputs
    # =============================================================================
    if not all_candidates:
        print("[WARN] No candidates found. Prefilters too strict or model paths failing.")
        pd.DataFrame([]).to_csv(output_csv, index=False)
        return

    print("\n" + "=" * 72)
    print(f"[FINAL] >>> Global selection from {len(all_candidates)} candidates")
    print("=" * 72)

    # Quota-based selection.
    final_selection = _quota_select(all_candidates, top_n=top_n)

    # Final hard filter (quota cannot override confidence/score).
    final_selection = [
        c for c in final_selection
        if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
        and float(c.get("final_score", 0.0)) >= MIN_FINAL_SCORE
    ]

    # Optional fallback fill if strict score floor caused quota to under-fill.
    if len(final_selection) < top_n and ALLOW_SCORE_FLOOR_FALLBACK:
        print(f"[FINAL] Quota could not fill {top_n}. Trying fallback score floor {FALLBACK_MIN_FINAL_SCORE} (conf unchanged).")
        extra = sorted(
            [
                c for c in all_candidates
                if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
                and float(c.get("final_score", 0.0)) >= FALLBACK_MIN_FINAL_SCORE
                and c not in final_selection
            ],
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        # Still enforce min_frame_gap even in fallback
        for c in extra:
            if len(final_selection) >= top_n:
                break
            
            # Check temporal gap against already selected frames
            ci = extract_frame_index(c["path"])
            ok = True
            for s in final_selection:
                si = extract_frame_index(s["path"])
                if abs(ci - si) < MIN_FRAME_GAP:  # Use same min_frame_gap as _quota_select
                    ok = False
                    break
            
            if ok:
                final_selection.append(c)

    print(f"[FINAL] <<< Selected: {len(final_selection)} / {top_n}\n")

    # -----------------------------------------------------------------------------
    # Save selected images + write final CSV.
    # The CSV includes all the individual score contributions so you can debug:
    # - content_source tells you what "content" was (face/celebration/closeup)
    # - w_* columns show EXACT boost amounts inside score_pre_mult
    # -----------------------------------------------------------------------------
    results = []
    for rank, c in enumerate(final_selection):
        pr = c.get("segment_priority", "NA")
        conf = float(c.get("model_confidence", 0.0))
        score = float(c.get("final_score", 0.0))

        out_path = os.path.join(output_dir, f"rank{rank+1:02d}_{pr}_conf{conf:.2f}_score{score:.3f}.jpg")

        img = cv2.imread(c["path"])
        if img is not None:
            cv2.imwrite(out_path, img)

        results.append({
            # --- identity / ranking ---
            "rank": rank + 1,  # Final rank after global quota selection and time-gap filtering
            "segment_id": c["segment_id"],  # Temporal segment ID this frame was selected from
            "segment_priority": pr,  # Semantic segment class (P1 player/referee, P2 corner, P3 staff, P4 behind goal)
            "segment_multiplier": round(c["segment_mult"], 3),  # Hierarchy boost applied to the score (higher = more important)

            # --- content decision ---
            "keyframe_priority": int(c["keyframe_priority"]),          # Which semantic signal dominated: 1=face, 2=celebration, 3=closeup fallback
            "content_source": c.get("content_source", "NA"),           # Human-interpretable: face / celebration / closeup
            "content_signal": round(c.get("content_signal", 0.0), 3),  # Normalized [0,1] semantic strength of chosen content signal

            # --- final score ---
            "final_score": round(score, 3),     # Final ranking score = score_pre_mult × segment_multiplier
            "score_pre_mult": round(c.get("score_pre_mult", 0.0), 4),     # Raw weighted score before applying segment hierarchy multiplier

            # --- weighted contributions (these SUM to score_pre_mult) ---
            "w_content": round(c.get("w_content", 0.0), 4),     # Contribution from semantic importance (faces / celebration / closeup)
            "w_musiq": round(c.get("w_musiq", 0.0), 4),         # Contribution from MUSIQ image quality assessment
            "w_sharp": round(c.get("w_sharp", 0.0), 4),         # Contribution from Laplacian-based sharpness (percentile-normalized)

            # --- normalized raw signals ---
            "musiq_norm": round(c.get("musiq_norm", 0.0), 3),     # Normalized MUSIQ score [0,1]
            "sharp_norm": round(c.get("sharp_norm", 0.0), 3),     # Normalized sharpness [0,1] (percentile-based, no saturation)
            "model_confidence": round(conf, 3),                   # Classifier confidence for this frame

            # --- semantic context ---
            "num_faces": int(c.get("num_faces", 0)),                # Number of detected faces
            "face_quality": round(c.get("face_quality", 0.0), 3),   # Combined face quality score (size, centering, detection confidence)

            # --- emotion ---
            "emotion_intensity": round(c.get("emotion_intensity", 0.0), 3),
            "emotion_label": c.get("emotion_label", "none"),

            # --- signal breakdown (super useful to debug) ---
            "face_signal": round(c.get("face_signal", 0.0), 3),
            "pose_signal": round(c.get("pose_signal", 0.0), 3),
            "close_signal": round(c.get("close_signal", 0.0), 3),
    
            # --- bookkeeping ---
            "quality_fallback": int(c.get("quality_fallback", 0)),     # Flag: MUSIQ gate fallback was used
            "input_path": c["path"],     # Original extracted frame path
            "saved_path": out_path,     # Output path of selected keyframe image

        })


    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    # -----------------------------------------------------------------------------
    # Print analytics summary
    # This tells if preprocessing + SOTA stages are actually filtering anything.
    # -----------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYTICS")
    print(f"{'='*60}")
    print(f"Frames processed (after conf gate): {analytics['frames_processed']}")
    print(f"Dropped by cut-mask:              {analytics['frames_dropped_cutmask']}")
    print(f"Filtered by conf (<{MIN_FINAL_CONF}):        {analytics['frames_filtered_conf']}")
    print(f"Filtered overlay:                {analytics['frames_filtered_overlay']}")
    print(f"Filtered blur:                   {analytics['frames_filtered_blur']}")
    print(f"Filtered texture:                {analytics['frames_filtered_texture']}")
    print(f"Filtered motion blur:            {analytics['frames_filtered_motion_blur']}")
    print(f"Filtered temporal                {analytics['frames_filtered_temporal']}")
    print(f"Filtered closeup (pose):         {analytics['frames_filtered_closeup']}")
    print(f"Segments empty after prefilter:  {analytics['segments_empty_after_prefilter']}")
    print(f"P1 MUSIQ gate fallbacks used:    {analytics['p1_failed_musiq_gate_fallbacks']}")
    print("\nCandidates by closeup type:")
    for k, v in analytics["candidates_by_segment_type"].items():
        print(f"  {k}: {v}")

    print("\nTiming breakdown:")
    for k, v in analytics["timing"].items():
        print(f"  {k}: {v:.2f}s")

    print(f"\nSaved {len(results_df)} keyframes -> {output_csv}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI entrypoint
# =============================================================================
if __name__ == "__main__":
    import argparse

    # Command-line interface for running this module directly.
    parser = argparse.ArgumentParser(description="Keyframe Selection")
    parser.add_argument("--pred_csv", required=True)        # predictions file (per frame)
    parser.add_argument("--seg_csv", required=True)         # segment ranges + priorities
    parser.add_argument("--output_csv", required=True)      # output CSV path
    parser.add_argument("--output_dir", required=True)      # where to save keyframe images
    parser.add_argument("--top_n", type=int, default=10)    # total output keyframes
    parser.add_argument("--device", type=str, default="cuda")  # cuda or cpu
    parser.add_argument("--yolo_pose_path", type=str, default="models/yolo/yolo11m-pose.pt")
    parser.add_argument("--debug", action="store_true", help="Verbose stage prints inside segments")
    args = parser.parse_args()

    # Run the selection.
    select_keyframes(
        pred_csv=args.pred_csv,
        seg_csv=args.seg_csv,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        device=args.device,
        yolo_pose_path=args.yolo_pose_path,
        debug=args.debug,
    )