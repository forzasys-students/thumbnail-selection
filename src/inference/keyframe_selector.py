# keyframe_selector.py
"""
SOTA KEYFRAME SELECTION PIPELINE

What this file does (high level):
1) Load frame-level predictions (pred_csv) + segment ranges (seg_csv)
2) For each segment:
   A) HECATE-style preprocessing filters (luminance/sharpness/uniformity + extensions) to remove bad frames
   B) Redundancy reduction to remove near-duplicates + Logo detection filter to remove branded frames 
   C) Scoring signals (face/emotion/pose - detection)
   D) Image quality assessment (TOPIQ) on segment candidates
   E) Compute final score and keep only strong candidates
   F) Redundancy reduction to remove near-duplicate frames
3) Global selection across all segments 
4) Save selected keyframes and a CSV with score breakdown

Close up shot types:
- P1_player_referee - BEST thumbnails
- P2_corner 
- P3_side_staff 
- P4_behind_goal 

"""

from __future__ import annotations
from sota_models import SOTAModels
from logo_detector import LogoDetector
from redundancy_reduction import reduce_redundancy, CLUSTER_DEBUG_DATA

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
    compute_frame_metrics,      # HECATE filters, plus overlay and texture proxies
)

# =============================================================================
# CONFIG (tuning knobs)
# =============================================================================

# Segment-specific multiplier: boosts or penalizes final score by segment type.
SEGMENT_SCORE_MULT = {
    "P1_player_referee": 1.10,   # Highest - action/emotion shots (best thumbnails)
    "P2_corner": 1.00,           # High - set pieces, tactical moments
    "P3_side_staff": 1.05,       # Medium - coach reactions, bench celebrations
    "P4_behind_goal": 1.00,      # Baseline - goalkeeper shots, different angle
}

# Global quota fractions: ensures variety across closeup types.
SEGMENT_QUOTA_FRAC = {
    "P1_player_referee": 0.70,   # 70% from player/referee closeups
    "P2_corner": 0.10,           # 10% from corner closeups
    "P3_side_staff": 0.10,       # 10% from staff/bench closeups
    "P4_behind_goal": 0.10,      # 10% from behind-goal closeups
}

# Shot-type-specific thresholds for filtering.
QUALITY_THRESHOLDS = {
    "P1_player_referee": {
        "min_luminance": 50.0,       # reject dark frames
        "min_sharpness": 15.0,       # reject blurry frames (gradient magnitude)
        "max_uniformity": 0.70,       # reject flat/uniform frames
        "min_texture": 5.0,          # edge density check
        "min_closeup_ratio": 0.05,   # pose-based closeup proxy (tune this based on your pose model's output)
        "min_iqa_norm": 0.05,        
    },
    "P2_corner": {
        "min_luminance": 50.0,       
        "min_sharpness": 15.0,       
        "max_uniformity": 0.70,       
        "min_texture": 5.0,
        "min_closeup_ratio": 0.05,    
        "min_iqa_norm": 0.0,        
    },
    "P3_side_staff": {
        "min_luminance": 50.0,
        "min_sharpness": 15.0,
        "max_uniformity": 0.70,
        "min_texture": 4.0,
        "min_closeup_ratio": 0.05,   
        "min_iqa_norm": 0.0,
    },
    "P4_behind_goal": {
        "min_luminance": 50.0,
        "min_sharpness": 15.0,
        "max_uniformity": 0.70,
        "min_texture": 4.0,
        "min_closeup_ratio": 0.05,    
        "min_iqa_norm": 0.0,
    },
}


ENABLE_REDUNDANCY_REDUCTION = True  # Removes near-duplicate frames. Turn off to disable.
USE_QUOTA_SELECTION = False         # True = quota-based, False = global ranking

# Fallback parameters 
ALLOW_SCORE_FLOOR_FALLBACK = True    # Enable/disable fallback
FALLBACK_MIN_FINAL_SCORE = 0.20      # Lower score floor for fallback

# Hard floors: frame must pass these to be eligible for final selection.
MIN_FINAL_CONF = 0.40
MIN_FINAL_SCORE = 0.30 

# Logo detection parameters
LOGO_CKPT_PATH = "models/logo/logo_sef_2024_resnet50.pth"
LOGO_THRESHOLD = 0.50   # logo presence threshold
LOGO_BATCH_SIZE = 32    # logo detection batch size

def normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

# weights as scaling factors to balance the contribution of each signal to the final score (before segment multiplier).
WEIGHTS = normalize_weights({
    #"content": 0.60,        # face/celebration/closeup (depends on keyframe_priority)
    "iqa": 0.35,             # Image quality assessment (TOPIQ)
    "face": 0.25,            # face quality signal
    "emotion": 0.20,         # emotion signal
    "pose": 0.20,            # pose signal
})

# Compute a single proxy score from the raw metrics to use for early redundancy reduction (before heavy models).
def compute_proxy_from_metrics(luminance, sharpness, texture):

    # Normalize signals
    sharp_norm = min(sharpness, 200.0) / 200.0
    lum_norm = np.clip((luminance - 40.0) / 140.0, 0.0, 1.0)
    tex_norm = np.clip(texture / 50.0, 0.0, 1.0)

    proxy = (
        0.6 * sharp_norm +
        0.2 * lum_norm +
        0.2 * tex_norm 
    )

    return float(np.clip(proxy, 0.0, 1.0))


# =============================================================================
# QUOTA SELECTION 
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
    top_n: int = 50,
    device: str = "cuda",
    yolo_pose_path: str = "models/yolo/yolo26m-pose.pt",
    debug: bool = True,
    video_id: str = "unknown",
    redundancy_reduction: bool = True,
    fps: float = 24.0,
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

    t_pipeline_start = time.time()   

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
        "frames_filtered_conf": 0,        # dropped by classifier confidence
    
        "frames_filtered_luminance": 0,
        "frames_filtered_sharpness": 0,
        "frames_filtered_uniformity": 0,

        "frames_filtered_overlay": 0,     # dropped by overlay proxy threshold
        "frames_filtered_texture": 0,     # dropped by texture (uniformity) threshold

        "frames_filtered_closeup": 0,     # dropped by pose-derived closeup threshold
        "frames_dropped_redundancy": 0,   # dropped by hybrid redundancy reduction (CLIP + temporal)
        "frames_filtered_logo": 0,        # dropped by logo detection

        "segments_empty_after_prefilter": 0,  # segments that become empty after early filtering
        "candidates_by_segment_type": {
            "P1_player_referee": 0,
            "P2_corner": 0,
            "P3_side_staff": 0,
            "P4_behind_goal": 0
        },
        "timing": {
            "preprocess_sec": 0.0,          # cheap stage
            "redundancy_sec": 0.0,
            "logo_sec": 0.0,
            "face_sec": 0.0,                # face quality + detection
            "emotion_sec": 0.0,             # emotion intensity
            "pose_sec": 0.0,                # pose detection
            "iqa_sec": 0.0,                 # TOPIQ only 
            "sota_heavy_sec": 0.0,          # total heavy stage time
        }
    }

    # Global pool of candidates (after all segment processing).
    all_candidates: List[dict] = []

    print(f"{'='*60}")
    print("CLOSEUP-ONLY KEYFRAME SELECTION")
    print(f"Processing {len(df_segs)} closeup segments")
    print(f"{'='*60}\n")

    # =============================================================================
    # Iterate over segments 
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

        # Segment frame range (paths contain frame indices in their filename).
        start = extract_frame_index(seg["start_frame"])
        end = extract_frame_index(seg["end_frame"])

        # Slice predictions to only frames within this segment.
        segment_frames = df_preds[(df_preds.frame_index >= start) & (df_preds.frame_index <= end)]

        if debug:
            print("\n" + "-" * 72)
            print(f"[SEGMENT] id={seg_id} priority={seg_priority} frames_in_range={len(segment_frames)}")
            print("-" * 72)

        # ======================================================================
        # STAGE 1: PREPROCESSING
        # ======================================================================
        if debug:
            print("[PREPROCESS] >>> Stage 1: cheap filters (overlay/blur/texture/conf)")

        t_pre = time.time()

        pre_items: List[dict] = []
        for _, row in segment_frames.iterrows():
            path = row["frame_path"]

            # Classifier confidence gate:
            # If the shot-type classifier isn't confident, don't waste compute on it.
            model_conf = float(row.get("confidence", 0.5))
            if model_conf < MIN_FINAL_CONF:
                analytics["frames_filtered_conf"] += 1
                continue

            analytics["frames_processed"] += 1

            metrics = compute_frame_metrics(path)
            if metrics is None:
                continue  # unreadable image — skip silently
 
            luminance  = metrics["luminance"]
            sharpness  = metrics["sharpness"]
            uniformity = metrics["uniformity"]
            overlay    = metrics["overlay"]
            tex        = metrics["texture"]
 
            if luminance < th["min_luminance"]:
                analytics["frames_filtered_luminance"] += 1
                continue
 
            if sharpness < th["min_sharpness"]:
                analytics["frames_filtered_sharpness"] += 1
                continue
 
            if uniformity > th["max_uniformity"]:
                analytics["frames_filtered_uniformity"] += 1
                continue
 
            if overlay < 18:
                analytics["frames_filtered_overlay"] += 1
                continue
 
            if tex < float(th.get("min_texture", 0.0)):
                analytics["frames_filtered_texture"] += 1
                continue


            proxy_score = compute_proxy_from_metrics(luminance,sharpness,tex)

            # If it passes all cheap filters, keep it for the next stage.
            pre_items.append({
                "path": path,
                "segment_id": seg_id,
                "segment_priority": seg_priority,
                "segment_mult": seg_mult,
                "model_confidence": model_conf,
                "luminance": float(luminance),
                "sharpness": float(sharpness),
                "uniformity": float(uniformity),
                "texture": float(tex),
                "proxy_score": float(proxy_score),
            })

        analytics["timing"]["preprocess_sec"] += (time.time() - t_pre)

        if debug:
            print(f"[PREPROCESS] <<< Stage 1 done: kept={len(pre_items)} / processed={len(segment_frames)} ")

        # If nothing survives cheap filters, segment contributes nothing.
        if not pre_items:
            analytics["segments_empty_after_prefilter"] += 1
            if debug:
                print("[PREPROCESS] !!! Segment empty after cheap filters. Skipping segment.")
            continue


        # ==============================
        # STAGE 2A: REDUNDANCY REDUCTION
        # ==============================
        # Hybrid: CLIP first (semantic near-duplicates), then temporal safety
        # net for edge cases CLIP lets through — frames that look just different
        # enough to survive the similarity threshold but whose frame indices are
        # too close to be distinct moments.
        # temporal_window is derived from fps so the window always represents
        # 0.5s of real video time regardless of extraction rate.
        #   5fps  -> window=3   (0.5s * 5  = 2.5, rounds to 3)
        #   12fps -> window=6   (0.5s * 12 = 6)
        #   24fps -> window=12  (0.5s * 24 = 12)
        t_rr = time.time()
        if redundancy_reduction and pre_items:
            before_rr = len(pre_items)
            temporal_window = max(1, round(0.5 * fps))
            if debug:
                print(f"[REDUNDANCY-EARLY] Before: {before_rr} | temporal_window={temporal_window} (0.5s @ {fps}fps)")
 
            pre_items = reduce_redundancy(
                pre_items,
                method="hybrid",
                visual_threshold=0.90,   # CLIP similarity 
                visual_method="clip",    # semantic embeddings
                temporal_window=temporal_window,
                score_key="proxy_score",
                top_k=1,
                clip_device=device,
                clip_batch_size=64,
                debug=debug,
            )

            analytics["timing"]["redundancy_sec"] += (time.time() - t_rr)

            if debug:
                print(f"[REDUNDANCY-EARLY] After: {len(pre_items)} "
                      f"(dropped {before_rr - len(pre_items)})")
            analytics["frames_dropped_redundancy"] += before_rr - len(pre_items)


        # ======================================================================
        # STAGE 2B : LOGO DETECTION FILTER
        # ======================================================================
        t_logo = time.time()
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
                analytics["frames_filtered_logo"] += 1
                continue
                        
            kept.append(it)

        pre_items = kept
        after_logo = len(pre_items)
        
        analytics["timing"]["logo_sec"] += (time.time() - t_logo)

        if debug:
            print(f"[LOGO] <<< Stage 2C logo: {before_logo} -> {after_logo} (thr={LOGO_THRESHOLD})")
            if before_logo > 0:
                avg_prob = sum(logo_probs) / len(logo_probs)
                print(f"[LOGO] Average logo prob: {avg_prob:.3f}")


        # ======================================================================
        # STAGE 3: SCORING SIGNALS (face quality, emotion intensity, pose) + PRIORITY-BASED RANKING
        # ======================================================================
        if debug:
            print("[SOTA] >>> Stage 3: Scoring signals (face, emotion, pose)")

        t_sota = time.time()

        candidates: List[dict] = []
        for it in pre_items:
            path = it["path"]
            
            closeup = models.closeup_ratio_from_pose(path)
            if closeup < th["min_closeup_ratio"]:
                analytics["frames_filtered_closeup"] += 1
                continue

            # Face quality (includes occlusion/visibility score)
            t0 = time.time()
            num_faces, face_area, face_q, face_det, face_visibility, detected_faces, img = models.face_quality(path)
            analytics["timing"]["face_sec"] += (time.time() - t0)
            
            # Emotion detecton
            t0 = time.time()
            emo_intensity, emo_label = models.emotion_intensity_from_faces(path, detected_faces, img=img, max_faces=5)
            analytics["timing"]["emotion_sec"] += (time.time() - t0)

            # Pose detection
            t0 = time.time()
            pose_score, pose_label, pose_breakdown = models.pose_signals(path)
            analytics["timing"]["pose_sec"] += (time.time() - t0)


            # ==================== CONFIG SECTION ===========================
            # Tier-specific signal weights (used in both STAGE 3 and STAGE 5)
            TIER_WEIGHTS = {
                1: {"face": 0.35, "emotion": 0.35, "pose": 0.30},  # Complete celebration
                2: {"face": 0.60, "emotion": 0.40},                # Expressive face
                3: {"face": 0.50, "pose": 0.50},                   # Pose-driven
                4: {"face": 1.0},                                  # face-only 
            }
            # =============================================================

            FACE_Q_THR = 0.40          # tune
            FACE_AREA_MIN = 0.02       # tune (if you normalized face_area)
            FACE_AREA_MAX = 0.40  
            EMO_THR = 0.20             # Expressive emotion threshold
            POSE_THR = 0.15            # Clear pose threshold

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
                rank_value =  w["face"] * face_q + w["pose"] * pose_score
                
            else:
                kf_priority = 4
                w = TIER_WEIGHTS[4]
                rank_value = w["face"] * face_q 


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
                "face_visibility": float(face_visibility),  
                            
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

        # Per-segment top-k BEFORE TOPIQ:
        # Sort by keyframe_priority (1 best) then rank_value (descending).
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (x["keyframe_priority"], -x["rank_value"]))

        # =====================================================================
        # STAGE 4: Image Quality Assessment 
        # ====================================================================
        if debug:
            print("[IQA] >>> Stage 4: Image Quality Assessment (TOPIQ)")

        t_m = time.time()
        
        iqa_paths = [c["path"] for c in candidates_sorted]
        iqa_scores = models.iqa_norm_batch(iqa_paths)
        
        for c, score in zip(candidates_sorted, iqa_scores):
            c["iqa_norm"] = score
            c["quality_fallback"] = 0

        analytics["timing"]["iqa_sec"] += (time.time() - t_m)

        if debug:
            print(f"[IQA] <<< Batched {len(iqa_scores)} in {time.time() - t_m:.2f}s")

        # =====================================================================
        # STAGE 5: Calculate final score 
        # ====================================================================
        if debug:
            print("[SCORING] >>> Stage 5: Final score calculation")

        t_sc = time.time()

        scored: List[dict] = []
        for c in candidates_sorted:

            # ============ BUILD SEMANTIC SIGNALS (face / emotion / pose) ===========
            
            # Validity conditions for each signal 
            face_valid = (
                c["num_faces"] > 0 and
                c["face_det_score"] > 0.60 # can tune this (face detection confidence threshold)
            )

            pose_valid = c["pose_score"] > 0.01

            emotion_valid = (
                face_valid and
                c["emotion_intensity"] > 0.01
            )

            # Signals: validity gates determine whether a signal contributes.
            # If valid, use the signal directly.
            face_signal    = c["face_quality"]       if face_valid    else 0.0
            pose_signal    = c["pose_score"]         if pose_valid    else 0.0
            emotion_signal = c["emotion_intensity"]  if emotion_valid else 0.0


            # ---------- PRIORITY-BASED MIX (4-tier system) ----------             

            # Apply tier-specific weights (SAME as STAGE 3)
            kf_priority = c["keyframe_priority"]
            w = TIER_WEIGHTS[kf_priority]

            if kf_priority == 1:
                content_src = f"celebration:{c.get('pose_label', 'none')}"
                content_signal = (w["face"] * face_signal + w["emotion"] * emotion_signal + w["pose"] * pose_signal)
                
            elif kf_priority == 2:
                content_src = f"expressive:{c.get('emotion_label', 'none')}"
                content_signal = w["face"] * face_signal + w["emotion"] * emotion_signal
                
            elif kf_priority == 3:
                content_src = f"pose:{c.get('pose_label', 'none')}"
                content_signal =  w["face"] * face_signal + w["pose"] * pose_signal
                
            else:
                content_src = "face_only"
                content_signal = w["face"] * face_signal 
                        
            content_signal = float(np.clip(content_signal, 0.0, 1.0))


            # ---------- NORMALIZE FEATURES ----------
            iqa_signal = float(np.clip(c.get("iqa_norm", 0.0), 0.0, 1.0))

            # ---------- WEIGHTED CONTRIBUTIONS ----------
            #w_content = WEIGHTS["content"] * content_signal
            
            w_face = WEIGHTS["face"] * face_signal  
            w_emotion = WEIGHTS["emotion"] * emotion_signal
            w_pose = WEIGHTS["pose"] * pose_signal
            w_iqa = WEIGHTS["iqa"] * iqa_signal
            score_pre = w_iqa + w_face + w_emotion + w_pose

            final_score = score_pre * c["segment_mult"]

            # ---------- FINAL SCORE ----------
            c["final_score"] = float(score_pre * c["segment_mult"])


            # ---------- STORE BREAKDOWN FOR CSV ----------
            c["face_signal"] = float(face_signal)
            c["pose_signal"] = float(pose_signal)
            c["emotion_signal"] = float(emotion_signal)
            c["emotion_label"] = c.get("emotion_label", "none")

            c["face_visibility"] = float(c.get("face_visibility", 0.0)) 

            c["content_source"] = content_src
            c["content_signal"] = content_signal
            #c["w_content"] = w_content

            c["w_face"] = w_face
            c["w_emotion"] = w_emotion
            c["w_pose"] = w_pose
            c["w_iqa"]   = w_iqa
            
            c["score_pre_mult"] = score_pre
            c["final_score"] = float(final_score)


            # ---------- HARD FLOORS ----------
            # A frame is only eligible if:
            # - confidence is above MIN_FINAL_CONF
            # - final_score is above MIN_FINAL_SCORE
            if c["model_confidence"] < MIN_FINAL_CONF:
                continue
            if c["final_score"] < MIN_FINAL_SCORE:
                continue

            scored.append(c)

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
    # Stage 6: Save selected keyframes 
    # =============================================================================

    if not all_candidates:
        print("[WARN] No candidates found. Prefilters too strict or model paths failing.")
        pd.DataFrame([]).to_csv(output_csv, index=False)
        return

    print("\n" + "=" * 72)
    print(f"[FINAL] >>> Global selection from {len(all_candidates)} candidates")
    print(f"[FINAL] Selection mode: {'QUOTA-BASED' if USE_QUOTA_SELECTION else 'GLOBAL RANKING'}")
    print("=" * 72)


    # -------------------------------------------------------------------------
    # Helper function for selecting final frames
    # -------------------------------------------------------------------------
    def select_final_frames(pool: List[dict], score_floor: float, max_frames: int) -> List[dict]:
        """
        Select frames from pool using global ranking:
        1. Apply hard floors (confidence + score)
        2. Sort by final_score descending
        
        Returns: list of selected candidates
        """
        # Step 1: Hard floors
        eligible = [
            c for c in pool
            if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
            and float(c.get("final_score", 0.0)) >= score_floor
        ]
        
        if not eligible:
            return []
        
        # Step 2: Sort by final_score (global ranking)
        eligible_sorted = sorted(eligible, key=lambda x: x["final_score"], reverse=True)
        
        return eligible_sorted[:max_frames]

    # -------------------------------------------------------------------------
    # Segment-Quota-Based vs Global Ranking Selection
    # -------------------------------------------------------------------------

    if USE_QUOTA_SELECTION:
        # QUOTA-BASED: Use segment quotas to ensure diversity
        print("[FINAL] Using quota-based selection (ensures per-segment diversity)")
        final_selection = _quota_select(all_candidates, top_n=top_n)
        
        # Apply hard filters (quota cannot override confidence/score)
        before_filter = len(final_selection)
        final_selection = [
            c for c in final_selection
            if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
            and float(c.get("final_score", 0.0)) >= MIN_FINAL_SCORE
        ]
        
        if before_filter > len(final_selection):
            print(f"[FINAL] Hard floors filtered: {before_filter} -> {len(final_selection)}")
        
        # Fallback: try to fill remaining slots with relaxed score floor
        if len(final_selection) < top_n and ALLOW_SCORE_FLOOR_FALLBACK:
            print(f"[FINAL] Quota selection: {len(final_selection)} / {top_n}")
            print(f"[FINAL] Trying fallback with score floor {FALLBACK_MIN_FINAL_SCORE}")
            
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
            
            for c in extra:
                if len(final_selection) >= top_n:
                    break
                final_selection.append(c)

    else:
        # GLOBAL RANKING: Select best frames overall (no segment quotas)
        print("[FINAL] Using global ranking (best frames win, segment mult already in score)")
        
        # Primary selection with strict score floor
        final_selection = select_final_frames(all_candidates, MIN_FINAL_SCORE, top_n)
        
        # Fallback: if we didn't get enough frames, try relaxed score floor
        if len(final_selection) < top_n and ALLOW_SCORE_FLOOR_FALLBACK:
            print(f"[FINAL] Primary selection: {len(final_selection)} / {top_n}")
            print(f"[FINAL] Trying fallback with score floor {FALLBACK_MIN_FINAL_SCORE} (conf unchanged)")
            
            # Re-run selection with relaxed floor
            final_selection = select_final_frames(all_candidates, FALLBACK_MIN_FINAL_SCORE, top_n)
    
    print(f"[FINAL] <<< Selected: {len(final_selection)} / {top_n}\n")



    # -----------------------------------------------------------------------------
    # Save selected images + write final CSV.
    # -----------------------------------------------------------------------------
    results = []
    for rank, c in enumerate(final_selection):
        pr = c.get("segment_priority", "NA")
        conf = float(c.get("model_confidence", 0.0))
        score = float(c.get("final_score", 0.0))

        #out_path = os.path.join(output_dir, f"rank{rank+1:02d}_{pr}_conf{conf:.2f}_score{score:.3f}.jpg")
        out_path = os.path.join(output_dir, f"video_{video_id}_rank{rank+1:02d}_{pr}_conf{conf:.2f}_score{score:.3f}.jpg")


        img = cv2.imread(c["path"])
        if img is not None:
            cv2.imwrite(out_path, img)

        results.append({
            # --- identity / ranking ---
            "video_id": video_id,                                       # Forzasys video asset ID — used by metadata API to match event
            "rank": rank + 1,                                           # Final rank after all selection stages
            "segment_id": c["segment_id"],                              # Temporal segment ID this frame was selected from
            "segment_priority": pr,                                     # Semantic segment class (P1 player/referee, P2 corner, P3 staff, P4 behind goal)
            "segment_multiplier": round(c["segment_mult"], 3),          # Hierarchy boost applied to the score (higher = more important)

            # --- content decision ---
            "keyframe_priority": int(c["keyframe_priority"]),           # Which semantic signal dominated: 1=face, 2=celebration, 3=closeup fallback
            "content_source": c.get("content_source", "NA"),            # Human-interpretable: face / celebration / closeup
            #"content_signal": round(c.get("content_signal", 0.0), 3),  # Normalized [0,1] semantic strength of chosen content signal

            # --- final score ---
            "final_score": round(score, 3),                             # Final ranking score = score_pre_mult × segment_multiplier
            "score_pre_mult": round(c.get("score_pre_mult", 0.0), 4),   # Raw weighted score before applying segment hierarchy multiplier

            # --- weighted contributions (these SUM to score_pre_mult) ---
            "w_content": round(c.get("w_content", 0.0), 4),             # Contribution from semantic importance (faces / celebration / closeup)
            "w_iqa": round(c.get("w_iqa", 0.0), 4),                     # Contribution from TOPIQ image quality assessment
            "iqa_signal": round(c.get("iqa_norm", 0.0), 3),
        
            # --- normalized raw signals ---
            "model_confidence": round(conf, 3),                         # Classifier confidence for this frame

            # --- semantic context ---
            "num_faces": int(c.get("num_faces", 0)),                    # Number of detected faces

            # --- emotion ---
            "w_emotion": round(c.get("w_emotion", 0.0), 4),             # Contribution of emotion signal to final score
            "emotion_signal": round(c.get("emotion_signal", 0.0), 3), # Normalized [0,1] emotion intensity
            "emotion_label": c.get("emotion_label", "none"),  #             # Emotion label (happy, sad, angry, etc.)

            # --- signal breakdown (super useful to debug) ---
            "w_face": round(c.get("w_face", 0.0), 4),                   # Contribution of face signal to final score
            "face_signal": round(c.get("face_signal", 0.0), 3),          # Normalized [0,1] face quality signal
            "face_visibility": round(c.get("face_visibility", 0.5), 3),  # Landmark-geometry occlusion score [0..1]; 1=unobstructed
            "w_pose": round(c.get("w_pose", 0.0), 4),                   # Contribution of pose signal to final score
            "pose_signal": round(c.get("pose_signal", 0.0), 3),          # Normalized [0,1] pose detection signal
    
            # --- bookkeeping ---
            "input_path": c["path"],                                     # Original extracted frame path
            "saved_path": out_path,                                      # Output path of selected keyframe image

        })


    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    if CLUSTER_DEBUG_DATA:
        df_clusters = pd.DataFrame(CLUSTER_DEBUG_DATA)
        df_clusters.to_csv("cluster_debug.csv", index=False)
        print(f"[DEBUG] Saved cluster_debug.csv with {len(df_clusters)} rows")

    # -----------------------------------------------------------------------------
    # Print analytics summary
    # -----------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYTICS")
    print(f"{'='*60}")
    print(f"Frames processed (after conf gate): {analytics['frames_processed']}")
    print(f"Filtered by conf (<{MIN_FINAL_CONF}):        {analytics['frames_filtered_conf']}")
    print(f"\nHECATE filters:")
    print(f"  Luminance (dark frames):         {analytics['frames_filtered_luminance']}")
    print(f"  Sharpness (blurry frames):       {analytics['frames_filtered_sharpness']}")
    print(f"  Uniformity (flat frames):        {analytics['frames_filtered_uniformity']}")
    print(f"\nExtension filters:")
    print(f"  Overlay:                         {analytics['frames_filtered_overlay']}")
    print(f"  Texture:                         {analytics['frames_filtered_texture']}")
    print(f"\nStage filters:")
    print(f"  Closeup ratio (pose):            {analytics['frames_filtered_closeup']}")
    print(f"  Redundancy reduction (visual + temporal): {analytics['frames_dropped_redundancy']}")
    print(f"  Logo detection:                  {analytics['frames_filtered_logo']}")

    print("\nCandidates by closeup type:")
    for k, v in analytics["candidates_by_segment_type"].items():
        print(f"  {k}: {v}")

    total_pipeline_sec = time.time() - t_pipeline_start

    print("\nTiming breakdown (cumulative across all segments):")
    print(f"  Preprocessing:     {analytics['timing']['preprocess_sec']:.2f}s")
    print(f"  Redundancy (CLIP): {analytics['timing']['redundancy_sec']:.2f}s")
    print(f"  Logo detection:    {analytics['timing']['logo_sec']:.2f}s")
    print(f"  Face detection:    {analytics['timing']['face_sec']:.2f}s")
    print(f"  Emotion detection: {analytics['timing']['emotion_sec']:.2f}s")
    print(f"  Pose detection:    {analytics['timing']['pose_sec']:.2f}s")
    print(f"  IQA (TOPIQ):       {analytics['timing']['iqa_sec']:.2f}s")
    print(f"  Heavy models total:{analytics['timing']['face_sec'] + analytics['timing']['emotion_sec'] + analytics['timing']['pose_sec'] + analytics['timing']['iqa_sec']:.2f}s")
    print(f"  Total pipeline:    {total_pipeline_sec:.2f}s")


# =============================================================================
# CLI entrypoint
# =============================================================================
if __name__ == "__main__":
    import argparse

    # Command-line interface for running this module directly.
    parser = argparse.ArgumentParser(description="Keyframe Selection")
    parser.add_argument("--pred_csv", required=True)            # predictions file (per frame)
    parser.add_argument("--seg_csv", required=True)             # segment ranges + priorities
    parser.add_argument("--output_csv", required=True)          # output CSV path
    parser.add_argument("--output_dir", required=True)          # where to save keyframe images
    parser.add_argument("--top_n", type=int, default=50)       # total output keyframes
    parser.add_argument("--device", type=str, default="cuda")   # cuda or cpu
    parser.add_argument("--yolo_pose_path", type=str, default="models/yolo/yolo26m-pose.pt")
    parser.add_argument("--debug", action="store_true", help="Verbose stage prints inside segments")
    parser.add_argument("--video_id", type=str, default="unknown",
                        help="Forzasys video asset ID — embedded in output filenames and keyframes.csv")
    parser.add_argument("--redundancy_reduction", type=str, default="true",
                        help="Enable redundancy reduction: 'true' or 'false' (default: true)")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Extraction FPS — used to derive temporal_window (0.5s window). "
                             "Must match the FPS passed to frame_extractor.py (default: 24.0)")
    args = parser.parse_args()

    enable_rr = args.redundancy_reduction.lower() not in ("false", "0", "no", "off")

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
        video_id=args.video_id,
        redundancy_reduction=enable_rr,
        fps=args.fps,
    )