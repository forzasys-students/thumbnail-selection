# keyframe_selector.py
"""
KEYFRAME SELECTION PIPELINE

What this file does (high level):

1) Load frame-level predictions (pred_csv) + segment ranges (seg_csv)
2) For each segment:
   A) HECATE-style preprocessing filters (luminance/sharpness/uniformity) to remove bad frames
   B) Redundancy reduction to remove near-duplicates + Logo detection filter to remove branded frames 
   C) Scoring signals (face/emotion/pose - detection)
   D) Image quality assessment (TOPIQ) on segment candidates
   E) Compute final score and keep only strong candidates
3) Global selection across all segments 
4) Save selected keyframes and a CSV with score breakdown

"""

from __future__ import annotations

import time as _time
_PROCESS_START = _time.time()

from sota_models import SOTAModels
from logo_detector import LogoDetector
from redundancy_reduction import reduce_redundancy, CLUSTER_DEBUG_DATA, _CLIPEmbedder
from concurrent.futures import ThreadPoolExecutor

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
    compute_frame_metrics,      # HECATE filters, plus texture proxies
)

# =============================================================================
# CONFIG (tuning knobs)
# =============================================================================

# Segment-specific multiplier: boosts or penalizes final score by segment type.
SEGMENT_SCORE_MULT = {
    "P1_player_referee": 1.20,   # Highest - action/emotion shots (best thumbnails)
    "P2_corner": 1.075,           # High - set pieces, tactical moments
    "P3_side_staff": 1.05,       # Medium - coach reactions, bench celebrations
    "P4_behind_goal": 1.00,      # Baseline - goalkeeper shots, different angle
}

# Shot-type-specific thresholds for filtering.
QUALITY_THRESHOLDS = {
    "P1_player_referee": {
        "min_luminance": 50.0,       # reject dark frames
        "min_sharpness": 15.0,       # reject blurry frames (gradient magnitude)
        "max_uniformity": 0.70,       # reject flat/uniform frames
        "min_texture": 5.0,          # edge density check
        "min_closeup_ratio": 0.15,   # pose-based closeup proxy (tune this based on your pose model's output)
    },
    "P2_corner": {
        "min_luminance": 50.0,       
        "min_sharpness": 15.0,       
        "max_uniformity": 0.70,       
        "min_texture": 5.0,
        "min_closeup_ratio": 0.10,    
    },
    "P3_side_staff": {
        "min_luminance": 50.0,
        "min_sharpness": 15.0,
        "max_uniformity": 0.70,
        "min_texture": 4.0,
        "min_closeup_ratio": 0.15,   
    },
    "P4_behind_goal": {
        "min_luminance": 50.0,
        "min_sharpness": 15.0,
        "max_uniformity": 0.70,
        "min_texture": 4.0,
        "min_closeup_ratio": 0.10,    
    },
}

ENABLE_REDUNDANCY_REDUCTION = True  # Removes near-duplicate frames. Turn off to disable.
TEMPORAL_WINDOW = 18  # frames (1 second at 24fps) - used in redundancy reduction and final selection to ensure temporal diversity.

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
    "emotion": 0.15,         # emotion signal
    "pose": 0.15,            # pose signal
})


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
    visual_threshold: float = 0.90,
    logo_threshold: float = 0.50,
    w_face: float = 0.25,
    w_emotion: float = 0.15,
    w_pose: float = 0.15,
    w_iqa: float = 0.35,
):
    """
    This is the main selection stage (STEP 4 in the pipeline).

    It expects:
    - pred_csv: per-frame classifier output (frame_path, predicted label, confidence, etc.)
    - seg_csv: segment table produced earlier (segment_id, start_frame, end_frame, priority)

    It produces:
    - output_dir: image files of selected keyframes
    - output_csv: CSV with keyframes + full score breakdown per selected frame

    """

    t_pipeline_start = time.time()   
    startup_sec = t_pipeline_start - _PROCESS_START  # import + interpreter startup

    # ---- Load heavy models ONCE (expensive startup) ----
    print("Initializing heavy models")
    t_init = time.time()

    models = SOTAModels(device=device, yolo_pose_path=yolo_pose_path, debug=debug)

    # ---- Load logo detection model ----
    logo_det = LogoDetector(LOGO_CKPT_PATH, device=device)

    init_sec = time.time() - t_init

    # ---- Load CSV inputs ----
    df_preds = pd.read_csv(pred_csv)
    df_segs = pd.read_csv(seg_csv)


    ui_weights = normalize_weights({
        "face": max(0.0, float(w_face)),
        "emotion": max(0.0, float(w_emotion)),
        "pose": max(0.0, float(w_pose)),
        "iqa": max(0.0, float(w_iqa)),
    })

    if debug:
        print(f"[CONFIG] redundancy_reduction={redundancy_reduction}")
        print(f"[CONFIG] visual_threshold={visual_threshold}")
        print(f"[CONFIG] logo_threshold={logo_threshold}")
        print(f"[CONFIG] normalized weights={ui_weights}")
    

    # ---- Analytics: counters and timing so you can tell what is happening ----
    analytics = {
        "frames_processed": 0,            
        "frames_filtered_conf": 0,        # dropped by classifier confidence
    
        "frames_filtered_luminance": 0,
        "frames_filtered_sharpness": 0,
        "frames_filtered_uniformity": 0,
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
            "data_load_sec": 0.0,           # reading CSVs / parsing inputs
            "preprocess_sec": 0.0,          # cheap frame-quality filters
            "redundancy_sec": 0.0,          # CLIP-based redundancy reduction
            "logo_sec": 0.0,                # logo detection filter
            "face_sec": 0.0,                # face detection / face quality
            "emotion_sec": 0.0,             # emotion detection
            "closeup_sec": 0.0,             # closeup_ratio_from_pose gate
            "pose_sec": 0.0,                # pose scoring
            "iqa_sec": 0.0,                 # IQA (TOPIQ)
            "scoring_sec": 0.0,             # final score computation
            "final_select_sec": 0.0,        # global ranking + temporal suppression
            "save_sec": 0.0,                # image + CSV writing
        }
    }


    # Global pool of candidates (after all segment processing).
    all_candidates: List[dict] = []


    t_data = time.time()

    # Parse frame index from frame filename so we can do segment slicing + time-gap checks.
    df_preds["frame_index"] = df_preds["frame_path"].apply(extract_frame_index)

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    analytics["timing"]["data_load_sec"] += (time.time() - t_data)


    print(f"{'='*60}")
    print("CLOSEUP-ONLY KEYFRAME SELECTION")
    print(f"Processing {len(df_segs)} closeup segments")
    print(f"{'='*60}\n")

    # =============================================================================
    # PRE-COMPUTATION — scoped to segment frames only
    # =============================================================================

    # --- Build a mask of only frames that fall within a segment range ---
    # df_preds has frame_index already parsed above.
    # We OR together the range masks for each segment to get a single filter.
    _seg_mask = pd.Series(False, index=df_preds.index)
    for _, _seg_row in df_segs.iterrows():
        _s = extract_frame_index(_seg_row["start_frame"])
        _e = extract_frame_index(_seg_row["end_frame"])
        _seg_mask |= (df_preds["frame_index"] >= _s) & (df_preds["frame_index"] <= _e)

    _df_seg_frames = df_preds[_seg_mask]
    _CLOSEUP_LABELS = {
        "Close-up_player_or_field_referee",
        "Close-up_corner",
        "Close-up_side_staff",
        "Close-up_behind_the_goal",
    }
    _df_seg_frames = _df_seg_frames[_df_seg_frames["pred_label"].isin(_CLOSEUP_LABELS)]

    print(f"[PREPROCESS] Scoped to {len(_df_seg_frames)} closeup frames inside segments "
          f"(skipping {len(df_preds) - len(_df_seg_frames)} non-closeup/non-segment frames)")

    # --- Collect paths passing confidence gate (within segments only) ---
    _all_candidate_paths = []
    _conf_map: dict = {}
    for _, _row in _df_seg_frames.iterrows():
        _conf = float(_row.get("confidence", 0.5))
        if _conf >= MIN_FINAL_CONF:
            _p = _row["frame_path"]
            _all_candidate_paths.append(_p)
            _conf_map[_p] = _conf

    # --- Parallel preprocessing ---
    print(f"[PREPROCESS] Pre-computing metrics for {len(_all_candidate_paths)} candidate frames...")
    _t_pre_global = time.time()
    metrics_cache: dict = {}

    def _safe_metrics(p: str):
        return p, compute_frame_metrics(p)

    with ThreadPoolExecutor(max_workers=8) as _pool:
        for _path, _result in _pool.map(_safe_metrics, _all_candidate_paths):
            if _result is not None:
                metrics_cache[_path] = _result

    analytics["timing"]["preprocess_sec"] += (time.time() - _t_pre_global)
    print(f"[PREPROCESS] Done in {analytics['timing']['preprocess_sec']:.2f}s "
          f"({len(metrics_cache)} frames kept)")

    # --- CLIP warm-up on quality-passing frames only ---
    # Apply a conservative global quality floor before deciding what to CLIP-embed.
    # Per-segment thresholds still apply inside the loop — this just avoids
    # wasting GPU time on frames that will definitely be filtered anyway.
    if redundancy_reduction:
        _GLOBAL_MIN_LUM   = min(t["min_luminance"] for t in QUALITY_THRESHOLDS.values())
        _GLOBAL_MIN_SHARP = min(t["min_sharpness"]  for t in QUALITY_THRESHOLDS.values())
        _GLOBAL_MAX_UNIF  = max(t["max_uniformity"] for t in QUALITY_THRESHOLDS.values())
        _GLOBAL_MIN_TEX   = min(t.get("min_texture", 0.0) for t in QUALITY_THRESHOLDS.values())

        _clip_paths = [
            p for p, m in metrics_cache.items()
            if (m["luminance"]  >= _GLOBAL_MIN_LUM
            and m["sharpness"]  >= _GLOBAL_MIN_SHARP
            and m["uniformity"] <= _GLOBAL_MAX_UNIF
            and m["texture"]    >= _GLOBAL_MIN_TEX)
        ]
        print(f"[CLIP] Pre-warming embeddings for {len(_clip_paths)} quality-passing frames "
              f"({len(metrics_cache) - len(_clip_paths)} skipped by quality floor)...")
        _t_clip = time.time()
        _clip_embedder = _CLIPEmbedder.get(device=device)
        _clip_embedder.embed_batch(_clip_paths, batch_size=64)
        analytics["timing"]["redundancy_sec"] += (time.time() - _t_clip)
        print(f"[CLIP] Warm-up done in {time.time() - _t_clip:.2f}s")

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
        #segment_frames = df_preds[(df_preds.frame_index >= start) & (df_preds.frame_index <= end)]
        segment_frames = df_preds[
            (df_preds.frame_index >= start) &
            (df_preds.frame_index <= end) &
            (df_preds["pred_label"].isin(_CLOSEUP_LABELS))
        ]

        if debug:
            print("\n" + "-" * 72)
            print(f"[SEGMENT] id={seg_id} priority={seg_priority} frames_in_range={len(segment_frames)}")
            print("-" * 72)

        # ======================================================================
        # STAGE 1: PREPROCESSING
        # ======================================================================
        if debug:
            print("[PREPROCESS] >>> Stage 1: cheap filters (blur/texture/conf)")

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

            #metrics = compute_frame_metrics(path)
            metrics = metrics_cache.get(path)
            if metrics is None:
                continue  # not in cache means unreadable or below conf gate — skip

            if metrics is None:
                continue  # unreadable image — skip silently
 
            luminance  = metrics["luminance"]
            sharpness  = metrics["sharpness"]
            uniformity = metrics["uniformity"]
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
 
            if tex < float(th.get("min_texture", 0.0)):
                analytics["frames_filtered_texture"] += 1
                continue

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


        # ============================================================
        # STAGE 2A: REDUNDANCY REDUCTION
        # ============================================================
        
        t_rr = time.time()
        if redundancy_reduction and pre_items:
            before_rr = len(pre_items)
            if debug:
                print(f"[REDUNDANCY-EARLY] Before: {before_rr} | temporal_window={TEMPORAL_WINDOW}")
 
            pre_items = reduce_redundancy(
                pre_items,
                method="hybrid",
                visual_threshold=float(visual_threshold),
                visual_method="clip",
                temporal_window=TEMPORAL_WINDOW,
                score_key="aesthetic_score",   
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
                        
            if p >= float(logo_threshold):
                analytics["frames_filtered_logo"] += 1
                continue
                        
            kept.append(it)

        pre_items = kept
        after_logo = len(pre_items)
        
        analytics["timing"]["logo_sec"] += (time.time() - t_logo)

        if debug:
            print(f"[LOGO] <<< Stage 2C logo: {before_logo} -> {after_logo} (thr={logo_threshold})")
            if before_logo > 0:
                avg_prob = sum(logo_probs) / len(logo_probs)
                print(f"[LOGO] Average logo prob: {avg_prob:.3f}")


        # ======================================================================
        # STAGE 3: SCORING SIGNALS (face, emotion, pose) + PRIORITY-BASED RANKING
        # ======================================================================
        if debug:
            print("[SOTA] >>> Stage 3: Scoring signals (face, emotion, pose)")

        candidates: List[dict] = []
        for it in pre_items:
            path = it["path"]
        

            # Face quality (includes occlusion/visibility score)
            t0 = time.time()
            num_faces, face_area, face_q, face_det, face_visibility, detected_faces, img = models.face_quality(path)
            analytics["timing"]["face_sec"] += (time.time() - t0)
            
            # Emotion detecton
            t0 = time.time()
            emo_intensity, emo_label = models.emotion_intensity_from_faces(path, detected_faces, img=img, max_faces=5)
            analytics["timing"]["emotion_sec"] += (time.time() - t0)

            # Pose detection (close up ratio is counted as a part of pose detection because we use the yolo bounding boxes to calculate it)
            # In the report we added their time up 
            t0 = time.time()
            closeup = models.closeup_ratio_from_pose(path)
            analytics["timing"]["closeup_sec"] += (time.time() - t0)

            if closeup < th["min_closeup_ratio"]:
                analytics["frames_filtered_closeup"] += 1
                continue

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
            
            w_face = ui_weights["face"] * face_signal 
            w_emotion = ui_weights["emotion"] * emotion_signal
            w_pose = ui_weights["pose"] * pose_signal
            w_iqa = ui_weights["iqa"] * iqa_signal
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

            scored.append(c)

        analytics["timing"]["scoring_sec"] += (time.time() - t_sc)

        if debug:
            print(f"[SCORING] <<< Stage 5 done in {time.time() - t_sc:.2f}s")

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

    print(f"\n[FINAL] >>> Selecting from {len(all_candidates)} candidates")
    

    def select_final_frames(pool: List[dict], score_floor: float, max_frames: int) -> List[dict]:
        eligible = sorted(
            [c for c in pool
             if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
             and float(c.get("final_score", 0.0)) >= score_floor],
            key=lambda x: x["final_score"], reverse=True
        )
        selected, selected_indices = [], []
        for c in eligible:
            idx = extract_frame_index(c["path"])
            if not any(abs(idx - ki) < TEMPORAL_WINDOW for ki in selected_indices):
                selected.append(c)
                selected_indices.append(idx)
            if len(selected) >= max_frames:
                break
        return selected

    t_final = time.time()
    final_selection = select_final_frames(all_candidates, MIN_FINAL_SCORE, top_n)

    if len(final_selection) < top_n and ALLOW_SCORE_FLOOR_FALLBACK:
        print(f"[FINAL] Only {len(final_selection)}/{top_n} with strict floor, trying fallback ({FALLBACK_MIN_FINAL_SCORE})")
        final_selection = select_final_frames(all_candidates, FALLBACK_MIN_FINAL_SCORE, top_n)

    analytics["timing"]["final_select_sec"] += (time.time() - t_final)

    print(f"[FINAL] <<< Selected: {len(final_selection)} / {top_n}\n")


    # -----------------------------------------------------------------------------
    # Save selected images + write final CSV.
    # -----------------------------------------------------------------------------
    t_save = time.time()
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
    analytics["timing"]["save_sec"] += (time.time() - t_save)


    #if CLUSTER_DEBUG_DATA:
    #    df_clusters = pd.DataFrame(CLUSTER_DEBUG_DATA)
    #    df_clusters.to_csv("cluster_debug.csv", index=False)
    #    print(f"[DEBUG] Saved cluster_debug.csv with {len(df_clusters)} rows")

    # -----------------------------------------------------------------------------
    # Print analytics summary
    # -----------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYTICS")
    print(f"{'='*60}")
    print(f"Frames processed (after conf gate): {analytics['frames_processed']}")
    print(f"Filtered by conf (<{MIN_FINAL_CONF}):        {analytics['frames_filtered_conf']}")
    print(f"\nPreprocessing filters:")
    print(f"  Luminance (dark frames):         {analytics['frames_filtered_luminance']}")
    print(f"  Sharpness (blurry frames):       {analytics['frames_filtered_sharpness']}")
    print(f"  Uniformity (flat frames):        {analytics['frames_filtered_uniformity']}")
    print(f"\nExtension filters:")
    print(f"  Texture:                         {analytics['frames_filtered_texture']}")
    print(f"\nStage filters:")
    print(f"  Closeup ratio (pose):            {analytics['frames_filtered_closeup']}")
    print(f"  Redundancy reduction:            {analytics['frames_dropped_redundancy']}")
    print(f"  Logo detection:                  {analytics['frames_filtered_logo']}")

    print("\nCandidates by closeup type:")
    for k, v in analytics["candidates_by_segment_type"].items():
        print(f"  {k}: {v}")

    full_pipeline_sec = time.time() - _PROCESS_START

    print("\nTiming breakdown (exclusive, additive):")
    print(f"  Python startup+imports: {startup_sec:.2f}s")
    print(f"  Model init:             {init_sec:.2f}s")
    print(f"  CSV/data load:          {analytics['timing']['data_load_sec']:.2f}s")
    print(f"  Preprocessing:          {analytics['timing']['preprocess_sec']:.2f}s")
    print(f"  Redundancy (CLIP):      {analytics['timing']['redundancy_sec']:.2f}s")
    print(f"  Logo detection:         {analytics['timing']['logo_sec']:.2f}s")
    print(f"  Face detection:         {analytics['timing']['face_sec']:.2f}s")
    print(f"  Emotion detection:      {analytics['timing']['emotion_sec']:.2f}s")
    print(f"  Closeup gate (pose):    {analytics['timing']['closeup_sec']:.2f}s")
    print(f"  Pose scoring:           {analytics['timing']['pose_sec']:.2f}s")
    print(f"  IQA (TOPIQ):            {analytics['timing']['iqa_sec']:.2f}s")
    print(f"  Final scoring:          {analytics['timing']['scoring_sec']:.2f}s")
    print(f"  Final selection:        {analytics['timing']['final_select_sec']:.2f}s")
    print(f"  Saving outputs:         {analytics['timing']['save_sec']:.2f}s")

    accounted = (
        startup_sec +
        init_sec +
        analytics['timing']['data_load_sec'] +
        analytics['timing']['preprocess_sec'] +
        analytics['timing']['redundancy_sec'] +
        analytics['timing']['logo_sec'] +
        analytics['timing']['face_sec'] +
        analytics['timing']['emotion_sec'] +
        analytics['timing']['closeup_sec'] +
        analytics['timing']['pose_sec'] +
        analytics['timing']['iqa_sec'] +
        analytics['timing']['scoring_sec'] +
        analytics['timing']['final_select_sec'] +
        analytics['timing']['save_sec']
    )

    other_sec = max(0.0, full_pipeline_sec - accounted)

    print(f"  Accounted for:          {accounted:.2f}s")
    print(f"  Total step 4:           {full_pipeline_sec:.2f}s")
    print(f"  Other / loop overhead:  {other_sec:.2f}s")

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
    parser.add_argument("--visual_threshold", type=float, default=0.85)
    parser.add_argument("--logo_threshold", type=float, default=0.50)
    parser.add_argument("--w_face", type=float, default=0.25)
    parser.add_argument("--w_emotion", type=float, default=0.15)
    parser.add_argument("--w_pose", type=float, default=0.15)
    parser.add_argument("--w_iqa", type=float, default=0.35)
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
        visual_threshold=args.visual_threshold,
        logo_threshold=args.logo_threshold,
        w_face=args.w_face,
        w_emotion=args.w_emotion,
        w_pose=args.w_pose,
        w_iqa=args.w_iqa,
    )

    