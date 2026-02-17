# redundancy_reduction.py
"""
POST-SCORING REDUNDANCY REDUCTION

This module provides visual + temporal clustering to remove near-duplicate keyframes
while preserving the highest-scoring representative from each cluster.

"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from collections import defaultdict
import os


def extract_frame_index(path: str) -> int:
    """Extract frame number from path like 'frame_00123.jpg'"""
    basename = os.path.basename(path)
    # Handle formats: frame_00123.jpg, frame_123.jpg, etc.
    parts = basename.replace(".jpg", "").replace(".png", "").split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return 0


# =============================================================================
# VISUAL SIMILARITY METHODS
# =============================================================================

def compute_color_histogram(img_path: str, bins: int = 16) -> np.ndarray:
    """
    Fast color histogram for visual similarity.
    Returns normalized 3D histogram (RGB).
    """
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((bins, bins, bins), dtype=np.float32)
    
    # Resize for speed
    img = cv2.resize(img, (128, 128))
    
    # Compute 3D histogram
    hist = cv2.calcHist(
        [img], 
        [0, 1, 2],           # B, G, R channels
        None, 
        [bins, bins, bins],  # bins per channel
        [0, 256, 0, 256, 0, 256]
    )
    
    # Normalize
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-6)
    
    return hist


def compute_perceptual_hash(img_path: str, hash_size: int = 8) -> str:
    """
    Difference hash (dHash) for perceptual similarity.
    Fast and robust to small changes.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "0" * (hash_size * hash_size)
    
    # Resize to hash_size + 1 for difference calculation
    img = cv2.resize(img, (hash_size + 1, hash_size))
    
    # Compute horizontal gradient
    diff = img[:, 1:] > img[:, :-1]
    
    # Convert to hex string
    hash_str = "".join(["1" if d else "0" for row in diff for d in row])
    return hash_str


def hamming_distance(hash1: str, hash2: str) -> int:
    """Hamming distance between two binary hash strings"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two vectors"""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / (norm + 1e-8)


# =============================================================================
# CLUSTERING STRATEGIES
# =============================================================================

def temporal_clustering(
    candidates: List[Dict],
    window_size: int = 60,
    debug: bool = False
) -> List[Dict]:
    """
    Group frames into temporal windows, keep best from each window.
    
    Args:
        candidates: List of candidate dicts with 'path' and 'final_score'
        window_size: Frames within this distance are considered a cluster
        debug: Print debug info
    
    Returns:
        Deduplicated list (best frame per temporal cluster)
    """
    if not candidates:
        return []
    
    # Sort by frame index
    sorted_cands = sorted(candidates, key=lambda x: extract_frame_index(x["path"]))
    
    clusters = []
    current_cluster = [sorted_cands[0]]
    
    for i in range(1, len(sorted_cands)):
        prev_idx = extract_frame_index(sorted_cands[i-1]["path"])
        curr_idx = extract_frame_index(sorted_cands[i]["path"])
        
        if curr_idx - prev_idx <= window_size:
            # Same cluster
            current_cluster.append(sorted_cands[i])
        else:
            # New cluster
            clusters.append(current_cluster)
            current_cluster = [sorted_cands[i]]
    
    # Don't forget last cluster
    clusters.append(current_cluster)
    
    # Pick best from each cluster
    result = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: x["final_score"])
        result.append(best)
    
    if debug:
        print(f"[TEMPORAL] {len(candidates)} → {len(result)} (window={window_size})")
        print(f"[TEMPORAL] Removed {len(candidates) - len(result)} duplicates")
    
    return result


def visual_clustering(
    candidates: List[Dict],
    similarity_threshold: float = 0.92,
    method: str = "histogram",  # "histogram" or "phash"
    debug: bool = False
) -> List[Dict]:
    """
    Group visually similar frames, keep best from each cluster.
    
    Args:
        candidates: List of candidate dicts
        similarity_threshold: Similarity above this = same cluster
        method: "histogram" (color-based) or "phash" (perceptual hash)
        debug: Print debug info
    
    Returns:
        Deduplicated list
    """
    if not candidates:
        return []
    
    if method == "histogram":
        # Compute histograms for all candidates
        features = {}
        for c in candidates:
            features[c["path"]] = compute_color_histogram(c["path"])
        
        # Greedy clustering by similarity
        used = set()
        clusters = []
        
        for i, c1 in enumerate(candidates):
            if c1["path"] in used:
                continue
            
            cluster = [c1]
            used.add(c1["path"])
            
            for j, c2 in enumerate(candidates):
                if i == j or c2["path"] in used:
                    continue
                
                sim = cosine_similarity(features[c1["path"]], features[c2["path"]])
                if sim >= similarity_threshold:
                    cluster.append(c2)
                    used.add(c2["path"])
            
            clusters.append(cluster)
    
    elif method == "phash":
        # Compute perceptual hashes
        hashes = {c["path"]: compute_perceptual_hash(c["path"]) for c in candidates}
        
        # Hamming distance threshold (for 8x8 hash, ~10% difference)
        max_hamming = int(64 * (1.0 - similarity_threshold))
        
        used = set()
        clusters = []
        
        for i, c1 in enumerate(candidates):
            if c1["path"] in used:
                continue
            
            cluster = [c1]
            used.add(c1["path"])
            
            for j, c2 in enumerate(candidates):
                if i == j or c2["path"] in used:
                    continue
                
                dist = hamming_distance(hashes[c1["path"]], hashes[c2["path"]])
                if dist <= max_hamming:
                    cluster.append(c2)
                    used.add(c2["path"])
            
            clusters.append(cluster)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'histogram' or 'phash'")
    
    # Pick best from each cluster
    result = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: x["final_score"])
        result.append(best)
    
    if debug:
        print(f"[VISUAL] {len(candidates)} → {len(result)} (thr={similarity_threshold}, method={method})")
        print(f"[VISUAL] Removed {len(candidates) - len(result)} duplicates")
    
    return result


def hybrid_clustering(
    candidates: List[Dict],
    temporal_window: int = 60,
    visual_threshold: float = 0.92,
    visual_method: str = "histogram",
    debug: bool = False
) -> List[Dict]:
    """
    Two-stage clustering:
    1. Temporal clustering (group nearby frames)
    2. Visual clustering within each temporal group
    
    This is the RECOMMENDED approach for keyframe selection.
    
    Args:
        candidates: List of candidate dicts
        temporal_window: Max frame distance for temporal grouping
        visual_threshold: Similarity threshold for visual clustering
        visual_method: "histogram" or "phash"
        debug: Print debug info
    
    Returns:
        Deduplicated list
    """
    if not candidates:
        return []
    
    if debug:
        print(f"\n[HYBRID] Starting with {len(candidates)} candidates")
        print(f"[HYBRID] Temporal window: {temporal_window}, Visual threshold: {visual_threshold}")
    
    # Step 1: Temporal clustering
    sorted_cands = sorted(candidates, key=lambda x: extract_frame_index(x["path"]))
    
    temporal_groups = []
    current_group = [sorted_cands[0]]
    
    for i in range(1, len(sorted_cands)):
        prev_idx = extract_frame_index(sorted_cands[i-1]["path"])
        curr_idx = extract_frame_index(sorted_cands[i]["path"])
        
        if curr_idx - prev_idx <= temporal_window:
            current_group.append(sorted_cands[i])
        else:
            temporal_groups.append(current_group)
            current_group = [sorted_cands[i]]
    
    temporal_groups.append(current_group)
    
    if debug:
        print(f"[HYBRID] Stage 1 (temporal): {len(temporal_groups)} groups")
    
    # Step 2: Visual clustering within each temporal group
    result = []
    total_removed = 0
    
    for group_idx, group in enumerate(temporal_groups):
        if len(group) == 1:
            # Single frame in group - keep it
            result.extend(group)
            continue
        
        # Apply visual clustering within this temporal group
        deduplicated = visual_clustering(
            group,
            similarity_threshold=visual_threshold,
            method=visual_method,
            debug=False
        )
        
        removed = len(group) - len(deduplicated)
        total_removed += removed
        
        if debug and removed > 0:
            frame_range = (
                extract_frame_index(group[0]["path"]),
                extract_frame_index(group[-1]["path"])
            )
            print(f"[HYBRID]   Group {group_idx} (frames {frame_range[0]}-{frame_range[1]}): "
                  f"{len(group)} → {len(deduplicated)} (-{removed})")
        
        result.extend(deduplicated)
    
    if debug:
        print(f"[HYBRID] Stage 2 (visual): Removed {total_removed} visual duplicates")
        print(f"[HYBRID] Final: {len(candidates)} → {len(result)} candidates\n")
    
    return result


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def reduce_redundancy(
    candidates: List[Dict],
    method: str = "hybrid",
    temporal_window: int = 48,
    visual_threshold: float = 0.80,
    visual_method: str = "histogram",
    debug: bool = False
) -> List[Dict]:
    """
    Main interface for redundancy reduction.
    
    Args:
        candidates: List of candidate dicts with 'path' and 'final_score'
        method: "temporal", "visual", or "hybrid" (recommended)
        temporal_window: Frame distance for temporal clustering
        visual_threshold: Similarity threshold (0-1, higher = stricter)
        visual_method: "histogram" (color-based) or "phash" (structure-based)
        debug: Print debug info
    
    Returns:
        Deduplicated candidates (best frame per cluster)
    
    Example:
        >>> deduplicated = reduce_redundancy(
        ...     all_candidates,
        ...     method="hybrid",
        ...     temporal_window=48,      
        ...     visual_threshold=0.90,   
        ...     debug=True
        ... )
    """
    if method == "temporal":
        return temporal_clustering(candidates, temporal_window, debug)
    
    elif method == "visual":
        return visual_clustering(candidates, visual_threshold, visual_method, debug)
    
    elif method == "hybrid":
        return hybrid_clustering(
            candidates,
            temporal_window,
            visual_threshold,
            visual_method,
            debug
        )
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'temporal', 'visual', or 'hybrid'")


# =============================================================================
# TUNING GUIDE
# =============================================================================

"""
TUNING RECOMMENDATIONS:

TEMPORAL_WINDOW (frame distance):
   - 24 fps footage:
     * 24 = 1 second window
     * 48 = 2 second window 
     * 72 = 3 second window 
   - Larger = more aggressive deduplication

VISUAL_THRESHOLD (0-1):
   - Lower = more aggressive deduplication

VISUAL_METHOD:
   - "histogram": Fast, good for color changes, lighting variations
   - "phash": Better for structural similarity, robust to color shifts

METHOD:
   - "hybrid": RECOMMENDED - combines temporal + visual
   - "temporal": Simple, fast, no visual analysis
   - "visual": Slow, ignores temporal proximity

EXAMPLE CONFIGS:

Conservative (keep variety):
    method="hybrid"
    temporal_window=45
    visual_threshold=0.95
    visual_method="histogram"

Balanced (recommended):
    method="hybrid"
    temporal_window=60
    visual_threshold=0.92
    visual_method="histogram"

Aggressive (minimize duplicates):
    method="hybrid"
    temporal_window=90
    visual_threshold=0.88
    visual_method="phash"
"""