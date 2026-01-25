# preprocess.py

"""

Cheap, deterministic preprocessing + redundancy reduction for keyframe selection.
Includes:
- Cut/transition detection and masking
- Cheap frame quality metrics
- Redundancy reduction via HSV histogram embeddings and greedy selection

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


# -----------------------
# Helpers
# -----------------------

def extract_frame_index(path: str) -> int:
    """Extract integer frame index from filename digits."""
    return int("".join(c for c in os.path.basename(path) if c.isdigit()))


# -----------------------
# Cheap metrics
# -----------------------

def transition_overlay_score(path: str) -> float:
    """Stddev of grayscale intensity. Cheap proxy to avoid flat frames."""
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.meanStdDev(gray)[1][0][0])


def motion_blur_score(path: str) -> float:
    """
    Detect motion blur using gradient variance ratio.
    
    Returns:
        score (0-1): Higher = less motion blur (better for thumbnails)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # ← Already grayscale
    if img is None:
        return 0.0
    
    # FIX: Grayscale images have shape (h, w), not (h, w, 3)
    h, w = img.shape[:2]  # ← Change this line
    
    # Downscale for speed
    if max(h, w) > 480:
        scale = 480.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # Sobel gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Variance of gradients
    var_gx = np.var(gx)
    var_gy = np.var(gy)
    
    if var_gx < 1e-6 or var_gy < 1e-6:
        return 0.0  # Flat image
    
    # Ratio should be near 1.0 for sharp, far from 1.0 for directional blur
    ratio = min(var_gx, var_gy) / max(var_gx, var_gy)
    
    # Also check gradient magnitude variance (sharp = high variance)
    grad_mag = np.sqrt(gx**2 + gy**2)
    mag_var = np.var(grad_mag)
    mag_norm = np.clip(mag_var / 5000.0, 0.0, 1.0)
    
    # Combined score
    score = 0.6 * ratio + 0.4 * mag_norm
    
    return float(np.clip(score, 0.0, 1.0))


def blur_laplacian_var(path: str) -> float:
    """Laplacian variance blur score (higher is sharper)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def texture_proxy(path: str) -> float:
    """
    Edge density proxy (percentage-ish).
    Higher => more structure; low => uniform/flat.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    scale = 160.0 / max(w, 1)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(img, 60, 180)
    return float(np.mean(edges > 0) * 100.0)


# -----------------------
# Cut/transition masking
# -----------------------

def _downscaled_gray(path: str, width: int = 160) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w <= 0:
        return None
    scale = float(width) / float(w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_and_mask_cuts(
    sorted_paths: Sequence[str],
    diff_thresh: float,
    drop_radius: int,
) -> Set[str]:
    """
    Return a set of paths to drop around detected cuts/transitions.

    diff_thresh: mean abs diff on downscaled grayscale [0-255]
    drop_radius: drop +/- this many extracted frames around a cut
    """
    to_drop: Set[str] = set()
    if len(sorted_paths) < 3:
        return to_drop

    prev = _downscaled_gray(sorted_paths[0])
    if prev is None:
        return to_drop

    diffs: List[float] = []
    for i in range(1, len(sorted_paths)):
        cur = _downscaled_gray(sorted_paths[i])
        if cur is None:
            diffs.append(0.0)
            prev = cur
            continue
        if cur.shape != prev.shape:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
        d = float(np.mean(cv2.absdiff(cur, prev)))
        diffs.append(d)
        prev = cur

    # diffs[i-1] corresponds to boundary between i-1 and i
    for i, d in enumerate(diffs, start=1):
        if d >= diff_thresh:
            for j in range(max(0, i - drop_radius), min(len(sorted_paths), i + drop_radius + 1)):
                to_drop.add(sorted_paths[j])
    return to_drop

