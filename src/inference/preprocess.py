# preprocess.py

"""

Preprocessing stages for video frame selection. 
Frame filtering based on cheap quality metrics.

Includes:
- Luminance filtering (detect dark frames)
- Sharpness filtering (detect blurry frames) 
- Uniformity filtering (detect flat/uniform frames)
- Cut/transition detection and masking

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
# HECATE-style Frame Filtering Metrics
# Following Song et al. (2016) "To Click or Not To Click"
# -----------------------

def compute_luminance(path: str) -> float:
    """
    Compute relative luminance following ITU-R BT.709 (sRGB standard).
    
    From paper Equation (1):
    Luminance(Irgb) = 0.2126*Ir + 0.7152*Ig + 0.0722*Ib
    
    Returns:
        Mean luminance value [0-255]. Lower values indicate darker frames.
        
    Usage:
        Dark frames should be filtered with threshold ~50-70
    """
    img = cv2.imread(path)
    if img is None:
        return 0.0
    
    # OpenCV loads as BGR, convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply ITU-R BT.709 coefficients
    luminance = (0.2126 * img_rgb[:, :, 0] + 
                 0.7152 * img_rgb[:, :, 1] + 
                 0.0722 * img_rgb[:, :, 2])
    
    return float(np.mean(luminance))


def compute_sharpness(path: str) -> float:
    """
    Compute sharpness using gradient magnitude (Sobel filter).
    
    From paper Equation (2):
    Sharpness(Igray) = sqrt((ΔxIgray)² + (ΔyIgray)²)
    
    Returns:
        Mean gradient magnitude. Higher values indicate sharper images.
        
    Usage:
        Blurry frames should be filtered with threshold ~100-200
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Compute Sobel gradients (this gives Δx and Δy)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude: sqrt(grad_x² + grad_y²)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return float(np.mean(gradient_magnitude))


def compute_uniformity(path: str) -> float:
    """
    Compute uniformity score to detect flat/uniform-colored frames.
    
    From paper Equation (3):
    Uniformity(Igray) = ∫[0 to 5%] cdf(sort(hist(Igray))) dp
    
    Implementation:
    1. Compute normalized intensity histogram
    2. Sort histogram bins in descending order
    3. Compute cumulative distribution at top 5% bins
    
    Returns:
        Uniformity score [0-1]. Higher values indicate more uniform frames.
        
    Usage:
        Uniform frames should be filtered with threshold ~0.7-0.9
        (frames where >70% of pixels are concentrated in just 5% of intensity bins)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Compute normalized histogram (256 bins for grayscale)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize to get probability distribution
    
    # Sort histogram values in descending order
    sorted_hist = np.sort(hist)[::-1]
    
    # Compute cumulative distribution
    cumsum = np.cumsum(sorted_hist)
    
    # Get cumulative sum at top 5% of bins (12.8 bins for 256 total)
    top_5_percent_idx = int(0.05 * len(sorted_hist))
    if top_5_percent_idx == 0:
        top_5_percent_idx = 1
    
    uniformity_score = cumsum[top_5_percent_idx - 1]  # -1 for 0-indexing
    
    return float(uniformity_score)


def filter_low_quality_frames(
    paths: Sequence[str],
    luminance_threshold: float = 60.0,
    sharpness_threshold: float = 150.0,
    uniformity_threshold: float = 0.8,
    verbose: bool = False
) -> List[str]:
    """
    Filter out low-quality frames using HECATE criteria.
    
    Args:
        paths: List of frame paths to evaluate
        luminance_threshold: Min luminance (filter dark frames)
        sharpness_threshold: Min sharpness (filter blurry frames)
        uniformity_threshold: Max uniformity (filter flat frames)
        verbose: Print filtering statistics
        
    Returns:
        List of paths that pass all quality checks
    """
    passed = []
    stats = {"dark": 0, "blurry": 0, "uniform": 0}
    
    for path in paths:
        lum = compute_luminance(path)
        sharp = compute_sharpness(path)
        unif = compute_uniformity(path)
        
        # Apply filters 
        if lum < luminance_threshold:
            stats["dark"] += 1
            continue
        if sharp < sharpness_threshold:
            stats["blurry"] += 1
            continue
        if unif > uniformity_threshold:
            stats["uniform"] += 1
            continue
            
        passed.append(path)
    
    if verbose:
        total = len(paths)
        filtered = total - len(passed)
        print(f"[HECATE Filter] Processed {total} frames:")
        print(f"  Passed: {len(passed)} ({100*len(passed)/total:.1f}%)")
        print(f"  Filtered: {filtered} ({100*filtered/total:.1f}%)")
        print(f"  - Dark: {stats['dark']}")
        print(f"  - Blurry: {stats['blurry']}")
        print(f"  - Uniform: {stats['uniform']}")
    
    return passed



# -----------------------
# Additional quality metrics
# -----------------------

def transition_overlay_score(path: str) -> float:
    """Stddev of grayscale intensity. Cheap proxy to avoid flat frames."""
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.meanStdDev(gray)[1][0][0])

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


def motion_blur_score(path: str) -> float:
    """
    Detect motion blur using gradient variance ratio.
    
    Returns:
        score (0-1): Higher = less motion blur (better for thumbnails)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # ← Already grayscale
    if img is None:
        return 0.0
    
    # Grayscale images have shape (h, w), not (h, w, 3)
    h, w = img.shape[:2]  
    
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


# -----------------------
# Single-read metric computation
# -----------------------

def compute_frame_metrics(path: str) -> Optional[Dict]:
    """
    Read the image once and compute all preprocessing metrics from the same
    in-memory array. This replaces calling compute_luminance, compute_sharpness,
    compute_uniformity, transition_overlay_score, and texture_proxy separately,
    which each re-read the image from disk.

    Returns None if the image cannot be read (treat as failed / drop frame).

    Returns a dict with keys:
        luminance   - ITU-R BT.709 mean luminance [0-255]
        sharpness   - mean Sobel gradient magnitude
        uniformity  - top-5% histogram CDF score [0-1]
        overlay     - grayscale stddev (transition proxy)
        texture     - Canny edge density [0-100]
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None

    # ---- Luminance (ITU-R BT.709) ----
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    luminance = float(np.mean(
        0.2126 * img_rgb[:, :, 0] +
        0.7152 * img_rgb[:, :, 1] +
        0.0722 * img_rgb[:, :, 2]
    ))

    # ---- Grayscale (shared for remaining metrics) ----
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Sharpness (Sobel gradient magnitude) ----
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))

    # ---- Uniformity (top-5% histogram CDF) ----
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    sorted_hist = np.sort(hist)[::-1]
    top_idx = max(1, int(0.05 * len(sorted_hist)))
    uniformity = float(np.cumsum(sorted_hist)[top_idx - 1])

    # ---- Overlay proxy (grayscale stddev) ----
    overlay = float(cv2.meanStdDev(gray)[1][0][0])

    # ---- Texture proxy (Canny edge density on downscaled image) ----
    h, w = gray.shape[:2]
    scale = 160.0 / max(w, 1)
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    edges = cv2.Canny(small, 60, 180)
    texture = float(np.mean(edges > 0) * 100.0)

    return {
        "luminance": luminance,
        "sharpness": sharpness,
        "uniformity": uniformity,
        "overlay": overlay,
        "texture": texture,
    }


