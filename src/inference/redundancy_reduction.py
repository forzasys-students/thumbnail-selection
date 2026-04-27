# redundancy_reduction.py
"""
MULTI-STAGE REDUNDANCY REDUCTION

This module supports redundancy reduction of the pipeline

It provides:
- temporal clustering
- visual clustering
- hybrid clustering (recommended)

For each cluster, it keeps top-K items according to score_key.
"""

from typing import List, Dict, Optional
import cv2
import numpy as np
import os
import torch

from torchvision import models, transforms
import torch.nn as nn 
from PIL import Image

# =============================================================================
# DEBUG STATE
# =============================================================================

CLUSTER_DEBUG_DATA = []

# Global counter — incremented every time a cluster is formed so IDs are
# unique across all segments, all calls, and all clustering stages.
_cluster_counter = 0


def _next_cluster_id(prefix: str) -> str:
    global _cluster_counter
    cid = f"{prefix}_{_cluster_counter}"
    _cluster_counter += 1
    return cid


def reset_cluster_debug():
    """Call this between videos to reset debug state."""
    global _cluster_counter
    CLUSTER_DEBUG_DATA.clear()
    _cluster_counter = 0


# =============================================================================
# CLIP EMBEDDER — loads once, reused across all segments
# =============================================================================
class _CLIPEmbedder:
    """
    Lazy singleton wrapper around CLIP ViT-B/32.

    Loading CLIP takes ~2s. This class ensures it loads exactly once
    regardless of how many times visual_clustering is called.

    Why CLIP instead of histograms:
        Color histograms fail on soccer footage because every frame shares
        nearly identical color distributions (green pitch, same jerseys).
        Two completely different moments — a goalkeeper dive vs a player
        celebrating — can easily hit 85%+ histogram similarity just because
        they share the same background. CLIP encodes *what is happening*,
        not just what colors are present, so semantically different frames
        stay in different clusters.
    """
    _instance: Optional["_CLIPEmbedder"] = None

    def __init__(self, device: str = "cuda"):
        try:
            import clip
            from PIL import Image
            self._clip = clip
            self._Image = Image
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        self.device = device
        print(f"[CLIP] Loading ViT-B/32 on {device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        #self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()

        self._embedding_cache: dict = {}   

        print("[CLIP] Ready.")

    @classmethod
    def get(cls, device: str = "cuda") -> "_CLIPEmbedder":
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def embed_batch(self, paths: list, batch_size: int = 64) -> dict:
        """
        Encode all paths in one or more batched GPU passes.
        Results are cached internally — repeated calls for the same path
        are instant dict lookups with no GPU work.

        Returns {path: np.ndarray} — L2-normalised unit vectors.
        Unreadable paths get a zero vector.
        """
        embeddings: dict = {}
        to_compute: list = []

        # Split paths into cache hits and misses
        for p in paths:
            if p in self._embedding_cache:
                embeddings[p] = self._embedding_cache[p]
            else:
                to_compute.append(p)

        # Only run the GPU forward pass for cache misses
        for start in range(0, len(to_compute), batch_size):
            batch_paths = to_compute[start: start + batch_size]
            tensors, valid_paths = [], []

            for p in batch_paths:
                try:
                    img = self._Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                    valid_paths.append(p)
                except Exception:
                    zero = np.zeros(512, dtype=np.float32)
                    self._embedding_cache[p] = zero
                    embeddings[p] = zero

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_tensor).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)

            for path, vec in zip(valid_paths, feats.cpu().numpy()):
                self._embedding_cache[path] = vec
                embeddings[path] = vec

        return embeddings



# =============================================================================
# AESTHETIC SCORER — runs on top of existing CLIP embeddings, zero extra cost
# =============================================================================

class _AestheticScorer:
    """
    LAION aesthetic predictor MLP. Runs on CLIP ViT-B/32 embeddings (512-dim).
    Since embeddings are already computed for clustering, scoring is free.
    Downloads ~4MB weights on first use, cached by torch.hub.
    """
    _instance: Optional["_AestheticScorer"] = None

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = nn.Linear(512, 1).to(device).eval()
        url = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
        state = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
        self.model.load_state_dict(state)

        print(f"[AestheticScorer] Ready on {device}")

    @classmethod
    def get(cls, device: str = "cuda") -> "_AestheticScorer":
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def score(self, embeddings: dict) -> dict:
        """
        Score all embeddings in one GPU pass.
        embeddings: {path -> np.ndarray (512,)} — already L2-normalised CLIP vectors
        Returns:    {path -> float} aesthetic score, normalised to [0, 1]
        """
        if not embeddings:
            return {}
        paths = list(embeddings.keys())
        vecs = torch.tensor(
            np.stack([embeddings[p] for p in paths]),
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            raw = self.model(vecs).squeeze(-1).cpu().numpy()

        # Raw scores are roughly in [1, 10] — normalise to [0, 1]
        normalised = np.clip((raw - 1.0) / 9.0, 0.0, 1.0)
        return {p: float(s) for p, s in zip(paths, normalised)}


# =============================================================================
# FRAME INDEX UTILITY
# =============================================================================

def extract_frame_index(path: str) -> int:
    """Extract frame number from path like 'frame_00123.jpg'"""
    basename = os.path.basename(path)
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
        return np.zeros((bins * bins * bins,), dtype=np.float32)
    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist(
        [img], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)
    return hist


def compute_perceptual_hash(img_path: str, hash_size: int = 8) -> str:
    """Difference hash (dHash) for perceptual similarity."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "0" * (hash_size * hash_size)
    img = cv2.resize(img, (hash_size + 1, hash_size))
    diff = img[:, 1:] > img[:, :-1]
    return "".join(["1" if d else "0" for row in diff for d in row])


def hamming_distance(hash1: str, hash2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot / (norm + 1e-8))


# =============================================================================
# CLUSTER SELECTION
# =============================================================================

def select_from_cluster(
    cluster,
    score_key="aesthetic_score",
    top_k=1,
    cluster_id=None,
    stage="unknown"
):
    if not cluster:
        return []

    cluster_sorted = sorted(
        cluster,
        key=lambda x: float(x.get(score_key, 0.0)),
        reverse=True
    )

    selected = cluster_sorted[:max(1, int(top_k))]
    selected_paths = set(x["path"] for x in selected)

    for item in cluster_sorted:
        CLUSTER_DEBUG_DATA.append({
            "cluster_id": cluster_id,
            "stage": stage,
            "path": item["path"],
            "score": float(item.get(score_key, 0.0)),
            "selected": item["path"] in selected_paths,
            "cluster_size": len(cluster),
            "segment_id": item.get("segment_id", -1)
        })

    return selected


# =============================================================================
# CLUSTERING STRATEGIES
# =============================================================================

def visual_clustering(
    candidates: List[Dict],
    similarity_threshold: float = 0.92,
    method: str = "histogram",
    score_key: str = "final_score",
    top_k: int = 1,
    clip_device: str = "cuda",
    clip_batch_size: int = 64,
    debug: bool = False
) -> List[Dict]:
    """
    Group visually similar frames, keep top-K per cluster.

    method options:
      "histogram" — fast, color-only. Fails on soccer footage where all frames
                    share the same color distribution (pitch, jerseys).
      "phash"     — structural similarity, robust to color shifts.
      "clip"      — semantic embeddings via CLIP ViT-B/32. Understands *what is
                    happening* in the frame, not just colors. Recommended for
                    soccer. Model loads once globally and is reused every call.

    clip_device:     "cuda" or "cpu"
    clip_batch_size: frames per GPU forward pass. Reduce to 32 if OOM.
    """
    if not candidates:
        return []

    # ---- Feature extraction (batched for CLIP, per-image for others) ----
    if method == "histogram":
        features = {c["path"]: compute_color_histogram(c["path"]) for c in candidates}

    elif method == "phash":
        features = {c["path"]: compute_perceptual_hash(c["path"]) for c in candidates}
        max_hamming = int(64 * (1.0 - similarity_threshold))

    elif method == "clip":
        paths = [c["path"] for c in candidates]
        embedder = _CLIPEmbedder.get(device=clip_device)
        features = embedder.embed_batch(paths, batch_size=clip_batch_size)

        # Score aesthetics from the same embeddings — no extra forward pass
        aesthetic_scores = _AestheticScorer.get(device=clip_device).score(features)
        for c in candidates:
            c["aesthetic_score"] = aesthetic_scores.get(c["path"], 0.0)

        if debug:
            print(f"[CLIP] Encoded {len(paths)} frames (batch_size={clip_batch_size})")

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose: histogram | phash | clip")

    # ---- Greedy clustering (same logic for all methods) ----
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

            if method == "phash":
                similar = hamming_distance(features[c1["path"]], features[c2["path"]]) <= max_hamming
            else:
                # histogram and clip both use cosine similarity on their vectors
                similar = cosine_similarity(features[c1["path"]], features[c2["path"]]) >= similarity_threshold

            if similar:
                cluster.append(c2)
                used.add(c2["path"])

        clusters.append(cluster)

    result = []
    for cluster in clusters:
        result.extend(select_from_cluster(
            cluster, score_key, top_k,
            cluster_id=_next_cluster_id("V"),
            stage="visual"
        ))

    if debug:
        print(f"[VISUAL] {len(candidates)} -> {len(result)} "
              f"(thr={similarity_threshold}, method={method}, score_key={score_key}, top_k={top_k})")
    return result


def temporal_clustering(
    candidates: List[Dict],
    min_frame_gap: int = 24,
    score_key: str = "aesthetic_score",
    top_k: int = 1,
    debug: bool = False,
) -> List[Dict]:
    """
    Group temporally close frames into clusters, then keep top_k per cluster.

    This is a cluster-then-select approach (not greedy-keep):
      1. Sort all candidates by frame index.
      2. Walk the sorted list and assign consecutive frames to the same cluster
         as long as each new frame is within `min_frame_gap` of the previous
         frame in the cluster.
      3. From each cluster, keep the top_k highest-scoring frames.

    This is the right complement to CLIP visual deduplication: CLIP catches
    semantically similar frames wherever they sit in time; temporal clustering
    catches temporally adjacent frames that CLIP let through because they look
    slightly different (e.g., mid-action vs peak-action one frame apart).

    Why cluster-then-select beats greedy-keep:
      - Greedy-keep scores frames before grouping, so it can mis-order
        candidates and leave gaps in coverage.
      - Cluster-then-select first finds the natural temporal groups, then
        applies scoring inside each group — the winner is always the best
        frame that actually belongs to that moment.

    min_frame_gap reference (frame-index units, independent of extraction FPS):
        6   — burst duplicates only (~0.25s at 24fps, ~0.5s at 12fps)
        12  — tight window (~0.5s at 24fps, ~1s at 12fps)
        24  — 1-second window at 24fps  /  2-second window at 12fps
        48  — 2-second window at 24fps  /  4-second window at 12fps

    top_k:
        1  — decisive: one winner per moment (recommended after heavy scoring)
        2+ — keep alternatives per moment (useful before scoring is final)
    """
    if not candidates:
        return []

    # ---- Step 1: sort by frame index so adjacent frames are neighbours ----
    sorted_by_idx = sorted(
        candidates,
        key=lambda x: extract_frame_index(x["path"])
    )

    # ---- Step 2: build temporal clusters (sliding-window grouping) ----
    # A new cluster starts whenever the gap to the previous frame exceeds
    # min_frame_gap. Because frames are sorted, this is a single linear pass.
    clusters: List[List[Dict]] = []
    current_cluster: List[Dict] = [sorted_by_idx[0]]

    for c in sorted_by_idx[1:]:
        prev_idx = extract_frame_index(current_cluster[-1]["path"])
        curr_idx = extract_frame_index(c["path"])

        if curr_idx - prev_idx < min_frame_gap:
            # Still within the same temporal burst → extend current cluster
            current_cluster.append(c)
        else:
            # Gap is large enough → start a new cluster
            clusters.append(current_cluster)
            current_cluster = [c]

    clusters.append(current_cluster)  # flush last cluster

    # ---- Step 3: select top_k from each cluster by score ----
    result: List[Dict] = []
    for cluster in clusters:
        result.extend(select_from_cluster(
            cluster,
            score_key=score_key,
            top_k=top_k,
            cluster_id=_next_cluster_id("T"),
            stage="temporal",
        ))

    if debug:
        dropped = len(candidates) - len(result)
        sizes = [len(cl) for cl in clusters]
        print(
            f"[TEMPORAL] {len(candidates)} -> {len(result)} "
            f"(min_frame_gap={min_frame_gap}, clusters={len(clusters)}, "
            f"dropped={dropped}, "
            f"cluster_sizes min/max/avg={min(sizes)}/{max(sizes)}/{sum(sizes)/len(sizes):.1f})"
        )

    return result


def hybrid_clustering(
    candidates: List[Dict],
    temporal_window: int = 12,
    visual_threshold: float = 0.90,
    visual_method: str = "clip",
    score_key: str = "aesthetic_score",
    top_k: int = 1,
    clip_device: str = "cuda",
    clip_batch_size: int = 64,
    debug: bool = False
) -> List[Dict]:
    """
    Two-pass redundancy reduction:

    Pass 1 — Visual (CLIP):
        Remove semantically similar frames.

    Pass 2 — Temporal:
        Remove frames that are too close together in time.
        temporal_window = min_frame_gap: two survivors must be at least
        this many frame indices apart in the filename.
    """
    if not candidates:
        return []

    after_visual = visual_clustering(
        candidates,
        similarity_threshold=visual_threshold,
        method=visual_method,
        score_key=score_key,
        top_k=top_k,
        clip_device=clip_device,
        clip_batch_size=clip_batch_size,
        debug=debug,
    )

    if debug:
        print(f"[HYBRID] Pass 1 visual: {len(candidates)} -> {len(after_visual)}")

    if not after_visual:
        return []

    result = temporal_clustering(
        after_visual,
        min_frame_gap=temporal_window,
        score_key=score_key,
        top_k=top_k,
        debug=debug,
    )

    if debug:
        print(f"[HYBRID] Pass 2 temporal: {len(after_visual)} -> {len(result)}")

    return result


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def reduce_redundancy(
    candidates: List[Dict],
    method: str = "hybrid",
    temporal_window: int = 12,
    visual_threshold: float = 0.90,
    visual_method: str = "clip",
    score_key: str = "aesthetic_score",
    top_k: int = 1,
    clip_device: str = "cuda",
    clip_batch_size: int = 64,
    debug: bool = False
) -> List[Dict]:
    """
    Main interface for redundancy reduction.

    visual_method options: "histogram" | "phash" | "clip"
    Use "clip" for semantic-aware deduplication (recommended for soccer footage).
    CLIP model loads once on first call and is reused across all subsequent calls.

    clip_device:     "cuda" or "cpu"
    clip_batch_size: frames per GPU forward pass (reduce to 32 if OOM)

    """
    if method == "visual":
        return visual_clustering(
            candidates, visual_threshold, visual_method, score_key, top_k,
            clip_device=clip_device, clip_batch_size=clip_batch_size, debug=debug
        )

    elif method == "temporal":
        return temporal_clustering(
            candidates,
            min_frame_gap=temporal_window,
            score_key=score_key,
            top_k=top_k,
            debug=debug,
        )

    elif method == "hybrid":
        return hybrid_clustering(
            candidates, temporal_window, visual_threshold, visual_method, score_key, top_k,
            clip_device=clip_device, clip_batch_size=clip_batch_size, debug=debug
        )

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# TUNING GUIDE
# =============================================================================

"""
METHOD:
   "visual":   CLIP only — semantic deduplication. Use when you have no frame-
               index signal or want CLIP to do all the work.
   "temporal": Frame-index clustering only — groups temporally adjacent frames
               and keeps the best. Use as a lightweight post-CLIP safety net
               or when CLIP is disabled.
   "hybrid":   CLIP first, then temporal. The recommended two-pass pipeline:
               CLIP removes semantic duplicates; temporal catches the edge cases
               CLIP misses because two adjacent frames look just different enough
               to fall below the similarity threshold.

VISUAL_METHOD (for "visual" and "hybrid"):
   "histogram": Fast, but fails on soccer — all frames share the same color
                distribution (green pitch, same jerseys). Use only for testing.
   "phash":     Structural similarity, better than histogram, no GPU needed.
   "clip":      Semantic embeddings — understands what is happening in the frame.
                Recommended for soccer. Loads once, batched per temporal group.

TEMPORAL_WINDOW / min_frame_gap reference (frame-index units, FPS-independent):
   The frame index in the filename is the source of truth. Two frames with
   indices 0012 and 0013 are always 1 frame apart regardless of whether you
   extracted at 6, 12, or 24fps.

VISUAL_THRESHOLD for clip: 0.90–0.93 is a tight semantic match
    that only groups near-identical frames. 0.80–0.85 is more aggressive and
    can group different moments that share some visual features (e.g., same
    player celebrating in the same part of the pitch, even if one is a mid-action
    frame and the other is a peak-action frame).

"""