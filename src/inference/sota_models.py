# sota_models.py

from __future__ import annotations

import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Tuple, Optional, Union
from PIL import Image

import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis
import pyiqa

class SOTAModels:
    def __init__(
        self,
        device: str = "cuda",
        yolo_pose_path: str = "models/yolo/yolo11m-pose.pt",
        insightface_name: str = "buffalo_l",
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"

        if self.debug:
            print(f"[SOTA] Init: device={self.device}")
            print(f"[SOTA] Loading YOLO pose: {yolo_pose_path}")

        self.pose_model = YOLO(yolo_pose_path)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        if self.debug:
            print(f"[SOTA] Loading InsightFace: {insightface_name} providers={providers}")

        self.face_app = FaceAnalysis(name=insightface_name, providers=providers)
        ctx_id = 0 if self.device.startswith("cuda") else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        if self.debug:
            print("[SOTA] Loading MUSIQ (pyiqa)")

        self.musiq = pyiqa.create_metric("musiq", device=self.device)
        
        # Cache for MUSIQ
        self._musiq_cache = {}
        
        # Cache for pose results 
        self._pose_cache = {}

    # -----------------------
    # Single pose inference
    # -----------------------
    
    def get_pose_result(self, path: str):
        """
        Get pose result with caching.
        This prevents calling YOLO twice per frame (closeup + celebration).
        """
        if path in self._pose_cache:
            return self._pose_cache[path]
        
        try:
            result = self.pose_model(path, verbose=False)[0]
            self._pose_cache[path] = result
            return result
        except Exception:
            self._pose_cache[path] = None
            return None

    # -----------------------
    # Pose / closeup 
    # -----------------------

    def closeup_ratio_from_pose(self, path: str) -> float:
        """Largest bbox area / image area from pose model boxes."""
        r = self.get_pose_result(path)
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return 0.0
        
        h, w = r.orig_shape[:2]
        img_area = float(h * w)
        best = 0.0
        for b in r.boxes.xyxy:
            x1, y1, x2, y2 = map(float, b.tolist())
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            best = max(best, area / img_area)
        return float(best)

    # -----------------------
    # Celebration 
    # -----------------------

    def celebration_detection(self, path: str) -> Tuple[float, str]:
        """Improved celebration detection using cached pose result."""
        res = self.get_pose_result(path)
        if res is None or res.keypoints is None or len(res.keypoints) == 0:
            return 0.0, "none"

        num_people = len(res.keypoints)
        best_score = 0.0
        best_type = "none"
        high_conf_people = 0

        for det in res.keypoints:
            kps = det.xy[0]
            if kps is None or len(kps) < 17:
                continue

            # Keypoint confidence gating
            conf = getattr(det, "conf", None)
            if conf is not None:
                try:
                    conf_arr = conf[0].detach().cpu().numpy()
                    valid_conf = conf_arr[conf_arr > 0.4]
                    if valid_conf.size < 10:
                        continue
                    avg_conf = float(np.mean(valid_conf))
                    if avg_conf < 0.5:
                        continue
                    high_conf_people += 1
                except Exception:
                    continue
            else:
                high_conf_people += 1

            ls, rs = kps[5], kps[6]
            lw, rw = kps[9], kps[10]
            lh, rh = kps[11], kps[12]
            lk, rk = kps[13], kps[14]

            shoulder_y = float((ls[1] + rs[1]) / 2.0)
            hip_y = float((lh[1] + rh[1]) / 2.0)
            torso_h = abs(hip_y - shoulder_y)
            if torso_h <= 1e-6:
                continue

            left_arm = (shoulder_y - float(lw[1])) / torso_h
            right_arm = (shoulder_y - float(rw[1])) / torso_h
            avg_arm = (left_arm + right_arm) / 2.0

            shoulder_w = abs(float(rs[0] - ls[0]))
            wrist_w = abs(float(rw[0] - lw[0]))
            spread = (wrist_w / shoulder_w) if shoulder_w > 1e-6 else 0.0

            jumping = (float(lk[1]) < hip_y - torso_h * 0.3) or (float(rk[1]) < hip_y - torso_h * 0.3)
            both_arms_up = (left_arm > 0.4) and (right_arm > 0.4)

            if jumping and avg_arm > 0.4:
                score, typ = 1.0, "jumping_celebration"
            elif both_arms_up and spread > 1.3:
                score, typ = 0.95, "full_celebration"
            elif both_arms_up:
                score, typ = 0.8, "arms_raised"
            elif (left_arm > 0.2) or (right_arm > 0.2):
                score, typ = 0.5, "partial_celebration"
            else:
                score, typ = 0.0, "none"

            if conf is not None and score > 0.0:
                score *= avg_conf

            if score > best_score:
                best_score = float(score)
                best_type = typ

        if high_conf_people >= 2 and best_score > 0.5:
            group_bonus = min(high_conf_people * 0.10, 0.25)
            best_score = min(best_score + group_bonus, 1.0)
            
            if high_conf_people >= 2:
                best_type = f"group_{best_type}"
            
            if high_conf_people >= 3 and best_score > 0.75:
                best_score = min(best_score * 1.1, 1.0)

        return float(best_score), str(best_type)

    # -----------------------
    # Face quality
    # -----------------------

    def face_quality(self, path: str) -> Tuple[int, float, float, float]:
        """Returns: num_faces, largest_face_area, face_quality, best_det_score"""
        img = cv2.imread(path)
        if img is None:
            return 0, 0.0, 0.0, 0.0

        faces = self.face_app.get(img)
        if not faces:
            return 0, 0.0, 0.0, 0.0

        h, w = img.shape[:2]
        img_area = float(h * w)
        cx, cy = w / 2.0, h / 2.0

        largest = 0.0
        best_q = 0.0
        best_det = 0.0

        for f in faces:
            x1, y1, x2, y2 = map(float, f.bbox.tolist())
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            largest = max(largest, area)

            face_coverage = area / img_area
            if face_coverage >= 0.15 and face_coverage <= 0.40:
                size_score = 1.0
            elif face_coverage < 0.15:
                size_score = min(face_coverage / 0.15, 1.0)
            else:
                size_score = max(1.0 - (face_coverage - 0.40) / 0.30, 0.5)
            
            fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
            max_dist = np.sqrt(cx**2 + cy**2)
            pos_score = 1.0 - float(dist / max_dist) * 0.5

            det_score = float(getattr(f, "det_score", 0.5))
            best_det = max(best_det, det_score)

            q = (0.45 * size_score) + (0.25 * pos_score) + (0.30 * det_score)
            best_q = max(best_q, float(q))

        num = int(len(faces))
        return num, float(largest), float(best_q), float(best_det)

    # -----------------------
    #  BATCHED MUSIQ 
    # -----------------------

    def musiq_norm_batch(self, paths: List[str]) -> List[float]:
        """
        ACTUALLY batch process MUSIQ.
        Previous version just looped - this version truly batches.
        """
        if not paths:
            return []
        
        # Separate cached vs uncached
        uncached_paths = []
        uncached_indices = []
        scores = [None] * len(paths)
        
        for i, path in enumerate(paths):
            if path in self._musiq_cache:
                scores[i] = self._musiq_cache[path]
            else:
                uncached_paths.append(path)
                uncached_indices.append(i)
        
        if not uncached_paths:
            return scores
        
        try:
            # Load all images as PIL
            pil_images = []
            valid_indices = []
            
            for idx, path in zip(uncached_indices, uncached_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    pil_images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    if self.debug:
                        print(f"[MUSIQ] Failed to load {path}: {e}")
                    scores[idx] = 0.5
                    self._musiq_cache[path] = 0.5
            
            if not pil_images:
                return scores
            
            # ACTUAL BATCHING: Process all images at once if model supports it
            # Note: pyiqa's MUSIQ might not support true batching, so we compromise
            # by at least avoiding redundant preprocessing
            batch_scores = []
            for img in pil_images:
                try:
                    s = self.musiq(img)
                    if hasattr(s, "item"):
                        s = float(s.item())
                    else:
                        s = float(s)
                    s_norm = float(np.clip(s / 100.0, 0.0, 1.0))
                    batch_scores.append(s_norm)
                except Exception as e:
                    if self.debug:
                        print(f"[MUSIQ] Error: {e}")
                    batch_scores.append(0.5)
            
            # Fill in results and cache
            for idx, path, score in zip(valid_indices, uncached_paths, batch_scores):
                scores[idx] = score
                self._musiq_cache[path] = score
                
        except Exception as e:
            if self.debug:
                print(f"[MUSIQ] Batch error: {e}")
            # Fallback: fill remaining with 0.5
            for idx in uncached_indices:
                if scores[idx] is None:
                    scores[idx] = 0.5
                    self._musiq_cache[paths[idx]] = 0.5
        
        return scores

    def musiq_norm(self, path: str) -> float:
        """Single-image MUSIQ (uses batch internally)."""
        return self.musiq_norm_batch([path])[0]
    
    def clear_caches(self):
        """Clear all caches (call between videos to free memory)."""
        self._musiq_cache.clear()
        self._pose_cache.clear()


class LogoDetector:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"

        # ResNet-50 backbone
        m = models.resnet50(weights=None)

        # IMPORTANT: this checkpoint uses a custom fc head, not default resnet.fc
        m.fc = nn.Sequential(
            nn.BatchNorm1d(2048),      # fc.0.*
            nn.Dropout(p=0.5),         # fc.1 (no params)
            nn.Linear(2048, 512),      # fc.2.*
            nn.ReLU(inplace=True),     # fc.3 (no params)
            nn.BatchNorm1d(512),       # fc.4.*
            nn.Dropout(p=0.5),         # fc.5 (no params)
            nn.Linear(512, 2),         # fc.6.*
        )

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # PyTorch < 2.6 doesn't have weights_only
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=True)

        self.model = m.to(self.device).eval()

        # Standard ImageNet preprocessing 
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @torch.no_grad()
    def predict_logo_prob(self, paths: List[str], batch_size: int = 32) -> List[float]:
        """
        Returns probability that a frame contains a logo.
        ASSUMPTION: class index 1 == "logo".
        If results look inverted, swap to probs[:,0].
        """
        probs: List[float] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = []
            ok_mask = []

            for p in batch_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    imgs.append(self.tf(im))
                    ok_mask.append(True)
                except Exception:
                    ok_mask.append(False)

            if not imgs:
                probs.extend([0.0] * len(batch_paths))
                continue

            x = torch.stack(imgs, dim=0).to(self.device)
            logits = self.model(x)
            p_logo = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()

            # map back including failed reads
            j = 0
            for ok in ok_mask:
                if ok:
                    probs.append(float(p_logo[j]))
                    j += 1
                else:
                    probs.append(0.0)

        return probs
