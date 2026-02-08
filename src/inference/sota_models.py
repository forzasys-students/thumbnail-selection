# sota_models.py

from __future__ import annotations

import time
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np

from hsemotion.facial_emotions import HSEmotionRecognizer
from insightface.app import FaceAnalysis
from ultralytics import YOLO
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
            print("[SOTA] Loading TOPIQ (pyiqa)")

        #self.musiq = pyiqa.create_metric("musiq", device=self.device)        
        self.iqa = pyiqa.create_metric("topiq_nr", device=self.device)

        # Cache for TOPIQ
        self._iqa_cache = {}
        
        # Cache for pose results 
        self._pose_cache = {}

        # Cache for emotion results
        self._emotion_cache = {} 

        if self.debug:
            print("[SOTA] Loading HSEmotion")

        # HSEmotion device should match your pipeline device
        # hsemotion accepts 'cpu' or 'cuda'
        emo_device = "cuda" if self.device.startswith("cuda") else "cpu"
        self.emotion_rec = HSEmotionRecognizer(
            model_name="enet_b0_8_best_afew",
            device=emo_device
        )


    # -----------------------
    # Single pose inference
    # -----------------------
    
    def get_pose_result(self, path: str):
        if path in self._pose_cache:
            return self._pose_cache[path]
        
        try:
            result = self.pose_model.predict(
                path,
                imgsz=640,       # Faster, still good for broadcast footage
                conf=0.30,       # Higher - fewer false positives
                iou=0.5,         # Lower - better for overlapping people (celebrations)
                max_det=10,      # Limit detections (group celebrations rarely >10 people in frame)
                verbose=False,
            )[0]
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


    def pose_signals(self, path: str) -> Tuple[float, str, dict]:
        """
        Detect soccer-specific poses with STRICT thresholds.
        
        Returns:
            pose_score: [0..1] confidence of detected pose
            pose_label: "arms_raised" | "jump" | "slide" | "hands_on_head" | "one_arm_up" | "none"
            pose_breakdown: dict of all pose scores
        """
        res = self.get_pose_result(path)
        if res is None or res.keypoints is None or len(res.keypoints) == 0:
            return 0.0, "none", {}

        best_score = 0.0
        best_label = "none"
        best_breakdown = {}

        for det in res.keypoints:
            xy = det.xy[0].detach().cpu().numpy()  # (17, 2)
            conf = getattr(det, "conf", None)
            
            if conf is not None:
                conf = conf[0].detach().cpu().numpy()
                # Require HIGH confidence on core keypoints
                core_conf = np.mean(conf[[5, 6, 11, 12]])
                if core_conf < 0.55:  # ← Was 0.45, now 0.55
                    continue
            else:
                conf = np.ones(17, dtype=np.float32)
                core_conf = 1.0

            # Keypoint indices
            ls, rs = xy[5], xy[6]
            lw, rw = xy[9], xy[10]
            lh, rh = xy[11], xy[12]
            lk, rk = xy[13], xy[14]

            shoulder_y = (ls[1] + rs[1]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            torso = abs(hip_y - shoulder_y)
            
            if torso < 1e-6:
                continue

            # --- POSE FEATURES (STRICT) ---
            
            lw_up = (shoulder_y - lw[1]) / torso
            rw_up = (shoulder_y - rw[1]) / torso
            knee_lift = max((hip_y - lk[1]) / torso, (hip_y - rk[1]) / torso)
            
            # Sliding detection
            knee_low = min(lk[1], rk[1]) > hip_y + 0.3 * torso
            body_horizontal = abs(shoulder_y - hip_y) < 0.3 * torso
            
            # Hands on head
            wrists_high = (lw[1] < shoulder_y) and (rw[1] < shoulder_y)
            wrists_close = abs(lw[0] - rw[0]) < abs(rs[0] - ls[0]) * 0.8

            # --- POSE SCORES (VERY STRICT) ---
            
            scores = {}
            
            # Arms raised - both wrists WELL above shoulders
            if lw_up > 0.55 and rw_up > 0.55:
                arm_avg = (lw_up + rw_up) / 2
                scores["arms_raised"] = float(np.clip((arm_avg - 0.45) / 0.5, 0.0, 1.0))
            
            # Jumping - knees notably lifted
            if knee_lift > 0.40:
                jump_score = float(np.clip((knee_lift - 0.30) / 0.4, 0.0, 1.0))
                # Bonus if arms also raised
                if max(lw_up, rw_up) > 0.45:
                    jump_score = min(jump_score * 1.15, 1.0)
                scores["jump"] = jump_score
            
            # Sliding celebration
            if knee_low and body_horizontal:
                scores["slide"] = 0.85
            
            # Hands on head
            if wrists_high and wrists_close:
                scores["hands_on_head"] = 0.75
            
            # One arm up (partial celebration)
            if (lw_up > 0.60) ^ (rw_up > 0.60):  # XOR
                scores["one_arm_up"] = float(np.clip((max(lw_up, rw_up) - 0.50) / 0.5, 0.0, 1.0))

            # Weight by keypoint confidence
            for k in scores:
                scores[k] *= core_conf

            # Pick best pose for this person
            if scores:
                label = max(scores, key=scores.get)
                score = scores[label]
                
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_breakdown = scores

        # NO GROUP LOGIC - just return best individual pose
        return float(best_score), str(best_label), dict(best_breakdown)

    # -----------------------
    # Face quality
    # -----------------------
    def face_quality(self, path: str) -> Tuple[int, float, float, float, List, Optional[np.ndarray]]:
        """Returns: num_faces, largest_face_area, face_quality, best_det_score"""
        img = cv2.imread(path)
        if img is None:
            return 0, 0.0, 0.0, 0.0, [], None

        faces = self.face_app.get(img)
        if not faces:
            return 0, 0.0, 0.0, 0.0, [], img

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
            if face_coverage >= 0.12 and face_coverage <= 0.50:
                size_score = 1.0
            elif face_coverage < 0.12:
                size_score = min(face_coverage / 0.12, 1.0)
            else:
                size_score = max(1.0 - (face_coverage - 0.50) / 0.30, 0.5)
            
            fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
            max_dist = np.sqrt(cx**2 + cy**2)
            pos_score = 1.0 - float(dist / max_dist) * 0.5

            det_score = float(getattr(f, "det_score", 0.5))
            best_det = max(best_det, det_score)

            q = (0.45 * size_score) + (0.25 * pos_score) + (0.30 * det_score)
            best_q = max(best_q, float(q))

        num = int(len(faces))
        largest_norm = float(largest / img_area)
        return num, largest_norm, float(best_q), float(best_det), faces, img # treat face_area as [0..1].



    # -----------------------
    #  BATCHED IQA 
    # -----------------------
    def iqa_norm_batch(self, paths: List[str]) -> List[float]:
        if not paths:
            return []

        uncached_paths = []
        uncached_indices = []
        scores = [None] * len(paths)

        for i, path in enumerate(paths):
            if path in self._iqa_cache:
                scores[i] = self._iqa_cache[path]
            else:
                uncached_paths.append(path)
                uncached_indices.append(i)

        if not uncached_paths:
            return scores

        pil_images = []
        valid = []
        for idx, path in zip(uncached_indices, uncached_paths):
            try:
                img = Image.open(path).convert("RGB")
                pil_images.append(img)
                valid.append((idx, path))
            except Exception as e:
                if self.debug:
                    print(f"[IQA] Failed to load {path}: {e}")
                scores[idx] = 0.5
                self._iqa_cache[path] = 0.5

        if not pil_images:
            return scores

        for (idx, path), img in zip(valid, pil_images):
            try:
                s = self.iqa(img)
                s = float(s.item()) if hasattr(s, "item") else float(s)

                # Make it consistent: higher = better
                if getattr(self.iqa, "lower_better", False):
                    s = -s

                # Clamp to a stable range, then map to [0,1]
                s = float(np.clip(s, -5.0, 5.0))
                s_norm = (s + 5.0) / 10.0
            except Exception as e:
                if self.debug:
                    print(f"[IQA] Error: {e}")
                s_norm = 0.5

            scores[idx] = s_norm
            self._iqa_cache[path] = s_norm

        return scores


    def iqa_norm(self, path: str) -> float:
        return self.iqa_norm_batch([path])[0]

    
    # -----------------------
    #  Clear caches
    # -----------------------
    def clear_caches(self):
        """Clear all caches (call between videos to free memory)."""
        self._iqa_cache.clear()
        self._pose_cache.clear()
        self._emotion_cache.clear() 

    # -----------------------
    #  Emotion detection from faces
    # -----------------------
    def emotion_intensity_from_faces(self, path: str, detected_faces: List, img: Optional[np.ndarray], max_faces: int = 2) -> Tuple[float, str]:
        if path in self._emotion_cache:
            return self._emotion_cache[path]
        
        if not detected_faces:
            return 0.0, "none"
        
        if img is None:
            img = cv2.imread(path)
        
        h, w = img.shape[:2]
        
        # Skip self.face_app.get() - use detected_faces directly
        faces = detected_faces

        def area(f):
            x1, y1, x2, y2 = map(float, f.bbox.tolist())
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        faces = sorted(faces, key=area, reverse=True)[:max_faces]

        # For enet_b0_8_* models (8 classes)
        idx = {
            "Anger": 0,
            "Contempt": 1,
            "Disgust": 2,
            "Fear": 3,
            "Happiness": 4,
            "Neutral": 5,
            "Sadness": 6,
            "Surprise": 7,
        }
        expressive = ["Happiness", "Surprise", "Anger"]

        best_intensity = 0.0
        best_label = "none"

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox.tolist())
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # IMPORTANT: pass numpy array, NOT PIL.
            # hsemotion will do Image.fromarray() internally.
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            emo_label, emo_scores = self.emotion_rec.predict_emotions(crop_rgb)  # emo_scores is np array
            intensity = max(float(emo_scores[idx[k]]) for k in expressive)

            # weight by face area (relative) so tiny faces don't dominate
            fa = float((x2 - x1) * (y2 - y1)) / float(w * h)
            area_w = min(fa / 0.25, 1.0)  # saturate at 25% of frame
            intensity *= area_w

            if intensity > best_intensity:
                best_intensity = intensity
                best_label = str(emo_label)

        result = (float(np.clip(best_intensity, 0.0, 1.0)), best_label)
        self._emotion_cache[path] = result 
        return result
    

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

