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



    def _select_dominant_person(self, res, img_width: int, img_height: int) -> Optional[int]:
        """
        Select the most relevant person for celebration detection.
        
        Strategy:
        1. Filter by minimum size (remove tiny background players)
        2. Score by: area (40%) + center proximity (30%) + confidence (30%)
        3. Return index of best person
        
        Only evaluate dominant player
        """
        if res is None or res.keypoints is None or len(res.keypoints) == 0:
            return None
        
        if res.boxes is None or len(res.boxes) == 0:
            return None
        
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones(len(boxes))
        
        img_area = float(img_width * img_height)
        center_x, center_y = img_width / 2.0, img_height / 2.0
        
        best_idx = None
        best_score = 0.0
        
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            
            # Compute area
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = area / img_area
            
            # Filter small persons (background players)
            # Require at least 4% of frame for valid detection (relaxed from 8%)
            if area_ratio < 0.04:
                continue
            
            # Compute center distance (normalized)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist_x = abs(cx - center_x) / center_x
            dist_y = abs(cy - center_y) / center_y
            center_dist = np.sqrt(dist_x**2 + dist_y**2)
            center_score = max(0.0, 1.0 - center_dist)
            
            # Composite score: area (40%) + center (30%) + confidence (30%)
            score = 0.40 * min(area_ratio / 0.25, 1.0) + 0.30 * center_score + 0.30 * conf
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx

    # Helper function to compute arm angle relative to torso for pose detection
    def _compute_arm_angle_relative_to_torso(
        self, 
        shoulder: np.ndarray, 
        wrist: np.ndarray, 
        hip: np.ndarray
    ) -> float:
        """
        Normalize relative to torso vector.
        
        Compute angle between arm vector and torso vector.
        More robust than pixel-based height comparison.
        
        Returns:
            angle in degrees [0-180]
        """
        # Torso vector (shoulder to hip)
        torso_vec = hip - shoulder
        
        # Arm vector (shoulder to wrist)
        arm_vec = wrist - shoulder
        
        # Normalize
        torso_norm = np.linalg.norm(torso_vec)
        arm_norm = np.linalg.norm(arm_vec)
        
        if torso_norm < 1e-6 or arm_norm < 1e-6:
            return 0.0
        
        torso_vec = torso_vec / torso_norm
        arm_vec = arm_vec / arm_norm
        
        # Cosine similarity
        cos_angle = np.dot(arm_vec, torso_vec)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        return float(angle)


    def pose_signals(self, path: str) -> Tuple[float, str, dict]:
        """
        Detect soccer-specific poses with ROBUST filtering.
    
        Returns:
            pose_score: [0..1] confidence of detected pose
            pose_label: "arms_raised" | "jump" | "slide" | "hands_on_head" | "one_arm_up" | "none"
            pose_breakdown: dict of all pose scores
        """
        res = self.get_pose_result(path)
        if res is None or res.keypoints is None or len(res.keypoints) == 0:
            return 0.0, "none", {}
        
        # Get image dimensions
        h, w = res.orig_shape[:2]
        
        # Select dominant person only
        dominant_idx = self._select_dominant_person(res, w, h)
        if dominant_idx is None:
            return 0.0, "none", {}
        
        # Extract keypoints for dominant person only
        det = res.keypoints[dominant_idx]
        xy = det.xy[0].detach().cpu().numpy()  # (17, 2)
        conf = getattr(det, "conf", None)
        
        if conf is not None:
            conf = conf[0].detach().cpu().numpy()
        else:
            conf = np.ones(17, dtype=np.float32)
        
        # Relaxed keypoint confidence gates for partial celebrations
        # COCO keypoints: 5=L_shoulder, 6=R_shoulder, 9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip
        
        # For partial celebrations (one-arm-up), we only need:
        # - At least ONE shoulder visible
        # - At least ONE wrist visible
        # - At least ONE hip visible
        # This allows side-profile shots where one side is occluded
        
        if max(conf[5], conf[6]) < 0.40:  # At least one shoulder
            return 0.0, "none", {}
        
        hips_visible = max(conf[11], conf[12]) >= 0.45
        
        if max(conf[9], conf[10]) < 0.30:  # At least one wrist (relaxed for partial)
            return 0.0, "none", {}
        
        # Debug: print keypoint confidences if debug mode
        if self.debug:
            print(f"[DEBUG] Keypoint confs: L_shoulder={conf[5]:.2f}, R_shoulder={conf[6]:.2f}, "
                  f"L_wrist={conf[9]:.2f}, R_wrist={conf[10]:.2f}, "
                  f"L_hip={conf[11]:.2f}, R_hip={conf[12]:.2f}")
        
        # Core confidence (average of visible shoulders + hips)
        # Use max to handle partial occlusion
        shoulder_conf = max(conf[5], conf[6])
        hip_conf = max(conf[11], conf[12])
        core_conf = (shoulder_conf + hip_conf) / 2.0
        
        if core_conf < 0.50:  # Relaxed from 0.60
            return 0.0, "none", {}
        
        # Extract keypoint positions
        ls, rs = xy[5], xy[6]  # shoulders
        lw, rw = xy[9], xy[10]  # wrists
        lh, rh = xy[11], xy[12]  # hips
        lk, rk = xy[13], xy[14]  # knees
        
        # Compute reference points
        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        torso = abs(hip_y - shoulder_y)
        
        if torso < 1e-6:
            return 0.0, "none", {}
        
        # --- POSE FEATURES ---
        
        # Original pixel-based features (kept for backward compatibility)
        lw_up = (shoulder_y - lw[1]) / torso
        rw_up = (shoulder_y - rw[1]) / torso
        knee_lift = max((hip_y - lk[1]) / torso, (hip_y - rk[1]) / torso)
        
        # Arm angles relative to torso (more robust)
        l_arm_angle = self._compute_arm_angle_relative_to_torso(ls, lw, lh)
        r_arm_angle = self._compute_arm_angle_relative_to_torso(rs, rw, rh)
        
        # Sliding detection
        knee_low = min(lk[1], rk[1]) > hip_y + 0.3 * torso
        body_horizontal = abs(shoulder_y - hip_y) < 0.3 * torso
        
        # Hands on head
        wrists_high = (lw[1] < shoulder_y) and (rw[1] < shoulder_y)
        wrists_close = abs(lw[0] - rw[0]) < abs(rs[0] - ls[0]) * 0.8
                
        scores = {}
        
        # Arms raised - use both pixel height AND angle
        # Arms raised means: wrists above shoulders (pixel) AND wide angle (>120 deg)
        if lw_up > 0.55 and rw_up > 0.55:
            # Check angles: arms should be raised (>100 degrees from torso)
            if l_arm_angle > 100 and r_arm_angle > 100:
                arm_avg = (lw_up + rw_up) / 2
                angle_factor = min((l_arm_angle + r_arm_angle) / 240, 1.0)  # normalize to [0,1]
                
                # Combined score: pixel height + angle
                pixel_score = np.clip((arm_avg - 0.45) / 0.5, 0.0, 1.0)
                scores["arms_raised"] = float(0.6 * pixel_score + 0.4 * angle_factor)
        

        # Jump detection (requires visible hips AND knees)
        if hips_visible and conf[13] > 0.40 and conf[14] > 0.40:
            knee_lift_l = (hip_y - lk[1]) / torso
            knee_lift_r = (hip_y - rk[1]) / torso

            both_knees_high = knee_lift_l > 0.60 and knee_lift_r > 0.60

            if both_knees_high:
                jump_score = float(np.clip((min(knee_lift_l, knee_lift_r) - 0.55) / 0.4, 0.0, 1.0))
                scores["jump"] = jump_score

       
        # Sliding celebration
        if knee_low and body_horizontal:
            scores["slide"] = 0.85
        
        # Hands on head
        if wrists_high and wrists_close:
            # Additional check: both wrists should have high confidence
            if conf[9] > 0.60 and conf[10] > 0.60:
                scores["hands_on_head"] = 0.75
        
        # One arm up (partial celebration) 
        # Check if ONE arm is raised significantly more than the other
        
        # Relaxed thresholds for partial celebrations
        if hips_visible:
            left_raised  = lw_up > 0.45 and l_arm_angle > 80
            right_raised = rw_up > 0.45 and r_arm_angle > 80
        else:
            # fallback: only pixel height check
            left_raised  = lw_up > 0.45
            right_raised = rw_up > 0.45

        
        # Only one arm should be raised (XOR logic)
        if left_raised != right_raised:  # XOR
            # Get the raised arm metrics
            if left_raised:
                raised_height = lw_up
                raised_angle = l_arm_angle
                raised_conf = conf[9]  # left wrist
            else:
                raised_height = rw_up
                raised_angle = r_arm_angle
                raised_conf = conf[10]  # right wrist
            
            # Only proceed if the raised arm has decent confidence
            if raised_conf > 0.30:  # Relaxed threshold
                height_score = np.clip((raised_height - 0.40) / 0.6, 0.0, 1.0)  # Relaxed from 0.50
                angle_score = min((raised_angle - 75) / 90, 1.0)  # Relaxed from 90
                
                # Combined score with confidence boost
                base_score = 0.6 * height_score + 0.4 * angle_score
                conf_boost = min(raised_conf / 0.50, 1.0)  # Boost for high confidence
                
                scores["one_arm_up"] = float(base_score * conf_boost)
        
        # Weight by keypoint confidence (confidence discount)
        for k in scores:
            scores[k] *= core_conf
        
        # Pick best pose
        if not scores:
            return 0.0, "none", {}
        
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        
        return float(best_score), str(best_label), dict(scores)


    # -----------------------
    # Face detection and quality
    # -----------------------
    def face_quality(self, path: str) -> Tuple[int, float, float, float, List, Optional[np.ndarray]]:
        """
        Single-frame face quality assessment.
        Returns: num_faces, largest_face_area, face_quality, best_det_score, detected_faces, img
        """
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
            x1, y1, x2, y2 = map(int, f.bbox.tolist())
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            largest = max(largest, area)

            face_coverage = area / img_area

            # ---------- SIZE SCORE ----------
            if 0.12 <= face_coverage <= 0.50:
                size_score = 1.0
            elif face_coverage < 0.12:
                size_score = min(face_coverage / 0.12, 1.0)
            else:
                size_score = max(1.0 - (face_coverage - 0.50) / 0.30, 0.5)

            # ---------- POSITION SCORE ----------
            fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
            max_dist = np.sqrt(cx**2 + cy**2)
            pos_score = 1.0 - float(dist / max_dist) * 0.5

            # ---------- DETECTION CONF ----------
            det_score = float(getattr(f, "det_score", 0.5))
            best_det = max(best_det, det_score)

            # ---------- FACE SHARPNESS ----------
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Normalize (tuneable)
                sharp_norm = min(lap_var / 150.0, 1.0)
            else:
                sharp_norm = 0.0

            # ---------- FINAL FACE QUALITY ----------
            q = (
                0.35 * size_score +
                0.20 * pos_score +
                0.25 * det_score +
                0.20 * sharp_norm
            )

            best_q = max(best_q, float(q))


        num = int(len(faces))
        largest_norm = float(largest / img_area)
        return num, largest_norm, float(best_q), float(best_det), faces, img
    
    # -----------------------
    # IQA 
    # -----------------------
    def iqa_norm_batch(self, paths: List[str]) -> List[float]:
        scores = [0.5] * len(paths)
        
        valid = []
        pil_images = []
        for idx, path in enumerate(paths):
            if path in self._iqa_cache:
                scores[idx] = self._iqa_cache[path]
            else:
                try:
                    img = Image.open(path).convert("RGB")
                    pil_images.append(img)
                    valid.append((idx, path))
                except Exception:
                    pass

        if not pil_images:
            return scores

        for (idx, path), img in zip(valid, pil_images):
            try:
                s = self.iqa(img)
                s = float(s.item()) if hasattr(s, "item") else float(s)

                if getattr(self.iqa, "lower_better", False):
                    s = -s

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
    def emotion_intensity_from_faces(self, path: str, detected_faces: List, img: Optional[np.ndarray], max_faces: int = 5) -> Tuple[float, str]:
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

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            emo_label, emo_scores = self.emotion_rec.predict_emotions(crop_rgb)
            intensity = max(float(emo_scores[idx[k]]) for k in expressive)

            fa = float((x2 - x1) * (y2 - y1)) / float(w * h)
            area_w = min(fa / 0.25, 1.0)
            intensity *= area_w

            if intensity > best_intensity:
                best_intensity = intensity
                best_label = str(emo_label)

        result = (float(np.clip(best_intensity, 0.0, 1.0)), best_label)
        self._emotion_cache[path] = result 
        return result
    