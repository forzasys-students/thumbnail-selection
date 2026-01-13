"""
SOTA KEYFRAME SELECTION PIPELINE
Integrates: YOLO11-Pose + InsightFace + MUSIQ (pyiqa)

Uses hierarchical priorities from segments.csv:
- P1A_closeup
- P1B_closeup
- P2_main_camera
- P3_public
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ultralytics import YOLO
from insightface.app import FaceAnalysis
import pyiqa


SEGMENT_SCORE_MULT = {
    "P1A_closeup": 1.45,
    "P1B_closeup": 1.10,
    "P2_main_camera": 1.05,
    "P3_public": 0.70,
}

SEGMENT_QUOTA_FRAC = {
    "P1A_closeup": 0.50,
    "P1B_closeup": 0.15,
    "P2_main_camera": 0.25,
    "P3_public": 0.10,
}

QUALITY_THRESHOLDS = {
    "P1A_closeup": {
        "min_sharpness": 30,
        "min_closeup_ratio": 0.10,
        "min_musiq_norm": 0.55,   # soft gate only for P1A
    },
    "P1B_closeup": {
        "min_sharpness": 28,
        "min_closeup_ratio": 0.06,
        "min_musiq_norm": 0.0,
    },
    "P2_main_camera": {
        "min_sharpness": 25,
        "min_closeup_ratio": 0.025,
        "min_musiq_norm": 0.0,
    },
    "P3_public": {
        "min_sharpness": 20,
        "min_closeup_ratio": 0.01,
        "min_musiq_norm": 0.0,
    }
}

PER_SEGMENT_KEEP = {
    "P1A_closeup": 6,
    "P1B_closeup": 6,
    "P2_main_camera": 10,
    "P3_public": 10,
}

WEIGHTS = {
    "content": 0.40,
    "musiq": 0.25,
    "sharp": 0.12,
    "composition": 0.10,
    "saturation": 0.08,
    "confidence": 0.05,
}

# Hard floors for anything that ends up in final keyframes
MIN_FINAL_CONF = 0.60
MIN_FINAL_SCORE = 0.60


# Optional: keep quotas from collapsing if a bucket is empty after filtering
ALLOW_SCORE_FLOOR_FALLBACK = True   # if True, will relax score floor slightly if needed
FALLBACK_MIN_FINAL_SCORE = 0.50     # only used if quotas can't fill



def extract_frame_index(path: str) -> int:
    return int("".join(c for c in os.path.basename(path) if c.isdigit()))


# =======================
# CHEAP METRICS
# =======================

def transition_overlay_score(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.meanStdDev(gray)[1][0][0])


def blur_laplacian_var(path: str) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def composition_score(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h_third = h // 3
    w_third = w // 3
    pts = [(w_third, h_third), (2*w_third, h_third), (w_third, 2*h_third), (2*w_third, 2*h_third)]

    total = 0.0
    for (px, py) in pts:
        radius = int(min(w, h) * 0.1)
        y1, y2 = max(0, py - radius), min(h, py + radius)
        x1, x2 = max(0, px - radius), min(w, px + radius)
        region = edges[y1:y2, x1:x2]
        if region.size:
            total += float(np.sum(region) / (region.size * 255.0))

    return float(min(total / 4.0, 1.0))


def saturation_score(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) / 255.0


# =======================
# State Of The Art Models
# =======================

class SOTAModels:
    def __init__(
        self,
        device: str = "cuda",
        yolo_pose_path: str = "models/yolo/yolo11m-pose.pt",
        insightface_name: str = "buffalo_l",
    ):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        self.pose_model = YOLO(yolo_pose_path)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"]
        self.face_app = FaceAnalysis(name=insightface_name, providers=providers)
        ctx_id = 0 if self.device.startswith("cuda") else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        self.musiq = pyiqa.create_metric("musiq", device=self.device)

    def closeup_ratio_from_pose(self, path: str) -> float:
        try:
            r = self.pose_model(path, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return 0.0
            h, w = r.orig_shape[:2]
            img_area = float(h * w)
            best = 0.0
            for b in r.boxes.xyxy:
                x1, y1, x2, y2 = map(float, b.tolist())
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                best = max(best, area / img_area)
            return float(best)
        except Exception:
            return 0.0

    def celebration_detection(self, path: str):
        try:
            res = self.pose_model(path, verbose=False)[0]
        except Exception:
            return 0.0, "none"

        if res.keypoints is None or len(res.keypoints) == 0:
            return 0.0, "none"

        num_people = len(res.keypoints)
        best_score = 0.0
        best_type = "none"

        for det in res.keypoints:
            kps = det.xy[0]
            if kps is None or len(kps) < 17:
                continue

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

            conf = getattr(det, "conf", None)
            if conf is not None:
                try:
                    conf_arr = conf[0].detach().cpu().numpy()
                    valid = conf_arr[conf_arr > 0]
                    if valid.size:
                        score *= float(np.mean(valid))
                except Exception:
                    pass

            if score > best_score:
                best_score = float(score)
                best_type = typ

        if num_people >= 2 and best_score > 0.5:
            best_score = min(best_score + min(num_people * 0.15, 0.3), 1.0)
            best_type = f"group_{best_type}"

        return float(best_score), str(best_type)

    def face_quality(self, path: str):
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

            size_score = min(area / (img_area * 0.3), 1.0)
            fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
            max_dist = np.sqrt(cx**2 + cy**2)
            pos_score = 1.0 - float(dist / max_dist)

            det_score = float(getattr(f, "det_score", 0.5))
            best_det = max(best_det, det_score)

            q = (0.40 * size_score) + (0.30 * pos_score) + (0.30 * det_score)
            best_q = max(best_q, float(q))

        num = int(len(faces))
        multi_bonus = min(num * 0.1, 0.3)
        final_q = min(best_q + multi_bonus, 1.0)

        return num, float(largest), float(final_q), float(best_det)

    def musiq_norm(self, path: str) -> float:
        try:
            s = self.musiq(path)
            if hasattr(s, "item"):
                s = float(s.item())
            s = float(s)
            return float(np.clip(s / 100.0, 0.0, 1.0))
        except Exception:
            return 0.5


# =======================
# QUOTA SELECT
# =======================

def _quota_select(all_candidates, top_n, min_frame_gap=10):
    buckets = {k: [] for k in SEGMENT_QUOTA_FRAC.keys()}
    for c in all_candidates:
        sp = c.get("segment_priority", None)
        if sp in buckets:
            buckets[sp].append(c)

    for sp in buckets:
        buckets[sp] = sorted(buckets[sp], key=lambda x: x["final_score"], reverse=True)

    keys = list(SEGMENT_QUOTA_FRAC.keys())
    quotas = {}
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

    final = []
    used = set()

    def add(sorted_list, need):
        nonlocal final
        added = 0
        for cand in sorted_list:
            if added >= need:
                break
            if cand["path"] in used:
                continue

            ci = extract_frame_index(cand["path"])
            ok = True
            for s in final:
                si = extract_frame_index(s["path"])
                if abs(ci - si) < min_frame_gap:
                    ok = False
                    break
            if not ok:
                continue

            final.append(cand)
            used.add(cand["path"])
            added += 1
        return added

    for sp in keys:
        add(buckets[sp], quotas[sp])

    if len(final) < top_n:
        overall = sorted(all_candidates, key=lambda x: x["final_score"], reverse=True)
        add(overall, top_n - len(final))

    if len(final) < top_n:
        overall = sorted(all_candidates, key=lambda x: x["final_score"], reverse=True)
        for c in overall:
            if len(final) >= top_n:
                break
            if c["path"] in used:
                continue
            final.append(c)
            used.add(c["path"])

    return final


# =======================
# MAIN
# =======================

def select_keyframes(
    pred_csv,
    seg_csv,
    output_csv,
    output_dir,
    top_n=15,
    device="cuda",
    yolo_pose_path="models/yolo/yolo11m-pose.pt",
):
    models = SOTAModels(device=device, yolo_pose_path=yolo_pose_path)

    df_preds = pd.read_csv(pred_csv)
    df_segs = pd.read_csv(seg_csv)
    df_preds["frame_index"] = df_preds["frame_path"].apply(extract_frame_index)
    os.makedirs(output_dir, exist_ok=True)

    analytics = {
        "frames_processed": 0,
        "frames_filtered_overlay": 0,
        "frames_filtered_blur": 0,
        "frames_filtered_closeup": 0,
        "segments_empty_after_prefilter": 0,
        "p1a_failed_musiq_gate_fallbacks": 0,
        "candidates_by_segment_type": {
            "P1A_closeup": 0,
            "P1B_closeup": 0,
            "P2_main_camera": 0,
            "P3_public": 0
        },
    }

    all_candidates = []

    print(f"\n{'='*60}")
    print("SOTA KEYFRAME SELECTION (YOLO11 + InsightFace + MUSIQ) — HIERARCHICAL")
    print(f"Processing {len(df_segs)} segments")
    print(f"{'='*60}\n")

    for _, seg in tqdm(df_segs.iterrows(), total=len(df_segs), desc="Processing segments"):
        seg_id = int(seg["segment_id"])
        seg_priority = str(seg.get("priority", "P2_main_camera"))

        seg_mult = float(SEGMENT_SCORE_MULT.get(seg_priority, 1.0))
        th = QUALITY_THRESHOLDS.get(seg_priority, QUALITY_THRESHOLDS["P2_main_camera"])
        keep_k = int(PER_SEGMENT_KEEP.get(seg_priority, 6))

        start = extract_frame_index(seg["start_frame"])
        end = extract_frame_index(seg["end_frame"])
        segment_frames = df_preds[(df_preds.frame_index >= start) & (df_preds.frame_index <= end)]

        candidates = []
        for _, row in segment_frames.iterrows():
            path = row["frame_path"]
            model_conf = float(row.get("confidence", 0.5))
            # HARD GATE: model confidence
            if model_conf < MIN_FINAL_CONF:
                continue
            analytics["frames_processed"] += 1

            overlay = transition_overlay_score(path)
            if overlay < 18:
                analytics["frames_filtered_overlay"] += 1
                continue

            lap = blur_laplacian_var(path)
            if lap < th["min_sharpness"]:
                analytics["frames_filtered_blur"] += 1
                continue

            closeup = models.closeup_ratio_from_pose(path)
            if closeup < th["min_closeup_ratio"]:
                analytics["frames_filtered_closeup"] += 1
                continue

            # heavy-ish but still acceptable here (you already do this)
            num_faces, face_area, face_q, face_det = models.face_quality(path)
            celebration, cel_type = models.celebration_detection(path)
            comp = composition_score(path)
            sat = saturation_score(path)

            if num_faces > 0:
                kf_priority = 1
                rank_value = face_area
            elif celebration > 0.5:
                kf_priority = 2
                rank_value = celebration
            else:
                kf_priority = 3
                rank_value = closeup

            candidates.append({
                "path": path,
                "segment_id": seg_id,
                "segment_priority": seg_priority,
                "segment_mult": seg_mult,
                "model_confidence": model_conf,

                "keyframe_priority": int(kf_priority),
                "rank_value": float(rank_value),

                "num_faces": int(num_faces),
                "face_area": float(face_area),
                "face_quality": float(face_q),
                "face_det_score": float(face_det),

                "celebration": float(celebration),
                "celebration_type": str(cel_type),

                "closeup": float(closeup),
                "sharpness": float(lap),

                "composition": float(comp),
                "saturation": float(sat),
            })

        if not candidates:
            analytics["segments_empty_after_prefilter"] += 1
            continue

        # top-k per segment (cheap-first philosophy preserved)
        candidates_sorted = sorted(candidates, key=lambda x: (x["keyframe_priority"], -x["rank_value"]))[:keep_k]

        # MUSIQ computed only on these
        for c in candidates_sorted:
            c["musiq_norm"] = models.musiq_norm(c["path"])
            c["quality_fallback"] = 0

        # Soft MUSIQ gate ONLY for P1A (but never delete segment)
        if seg_priority == "P1A_closeup":
            gated = [c for c in candidates_sorted if c["musiq_norm"] >= th["min_musiq_norm"]]
            if not gated:
                analytics["p1a_failed_musiq_gate_fallbacks"] += 1
                # fallback: keep best 2 based on your content proxy + sharpness
                candidates_sorted = sorted(
                    candidates_sorted,
                    key=lambda x: (x["keyframe_priority"], -x["rank_value"], -x["sharpness"])
                )[:2]
                for c in candidates_sorted:
                    c["quality_fallback"] = 1
            else:
                candidates_sorted = gated

        # scoring + HARD FLOORS (do NOT mutate list while iterating)
        scored = []
        for c in candidates_sorted:
            if c["keyframe_priority"] == 1:
                content_signal = (0.60 * c["face_quality"]) + (0.40 * min(c["num_faces"] / 6.0, 1.0))
            elif c["keyframe_priority"] == 2:
                content_signal = c["celebration"]
            else:
                content_signal = c["closeup"]

            sharp_norm = float(np.clip(c["sharpness"] / 200.0, 0.0, 1.0))

            score_pre = (
                WEIGHTS["content"] * float(np.clip(content_signal, 0.0, 1.0)) +
                WEIGHTS["musiq"] * float(np.clip(c["musiq_norm"], 0.0, 1.0)) +
                WEIGHTS["sharp"] * sharp_norm +
                WEIGHTS["composition"] * float(np.clip(c["composition"], 0.0, 1.0)) +
                WEIGHTS["saturation"] * float(np.clip(c["saturation"], 0.0, 1.0)) +
                WEIGHTS["confidence"] * float(np.clip(c["model_confidence"], 0.0, 1.0))
            )

            c["final_score"] = float(score_pre * c["segment_mult"])

            # HARD FLOORS
            if c["model_confidence"] < MIN_FINAL_CONF:
                continue
            if c["final_score"] < MIN_FINAL_SCORE:
                continue

            scored.append(c)

        candidates_sorted = scored
        if not candidates_sorted:
            continue


        if seg_priority in analytics["candidates_by_segment_type"]:
            analytics["candidates_by_segment_type"][seg_priority] += len(candidates_sorted)

        all_candidates.extend(candidates_sorted)

    if not all_candidates:
        print("[WARN] No candidates found. Prefilters too strict or model paths failing.")
        pd.DataFrame([]).to_csv(output_csv, index=False)
        return

    final_selection = _quota_select(all_candidates, top_n=top_n, min_frame_gap=10)
    # FINAL HARD FILTER (quota can't override this)
    final_selection = [
        c for c in final_selection
        if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
        and float(c.get("final_score", 0.0)) >= MIN_FINAL_SCORE
    ]   

    if len(final_selection) < top_n and ALLOW_SCORE_FLOOR_FALLBACK:
        # Relax ONLY score floor a bit, but NEVER relax confidence
        extra = sorted(
            [c for c in all_candidates
            if float(c.get("model_confidence", 0.0)) >= MIN_FINAL_CONF
            and float(c.get("final_score", 0.0)) >= FALLBACK_MIN_FINAL_SCORE
            and c not in final_selection],
            key=lambda x: x["final_score"],
            reverse=True
        )
        for c in extra:
            if len(final_selection) >= top_n:
                break
            final_selection.append(c)



    # Save
    results = []
    for rank, c in enumerate(final_selection):
        kf_label = f"P{c['keyframe_priority']}"
        pr = c.get("segment_priority", "NA")
        conf = float(c.get("model_confidence", 0.0))
        score = float(c.get("final_score", 0.0))

        out_path = os.path.join(
            output_dir,
            f"rank{rank+1:02d}_{pr}_conf{conf:.2f}_score{score:.3f}.jpg"
        )

        img = cv2.imread(c["path"])
        if img is not None:
            cv2.imwrite(out_path, img)

        results.append({
            "rank": rank + 1,
            "segment_id": c["segment_id"],
            "segment_priority": pr,
            "segment_multiplier": round(float(c["segment_mult"]), 3),
            "keyframe_priority": int(c["keyframe_priority"]),
            "model_confidence": round(conf, 3),
            "final_score": round(score, 3),

            "celebration": round(float(c["celebration"]), 3),
            "celebration_type": c["celebration_type"],

            "num_faces": int(c["num_faces"]),
            "face_quality": round(float(c["face_quality"]), 3),
            "face_det_score": round(float(c["face_det_score"]), 3),

            "closeup": round(float(c["closeup"]), 3),
            "sharpness": round(float(c["sharpness"]), 3),
            "composition": round(float(c["composition"]), 3),
            "saturation": round(float(c["saturation"]), 3),

            "musiq_norm": round(float(c.get("musiq_norm", 0.0)), 3),
            "quality_fallback": int(c.get("quality_fallback", 0)),

            "input_path": c["path"],
            "saved_path": out_path
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    # Analytics
    print(f"\n{'='*60}")
    print("ANALYTICS")
    print(f"{'='*60}")
    print(f"Frames processed: {analytics['frames_processed']}")
    print(f"Filtered overlay: {analytics['frames_filtered_overlay']}")
    print(f"Filtered blur: {analytics['frames_filtered_blur']}")
    print(f"Filtered closeup: {analytics['frames_filtered_closeup']}")
    print(f"Segments empty after prefilter: {analytics['segments_empty_after_prefilter']}")
    print(f"P1A MUSIQ gate fallbacks used: {analytics['p1a_failed_musiq_gate_fallbacks']}")
    print("\nCandidates by segment type:")
    for k, v in analytics["candidates_by_segment_type"].items():
        print(f"  {k}: {v}")

    print(f"\nSaved {len(results_df)} keyframes -> {output_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keyframe Selection")
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--seg_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--yolo_pose_path", type=str, default="models/yolo/yolo11m-pose.pt")
    args = parser.parse_args()

    select_keyframes(
        pred_csv=args.pred_csv,
        seg_csv=args.seg_csv,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        device=args.device,
        yolo_pose_path=args.yolo_pose_path,
    )
