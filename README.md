# Master's Thesis

## Thumbnail Selection in Soccer Broadcasts

### Project Overview

This project focuses on automatic thumbnail selection in soccer broadcasts using AI.

The pipeline is designed in **stages**:

1. **Shot-type classification**  
   Classify broadcast frames into camera types (e.g., close-up, main camera center/left/right, public, staff, etc.).  
   This stage is now implemented and evaluated on both Norwegian and Swedish league data.

2. **Keyframe selection within close-up shots**  
   Given a goal clip, the system first uses the shot-type model to isolate **close-up segments** (preferably player close-ups).  
   Within those segments, the next step is to select a small set of **interesting, non-blurry keyframes** suitable for thumbnails.

3. **Segmentation and graphic overlays**  
   After a keyframe is chosen, apply segmentation and graphic overlays to generate a final thumbnail proposal.

The first stage (shot-type classification) is now largely complete. The current work is transitioning into **Stage 2: keyframe selection**.

---

## Current Status

### 1. Shot-Type Dataset(s)

 Implemented pipelines to build internal frame datasets for camera-type classification:

- Extracted frames from internal broadcasts (Eliteserien + Allsvenskan).  
- Defined **8 camera classes**:  
  `Close-up_behind_the_goal`, `Close-up_corner`, `Close-up_player_or_field_referee`,  
  `Close-up_side_staff`, `Main_camera_center`, `Main_camera_left`, `Main_camera_right`, `Public`.
- Built two iterations of the dataset:
  - **`frames_global` (v1)** – initial internal dataset.
  - **`frames_global_v2` (v2)** – expanded with additional Allsvenskan matches to reduce domain gap.
- Stored metadata in a central `metadata.csv` with:
  - game / match id  
  - half  
  - label  
  - frame index  
  - file path

### 2. Model Training and Evaluation (Shot-Type Classification)

 Implemented a **training pipeline** (`train.py`) to train and evaluate:

- `ResNet18`
- `ResNet50`
- `ViT-B/16`
- `EfficientNet-B3`
- `ConvNeXt-Base`

Key points:

- All models trained with:
  - 10 epochs  
  - batch size 32  
  - learning rate fixed per model (as previously tuned)  
  - Adam optimizer, CrossEntropy loss  
- Best checkpoint per model = epoch with highest **validation accuracy**.
- Final evaluation is done with a separate script (`evaluate.py`) on:
  - A **held-out Allsvenskan test set** of 1000 images (8 classes).  
  - Metrics: per-class precision, recall, F1, macro F1, confusion matrix, and average inference time (ms/frame).

**Latest results on `frames_global_v2` (Allsvenskan test set):**

| Model           | Accuracy (Test) | Macro F1 | Avg Inference (ms/frame) |
|-----------------|-----------------|----------|---------------------------|
| ResNet18        | 75.9%           | 0.754    | 35.32                     |
| ResNet50        | **84.2%**       | **0.843**| 23.92                     |
| ViT-B/16        | 80.4%           | 0.806    | 25.74                     |
| EfficientNet-B3 | 80.7%           | 0.807    | 24.43                     |
| ConvNeXt-Base   | 79.9%           | 0.791    | 29.64                     |

- **Best overall backbone (accuracy + F1):** currently **ResNet50**.  
- **Best speed–accuracy trade-off for near real-time:** **EfficientNet-B3** (slightly lower F1 than ResNet50, but very fast).

The evaluation scripts also generate:

- `metrics_<model>.csv` → full per-class precision/recall/F1.  
- `confusion_matrix_<model>.png` → confusion matrices for qualitative analysis.

---

## Before vs After Dataset Expansion

The expansion from `frames_global` → `frames_global_v2` (adding Swedish Allsvenskan data) significantly improved generalization to the Allsvenskan test set:

| Model        | Acc Before | F1 Before | Acc After | F1 After |
|-------------|------------|-----------|-----------|----------|
| ResNet18    | 68.5%      | 0.666     | **75.9%** | **0.754** |
| ResNet50    | 62.6%      | 0.614     | **84.2%** | **0.843** |
| ViT-B/16    | 52.5%      | 0.523     | **80.4%** | **0.806** |
| EfficientNet| 69.1%      | 0.667     | **80.7%** | **0.807** |
| ConvNeXt    | 67.5%      | 0.649     | **79.9%** | **0.791** |

Observations:

- All models gained **~10–20 percentage points** in accuracy and macro F1.  
- The **“Main camera” classes** (center/left/right), which were previously heavily confused, are now much better separated in the confusion matrices.  
- The models still confuse some visually similar close-up classes (e.g., player/referee vs side staff), but overall diagonals are much stronger after expansion.

---

## Next Steps / Roadmap

### Stage 1 – Shot-Type Classification (status)

* [x] Build internal dataset `frames_global` from broadcast clips.
* [x] Expand to `frames_global_v2` with Allsvenskan matches to reduce domain gap.
* [x] Train and evaluate 5 architectures (ResNet18, ResNet50, ViT-B/16, EfficientNet-B3, ConvNeXt-Base).
* [x] Compare accuracy, macro F1, and inference time on Allsvenskan test set.
* [x] Analyze per-class results and confusion matrices.
* [x] Select **best-performing model** as main backbone (currently: **ResNet50**, with EfficientNet-B3 as a strong real-time candidate).

### Stage 2 – Keyframe Selection Within Close-Up Shots (current focus)

Goal: given a **goal clip** as input:

1. Use the trained **shot-type model** to:
   - Predict shot type per frame.
   - Extract **segments classified as close-ups** (ideally player close-ups).

2. Within each close-up segment, select **one or a few keyframes** that are good thumbnail candidates.

Planned components / ideas (based on Mehdi’s feedback + upcoming literature review):

* **Frame quality filtering**
  - Detect and discard **blurry** frames (e.g., using variance of Laplacian or other sharpness metrics).
  - Skip **distorted** or heavily motion-blurred frames.

* **Face visibility and quality**
  - Use a **face detection model** to:
    - Ensure at least one **visible face** (for story/emotion).
    - Prefer frames where the face is **large enough**, not heavily occluded, and not blurred.

* **Pose and celebration detection**

* **Scoring logic for keyframe selection**

* **Literature review**
  - Read and summarize **recent work on highlight thumbnail / keyframe selection** in sports and broadcast video  
    (including the paper suggested by Mehdi and more recent works).  
  - Integrate this into the **Background / Related Work** section of the thesis:
    - What others do for keyframe/thumbnail selection,
    - How they use faces, pose, motion, attention, etc.,
    - How the proposed approach fits / differs.

Ideas for checklist for Stage 2:

* [ ] Implement shot-type–based segment extraction for goal clips.  
* [ ] Implement frame quality (blur/distortion) filtering.  
* [ ] Integrate a face detection module and basic face-quality scoring.  
* [ ] Integrate a pose estimation module and design a simple pose-based scoring rule (e.g., celebration-like postures).  
* [ ] Design and tune a combined scoring function for keyframe ranking.  
* [ ] Evaluate qualitatively (visual examples) and, if possible, with user study or proxy metrics.

### Stage 3 – Segmentation and Graphic Overlays (planned)

* [ ] Apply player/background segmentation on the selected keyframe(s).  
* [ ] Design overlay templates (club logos, text, graphics).  
* [ ] Compose final thumbnail candidates with:
  - segmented player,  
  - clean background / blur,  
  - graphical elements.  
* [ ] Evaluate final thumbnails visually and/or via small user study.

---


## Deliverables (Updated)

* **Shot-Type Classifier (Stage 1)**
  - Internal dataset(s) with labeled frames and metadata (`frames_global`, `frames_global_v2`).
  - Trained models (ResNet18, ResNet50, ViT-B/16, EfficientNet-B3, ConvNeXt-Base).
  - Quantitative evaluation on Allsvenskan test set (accuracy, macro F1, per-class results, confusion matrices, inference speed).

* **Keyframe Selection Prototype (Stage 2)**
  - Pipeline that:
    1. Takes a goal clip as input,
    2. Uses the shot-type model to find close-up segments,
    3. Ranks frames within these segments using blur, face, and pose signals,
    4. Outputs 1–3 thumbnail candidates per event.

* **Segmentation + Overlays (Stage 3)**
  - Prototype of segmentation + graphic overlay on top of the selected keyframes.

* **Thesis Integration**
  - Background/related work on thumbnail and keyframe selection.
  - Detailed description of dataset construction, model training, evaluation, and iterative improvements.
  - Discussion of trade-offs between accuracy, robustness, and computational efficiency for practical deployment.
