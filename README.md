# Master's Thesis

## Thumbnail Selection

### Project Overview

This project focuses on automatic thumbnail selection in soccer broadcasts using AI.  
The first step is **shot-type classification**: recognizing broadcast camera types (wide, close-up, replay, etc.) from the SoccerNet-v2 dataset.  

The extracted shot labels and representative frames will then be used to train models that can later be validated on our **internal dataset**.  

The classifier must be both **accurate** and **efficient enough for near real-time use**.

---

## Current Status

✅ Implemented pipeline to:  
- Parse **SoccerNet-v2** `Labels-cameras.json` annotations.  
- Treat each annotation as a shot **start**, ending at the next annotation in the same half.  
- Extract multiple frames per shot (e.g., every 2s), trimming edges and skipping logos/blurry frames.  
- Save extracted frames into per-label folders.  
- Write out a central `metadata.csv` with `(game, half, label, frame_idx, filepath)`.  
- Generate a `label_counts.csv` report for dataset balancing.  

This defines **Dataset v1**, which can now be used directly in PyTorch.

---

## Next Steps / Roadmap

### 1. Dataset Preparation (done for v1, future refinements possible)

* [x] Parse SoccerNet-v2 annotations.
* [x] Extract multiple frames per shot (configurable step size, blur filter, edge trim).
* [x] Save frames in structured folders.
* [x] Write `metadata.csv` for PyTorch integration.
* [x] Document dataset statistics (class counts, balance).
* [ ] Experiment with different sampling configs (`step_sec`, `blur_thresh`) → define Dataset v2, v3…  
* [ ] Optional: remap 13 raw broadcast shot labels into fewer categories (Wide, Close-up, Outer, Replay).

### 2. Model Training

Train and compare different architectures:

* **ResNet-18/50** (fast baselines).
* **Vision Transformer (ViT-S)** (transformer baseline).
* **3D CNN (R(2+1)D / TimeSformer)** (temporal modeling).

For each model, record:

* Accuracy/F1 per class.
* Model size (MB).
* Inference speed (ms/frame).
* Notes on trade-offs (accuracy vs speed).

### 3. Testing on Internal Dataset

* [ ] Evaluate trained models on the internal Allsvenskan dataset.
* [ ] Check generalization (close-up, wide, replay predictions).
* [ ] If poor generalization → fine-tune with additional internal frames.

### 4. Documentation & Tracking

* [x] Dataset v1 pipeline + metadata CSV.
* [ ] Track future dataset versions (v2, v3).
* [ ] Log all experiments (configs, results).
* [ ] Summarize trade-offs between accuracy and speed.
* [ ] Keep results visible in this repo (Markdown tables / CSV).

---

## Deliverables

* **Dataset v1**: SoccerNet-v2 frames + `metadata.csv` + `label_counts.csv`.
* Baseline models (ResNet, ViT, 3D CNN) with accuracy/speed benchmarks.
* Validation results on internal dataset.
* Documented pipeline for dataset preparation and training.
* Final integration of shot-type classifier into downstream thumbnail selection.

---
