# Master's thesis

## Thumbnail Selection

### Project Overview

This project focuses on automatic thumbnail selection using AI. In order to achieve this, we need an AI model that can classify different types of shots in every event, **shot-type classification** for soccer broadcasts, as part of the thumbnail selection pipeline. The goal is to train and evaluate models on **SoccerNet-v2**, and later validate their performance on our **internal dataset**.

The classifier should not only be **accurate** but also **efficient enough for real-time or near real-time use**.

---

## Next Steps / Roadmap

### 1. Dataset Preparation

* [ ] Parse SoccerNet-v2 annotations (JSON).
* [ ] Map 13 broadcast shot classes → 5 categories:

  * Wide (main left, right, center)
  * Medium (medium left, right)
  * Close-up (player, referee)
  * Outer (coach, bench, public)
  * Replay/Logo
* [ ] Extract representative frames per shot (e.g. middle frame).
* [ ] Create training/validation/test splits.
* [ ] Document dataset statistics (class counts, balance).

### 2. Model Training

Train and compare different architectures:

* **ResNet-18/50** (baseline, varying speed/accuracy).
* **Vision Transformer (ViT-S)** (transformer baseline).
* **3D CNN (R(2+1)D / TimeSformer)** (temporal modeling).

For each model, record:

* Accuracy/F1 per class.
* Model size (MB).
* Inference speed (ms/frame).

### 3. Testing on Internal Dataset

* [ ] Evaluate trained models on the internal Allsvenskan dataset.
* [ ] Check generalization (close-up, wide, replay predictions).
* [ ] If generalization is poor → fine-tune with additional internal frames.

### 4. Documentation & Tracking

* [ ] Log all experiments (dataset versions, model configs, results).
* [ ] Summarize trade-offs between accuracy and speed.
* [ ] Keep results visible in this repo (Markdown tables / CSV).

---

## Deliverables

* Clean SoccerNet-v2 training dataset (mapped + split).
* Baseline models (ResNet, ViT, 3D CNN) with accuracy/speed benchmarks.
* Validation results on internal dataset.
* Documented pipeline that integrates shot-type classifier for downstream thumbnail selection.

---
