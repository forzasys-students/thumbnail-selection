Inference Pipeline — How to Run: |
  This script runs the full inference pipeline on any goal-event clip.
  It handles frame extraction, shot-type classification, segment creation,
  and keyframe selection automatically.

  You don't need to modify the game name or clip name — they are just folder
  labels used for organizing output. What actually matters is the --video
  path and --model. The video file must be placed inside the `data/` directory, which sits
  next to the `thumbnail-selection/` repository directory.

Directory Structure Example: |
  project_root/
     ├── thumbnail-selection/
     └── data/
          └── goalclip.mp4

Note: |
  Create the data/ directory if you do not have it already.

Run Example: |
  bash run_inference_pipeline.sh --video "../data/goalclip.mp4"

Windows Example: |
  "C:/Program Files/Git/bin/bash.exe" run_inference_pipeline.sh --video "../data/goalclip.mp4"

Model Selection: |
  You can choose which model checkpoint to run via --model.
  Default model = resnet50

Model Examples:
  - bash run_inference_pipeline.sh --video "../data/goalclip.mp4" --model resnet50
  - bash run_inference_pipeline.sh --video "../data/goalclip.mp4" --model vit
  - bash run_inference_pipeline.sh --video "../data/goalclip.mp4" --model efficientnet
  - bash run_inference_pipeline.sh --video "../data/goalclip.mp4" --model convnext

Model File Requirement: |
  Model files must exist at:
    thumbnail-selection/checkpoints/<model>_best.pt

Pipeline Steps:
  1: Extract frames from the input video (default 10 FPS - adjustable inside run_inference_pipeline.sh)
  2: Run the selected model for shot-type classification and save predictions to predictions_<model>.csv
  3: Build priority-based segments (Close-up → Main → Public) and save to segments.csv
  4: Score and rank candidate frames, and export the top-ranked keyframes

Output Location: |
  ../data/inference_output/
    ├── frames/
    ├── predictions/
    ├── segments/
    └── keyframes/

Usage Summary: |
  bash run_inference_pipeline.sh --video "../data/<clipname>.mp4" --model resnet50

Optional Arguments:
  --game_name: "Folder label for grouping outputs"
  --clip: "Folder label for clip name"
  --model: "resnet50 | resnet18 | vit | efficientnet | convnext"

Defaults:
  game_name: "SomeGameName"
  clip: "SomeClipName"
  video: "../data/goalclip.mp4"
  model: "resnet50"
