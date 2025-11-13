#!/bin/bash

#  DEFAULT INPUTS
DEFAULT_GAME_NAME="2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
DEFAULT_CLIP_NAME="1_224p"
DEFAULT_VIDEO_PATH="../data/SoccerNet/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"

#  Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --game_name) GAME_NAME="$2"; shift ;;
        --clip) CLIP_NAME="$2"; shift ;;
        --video) VIDEO_PATH="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

GAME_NAME="${GAME_NAME:-$DEFAULT_GAME_NAME}"
CLIP_NAME="${CLIP_NAME:-$DEFAULT_CLIP_NAME}"
VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO_PATH}"

echo "[INFO] Using game: $GAME_NAME"
echo "[INFO] Using clip: $CLIP_NAME"
echo "[INFO] Using video: $VIDEO_PATH"

#  Paths
INFER_ROOT="../data/inference_output"
FRAMES_DIR="$INFER_ROOT/frames/${GAME_NAME}/${CLIP_NAME}"
PRED_DIR="$INFER_ROOT/predictions"
SEG_DIR="$INFER_ROOT/closeup_segments"
KEYFRAME_DIR="$INFER_ROOT/keyframes"

PRED_CSV="$PRED_DIR/predictions_resnet50.csv"
SEG_CSV="$SEG_DIR/closeup_segments.csv"
KEYFRAME_CSV="$KEYFRAME_DIR/keyframes.csv"

WEIGHTS_PATH="checkpoints/resnet50_best.pt"

mkdir -p "$FRAMES_DIR" "$PRED_DIR" "$SEG_DIR" "$KEYFRAME_DIR"

#  STEP 1 — Frames
echo "[STEP 1] Extracting frames..."
python src/utils/frame_extractor.py \
    --input_path "$VIDEO_PATH" \
    --output_dir "$FRAMES_DIR" \
    --fps 1

#  STEP 2 — Inference
echo "[STEP 2] Running ResNet50 inference..."
python src/inference/inference_model.py \
    --frames_root "$FRAMES_DIR" \
    --weights "$WEIGHTS_PATH" \
    --output_csv "$PRED_CSV"

#  STEP 3 — Close-up Segments
echo "[STEP 3] Extracting close-up segments..."
python src/inference/extract_closeup_frames.py \
    --pred_csv "$PRED_CSV" \
    --min_length 3 \
    --copy_dir "$SEG_DIR" \
    --output_csv "$SEG_CSV"

#  STEP 4 — Keyframe Selection
echo "[STEP 4] Selecting keyframes..."
python src/inference/keyframe_selector.py \
    --pred_csv "$PRED_CSV" \
    --seg_csv "$SEG_CSV" \
    --output_csv "$KEYFRAME_CSV" \
    --output_dir "$KEYFRAME_DIR"


echo "[DONE] Pipeline completed successfully."
