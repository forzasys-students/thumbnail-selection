#!/bin/bash

#  DEFAULT INPUTS (override with arguments)
DEFAULT_GAME_NAME="2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
DEFAULT_CLIP_NAME="1_224p"
DEFAULT_VIDEO_PATH="../data/SoccerNet/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"



#  Argument Parsing
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


#  DIRECTORY SETUP (data folder is OUTSIDE)
INFER_ROOT="../data/inference_output"
FRAMES_DIR="$INFER_ROOT/frames/${GAME_NAME}/${CLIP_NAME}"
PRED_DIR="$INFER_ROOT/predictions"
CLOSEUP_DIR="$INFER_ROOT/closeup_segments"

PREDICTIONS_CSV="$PRED_DIR/predictions_resnet50.csv"
CLOSEUP_SEGMENTS_CSV="$CLOSEUP_DIR/closeup_segments.csv"
WEIGHTS_PATH="checkpoints/resnet50_best.pt"

mkdir -p "$FRAMES_DIR"
mkdir -p "$PRED_DIR"
mkdir -p "$CLOSEUP_DIR"


#  STEP 1 — Frame Extraction
echo "[STEP 1] Extracting frames..."
python src/utils/frame_extractor.py \
    --input_path "$VIDEO_PATH" \
    --output_dir "$FRAMES_DIR" \
    --fps 1


#  STEP 2 — ResNet50 Inference
echo "[STEP 2] Running ResNet50 inference..."
python src/inference/inference_model.py \
    --frames_root "$FRAMES_DIR" \
    --weights "$WEIGHTS_PATH" \
    --output_csv "$PREDICTIONS_CSV"


#  STEP 3 — Close-up Segment Extraction
echo "[STEP 3] Extracting close-up segments..."
python src/inference/extract_closeup_frames.py \
    --pred_csv "$PREDICTIONS_CSV" \
    --min_length 3 \
    --copy_dir "$CLOSEUP_DIR" \
    --output_csv "$CLOSEUP_SEGMENTS_CSV"


echo "[DONE] Pipeline completed successfully."
