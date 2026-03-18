#!/bin/bash

export PYTHONIOENCODING=utf-8

DEFAULT_VIDEO_PATH="../goalclip.mp4"
DEFAULT_MODEL="resnet18"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video)         VIDEO_PATH="$2";  shift ;;
        --model)         MODEL_NAME="$2";  shift ;;
        --video_id)      VIDEO_ID="$2";    shift ;;
        --fps)           FPS="$2";         shift ;;
        --no_redundancy) NO_REDUNDANCY=1         ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO_PATH}"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
FPS="${FPS:-5}"
REDUNDANCY_ARG="true"
if [ -n "$NO_REDUNDANCY" ]; then
    REDUNDANCY_ARG="false"
fi

if [ -z "$VIDEO_ID" ]; then
    VIDEO_ID=$(basename "$VIDEO_PATH" | grep -oE 'video_[0-9]+' | grep -oE '[0-9]+' | head -1)
fi
if [ -z "$VIDEO_ID" ]; then
    VIDEO_ID=$(echo "$VIDEO_PATH" | grep -oE '/[0-9]+:[0-9]+:[0-9]+/' | grep -oE '^/[0-9]+' | tr -d '/' | head -1)
fi
VIDEO_ID="${VIDEO_ID:-unknown}"

echo "[INFO] Using video:    $VIDEO_PATH"
echo "[INFO] Using model:    $MODEL_NAME"
echo "[INFO] Using video_id: $VIDEO_ID"
echo "[INFO] Using fps:      $FPS"
echo "[INFO] Redundancy reduction: $REDUNDANCY_ARG"

INFER_ROOT="data/inference_output"
FRAMES_DIR="$INFER_ROOT/frames"
PRED_DIR="$INFER_ROOT/predictions"
SEG_DIR="$INFER_ROOT/segments"
KEYFRAME_DIR="$INFER_ROOT/keyframes"

PRED_CSV="$PRED_DIR/predictions_${MODEL_NAME}.csv"
SEG_CSV="$SEG_DIR/segments.csv"
KEYFRAME_CSV="$KEYFRAME_DIR/keyframes.csv"
WEIGHTS_PATH="models/${MODEL_NAME}_best.pt"

mkdir -p "$FRAMES_DIR" "$PRED_DIR" "$SEG_DIR" "$KEYFRAME_DIR"

if [ -d "$FRAMES_DIR" ]; then
    echo "[INFO] Clearing previous frames from $FRAMES_DIR"
    rm -rf "$FRAMES_DIR"
    mkdir -p "$FRAMES_DIR"
fi

echo "[STEP 1] Extracting frames..."
python src/utils/frame_extractor.py \
    --input_path "$VIDEO_PATH" \
    --output_dir "$FRAMES_DIR" \
    --fps "$FPS" \
    --video_id "$VIDEO_ID" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Frame extraction failed"; exit 1; fi

echo "[STEP 2] Running inference ($MODEL_NAME)..."
python src/inference/inference_model.py \
    --frames_root "$FRAMES_DIR" \
    --weights "$WEIGHTS_PATH" \
    --model "$MODEL_NAME" \
    --output_csv "$PRED_CSV" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Inference failed"; exit 1; fi

echo "[STEP 3] Extracting segments..."
python src/inference/extract_priority_segments.py \
    --pred_csv "$PRED_CSV" \
    --min_length 5 \
    --min_segments_per_clip 6 \
    --copy_dir "$SEG_DIR" \
    --output_csv "$SEG_CSV" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Segment extraction failed"; exit 1; fi

echo "[STEP 4] Selecting keyframes..."
python src/inference/keyframe_selector.py \
    --pred_csv "$PRED_CSV" \
    --seg_csv "$SEG_CSV" \
    --output_csv "$KEYFRAME_CSV" \
    --output_dir "$KEYFRAME_DIR" \
    --video_id "$VIDEO_ID" \
    --fps "$FPS" \
    --redundancy_reduction "$REDUNDANCY_ARG" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Keyframe selection failed"; exit 1; fi

echo "[DONE] Pipeline completed successfully."