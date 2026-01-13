#!/bin/bash

# Set UTF-8 encoding for Python to handle Unicode characters
export PYTHONIOENCODING=utf-8

#  DEFAULT INPUTS
DEFAULT_GAME_NAME="SomeGameName"
DEFAULT_CLIP_NAME="SomeName"
DEFAULT_VIDEO_PATH="../data/goalclip.mp4"
DEFAULT_MODEL="resnet18"

#  Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --game_name) GAME_NAME="$2"; shift ;;
        --clip) CLIP_NAME="$2"; shift ;;
        --video) VIDEO_PATH="$2"; shift ;;
        --model) MODEL_NAME="$2"; shift ;;   
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

GAME_NAME="${GAME_NAME:-$DEFAULT_GAME_NAME}"
CLIP_NAME="${CLIP_NAME:-$DEFAULT_CLIP_NAME}"
VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO_PATH}"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"   

echo "[INFO] Using game: $GAME_NAME"
echo "[INFO] Using clip: $CLIP_NAME"
echo "[INFO] Using video: $VIDEO_PATH"
echo "[INFO] Using model: $MODEL_NAME"       

#  Paths
INFER_ROOT="data/inference_output"
FRAMES_DIR="$INFER_ROOT/frames/${GAME_NAME}/${CLIP_NAME}"
PRED_DIR="$INFER_ROOT/predictions"
SEG_DIR="$INFER_ROOT/segments"
KEYFRAME_DIR="$INFER_ROOT/keyframes"

PRED_CSV="$PRED_DIR/predictions_${MODEL_NAME}.csv"    
SEG_CSV="$SEG_DIR/segments.csv"
KEYFRAME_CSV="$KEYFRAME_DIR/keyframes.csv"

WEIGHTS_PATH="models/${MODEL_NAME}_best.pt"       

mkdir -p "$FRAMES_DIR" "$PRED_DIR" "$SEG_DIR" "$KEYFRAME_DIR"

#  STEP 1 – Frames
echo "[STEP 1] Extracting frames..."
python src/utils/frame_extractor.py \
    --input_path "$VIDEO_PATH" \
    --output_dir "$FRAMES_DIR" \
    --fps 5 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] Frame extraction failed"
    exit 1
fi

#  STEP 2 – Inference
echo "[STEP 2] Running inference ($MODEL_NAME)..."
python src/inference/inference_model.py \
    --frames_root "$FRAMES_DIR" \
    --weights "$WEIGHTS_PATH" \
    --model "$MODEL_NAME" \
    --output_csv "$PRED_CSV" 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed"
    exit 1
fi

#  STEP 3 – Extracting Priority based Segments
echo "[STEP 3] Extracting segments..."
python src/inference/extract_priority_segments.py \
    --pred_csv "$PRED_CSV" \
    --min_length 3 \
    --min_segments_per_clip 20 \
    --copy_dir "$SEG_DIR" \
    --output_csv "$SEG_CSV" 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] Segment extraction failed"
    exit 1
fi

#  STEP 4 – Keyframe Selection
echo "[STEP 4] Selecting keyframes..."
python src/inference/keyframe_selector.py \
    --pred_csv "$PRED_CSV" \
    --seg_csv "$SEG_CSV" \
    --output_csv "$KEYFRAME_CSV" \
    --output_dir "$KEYFRAME_DIR" 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] Keyframe selection failed"
    exit 1
fi

echo "[DONE] Pipeline completed successfully."