#!/bin/bash

export PYTHONIOENCODING=utf-8

# ── Pre-download all required models on first run, then go offline ────────────
echo "[INFO] Checking model cache..."
python - <<'EOF'
import sys
try:
    # This triggers the timm/HuggingFace download if not cached
    import timm
    timm.create_model('resnet50.a1_in1k', pretrained=True)
    print("[INFO] Model cache OK.")
except Exception as e:
    print(f"[WARN] Model pre-check failed: {e}", file=sys.stderr)
EOF

# Now that models are cached, block all further network calls to HuggingFace.
# This prevents the 5-retry delay on every run.
export HF_HUB_OFFLINE=1
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_START=$SECONDS

DEFAULT_VIDEO_PATH="../goalclip.mp4"
DEFAULT_MODEL="resnet18"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video)             VIDEO_PATH="$2";         shift ;;
        --model)             MODEL_NAME="$2";         shift ;;
        --video_id)          VIDEO_ID="$2";           shift ;;
        --fps)               FPS="$2";                shift ;;
        --visual_threshold)  VISUAL_THRESHOLD="$2";   shift ;;
        --logo_p)            LOGO_P="$2";             shift ;;
        --w_face)            W_FACE="$2";             shift ;;
        --w_emotion)         W_EMOTION="$2";          shift ;;
        --w_pose)            W_POSE="$2";             shift ;;
        --w_iqa)             W_IQA="$2";              shift ;;
        --no_redundancy)     NO_REDUNDANCY=1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done


VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO_PATH}"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
FPS="${FPS:-12}"
VISUAL_THRESHOLD="${VISUAL_THRESHOLD:-0.90}"
LOGO_P="${LOGO_P:-0.50}"
W_FACE="${W_FACE:-0.25}"
W_EMOTION="${W_EMOTION:-0.15}"
W_POSE="${W_POSE:-0.15}"
W_IQA="${W_IQA:-0.35}"

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
echo "[INFO] Using visual_threshold: $VISUAL_THRESHOLD"
echo "[INFO] Using logo_p:           $LOGO_P"
echo "[INFO] Using w_face:           $W_FACE"
echo "[INFO] Using w_emotion:        $W_EMOTION"
echo "[INFO] Using w_pose:           $W_POSE"
echo "[INFO] Using w_iqa:            $W_IQA"


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

# =============================================================================
# STEP 1 — Frame extraction
# =============================================================================
echo "[STEP 1] Extracting frames..."
T1=$SECONDS
python src/utils/frame_extractor.py \
    --input_path "$VIDEO_PATH" \
    --output_dir "$FRAMES_DIR" \
    --fps "$FPS" \
    --video_id "$VIDEO_ID" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Frame extraction failed"; exit 1; fi
echo "[STEP 1] Done in $((SECONDS - T1))s"

# =============================================================================
# STEP 2 — Model inference
# =============================================================================
echo "[STEP 2] Running inference ($MODEL_NAME)..."
T2=$SECONDS
python src/inference/inference_model.py \
    --frames_root "$FRAMES_DIR" \
    --weights "$WEIGHTS_PATH" \
    --model "$MODEL_NAME" \
    --output_csv "$PRED_CSV" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Inference failed"; exit 1; fi
echo "[STEP 2] Done in $((SECONDS - T2))s"

# =============================================================================
# STEP 3 — Segment extraction
# =============================================================================
echo "[STEP 3] Extracting segments..."
T3=$SECONDS
python src/inference/extract_priority_segments.py \
    --pred_csv "$PRED_CSV" \
    --min_length 5 \
    --min_segments_per_clip 6 \
    --copy_dir "$SEG_DIR" \
    --output_csv "$SEG_CSV" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Segment extraction failed"; exit 1; fi
echo "[STEP 3] Done in $((SECONDS - T3))s"

# =============================================================================
# STEP 4 — Keyframe selection
# =============================================================================
echo "[STEP 4] Selecting keyframes..."
T4=$SECONDS
python src/inference/keyframe_selector.py \
    --pred_csv "$PRED_CSV" \
    --seg_csv "$SEG_CSV" \
    --output_csv "$KEYFRAME_CSV" \
    --output_dir "$KEYFRAME_DIR" \
    --video_id "$VIDEO_ID" \
    --redundancy_reduction "$REDUNDANCY_ARG" \
    --visual_threshold "$VISUAL_THRESHOLD" \
    --logo_threshold "$LOGO_P" \
    --w_face "$W_FACE" \
    --w_emotion "$W_EMOTION" \
    --w_pose "$W_POSE" \
    --w_iqa "$W_IQA" 2>&1

if [ $? -ne 0 ]; then echo "[ERROR] Keyframe selection failed"; exit 1; fi
echo "[STEP 4] Done in $((SECONDS - T4))s"

# =============================================================================
# TIMING SUMMARY
# =============================================================================
echo ""
echo "================================================"
echo "PIPELINE TIMING SUMMARY"
echo "================================================"
echo "  Step 1 - Frame extraction:    $((T2 - T1))s"
echo "  Step 2 - Model inference:     $((T3 - T2))s"
echo "  Step 3 - Segment extraction:  $((T4 - T3))s"
echo "  Step 4 - Keyframe selection:  $((SECONDS - T4))s"
echo "  Total:                        $((SECONDS - PIPELINE_START))s"
echo "================================================"

echo "[DONE] Pipeline completed successfully."