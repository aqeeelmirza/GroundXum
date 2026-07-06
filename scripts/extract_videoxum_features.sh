#!/bin/bash
################################################################################
# Extract All Vision-Language Features for VideoXum Dataset
################################################################################
set -e

VIDEO_DIR="./dataset/videoxum/anet_videos/Anet_videos_15fps_short256"
OUTPUT_BASE_DIR="./dataset/videoxum/extracted_features"
SCRIPT_DIR="."
NUM_FRAMES=50
DEVICE="cuda"
BATCH_SIZE_RESNET=16
BATCH_SIZE_CLIP_B32=16
BATCH_SIZE_CLIP_B16=12

echo "========================================================================"
echo "VideoXum FEATURE EXTRACTION - ALL MODELS"
echo "========================================================================"
VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
echo "Total videos found: $VIDEO_COUNT"
mkdir -p "$OUTPUT_BASE_DIR"

extract_features() {
    local MODEL_NAME=$1
    local SCRIPT=$2
    local OUTPUT_DIR=$3
    local BATCH_SIZE=$4
    local EXTRA_ARGS=${5:-""}
    echo "======== EXTRACTING: $MODEL_NAME | $(date '+%Y-%m-%d %H:%M:%S') ========"
    mkdir -p "$OUTPUT_DIR"
    python "$SCRIPT" \
        --video-dir "$VIDEO_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-frames "$NUM_FRAMES" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        $EXTRA_ARGS
    echo "✅ $MODEL_NAME done: $(find $OUTPUT_DIR -name '*.npy' | wc -l) files"
}

START=$(date +%s)

# ResNet-152
extract_features "ResNet-152" \
    "$SCRIPT_DIR/extract_resnet152_features.py" \
    "$OUTPUT_BASE_DIR/resnet152" \
    "$BATCH_SIZE_RESNET"

# CLIP ViT-B/32
extract_features "CLIP ViT-B/32" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/clip_vit_b32" \
    "$BATCH_SIZE_CLIP_B32" \
    "--model clip-vit-b32"

# CLIP ViT-B/16
extract_features "CLIP ViT-B/16" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/clip_vit_b16" \
    "$BATCH_SIZE_CLIP_B16" \
    "--model clip-vit-b16"

END=$(date +%s)
echo "Done in $(( (END-START)/3600 ))h $(( ((END-START)%3600)/60 ))m"
echo "Feature counts:"
for dir in "$OUTPUT_BASE_DIR"/*/; do
    echo "  $(basename $dir): $(find $dir -name '*.npy' | wc -l) files"
done
