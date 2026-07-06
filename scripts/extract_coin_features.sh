#!/bin/bash
################################################################################
# Extract All Vision-Language Features for COIN Dataset
################################################################################
#
# This script extracts features from multiple VL models for COIN videos:
#   - ResNet-152 (CNN baseline)
#   - CLIP ViT-B/32 (VL baseline)
#   - CLIP ViT-B/16 (finer patches) - BEST PERFORMER
#   - BLIP Base
#
# All features: 50 frames per video
#
# Usage:
#   bash extract_coin_features.sh
#
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
VIDEO_DIR="./annotations/videos"
OUTPUT_BASE_DIR="./extracted_features/coin"
SCRIPT_DIR="."

# Parameters
NUM_FRAMES=50
DEVICE="cuda"

# Batch sizes
BATCH_SIZE_RESNET=16
BATCH_SIZE_CLIP_B32=16
BATCH_SIZE_CLIP_B16=12
BATCH_SIZE_BLIP=8

# ============================================================================
# SETUP
# ============================================================================

echo "========================================================================"
echo "COIN FEATURE EXTRACTION - ALL MODELS"
echo "========================================================================"
echo ""
echo "Video directory: $VIDEO_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Number of frames: $NUM_FRAMES"
echo "Device: $DEVICE"
echo ""

# Count videos
VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" | wc -l)
echo "Total videos found: $VIDEO_COUNT"
echo ""

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# ============================================================================
# HELPER FUNCTION
# ============================================================================

extract_features() {
    local MODEL_NAME=$1
    local SCRIPT=$2
    local OUTPUT_DIR=$3
    local BATCH_SIZE=$4
    local EXTRA_ARGS=${5:-""}
    
    echo ""
    echo "========================================================================"
    echo "EXTRACTING: $MODEL_NAME"
    echo "========================================================================"
    echo "Output: $OUTPUT_DIR"
    echo "Batch size: $BATCH_SIZE"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run extraction
    python "$SCRIPT" \
        --video-dir "$VIDEO_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-frames "$NUM_FRAMES" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        $EXTRA_ARGS
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ $MODEL_NAME extraction completed successfully!"
        echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
        
        # Count extracted features
        local NUM_FEATURES=$(find "$OUTPUT_DIR" -name "*.npy" | wc -l)
        echo "Total features: $NUM_FEATURES"
    else
        echo ""
        echo "❌ $MODEL_NAME extraction FAILED with exit code $EXIT_CODE"
        echo "Check the error messages above."
    fi
    
    echo ""
}

# ============================================================================
# MAIN EXTRACTION PROCESS
# ============================================================================

START_TIME=$(date +%s)

echo "Starting feature extraction at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. ResNet-152 (CNN Baseline) - 2048-dim
extract_features \
    "ResNet-152" \
    "$SCRIPT_DIR/extract_resnet152_features.py" \
    "$OUTPUT_BASE_DIR/resnet152" \
    "$BATCH_SIZE_RESNET"

# 2. CLIP ViT-B/32 (VL Baseline) - 512-dim
extract_features \
    "CLIP ViT-B/32" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/clip_vit_b32" \
    "$BATCH_SIZE_CLIP_B32" \
    "--model clip-vit-b32"

# 3. CLIP ViT-B/16 (Best Performer) - 512-dim
extract_features \
    "CLIP ViT-B/16" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/clip_vit_b16" \
    "$BATCH_SIZE_CLIP_B16" \
    "--model clip-vit-b16"

# 4. BLIP Base - 768-dim
extract_features \
    "BLIP Base" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/blip_base" \
    "$BATCH_SIZE_BLIP" \
    "--model blip-base"

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "========================================================================"
echo "EXTRACTION COMPLETE!"
echo "========================================================================"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Feature counts:"
for dir in "$OUTPUT_BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        NUM_FILES=$(find "$dir" -name "*.npy" | wc -l)
        DIR_NAME=$(basename "$dir")
        printf "  %-20s : %4d files\n" "$DIR_NAME" "$NUM_FILES"
    fi
done

echo ""
echo "Expected: $VIDEO_COUNT features per model"
echo ""
echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Generate COIN summaries:"
echo "   python generate_coin_summaries.py"
echo ""
echo "2. Train models on COIN:"
echo "   bash train_coin_models.sh"
echo ""
echo "3. Compare with YouCook2 results!"
echo ""
echo "========================================================================"