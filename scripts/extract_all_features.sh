#!/bin/bash
################################################################################
# Extract All Vision-Language Features for YouCook2
################################################################################
#
# This script extracts features from multiple VL models:
#   - ResNet-152 (CNN baseline)
#   - CLIP ViT-B/32 (VL baseline)
#   - CLIP ViT-B/16 (finer patches)
#   - BLIP Base
#   - BLIP-2 OPT-2.7B (state-of-the-art)
#   - SigLIP Base (Google's improved CLIP)
#
# All features: 50 frames per video
#
# Usage:
#   bash extract_all_features.sh
#
# Requirements:
#   - Python 3.8+
#   - PyTorch with CUDA
#   - transformers, opencv-python, pillow, tqdm, numpy
#
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (MODIFY THESE FOR YOUR SETUP)
VIDEO_DIR="./youcook2_videos/raw_videos"
OUTPUT_BASE_DIR="./extracted_features"
SCRIPT_DIR="."  # Directory containing the extraction scripts

# Parameters
NUM_FRAMES=50
DEVICE="cuda"

# Batch sizes (adjust based on your GPU memory)
BATCH_SIZE_RESNET=16
BATCH_SIZE_CLIP_B32=16
BATCH_SIZE_CLIP_B16=12
BATCH_SIZE_BLIP=8
BATCH_SIZE_BLIP2=4
BATCH_SIZE_SIGLIP=8

# ============================================================================
# SETUP
# ============================================================================

echo "========================================================================"
echo "YOUCOOK2 FEATURE EXTRACTION - ALL MODELS"
echo "========================================================================"
echo ""
echo "Video directory: $VIDEO_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Number of frames: $NUM_FRAMES"
echo "Device: $DEVICE"
echo ""

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "ERROR: Video directory not found: $VIDEO_DIR"
    echo "Please update VIDEO_DIR in this script."
    exit 1
fi

# Check if extraction scripts exist
if [ ! -f "$SCRIPT_DIR/extract_vl_features.py" ]; then
    echo "ERROR: extract_vl_features.py not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/extract_resnet152_features.py" ]; then
    echo "ERROR: extract_resnet152_features.py not found in $SCRIPT_DIR"
    exit 1
fi

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
        local NUM_FEATURES=$(ls "$OUTPUT_DIR"/*.npy 2>/dev/null | wc -l)
        echo "Total features: $NUM_FEATURES"
    else
        echo ""
        echo "❌ $MODEL_NAME extraction FAILED with exit code $EXIT_CODE"
        echo "Check the error messages above."
        # Don't exit, continue with other models
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

# 2. CLIP ViT-B/32 (Main VL Baseline) - 512-dim
extract_features \
    "CLIP ViT-B/32" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/clip_vit_b32" \
    "$BATCH_SIZE_CLIP_B32" \
    "--model clip-vit-b32"

# 3. CLIP ViT-B/16 (Finer Patches) - 512-dim
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

# 5. BLIP-2 OPT-2.7B (State-of-the-Art) - 768-dim
extract_features \
    "BLIP-2 OPT-2.7B" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/blip2_opt_2.7b" \
    "$BATCH_SIZE_BLIP2" \
    "--model blip2-opt-2.7b"

# 6. SigLIP Base (Google's Improved CLIP) - 768-dim
extract_features \
    "SigLIP Base" \
    "$SCRIPT_DIR/extract_vl_features.py" \
    "$OUTPUT_BASE_DIR/siglip_base" \
    "$BATCH_SIZE_SIGLIP" \
    "--model siglip-base"

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "========================================================================"
echo "EXTRACTION COMPLETE!"
echo "========================================================================"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Extracted features saved to:"
echo "  $OUTPUT_BASE_DIR/"
echo ""
echo "Directory structure:"
tree -L 1 "$OUTPUT_BASE_DIR" 2>/dev/null || ls -lh "$OUTPUT_BASE_DIR"
echo ""

# Count features in each directory
echo "Feature counts:"
for dir in "$OUTPUT_BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        NUM_FILES=$(ls "$dir"/*.npy 2>/dev/null | wc -l)
        DIR_NAME=$(basename "$dir")
        printf "  %-20s : %4d files\n" "$DIR_NAME" "$NUM_FILES"
    fi
done

echo ""
echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Verify extraction:"
echo "   python verify_extracted_features.py --base-dir $OUTPUT_BASE_DIR"
echo ""
echo "2. Train models with different features:"
echo "   bash train_all_models.sh"
echo ""
echo "3. Compare results for your journal paper!"
echo ""
echo "========================================================================"