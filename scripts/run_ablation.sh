#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

TRAIN_SRC="./dataset/youcook2_summarization/sum_train/tran.tok.txt"
TRAIN_TGT="./dataset/youcook2_summarization/sum_train/desc.tok.txt"
VAL_SRC="./dataset/youcook2_summarization/sum_cv/tran.tok.txt"
VAL_TGT="./dataset/youcook2_summarization/sum_cv/desc.tok.txt"
TEST_SRC="./dataset/youcook2_summarization/sum_cv/tran.tok.txt"
TEST_TGT="./dataset/youcook2_summarization/sum_cv/desc.tok.txt"
FEAT="./extracted_features/blip_base"

BATCH=16
LR=1e-5
EPOCHS=30
GRAD_ACC=4
DIM=256
VISUAL_DIM=768

train_ablation() {
    local LOG_NAME=$1
    local ATTN_TYPE=$2
    local FUSION_LAYER=$3
    local HEADS_=$4
    local FRAMES_=$5
    local MAX_IMG_=$6

    local METRICS="./evaluation1/results/${LOG_NAME}_youcook2_test_metrics.json"
    if [ -f "$METRICS" ] && [ -s "$METRICS" ]; then
        echo "✅ SKIP $LOG_NAME: already done"
        return 0
    fi

    echo "========================================================"
    echo "ABLATION: $LOG_NAME | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  type=$ATTN_TYPE  layer=$FUSION_LAYER  heads=$HEADS_  frames=$FRAMES_"
    echo "========================================================"

    python src/run.py \
        -model multi_modal_bart \
        -train_src_path "$TRAIN_SRC" \
        -train_tgt_path "$TRAIN_TGT" \
        -val_src_path   "$VAL_SRC" \
        -val_tgt_path   "$VAL_TGT" \
        -test_src_path  "$TEST_SRC" \
        -test_tgt_path  "$TEST_TGT" \
        -image_feature_path "${FEAT}/" \
        -visual_hidden_size $VISUAL_DIM \
        -fusion_layer $FUSION_LAYER \
        -cross_attn_type $ATTN_TYPE \
        -dim_common $DIM \
        -n_attn_heads $HEADS_ \
        -batch_size $BATCH \
        -learning_rate $LR \
        -num_epochs $EPOCHS \
        -num_frames $FRAMES_ \
        -max_input_len 512 \
        -max_output_len 128 \
        -max_img_len $MAX_IMG_ \
        -n_beams 5 \
        -no_repeat_ngram_size 3 \
        -do_train True \
        -do_test True \
        -checkpoint None \
        -auto_resume True \
        -skip_if_trained True \
        -log_name "$LOG_NAME" \
        -gpus 1 \
        -grad_accumulate $GRAD_ACC \
        -val_save_file  "./evaluation1/${LOG_NAME}_valid" \
        -test_save_file "./evaluation1/results/${LOG_NAME}_youcook2_test.txt"

    [ $? -eq 0 ] && echo "✅ $LOG_NAME done!" || echo "❌ $LOG_NAME FAILED!"
}

echo "========================================================"
echo "ABLATION STUDIES"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ============================================================
# 1. FUSION LAYER ABLATION
#    Fixed: type=6, heads=4, frames=50
#    Layer 5 already done (49.31)
# ============================================================
echo ""
echo "--- Fusion Layer Ablation ---"
train_ablation "ablat_layer1_youcook2"  6  1  4  50  50
train_ablation "ablat_layer2_youcook2"  6  2  4  50  50
train_ablation "ablat_layer3_youcook2"  6  3  4  50  50
train_ablation "ablat_layer4_youcook2"  6  4  4  50  50
train_ablation "ablat_layer6_youcook2"  6  6  4  50  50

# ============================================================
# 2. FUSION TYPE ABLATION
#    Fixed: layer=5, heads=4, frames=50
#    type0 done (38.97), type5 done (22.49), type6 done (49.31)
# ============================================================
echo ""
echo "--- Fusion Type Ablation ---"
train_ablation "ablat_type1_youcook2"  1  5  4  50  50
train_ablation "ablat_type2_youcook2"  2  5  4  50  50
train_ablation "ablat_type3_youcook2"  3  5  4  50  50
train_ablation "ablat_type4_youcook2"  4  5  4  50  50

# ============================================================
# 3. FRAME SAMPLING ABLATION
#    Fixed: type=6, layer=5, heads=4
#    frames=50 already done (49.31)
# ============================================================
echo ""
echo "--- Frame Sampling Ablation ---"
train_ablation "ablat_frames10_youcook2"   6  5  4  10   10
train_ablation "ablat_frames25_youcook2"   6  5  4  25   25
train_ablation "ablat_frames100_youcook2"  6  5  4  100  100

# ============================================================
# 4. HEAD ABLATION — ALL ALREADY DONE
#    H=1 (29.52), H=2 (23.31), H=4 (49.31), H=8 (41.53)
#    MHA no gating (22.49)
# ============================================================
echo ""
echo "--- Head Ablation: all already done ---"

echo ""
echo "========================================================"
echo "ALL ABLATIONS DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total new runs: 12 (5 layer + 4 type + 3 frame)"
echo "========================================================"
