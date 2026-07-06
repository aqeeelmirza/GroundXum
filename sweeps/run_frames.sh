#!/usr/bin/env bash
# SWEEP 3 (driver) - Keyframe-count sensitivity.   [NEEDS GPU]
#
# Re-runs grounding/ground.py at several frame counts. Changing --frames changes
# WHICH frames are sampled, so this cannot be done from cache - it is the one
# sweep that needs fresh BLIP/CLIP inference. Outputs go to a frames/ subtree
# that sweep_frames.py then analyzes.
#
# Edit the paths in the CONFIG block, then:  bash sweeps/run_frames.sh
set -euo pipefail

# ---------------- CONFIG (edit these) ----------------
DATASET="youcook2"                                  # youcook2 | videoxum
SCORER="blip"                                       # blip | clip
ID_SOURCE="data/youcook2/test/tran.tok.txt"         # first token per line = video id
VIDEO_DIR="data/videos/youcook2"
FRAMES_LIST="1 2 4 8 16 32"                          # 16 is the paper default
# label -> KG file (the extracted entities, reused across all frame counts)
declare -A KG=(
  ["S1"]="outputs/kg/${DATASET}/mm_blip.jsonl"
  ["S2"]="outputs/kg/${DATASET}/blip_base.jsonl"
)
OUT_ROOT="outputs/grounding/${DATASET}/frames"
# -----------------------------------------------------

mkdir -p "${OUT_ROOT}"
for label in "${!KG[@]}"; do
  kg="${KG[$label]}"
  for F in ${FRAMES_LIST}; do
    out="${OUT_ROOT}/${label}_${SCORER}_f${F}.jsonl"
    if [[ -f "${out}" ]]; then
      echo "SKIP ${out} (exists)"; continue
    fi
    echo "=== ${label} | frames=${F} | scorer=${SCORER} ==="
    python grounding/ground.py \
      --kg "${kg}" \
      --dataset "${DATASET}" \
      --id-source "${ID_SOURCE}" \
      --video-dir "${VIDEO_DIR}" \
      --scorer "${SCORER}" \
      --frames "${F}" \
      --output "${out}"
  done
done
echo "Done. Now run: python sweeps/sweep_frames.py --root ${OUT_ROOT} --scorer ${SCORER} ..."