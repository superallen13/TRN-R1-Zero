#!/usr/bin/env bash
# Build eval-prompt DatasetDicts from cleaned TAG .pt files using the SGC
# neighbour-aware hardness pipeline.
#
# Each source dataset is built into its own DatasetDict at
#   $SAVE_DIR/<dataset>_eval_nei${NEIGHBORS}_prompts/
#
# Env vars:
#   TAG_ROOT       Directory holding <dataset>.pt files (required)
#   DATASETS       Space-separated aliases (default: citeseer cora wikics instagram photo)
#   SAVE_DIR       Output directory (default: ./datasets/prompts)
#   NEIGHBORS      Number of 1-hop neighbours per prompt (default: 3)
#   SCORE_DEVICE   cuda | cpu (default: cuda)
#   SCORE_ENCODER  HF encoder for hardness scoring
#                  (default: sentence-transformers/all-MiniLM-L6-v2)
#
# Example:
#   TAG_ROOT=./datasets/tags DATASETS="cora wikics" \
#   bash scripts/eval/build_prompts.sh

set -euo pipefail

: "${TAG_ROOT:?Set TAG_ROOT (directory containing <dataset>.pt)}"
DATASETS=${DATASETS:-"citeseer cora wikics instagram photo"}
SAVE_DIR=${SAVE_DIR:-"./datasets/prompts"}
NEIGHBORS=${NEIGHBORS:-3}
SCORE_DEVICE=${SCORE_DEVICE:-"cuda"}
SCORE_ENCODER=${SCORE_ENCODER:-"sentence-transformers/all-MiniLM-L6-v2"}

mkdir -p "$SAVE_DIR"
# shellcheck disable=SC2206
ds_arr=( $DATASETS )

for ds in "${ds_arr[@]}"; do
  echo
  echo "==== build eval prompts: $ds ===="
  python -m trn_r1_zero.prompts.build_training_dataset \
    --mode eval \
    --datasets "$ds" \
    --tag-root "$TAG_ROOT" \
    --neighbors "$NEIGHBORS" \
    --fix_k \
    --score-device "$SCORE_DEVICE" \
    --score-encoder "$SCORE_ENCODER" \
    --save-dir "$SAVE_DIR"
done
