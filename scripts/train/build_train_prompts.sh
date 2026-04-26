#!/usr/bin/env bash
# Build training-prompt DatasetDicts from cleaned TAG .pt files using the SGC
# neighbour-aware hardness pipeline.
#
# Each source dataset is built into its own DatasetDict at
#   $SAVE_DIR/<dataset>_train_nei${NEIGHBORS}_prompts/
#
# Env vars:
#   TAG_ROOT          Directory holding <dataset>.pt files (required)
#   DATASETS          Space-separated aliases (default: citeseer history)
#   SAVE_DIR          Output directory (default: ./datasets/prompts)
#   NEIGHBORS         Number of 1-hop neighbours per prompt (default: 3)
#   AUGMENTATIONS     Variants per train node (default: 10, paper recipe)
#   SCORE_DEVICE      cuda | cpu (default: cuda)
#   SCORE_ENCODER     HF encoder for hardness scoring
#                     (default: sentence-transformers/all-MiniLM-L6-v2)
#   SCORE_SGC_LAYERS  SGC propagation layers (default: 1; 0 disables hardness)
#
# Example:
#   TAG_ROOT=./datasets/tags DATASETS="citeseer history" AUGMENTATIONS=10 \
#   bash scripts/train/build_train_prompts.sh

set -euo pipefail

: "${TAG_ROOT:?Set TAG_ROOT (directory containing <dataset>.pt)}"
DATASETS=${DATASETS:-"citeseer history"}
SAVE_DIR=${SAVE_DIR:-"./datasets/prompts"}
NEIGHBORS=${NEIGHBORS:-3}
AUGMENTATIONS=${AUGMENTATIONS:-10}
SCORE_DEVICE=${SCORE_DEVICE:-"cuda"}
SCORE_ENCODER=${SCORE_ENCODER:-"sentence-transformers/all-MiniLM-L6-v2"}
SCORE_SGC_LAYERS=${SCORE_SGC_LAYERS:-1}

mkdir -p "$SAVE_DIR"
# shellcheck disable=SC2206
ds_arr=( $DATASETS )

for ds in "${ds_arr[@]}"; do
  echo
  echo "==== build train prompts: $ds ===="
  python -m trn_r1_zero.prompts.build_training_dataset \
    --mode train \
    --datasets "$ds" \
    --tag-root "$TAG_ROOT" \
    --neighbors "$NEIGHBORS" \
    --augmentations "$AUGMENTATIONS" \
    --fix_k \
    --score-device "$SCORE_DEVICE" \
    --score-encoder "$SCORE_ENCODER" \
    --score-sgc-layers "$SCORE_SGC_LAYERS" \
    --save-dir "$SAVE_DIR"
done
