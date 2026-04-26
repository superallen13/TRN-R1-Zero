#!/usr/bin/env bash
# Convert HuggingFace prompt DatasetDicts (produced by build_training_dataset.py)
# into verl-friendly parquet files.
#
# Env vars:
#   TRAIN_DATASETS   Space-separated paths or HF repo ids whose 'train' split
#                    is merged into train.parquet
#   TEST_DATASETS    Space-separated whose 'test' split is merged into test.parquet
#   OUT_DIR          Output directory (default: ./verl_data/run)
#   SYSTEM_KEY       System-prompt key (default: simple)
#
# Example:
#   TRAIN_DATASETS="datasets/prompts/citeseer_train_nei3_prompts datasets/prompts/history_train_nei3_prompts" \
#   TEST_DATASETS="datasets/prompts/citeseer_eval_nei3_prompts datasets/prompts/cora_eval_nei3_prompts" \
#   OUT_DIR=verl_data/run \
#   bash scripts/train/prepare_data.sh

set -euo pipefail

TRAIN_DATASETS=${TRAIN_DATASETS:-""}
TEST_DATASETS=${TEST_DATASETS:-""}
OUT_DIR=${OUT_DIR:-"./verl_data/run"}
SYSTEM_KEY=${SYSTEM_KEY:-"simple"}

if [[ -z "$TRAIN_DATASETS" && -z "$TEST_DATASETS" ]]; then
  echo "Error: set TRAIN_DATASETS and/or TEST_DATASETS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

args=( -m trn_r1_zero.prompts.data_preprocessing --out_dir "$OUT_DIR" --system_prompt_key "$SYSTEM_KEY" )
if [[ -n "$TRAIN_DATASETS" ]]; then
  # shellcheck disable=SC2206
  td=( $TRAIN_DATASETS )
  args+=( --train_datasets "${td[@]}" )
fi
if [[ -n "$TEST_DATASETS" ]]; then
  # shellcheck disable=SC2206
  te=( $TEST_DATASETS )
  args+=( --test_datasets "${te[@]}" )
fi

python "${args[@]}"
