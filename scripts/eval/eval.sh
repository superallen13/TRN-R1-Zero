#!/usr/bin/env bash
# Inference: launch a vLLM server, then iterate the eval client over datasets.
#
# Env vars:
#   MODEL_NAME       HF model id or local path (default: Allen-UQ/trn-r1-zero-7b)
#   DATASETS         Space-separated local prompt-set paths (or HF DatasetDict ids).
#                    Built by scripts/eval/build_prompts.sh from datasets/tags/.
#                    Falls back to DATASET_NAME, then to eval.py default.
#   DATASET_SPLIT    Split name (default: all)
#   SYSTEM_KEY       System-prompt key (default: simple)
#   PORT             vLLM server port (default: 21000)
#   TENSOR_PARALLEL  vLLM TP size (default: 1)
#   CONCURRENCY      Async concurrency factor (default: 8)
#   VLLM_LOG         vLLM server log path (default: /tmp/vllm-$PORT.log)
#
# Example:
#   MODEL_NAME=Allen-UQ/trn-r1-zero-7b \
#   DATASETS="datasets/prompts/cora_eval_nei3_prompts datasets/prompts/wikics_eval_nei3_prompts" \
#   bash scripts/eval/eval.sh

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"Allen-UQ/trn-r1-zero-7b"}
if [[ -z "${DATASETS:-}" ]]; then
  DATASETS="${DATASET_NAME:-datasets/prompts/cora_eval_nei3_prompts}"
fi
DATASET_SPLIT=${DATASET_SPLIT:-"all"}
SYSTEM_KEY=${SYSTEM_KEY:-"simple"}
PORT=${PORT:-21000}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
CONCURRENCY=${CONCURRENCY:-8}
VLLM_LOG=${VLLM_LOG:-"/tmp/vllm-$PORT.log"}

echo "Launching vLLM server: $MODEL_NAME on port $PORT (tp=$TENSOR_PARALLEL)"
echo "  log: $VLLM_LOG"
vllm serve "$MODEL_NAME" --dtype bfloat16 --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL" >"$VLLM_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# eval.py polls /v1/models itself before sending requests.
# shellcheck disable=SC2206
ds_arr=( $DATASETS )
for ds in "${ds_arr[@]}"; do
  echo
  echo "==== eval: $ds ===="
  python -m trn_r1_zero.evaluate.eval \
    --num_gpus 1 \
    --concurrency_factor "$CONCURRENCY" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$ds" \
    --dataset_split "$DATASET_SPLIT" \
    --base_port "$PORT" \
    --system_prompt_key "$SYSTEM_KEY"
done
