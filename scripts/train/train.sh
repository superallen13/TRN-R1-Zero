#!/usr/bin/env bash
# GRPO training entry point. Wraps verl.trainer.main_ppo with sensible defaults.
#
# Required (run scripts/train/prepare_data.sh first):
#   $VERL_OUT_DIR/train.parquet
#   $VERL_OUT_DIR/test.parquet
#
# Env vars:
#   MODEL_NAME       HF model id or local path (default: Qwen/Qwen2.5-7B-Instruct)
#   REWARD           "margin" (paper headline, default) | "base"
#   VERL_OUT_DIR     Where train/test.parquet live (default: ./verl_data/run)
#   NUM_GPUS         GPUs to use (default: from CUDA_VISIBLE_DEVICES, else 1)
#   PROJECT_NAME     wandb project name (default: trn-r1-exp)
#   EXPERIMENT_NAME  wandb run name (default: derived from MODEL_NAME + REWARD)
#   TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE / PPO_MICRO_BATCH_SIZE /
#   LOGPROB_MICRO_BATCH_SIZE / ROLLOUT_N / TOTAL_EPOCHS / LR /
#   SAVE_FREQ / TEST_FREQ / GPU_MEM_UTIL — see defaults below
#
# Example:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct REWARD=margin NUM_GPUS=4 \
#   bash scripts/train/train.sh

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
REWARD=${REWARD:-"margin"}
VERL_OUT_DIR=${VERL_OUT_DIR:-"./verl_data/run"}

if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_GPUS=1
  fi
fi

case "$REWARD" in
  margin) REWARD_PATH="trn_r1_zero/reward/nc_margin.py" ;;
  base)   REWARD_PATH="trn_r1_zero/reward/nc.py" ;;
  *) echo "Unknown REWARD=$REWARD; expected 'margin' or 'base'" >&2; exit 1 ;;
esac

PROJECT_NAME=${PROJECT_NAME:-"trn-r1-exp"}
short_model=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/^_//;s/_$//')
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${short_model}_grpo_${REWARD}"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-16}
LOGPROB_MICRO_BATCH_SIZE=${LOGPROB_MICRO_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}

train_files="$VERL_OUT_DIR/train.parquet"
val_files="$VERL_OUT_DIR/test.parquet"

if [[ ! -f "$train_files" || ! -f "$val_files" ]]; then
  echo "Missing $train_files or $val_files. Run scripts/train/prepare_data.sh first." >&2
  exit 1
fi

echo "MODEL=$MODEL_NAME REWARD=$REWARD ($REWARD_PATH)"
echo "DATA=$VERL_OUT_DIR  GPUS=$NUM_GPUS  EXP=$EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=16 \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_NAME" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.strategy="fsdp" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE" \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$REWARD_PATH" \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS"
