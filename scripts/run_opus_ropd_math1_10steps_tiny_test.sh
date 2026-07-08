#!/usr/bin/env bash
# Tiny smoke test for the Opus-backed ROPD path.
# Uses one DAPO-Math prompt, 10 training steps, local Qwen3-4B student,
# offline Opus teacher answers, and Opus rubricator/verifier.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-ropd_opus_math1_10steps_tiny_test}"
export TRAIN_SAMPLES="${TRAIN_SAMPLES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export TRAINING_STEPS="${TRAINING_STEPS:-10}"
export SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-5}"
export STUDENT_ROLLOUT_N="${STUDENT_ROLLOUT_N:-4}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export WANDB_MODE="${WANDB_MODE:-offline}"

exec bash "$SCRIPT_DIR/run_opus_ropd_math10_10steps.sh"     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1     "$@"
