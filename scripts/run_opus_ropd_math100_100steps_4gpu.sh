#!/usr/bin/env bash
# Run a 4-GPU ROPD experiment on the 100-sample Opus teacher-index set.
#
# This script intentionally wraps run_opus_ropd_math10_20steps.sh so logging,
# summary generation, checkpoint layout, and teacher-index filtering stay in
# one place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EXPERIMENT="${EXPERIMENT:-ropd_opus_math100_100steps_b8_4gpu}"
export TRAIN_SAMPLES="${TRAIN_SAMPLES:-100}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export TRAINING_STEPS="${TRAINING_STEPS:-100}"
export SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-10}"
export STUDENT_ROLLOUT_N="${STUDENT_ROLLOUT_N:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

exec bash "$SCRIPT_DIR/run_opus_ropd_math10_20steps.sh" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    "$@"
