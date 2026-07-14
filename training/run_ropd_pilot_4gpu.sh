#!/usr/bin/env bash
# Four-GPU ROPD pilot run for observing early learning signals.
#
# This is a parameterized launcher over the normal training/train.sh ->
# verl.trainer.main_ppo path.  It does not replace the trainer, Ray, FSDP,
# vLLM, offline teacher, or ROPD reward implementations.
#
# Required before launch (kept out of this file):
#   ANTHROPIC_AUTH_TOKEN
# Optional overrides:
#   ROPD_PILOT_STEPS        default 300
#   ROPD_PILOT_BATCH_SIZE   default 8
#   ROPD_PILOT_ROLLOUT_N    default 4
#   ROPD_PILOT_RESPONSE_LEN default 2048
#   ROPD_PILOT_GPU_MEMORY_UTILIZATION default 0.45
#   EXPERIMENT              default ropd-pilot-4gpu-<timestamp>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${ANTHROPIC_AUTH_TOKEN:?Set ANTHROPIC_AUTH_TOKEN before starting this run.}"

# Keep this launch reproducible and prevent a local .env from replacing the
# explicit model, index, or judge role configuration below.
export ROPD_SKIP_REPO_DOTENV=true
export ROPD_MODEL_PATH="${ROPD_MODEL_PATH:-/home/work/migoo_ai_public/ruiyu_lu/models/Qwen3-4B}"
export ROPD_TEACHER_INDEX_PATH="${ROPD_TEACHER_INDEX_PATH:-$PROJECT_ROOT/tmp/opus_teacher_index_math100_n4.jsonl}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://inner-api.us.migoo.shopee.io/inbeeai/compass-api/v1}"

# Teacher stays offline.  Its profile/model must match the fingerprint written
# into the supplied index. Rubricator and verifier are online Anthropic roles.
export ROPD_TEACHER_PROFILE="${ROPD_TEACHER_PROFILE:-claude_compass}"
export ROPD_TEACHER_MODEL="${ROPD_TEACHER_MODEL:-claude-opus-4-6}"
export ROPD_RUBRICATOR_PROVIDER="${ROPD_RUBRICATOR_PROVIDER:-anthropic}"
export ROPD_RUBRICATOR_PROFILE="${ROPD_RUBRICATOR_PROFILE:-claude_compass}"
export ROPD_RUBRICATOR_MODEL="${ROPD_RUBRICATOR_MODEL:-claude-opus-4-6}"
export ROPD_VERIFIER_PROVIDER="${ROPD_VERIFIER_PROVIDER:-anthropic}"
export ROPD_VERIFIER_PROFILE="${ROPD_VERIFIER_PROFILE:-claude_compass}"
export ROPD_VERIFIER_MODEL="${ROPD_VERIFIER_MODEL:-claude-opus-4-6}"

steps="${ROPD_PILOT_STEPS:-300}"
batch_size="${ROPD_PILOT_BATCH_SIZE:-8}"
rollout_n="${ROPD_PILOT_ROLLOUT_N:-4}"
response_len="${ROPD_PILOT_RESPONSE_LEN:-2048}"
gpu_memory_utilization="${ROPD_PILOT_GPU_MEMORY_UTILIZATION:-0.45}"
export EXPERIMENT="${EXPERIMENT:-ropd-pilot-4gpu-$(date +%Y%m%d-%H%M%S)}"

if (( batch_size < 4 || batch_size % 4 != 0 )); then
    echo "ROPD_PILOT_BATCH_SIZE must be divisible by four GPUs and at least 4; got $batch_size." >&2
    exit 2
fi
if (( rollout_n < 2 )); then
    echo "ROPD_PILOT_ROLLOUT_N must be at least 2 for GRPO; got $rollout_n." >&2
    exit 2
fi

run_root="$PROJECT_ROOT/outputs/$EXPERIMENT"
ckpt_root="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
mkdir -p "$run_root"

echo "Experiment: $EXPERIMENT"
echo "Steps: $steps | prompt batch: $batch_size | rollouts/prompt: $rollout_n"
echo "Expected training trajectories: $((steps * batch_size * rollout_n))"
echo "Model: $ROPD_MODEL_PATH"
echo "Teacher index: $ROPD_TEACHER_INDEX_PATH"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
bash training/train.sh \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$steps" \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.logger='[console]' \
  trainer.default_local_dir="$ckpt_root" \
  data.train_files="[$PROJECT_ROOT/tmp/dapo-math-17k.train.teacher_index_aligned.smoke1.parquet]" \
  data.train_max_samples=800 \
  data.train_batch_size="$batch_size" \
  data.max_prompt_length=2048 \
  data.max_response_length="$response_len" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$batch_size" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="$rollout_n" \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length="$response_len" \
  actor_rollout_ref.rollout.max_model_len="$((2048 + response_len))" \
  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization" \
  rollout.n="$rollout_n" \
  reward_model.reward_kwargs.ropd.teacher_answer_count=4 \
  reward_model.reward_kwargs.ropd.max_concurrency=4 \
  reward_model.reward_kwargs.ropd.provider_limits.max_concurrent_requests=8 \
  reward_model.reward_kwargs.ropd.transport.max_in_flight_requests=8 \
  +reward_model.reward_kwargs.ropd.debug.output_dir="$run_root/reward_debug"
