#!/usr/bin/env bash
# Run a tiny ROPD experiment:
#   - train on the first 10 DAPO-Math prompts
#   - 20 PPO/GRPO steps
#   - local student: /home/work/models/Qwen3-4B
#   - offline teacher answers from script 1
#   - Opus/Anthropic rubricator and verifier
#   - student rollout n defaults to 4 for memory; set STUDENT_ROLLOUT_N=8 if the GPU can hold it
#
# The script captures stdout/stderr to a timestamped log and writes a Markdown
# summary after training finishes, even when training exits with an error.
# It also extracts console metrics every SUMMARY_INTERVAL steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

is_true() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

if ! is_true "${ROPD_SKIP_REPO_DOTENV:-false}" && [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets/unified}"
export ROPD_TRAIN_TASK="${ROPD_TRAIN_TASK:-math/dapo-math-17k}"
export ROPD_VAL_TASK="${ROPD_VAL_TASK:-math_eval/aime24}"
export ROPD_MODEL_PATH="${ROPD_MODEL_PATH:-/home/work/models/Qwen3-4B}"
SOURCE_TRAIN_PARQUET="${SOURCE_TRAIN_PARQUET:-$PROJECT_ROOT/tmp/dapo-math-17k.train.first100.parquet}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_ROOT/tmp/dapo-math-17k.train.teacher_index_aligned.parquet}"
export ROPD_TEACHER_INDEX_PATH="${ROPD_TEACHER_INDEX_PATH:-$PROJECT_ROOT/tmp/opus_teacher_index_math100_n4.jsonl}"
export ROPD_TEACHER_ANSWER_COUNT="${ROPD_TEACHER_ANSWER_COUNT:-4}"
OPUS_MODEL="${OPUS_MODEL:-claude-opus-4-6}"

# Training reads teacher answers from offline_index, but the teacher role still
# needs the same profile/model used when the index was built so fingerprint
# validation can match. Do not set ROPD_TEACHER_PROVIDER=anthropic here.
export ROPD_TEACHER_PROFILE="${ROPD_TEACHER_PROFILE:-claude_compass}"
export ROPD_TEACHER_MODEL="$OPUS_MODEL"

export ROPD_RUBRICATOR_PROVIDER="${ROPD_RUBRICATOR_PROVIDER:-anthropic}"
export ROPD_RUBRICATOR_PROFILE="${ROPD_RUBRICATOR_PROFILE:-claude_compass}"
export ROPD_RUBRICATOR_MODEL="$OPUS_MODEL"
export ROPD_VERIFIER_PROVIDER="${ROPD_VERIFIER_PROVIDER:-anthropic}"
export ROPD_VERIFIER_PROFILE="${ROPD_VERIFIER_PROFILE:-claude_compass}"
export ROPD_VERIFIER_MODEL="$OPUS_MODEL"

if [[ -n "${ANTHROPIC_COMPASS_AUTH_TOKEN:-}" || -n "${ANTHROPIC_COMPASS_BASE_URL:-}" ]]; then
    export ANTHROPIC_PROFILE="${ANTHROPIC_PROFILE:-COMPASS}"
fi

export EXPERIMENT="${EXPERIMENT:-ropd_opus_math10_20steps}"
export WANDB_MODE="${WANDB_MODE:-offline}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAINING_STEPS="${TRAINING_STEPS:-20}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-5}"
STUDENT_ROLLOUT_N="${STUDENT_ROLLOUT_N:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/outputs/script_runs/${EXPERIMENT}/${RUN_ID}}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/ropd_script_runs/${EXPERIMENT}/${RUN_ID}}"
LOG_PATH="$RUN_DIR/train.log"
SUMMARY_TIMESTAMP="${SUMMARY_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUMMARY_PATH="$RUN_DIR/summary_${SUMMARY_TIMESTAMP}.md"
DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-$RUN_DIR/ropd_debug}"

if [[ ! -s "$ROPD_TEACHER_INDEX_PATH" ]]; then
    echo "Teacher index not found: $ROPD_TEACHER_INDEX_PATH" >&2
    echo "Run: bash scripts/build_opus_teacher_index_100.sh" >&2
    exit 1
fi

if [[ ! -s "$SOURCE_TRAIN_PARQUET" ]]; then
    echo "Source train parquet not found: $SOURCE_TRAIN_PARQUET" >&2
    echo "Run: bash scripts/build_opus_teacher_index_100.sh" >&2
    exit 1
fi

uv run --no-sync python scripts/filter_parquet_by_teacher_index.py \
    --input "$SOURCE_TRAIN_PARQUET" \
    --teacher-index "$ROPD_TEACHER_INDEX_PATH" \
    --output "$TRAIN_PARQUET" \
    --min-rows "$TRAIN_SAMPLES"

mkdir -p "$RUN_DIR"

cmd=(
    bash training/train.sh
    "data.train_files=[$TRAIN_PARQUET]"
    data.train_max_samples="$TRAIN_SAMPLES"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=false
    actor_rollout_ref.actor.self_distillation.distillation_topk=null
    trainer.total_training_steps="$TRAINING_STEPS"
    trainer.total_epochs=1
    trainer.test_freq=-1
    trainer.save_freq="$SUMMARY_INTERVAL"
    trainer.default_local_dir="$CKPT_DIR"
    actor_rollout_ref.model.path="$ROPD_MODEL_PATH"
    actor_rollout_ref.rollout.n="$STUDENT_ROLLOUT_N"
    rollout.n="$STUDENT_ROLLOUT_N"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH"
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false
    +reward_model.reward_kwargs.ropd.debug.output_dir="$DEBUG_OUTPUT_DIR"
    reward_model.reward_kwargs.ropd.teacher_answer_count="$ROPD_TEACHER_ANSWER_COUNT"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.teacher.profile="$ROPD_TEACHER_PROFILE"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.teacher.model="$ROPD_TEACHER_MODEL"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.rubricator.provider="$ROPD_RUBRICATOR_PROVIDER"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.rubricator.profile="$ROPD_RUBRICATOR_PROFILE"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.rubricator.model="$ROPD_RUBRICATOR_MODEL"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.verifier.provider="$ROPD_VERIFIER_PROVIDER"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.verifier.profile="$ROPD_VERIFIER_PROFILE"
    +reward_model.reward_kwargs.ropd.provider_resolution.overrides.verifier.model="$ROPD_VERIFIER_MODEL"
    reward_model.reward_kwargs.ropd.max_concurrency=4
    reward_model.reward_kwargs.ropd.provider_limits.max_concurrent_requests=8
    reward_model.reward_kwargs.ropd.transport.max_in_flight_requests=8
)

if (($# > 0)); then
    cmd+=("$@")
fi

print_command() {
    printf '%q ' "${cmd[@]}"
}

extract_step_metrics() {
    local step="$1"
    if [[ ! -f "$LOG_PATH" ]]; then
        echo "Log file was not created."
        return
    fi

    local exact_lines
    exact_lines="$(grep -E "(^|[[:space:]-])step:${step}([[:space:]-]|$)|training/global_step:${step}(\.0)?([[:space:]-]|$)|global_step_${step}([/[:space:]]|$)" "$LOG_PATH" | tail -20 || true)"
    if [[ -n "$exact_lines" ]]; then
        printf '%s\n' "$exact_lines"
    else
        echo "No console metric/checkpoint lines found for step $step."
    fi
}

write_interval_sections() {
    local step
    step="$SUMMARY_INTERVAL"
    while ((step <= TRAINING_STEPS)); do
        echo
        echo "### Step $step"
        echo
        echo '```text'
        extract_step_metrics "$step"
        echo '```'
        step=$((step + SUMMARY_INTERVAL))
    done
}

write_summary() {
    local train_status="$1"
    local status_label="failed"
    if [[ "$train_status" -eq 0 ]]; then
        status_label="succeeded"
    fi

    SUMMARY_PATH="$(uv run --no-sync python scripts/summarize_ropd_run.py \
        --run-dir "$RUN_DIR" \
        --log-path "$LOG_PATH" \
        --output "$SUMMARY_PATH" \
        --timestamp "$SUMMARY_TIMESTAMP" \
        --status "$status_label" \
        --exit-code "$train_status")"
}

{
    echo "# Command"
    print_command
    echo
    echo
    echo "# Environment"
    echo "RUN_ID=$RUN_ID"
    echo "RUN_DIR=$RUN_DIR"
    echo "CKPT_DIR=$CKPT_DIR"
    echo "SUMMARY_TIMESTAMP=$SUMMARY_TIMESTAMP"
    echo "DATA_ROOT=$DATA_ROOT"
    echo "ROPD_TRAIN_TASK=$ROPD_TRAIN_TASK"
    echo "ROPD_VAL_TASK=$ROPD_VAL_TASK"
    echo "ROPD_MODEL_PATH=$ROPD_MODEL_PATH"
    echo "SOURCE_TRAIN_PARQUET=$SOURCE_TRAIN_PARQUET"
    echo "TRAIN_PARQUET=$TRAIN_PARQUET"
    echo "ROPD_TEACHER_INDEX_PATH=$ROPD_TEACHER_INDEX_PATH"
    echo "ROPD_TEACHER_ANSWER_COUNT=$ROPD_TEACHER_ANSWER_COUNT"
    echo "ROPD_TEACHER_PROFILE=$ROPD_TEACHER_PROFILE"
    echo "ROPD_TEACHER_MODEL=$ROPD_TEACHER_MODEL"
    echo "ROPD_RUBRICATOR_PROVIDER=$ROPD_RUBRICATOR_PROVIDER"
    echo "ROPD_RUBRICATOR_PROFILE=$ROPD_RUBRICATOR_PROFILE"
    echo "ROPD_RUBRICATOR_MODEL=$ROPD_RUBRICATOR_MODEL"
    echo "ROPD_VERIFIER_PROVIDER=$ROPD_VERIFIER_PROVIDER"
    echo "ROPD_VERIFIER_PROFILE=$ROPD_VERIFIER_PROFILE"
    echo "ROPD_VERIFIER_MODEL=$ROPD_VERIFIER_MODEL"
    echo "WANDB_MODE=$WANDB_MODE"
    echo "TRAIN_SAMPLES=$TRAIN_SAMPLES"
    echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    echo "PPO_MINI_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    echo "TRAINING_STEPS=$TRAINING_STEPS"
    echo "SUMMARY_INTERVAL=$SUMMARY_INTERVAL"
    echo "STUDENT_ROLLOUT_N=$STUDENT_ROLLOUT_N"
    echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
    echo "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
    echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
    echo "DEBUG_OUTPUT_DIR=$DEBUG_OUTPUT_DIR"
    echo "ADAMW_FOREACH=false"
    echo
    echo "# Training Log"
} >"$LOG_PATH"

echo "Student model:    $ROPD_MODEL_PATH"
echo "Source parquet:   $SOURCE_TRAIN_PARQUET"
echo "Train parquet:    $TRAIN_PARQUET"
echo "Teacher index:    $ROPD_TEACHER_INDEX_PATH"
echo "Teacher profile:  $ROPD_TEACHER_PROFILE"
echo "Teacher model:    $ROPD_TEACHER_MODEL"
echo "Rubricator model: $ROPD_RUBRICATOR_MODEL"
echo "Verifier model:   $ROPD_VERIFIER_MODEL"
echo "Experiment:       $EXPERIMENT"
echo "Run dir:          $RUN_DIR"
echo "Checkpoint dir:   $CKPT_DIR"
echo "Log path:         $LOG_PATH"
echo "Summary path:     $SUMMARY_PATH"
echo "Training steps:   $TRAINING_STEPS"
echo "Stat interval:    $SUMMARY_INTERVAL"
echo "Batch size:       $TRAIN_BATCH_SIZE"
echo "PPO mini batch:   $TRAIN_BATCH_SIZE"
echo "Rollout n:        $STUDENT_ROLLOUT_N"
echo "Response length:  $MAX_RESPONSE_LENGTH"
echo "vLLM GPU util:    $VLLM_GPU_MEMORY_UTILIZATION"
echo "Max batch tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Debug output:     $DEBUG_OUTPUT_DIR"
echo "AdamW foreach:   false"
echo "Command:          $(print_command)"

set +e
"${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
train_status=${PIPESTATUS[0]}
set -e

write_summary "$train_status"
echo "Summary written to: $SUMMARY_PATH"
echo "Full log written to: $LOG_PATH"
exit "$train_status"
