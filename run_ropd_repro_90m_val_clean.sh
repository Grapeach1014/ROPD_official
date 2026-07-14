#!/usr/bin/env bash
# Evaluate base + all saved ROPD FSDP checkpoints with ground-truth-only val.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export ROPD_SKIP_REPO_DOTENV=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ROPD_MODEL_PATH="/home/work/migoo_ai_public/ruiyu_lu/models/Qwen3-4B"
export ROPD_EVAL_VAL_FILE="${ROPD_EVAL_VAL_FILE:-$ROOT/data/math/val.parquet}"

CKPT_ROOT="${ROPD_EVAL_CKPT_ROOT:-$ROOT/checkpoints/ropd-repro-90m-4gpu}"
OUT_ROOT="${ROPD_EVAL_OUT_ROOT:-$ROOT/outputs/ropd-repro-90m-4gpu/validation}"
PYTHON="${ROPD_EVAL_PYTHON:-$ROOT/.venv/bin/python}"
TIMEOUT="${ROPD_EVAL_TIMEOUT:-45m}"

test -x "$PYTHON"
test -f "$ROPD_MODEL_PATH/tokenizer_config.json"
test -f "$ROPD_EVAL_VAL_FILE"
for step in 5 10 15 20 25; do
  test -f "$CKPT_ROOT/global_step_${step}/actor/model_world_size_4_rank_0.pt"
done

mkdir -p "$OUT_ROOT"

run_candidate() {
  local name="$1"
  local resume_mode="$2"
  local resume_path="${3:-}"
  local candidate_dir="$OUT_ROOT/$name"
  local metrics_path="$candidate_dir/metrics.json"
  local generations_dir="$candidate_dir/generations"

  mkdir -p "$candidate_dir"
  echo "=== Evaluating $name ==="

  local args=(
    -m verl.trainer.main_ppo
    --config-name ropd
    --config-name ropd_repro_90m_val_groundtruth_run
    "trainer.experiment_name=ropd-repro-90m-val-$name"
    "trainer.resume_mode=$resume_mode"
    "trainer.validation_metrics_path=$metrics_path"
    "trainer.validation_data_dir=$generations_dir"
  )
  if [[ "$resume_mode" == "resume_path" ]]; then
    args+=("trainer.resume_from_path=$resume_path")
  fi

  timeout "$TIMEOUT" "$PYTHON" "${args[@]}" 2>&1 | tee "$candidate_dir/main_ppo.log"
}

run_candidate base disable
for step in 5 10 15 20 25; do
  run_candidate "step_$step" resume_path "$CKPT_ROOT/global_step_$step"
done

echo "Validation finished. Metrics: $OUT_ROOT/*/metrics.json"
