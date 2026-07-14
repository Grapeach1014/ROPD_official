#!/usr/bin/env bash
# Single-entrypoint training script for the ROPD trainer.
#
# Usage:
#   bash training/train.sh [extra hydra overrides...]
#
# Configure your run with environment variables (a `.env` at the repo root is
# auto-sourced; see `.env.example`).
#   DATA_ROOT                   parquet dataset root              (default datasets/unified)
#   ROPD_TRAIN_TASK        training task subpath             (default math/dapo-math-17k)
#   ROPD_VAL_TASK          validation task subpath           (default math_eval/aime24)
#   ROPD_MODEL_PATH        path to the policy / actor model  (default models/Qwen3-4B)
#   ROPD_CKPT_DIR          checkpoint root                   (default <repo>/checkpoints/ropd)
#   ROPD_TEACHER_INDEX_PATH offline teacher index jsonl       (required)
#   ROPD_VLLM_BASE_URL     local vLLM judge URL              (default http://127.0.0.1:18080/v1)
#   ROPD_VLLM_MODEL        served name on the vLLM judge     (default Qwen3-30B-A3B-Instruct-2507)
#   ROPD_VLLM_API_KEY      API key for the vLLM judge        (default EMPTY)
#   EXPERIMENT                  W&B experiment / group name       (default ropd)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

is_true() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

# Source repo .env (skip with ROPD_SKIP_REPO_DOTENV=true).
if ! is_true "${ROPD_SKIP_REPO_DOTENV:-false}"; then
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env"
        set +a
    fi
fi

export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets/unified}"
export ROPD_TRAIN_TASK="${ROPD_TRAIN_TASK:-math/dapo-math-17k}"
export ROPD_VAL_TASK="${ROPD_VAL_TASK:-math_eval/aime24}"
export EXPERIMENT="${EXPERIMENT:-ropd}"
export ROPD_VLLM_BASE_URL="${ROPD_VLLM_BASE_URL:-http://127.0.0.1:18080/v1}"
export ROPD_VLLM_MODEL="${ROPD_VLLM_MODEL:-Qwen3-30B-A3B-Instruct-2507}"
export ROPD_VLLM_API_KEY="${ROPD_VLLM_API_KEY:-EMPTY}"

if command -v uv >/dev/null 2>&1; then
    # Respect an explicitly activated environment (for example .venv-b300).
    # Without --active, uv ignores VIRTUAL_ENV when it differs from the
    # project's default .venv and emits a warning.
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        cmd=(uv run --active --no-sync python -m verl.trainer.main_ppo --config-name ropd)
    else
        cmd=(uv run --no-sync python -m verl.trainer.main_ppo --config-name ropd)
    fi
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    cmd=("$PROJECT_ROOT/.venv/bin/python" -m verl.trainer.main_ppo --config-name ropd)
else
    echo "Neither uv nor $PROJECT_ROOT/.venv/bin/python was found." >&2
    exit 127
fi

if (($# > 0)); then
    cmd+=("$@")
fi

echo "DATA_ROOT=$DATA_ROOT"
echo "ROPD_TRAIN_TASK=$ROPD_TRAIN_TASK"
echo "ROPD_VAL_TASK=$ROPD_VAL_TASK"
echo "EXPERIMENT=$EXPERIMENT"
echo "Config: verl/trainer/config/ropd.yaml"
if [[ -n "${ROPD_MODEL_PATH:-}" ]]; then
    echo "ROPD_MODEL_PATH=$ROPD_MODEL_PATH"
fi
if [[ -n "${ROPD_TEACHER_INDEX_PATH:-}" ]]; then
    echo "ROPD_TEACHER_INDEX_PATH=$ROPD_TEACHER_INDEX_PATH"
fi
echo "ROPD_VLLM_BASE_URL=$ROPD_VLLM_BASE_URL"
echo "ROPD_VLLM_MODEL=$ROPD_VLLM_MODEL"
echo "Command: ${cmd[*]}"

if is_true "${ROPD_DRYRUN:-false}"; then
    exit 0
fi

exec "${cmd[@]}"
