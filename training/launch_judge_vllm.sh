#!/usr/bin/env bash
# Launch a local vLLM server that serves the rubricator + verifier judge for
# ROPD training.
#
# Usage:
#   bash training/launch_judge_vllm.sh
#
# Common environment variables:
#   ROPD_JUDGE_CUDA_VISIBLE_DEVICES   GPUs to use            (default 1)
#   ROPD_JUDGE_HOST                   bind host              (default 127.0.0.1)
#   ROPD_JUDGE_PORT                   port                   (default 18080)
#   ROPD_JUDGE_MODEL_PATH             local HF / safetensors model path (required)
#   ROPD_JUDGE_SERVED_MODEL_NAME      name advertised by vLLM (default Qwen3-30B-A3B-Instruct-2507)
#   ROPD_JUDGE_DTYPE                  weight dtype           (default bfloat16)
#   ROPD_JUDGE_MAX_MODEL_LEN          max context length     (default 98304)
#   ROPD_JUDGE_GPU_MEMORY_UTILIZATION GPU memory fraction    (default 0.92)
#   ROPD_JUDGE_TENSOR_PARALLEL_SIZE   TP size                (default 1)
#   ROPD_JUDGE_API_KEY                optional API key for the server

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

ROPD_JUDGE_CUDA_VISIBLE_DEVICES="${ROPD_JUDGE_CUDA_VISIBLE_DEVICES:-1}"
ROPD_JUDGE_VLLM_LOGGING_LEVEL="${ROPD_JUDGE_VLLM_LOGGING_LEVEL:-INFO}"
ROPD_JUDGE_HOST="${ROPD_JUDGE_HOST:-127.0.0.1}"
ROPD_JUDGE_PORT="${ROPD_JUDGE_PORT:-18080}"
ROPD_JUDGE_MODEL_PATH="${ROPD_JUDGE_MODEL_PATH:-}"
ROPD_JUDGE_SERVED_MODEL_NAME="${ROPD_JUDGE_SERVED_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
ROPD_JUDGE_DTYPE="${ROPD_JUDGE_DTYPE:-bfloat16}"
ROPD_JUDGE_MAX_MODEL_LEN="${ROPD_JUDGE_MAX_MODEL_LEN:-98304}"
ROPD_JUDGE_GPU_MEMORY_UTILIZATION="${ROPD_JUDGE_GPU_MEMORY_UTILIZATION:-0.92}"
ROPD_JUDGE_TENSOR_PARALLEL_SIZE="${ROPD_JUDGE_TENSOR_PARALLEL_SIZE:-1}"
ROPD_JUDGE_MAX_NUM_SEQS="${ROPD_JUDGE_MAX_NUM_SEQS:-16}"
ROPD_JUDGE_MAX_NUM_BATCHED_TOKENS="${ROPD_JUDGE_MAX_NUM_BATCHED_TOKENS:-131072}"
ROPD_JUDGE_UVICORN_LOG_LEVEL="${ROPD_JUDGE_UVICORN_LOG_LEVEL:-info}"

if [[ -z "$ROPD_JUDGE_MODEL_PATH" ]]; then
    echo "Error: ROPD_JUDGE_MODEL_PATH must be set to a local HF model path." >&2
    exit 1
fi

cmd=(
    uv run --no-sync python -m vllm.entrypoints.openai.api_server
    --host "$ROPD_JUDGE_HOST"
    --port "$ROPD_JUDGE_PORT"
    --model "$ROPD_JUDGE_MODEL_PATH"
    --served-model-name "$ROPD_JUDGE_SERVED_MODEL_NAME"
    --dtype "$ROPD_JUDGE_DTYPE"
    --max-model-len "$ROPD_JUDGE_MAX_MODEL_LEN"
    --gpu-memory-utilization "$ROPD_JUDGE_GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$ROPD_JUDGE_TENSOR_PARALLEL_SIZE"
    --max-num-seqs "$ROPD_JUDGE_MAX_NUM_SEQS"
    --max-num-batched-tokens "$ROPD_JUDGE_MAX_NUM_BATCHED_TOKENS"
    --uvicorn-log-level "$ROPD_JUDGE_UVICORN_LOG_LEVEL"
)

if is_true "${ROPD_JUDGE_TRUST_REMOTE_CODE:-false}"; then
    cmd+=(--trust-remote-code)
fi
if is_true "${ROPD_JUDGE_ENFORCE_EAGER:-false}"; then
    cmd+=(--enforce-eager)
fi

if [[ -n "${ROPD_JUDGE_API_KEY:-}" ]]; then
    cmd+=(--api-key "$ROPD_JUDGE_API_KEY")
fi

if [[ -n "${ROPD_JUDGE_REASONING_PARSER:-}" ]]; then
    cmd+=(--reasoning-parser "$ROPD_JUDGE_REASONING_PARSER")
fi

if [[ -n "${ROPD_JUDGE_CHAT_TEMPLATE:-}" ]]; then
    cmd+=(--chat-template "$ROPD_JUDGE_CHAT_TEMPLATE")
fi

echo "ROPD_JUDGE_CUDA_VISIBLE_DEVICES=$ROPD_JUDGE_CUDA_VISIBLE_DEVICES"
echo "ROPD_JUDGE_HOST=$ROPD_JUDGE_HOST"
echo "ROPD_JUDGE_PORT=$ROPD_JUDGE_PORT"
echo "ROPD_JUDGE_MODEL_PATH=$ROPD_JUDGE_MODEL_PATH"
echo "ROPD_JUDGE_SERVED_MODEL_NAME=$ROPD_JUDGE_SERVED_MODEL_NAME"
echo "Command: CUDA_VISIBLE_DEVICES=$ROPD_JUDGE_CUDA_VISIBLE_DEVICES VLLM_LOGGING_LEVEL=$ROPD_JUDGE_VLLM_LOGGING_LEVEL ${cmd[*]}"

if is_true "${ROPD_DRYRUN:-false}"; then
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$ROPD_JUDGE_CUDA_VISIBLE_DEVICES"
export VLLM_LOGGING_LEVEL="$ROPD_JUDGE_VLLM_LOGGING_LEVEL"
exec "${cmd[@]}"
