#!/usr/bin/env bash
# Four-H100, 25-step ROPD math pilot.
#
# All non-secret runtime variables live here so model-path fallback cannot send
# Transformers to Hugging Face.  The only interactive input is the API token.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export ROPD_SKIP_REPO_DOTENV=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export ROPD_MODEL_PATH="/home/work/migoo_ai_public/ruiyu_lu/models/Qwen3-4B"
export ROPD_TEACHER_INDEX_PATH="$ROOT/data/math/teacher_index_train.jsonl"
export ROPD_REPRO_TRAIN_FILE="$ROOT/data/math/train.parquet"
export ROPD_REPRO_VAL_FILE="$ROOT/data/math/val.parquet"

export ROPD_TEACHER_PROVIDER="offline"
export ROPD_TEACHER_PROFILE="claude_compass"
export ROPD_TEACHER_MODEL="claude-opus-4-6"
export ROPD_RUBRICATOR_PROVIDER="anthropic"
export ROPD_RUBRICATOR_PROFILE="claude_compass"
export ROPD_RUBRICATOR_MODEL="claude-opus-4-6"
export ROPD_VERIFIER_PROVIDER="anthropic"
export ROPD_VERIFIER_PROFILE="claude_compass"
export ROPD_VERIFIER_MODEL="claude-opus-4-6"
export ANTHROPIC_BASE_URL="http://inner-api.us.migoo.shopee.io/inbeeai/compass-api/v1"

export EXPERIMENT="${EXPERIMENT:-ropd-repro-90m-4gpu}"
export ROPD_REPRO_CKPT_DIR="${ROPD_REPRO_CKPT_DIR:-$ROOT/checkpoints/$EXPERIMENT}"
export ROPD_REPRO_DEBUG_DIR="${ROPD_REPRO_DEBUG_DIR:-$ROOT/outputs/$EXPERIMENT/reward_debug}"

test -f "$ROPD_MODEL_PATH/tokenizer_config.json"
test -f "$ROPD_TEACHER_INDEX_PATH"
test -f "$ROPD_REPRO_TRAIN_FILE"
test -f "$ROPD_REPRO_VAL_FILE"

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  read -rsp "ANTHROPIC_AUTH_TOKEN: " ANTHROPIC_AUTH_TOKEN
  echo
  export ANTHROPIC_AUTH_TOKEN
fi

echo "ROPD_MODEL_PATH=$ROPD_MODEL_PATH"
echo "ROPD_TEACHER_INDEX_PATH=$ROPD_TEACHER_INDEX_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec timeout "${ROPD_TIMEOUT:-110m}" \
  bash training/train.sh --config-name ropd_repro_90m "$@"
