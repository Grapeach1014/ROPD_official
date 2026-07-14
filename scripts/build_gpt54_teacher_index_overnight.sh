#!/usr/bin/env bash
# Incrementally build a separate GPT-5.4 Teacher Index from DAPO-Math.
#
# This is deliberately a wrapper around build_opus_teacher_index_overnight.sh:
# it retains the proven UID de-duplication, resume, bounded-window and pacing
# logic, while using an independent output file. Never mix this file with an
# Opus index: an offline teacher index has one immutable teacher fingerprint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Use a new path by default. `touch` is non-destructive: re-running resumes
# and appends only missing UID + prompt-hash records.
export OUTPUT_INDEX="${OUTPUT_INDEX:-$PROJECT_ROOT/tmp/gpt54_teacher_index_dapo_math17k_n4.jsonl}"
export PENDING_PARQUET="${PENDING_PARQUET:-$PROJECT_ROOT/tmp/dapo-math-17k.gpt54.pending.start${WINDOW_START:-0}.size${WINDOW_SIZE:-10000}.parquet}"
export LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/outputs/teacher_index/gpt54_math_window_${WINDOW_START:-0}_${WINDOW_SIZE:-10000}.log}"
mkdir -p "$(dirname "$OUTPUT_INDEX")"
touch "$OUTPUT_INDEX"

# The gateway routes GPT-5.4 through the same Compass /messages endpoint. The
# underlying provider translates its completion-token field automatically.
export ROPD_TEACHER_PROVIDER="${ROPD_TEACHER_PROVIDER:-anthropic}"
export ROPD_TEACHER_PROFILE="${ROPD_TEACHER_PROFILE:-claude_compass}"
export TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.4}"

# GPT-5.4 is much faster than Opus on this gateway. Four concurrent prompts
# keep an overnight job productive while staying well below the configured
# 16-request provider cap. If the log reports HTTP 429, rerun with
# MAX_WORKERS=1 BATCH_SIZE=1 INTER_BATCH_DELAY_SECONDS=10; --resume preserves
# all completed rows.
export MAX_WORKERS="${MAX_WORKERS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export INTER_BATCH_DELAY_SECONDS="${INTER_BATCH_DELAY_SECONDS:-2}"

exec bash "$SCRIPT_DIR/build_opus_teacher_index_overnight.sh"
