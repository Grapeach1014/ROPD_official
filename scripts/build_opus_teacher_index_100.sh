#!/usr/bin/env bash
# Build a 100-prompt offline teacher index with an Opus/Anthropic teacher.
#
# Required credentials:
#   ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL
# or:
#   ANTHROPIC_COMPASS_AUTH_TOKEN / ANTHROPIC_COMPASS_BASE_URL

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

INPUT_PARQUET="${INPUT_PARQUET:-$PROJECT_ROOT/datasets/unified/math/dapo-math-17k/train.parquet}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
TEACHER_ANSWER_COUNT="${TEACHER_ANSWER_COUNT:-4}"
SAMPLE_PARQUET="${SAMPLE_PARQUET:-$PROJECT_ROOT/tmp/dapo-math-17k.train.first${SAMPLE_SIZE}.parquet}"
OUTPUT_INDEX="${OUTPUT_INDEX:-$PROJECT_ROOT/tmp/opus_teacher_index_math${SAMPLE_SIZE}_n${TEACHER_ANSWER_COUNT}.jsonl}"
MAX_WORKERS="${MAX_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OPUS_MODEL="${OPUS_MODEL:-claude-opus-4-6}"

export ROPD_TEACHER_PROVIDER="${ROPD_TEACHER_PROVIDER:-anthropic}"
export ROPD_TEACHER_PROFILE="${ROPD_TEACHER_PROFILE:-claude_compass}"
export ROPD_TEACHER_MODEL="$OPUS_MODEL"
export ROPD_TEACHER_ANSWER_COUNT="$TEACHER_ANSWER_COUNT"

if [[ -n "${ANTHROPIC_COMPASS_AUTH_TOKEN:-}" || -n "${ANTHROPIC_COMPASS_BASE_URL:-}" ]]; then
    export ANTHROPIC_PROFILE="${ANTHROPIC_PROFILE:-COMPASS}"
fi

mkdir -p "$(dirname "$SAMPLE_PARQUET")" "$(dirname "$OUTPUT_INDEX")"

uv run --no-sync python - "$INPUT_PARQUET" "$SAMPLE_PARQUET" "$SAMPLE_SIZE" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
sample_size = int(sys.argv[3])

if not input_path.exists():
    raise FileNotFoundError(f"Input parquet not found: {input_path}")

table = pq.read_table(input_path).slice(0, sample_size)
if table.num_rows < sample_size:
    raise ValueError(f"{input_path} only has {table.num_rows} rows; requested {sample_size}.")

pq.write_table(table, output_path)
print(f"Wrote {table.num_rows} rows to {output_path}")
PY

echo "Teacher provider: $ROPD_TEACHER_PROVIDER"
echo "Teacher profile:  $ROPD_TEACHER_PROFILE"
echo "Teacher model:    $ROPD_TEACHER_MODEL"
echo "Teacher answers:  $ROPD_TEACHER_ANSWER_COUNT"
echo "Input sample:     $SAMPLE_PARQUET"
echo "Output index:     $OUTPUT_INDEX"
echo "Resume existing:  yes; completed uid/prompt rows with enough teacher_answers are skipped"

uv run --no-sync python scripts/build_teacher_index.py \
    --input "$SAMPLE_PARQUET" \
    --output "$OUTPUT_INDEX" \
    --teacher-answer-count "$ROPD_TEACHER_ANSWER_COUNT" \
    --batch-size "$BATCH_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --resume

echo "Done: $OUTPUT_INDEX"
