#!/usr/bin/env bash
# Incrementally extend the existing Opus Teacher Index from a bounded DAPO-Math
# window. Run this script inside tmux for an overnight job.
#
# Safety: it never truncates OUTPUT_INDEX; it filters existing UIDs before
# generation and the underlying builder additionally resumes by UID+prompt hash.

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

# Shell/.env line wrapping can accidentally insert a newline into a gateway
# URL, which httpx rejects before sending any request. URLs cannot contain
# whitespace, so normalize only these endpoint variables after all dotenv
# values have been loaded.
for url_var in ANTHROPIC_BASE_URL ANTHROPIC_COMPASS_BASE_URL; do
    if [[ -n "${!url_var:-}" ]]; then
        printf -v "$url_var" '%s' "$(printf '%s' "${!url_var}" | tr -d '[:space:]')"
        export "$url_var"
    fi
done

# The source has 1,791,700 rows. Keep a bounded, resumable window rather than
# submitting the entire corpus accidentally. The initial window includes the
# existing 500 UIDs, which are filtered out before generation.
SOURCE_PARQUET="${SOURCE_PARQUET:-$PROJECT_ROOT/data/DAPO-Math-17k/data/dapo-math-17k.parquet}"
WINDOW_START="${WINDOW_START:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-10000}"
WINDOW_END_EXCLUSIVE=$((WINDOW_START + WINDOW_SIZE))

# Required by the request: extend this index, never overwrite it.
OUTPUT_INDEX="${OUTPUT_INDEX:-$PROJECT_ROOT/tmp/opus_teacher_index_math100_n4.jsonl}"
PENDING_PARQUET="${PENDING_PARQUET:-$PROJECT_ROOT/tmp/dapo-math-17k.pending.start${WINDOW_START}.size${WINDOW_SIZE}.parquet}"

TEACHER_ANSWER_COUNT="${TEACHER_ANSWER_COUNT:-4}"
# TEACHER_MODEL is the neutral name used by wrappers for other providers.
# Keep OPUS_MODEL as a backward-compatible fallback for existing invocations.
TEACHER_MODEL="${TEACHER_MODEL:-${OPUS_MODEL:-claude-opus-4-6}}"

# The Compass gateway rate-limits this model aggressively. Process one prompt
# at a time and pause between prompts so a transient 429 cannot trip the
# circuit breaker and reject the rest of a large overnight window.
MAX_WORKERS="${MAX_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INTER_BATCH_DELAY_SECONDS="${INTER_BATCH_DELAY_SECONDS:-20}"

export ROPD_TEACHER_PROVIDER="${ROPD_TEACHER_PROVIDER:-anthropic}"
export ROPD_TEACHER_PROFILE="${ROPD_TEACHER_PROFILE:-claude_compass}"
export ROPD_TEACHER_MODEL="$TEACHER_MODEL"
export ROPD_TEACHER_ANSWER_COUNT="$TEACHER_ANSWER_COUNT"

if [[ -n "${ANTHROPIC_COMPASS_AUTH_TOKEN:-}" || -n "${ANTHROPIC_COMPASS_BASE_URL:-}" ]]; then
    export ANTHROPIC_PROFILE="${ANTHROPIC_PROFILE:-COMPASS}"
fi

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${ANTHROPIC_COMPASS_AUTH_TOKEN:-}" ]]; then
    read -rsp "ANTHROPIC_AUTH_TOKEN: " ANTHROPIC_AUTH_TOKEN
    echo
    export ANTHROPIC_AUTH_TOKEN
fi

test -s "$SOURCE_PARQUET"
test -f "$OUTPUT_INDEX"
mkdir -p "$(dirname "$PENDING_PARQUET")" "$(dirname "$OUTPUT_INDEX")" "$PROJECT_ROOT/outputs/teacher_index"

# Prefer the project virtual environment. This keeps the overnight job
# independent of whether the web shell has ~/.local/bin on PATH (where uv is
# normally installed). PYTHON_BIN can be overridden for a different venv.
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Create the project environment first, or pass PYTHON_BIN=/path/to/python." >&2
    exit 127
fi

LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/outputs/teacher_index/opus_math_window_${WINDOW_START}_${WINDOW_SIZE}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Source parquet: $SOURCE_PARQUET"
echo "Source window:  rows ${WINDOW_START}..$((WINDOW_END_EXCLUSIVE - 1))"
echo "Output index:   $OUTPUT_INDEX (append only)"
echo "Pending parquet:$PENDING_PARQUET"
echo "Teacher:        $ROPD_TEACHER_PROVIDER/$ROPD_TEACHER_PROFILE/$ROPD_TEACHER_MODEL"
echo "Answers/prompt: $TEACHER_ANSWER_COUNT"
echo "API workers:    $MAX_WORKERS"
echo "Prompt pacing:  ${INTER_BATCH_DELAY_SECONDS}s between batches of ${BATCH_SIZE}"
echo "Log file:       $LOG_FILE"

# Create a small pending parquet for only this window. UID de-duplication is
# intentionally performed here before calling the builder's stricter resume.
"$PYTHON_BIN" - "$SOURCE_PARQUET" "$OUTPUT_INDEX" "$PENDING_PARQUET" "$WINDOW_START" "$WINDOW_SIZE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

source_path = Path(sys.argv[1])
index_path = Path(sys.argv[2])
pending_path = Path(sys.argv[3])
window_start = int(sys.argv[4])
window_size = int(sys.argv[5])
window_end = window_start + window_size
if window_start < 0 or window_size < 1:
    raise ValueError("WINDOW_START must be >= 0 and WINDOW_SIZE must be >= 1")

existing_uids = set()
with index_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        uid = json.loads(line).get("uid")
        if uid is None:
            raise ValueError(f"Index row {line_number} has no uid")
        existing_uids.add(str(uid))

parquet = pq.ParquetFile(source_path)
total_rows = parquet.metadata.num_rows
if window_end > total_rows:
    raise ValueError(f"Requested rows ending at {window_end - 1}, source has {total_rows} rows")

batches = []
offset = 0
for batch in parquet.iter_batches(batch_size=65536):
    batch_end = offset + batch.num_rows
    left, right = max(window_start, offset), min(window_end, batch_end)
    if left < right:
        batches.append(batch.slice(left - offset, right - left))
    offset = batch_end
    if offset >= window_end:
        break
window = pa.Table.from_batches(batches)

def uid_of(row):
    for key in ("uid", "index"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("index") is not None:
        return str(extra["index"])
    raise ValueError("Source row has no uid, index, or extra_info.index")

mask = [uid_of(row) not in existing_uids for row in window.to_pylist()]
pending = window.filter(pa.array(mask))
pq.write_table(pending, pending_path)
print(
    f"Window rows={window.num_rows}; existing-index UID skips={window.num_rows - pending.num_rows}; "
    f"pending rows={pending.num_rows}; existing index UIDs={len(existing_uids)}"
)
PY

PENDING_COUNT="$("$PYTHON_BIN" - "$PENDING_PARQUET" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)"

if [[ "$PENDING_COUNT" == "0" ]]; then
    echo "No missing UIDs in this window. Next window: WINDOW_START=$WINDOW_END_EXCLUSIVE"
    exit 0
fi

"$PYTHON_BIN" scripts/build_teacher_index.py \
    --input "$PENDING_PARQUET" \
    --output "$OUTPUT_INDEX" \
    --teacher-answer-count "$TEACHER_ANSWER_COUNT" \
    --batch-size "$BATCH_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --inter-batch-delay-seconds "$INTER_BATCH_DELAY_SECONDS" \
    --resume

echo "Completed this window. Next window: WINDOW_START=$WINDOW_END_EXCLUSIVE"
