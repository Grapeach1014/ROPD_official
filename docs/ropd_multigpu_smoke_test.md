# ROPD four-GPU smoke test

## Purpose

This test runs the repository's normal `training/train.sh ->
verl.trainer.main_ppo -> RayPPOTrainer` path for one GRPO update.  It is an
integration test, not an evaluation of model quality.  It exercises FSDP actor
initialization, colocated vLLM rollout, offline teacher-index lookup, online
rubric generation, online verification, GRPO advantage calculation, and the
actor update.

## Design

`verl/trainer/config/ropd_smoke.yaml` composes `ropd.yaml`; it does not replace
trainer code.  The only scale reductions are four examples, two rollouts per
prompt, 256 generated tokens, a 768-token context window, and one training
step.  Four training GPUs remain intentional: FSDP/Ray placement is part of
the test.

The data file is `tmp/dapo-math-17k.train.teacher_index_aligned.smoke1.parquet`.
Its 800 rows were verified against `tmp/opus_teacher_index_math100_n4.jsonl`:
every row's `extra_info.index` and canonical prompt hash match an offline
teacher record.  The smoke configuration reads four rows; it therefore tests
the real `OfflineTeacherIndex` rather than a fallback or mock.

The judge models are selected through environment variables.  No credential or
model name is hard-coded in Python business logic or committed to this file.

## Prerequisites

Run this on the prepared four-H100 host from the repository root.  The project
environment must have been created with:

```bash
uv sync --extra math --extra gpu-generic
```

Set `ROPD_MODEL_PATH` to the local Qwen3-4B checkpoint and
`ROPD_TEACHER_INDEX_PATH` to the supplied index.  Export the Anthropic token
and endpoint by a secure mechanism; do not place credentials in `.env`, shell
history, or this report.

## Preflight: provider and index resolution

Use the following environment selections:

```bash
export ROPD_MODEL_PATH=/home/work/migoo_ai_public/ruiyu_lu/models/Qwen3-4B
export ROPD_TEACHER_INDEX_PATH="$PWD/tmp/opus_teacher_index_math100_n4.jsonl"
export ROPD_TEACHER_PROFILE=claude_compass
export ROPD_TEACHER_MODEL=claude-opus-4-6
export ROPD_RUBRICATOR_PROVIDER=anthropic
export ROPD_RUBRICATOR_PROFILE=claude_compass
export ROPD_RUBRICATOR_MODEL=claude-opus-4-6
export ROPD_VERIFIER_PROVIDER=anthropic
export ROPD_VERIFIER_PROFILE=claude_compass
export ROPD_VERIFIER_MODEL=claude-opus-4-6
```

`ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL` must already be exported.  The
teacher remains `offline_index`; assigning its `claude_compass` profile is
required only because the index fingerprint records the Anthropic API style,
base URL, model, and timeout.

Before the training run, inspect the resolved configuration without starting
Ray:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python -m verl.trainer.main_ppo \
  --config-name ropd_smoke --cfg job > /tmp/ropd_smoke_resolved.yaml
```

## Run

```bash
export EXPERIMENT=ropd-smoke-$(date +%Y%m%d-%H%M%S)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  timeout 45m bash training/train.sh --config-name ropd_smoke
```

`training/train.sh` forwards the Hydra override to the official entrypoint.
The expected terminal condition is exit status 0 after a single step.

## Evidence of a successful run

The log must show all of the following:

1. Ray starts and reserves four GPUs.
2. FSDP actor and vLLM rollout workers initialize.
3. The dataset reports four selected train samples.
4. ROPD reward debug output exists at
   `outputs/ropd_smoke/reward_debug/ropd_reward_debug.jsonl` and its group
   records have `final_status: "success"`, `fallback_used: false`, and a
   non-empty `rubric_hash`.
5. Trainer metrics include one training step and the process exits normally.

Inspect reward evidence without exposing prompts or credentials:

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path

path = Path('outputs/ropd_smoke/reward_debug/ropd_reward_debug.jsonl')
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert records, f'No reward records in {path}'
groups = records[-1]['groups']
assert groups, 'Reward record has no groups'
assert all(item['final_status'] == 'success' for item in groups), groups
assert all(not item['fallback_used'] for item in groups), groups
assert all(item['rubric_hash'] for item in groups), groups
print(f'PASS: {len(groups)} ROPD groups completed without fallback.')
PY
```

## Result record

At document creation time, environment and multi-GPU Ray worker scheduling
were validated separately.  Populate this section with the actual one-step
run's exit status and the relevant redacted log lines after executing the Run
command above.  A nonzero exit must be reported as a failed smoke test, with
the first failing module and log excerpt retained.
