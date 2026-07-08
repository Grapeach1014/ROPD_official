#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RAY_PREFIX_RE = re.compile(r"^\([^)]* pid=\d+\)\s*")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+(.*)$")
PAIR_RE = re.compile(r"([^\s:]+):([^\s]+)")

KEY_METRICS = [
    "train/reward_mean",
    "reward/student_score_mean",
    "reward/judge_error_rate",
    "reward/fallback_rate",
    "reward_quality/effective_uid_count",
    "reward_quality/group_excluded_count",
    "reward_quality/step_retry_exhausted",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "perf/max_memory_reserved_gb",
    "perf/time_per_step",
    "timing_s/gen",
    "timing_s/reward",
    "timing_s/update_actor",
    "actor/grad_norm",
    "actor/entropy",
]

TABLE_METRICS = [
    ("reward", "train/reward_mean"),
    ("student", "reward/student_score_mean"),
    ("judge_err", "reward/judge_error_rate"),
    ("fallback", "reward/fallback_rate"),
    ("eff_uid", "reward_quality/effective_uid_count"),
    ("resp_mean", "response_length/mean"),
    ("resp_max", "response_length/max"),
    ("mem_reserved_gb", "perf/max_memory_reserved_gb"),
    ("step_s", "perf/time_per_step"),
]


def strip_log_line(line: str) -> str:
    line = ANSI_RE.sub("", line.rstrip("\n"))
    line = RAY_PREFIX_RE.sub("", line)
    return line.replace("\r", "").strip()


def parse_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return int(value)
    return value


def parse_step_metrics(log_path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if not log_path.exists():
        return steps
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_log_line(raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        payload = match.group(2)
        metrics: dict[str, Any] = {"step": step}
        for key, raw_value in PAIR_RE.findall(payload):
            metrics[key] = parse_value(raw_value)
        steps.append(metrics)
    deduped: dict[int, dict[str, Any]] = {}
    for metrics in steps:
        deduped[int(metrics["step"])] = metrics
    return [deduped[key] for key in sorted(deduped)]


def read_header_sections(log_path: Path) -> tuple[str, dict[str, str]]:
    command = ""
    env: dict[str, str] = {}
    if not log_path.exists():
        return command, env
    section = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_log_line(raw_line)
        if line == "# Training Log":
            break
        if line in {"# Command", "# Environment"}:
            section = line
            continue
        if section == "# Command" and line:
            command = line
        elif section == "# Environment" and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return command, env


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def metric_series(steps: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for metrics in steps:
        value = metrics.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def summarize_metric(steps: list[dict[str, Any]], key: str) -> tuple[str, str, str]:
    values = metric_series(steps, key)
    if not values:
        return "-", "-", "-"
    return fmt(values[-1]), fmt(mean(values)), fmt(max(values))


def checkpoints(run_dir: Path, experiment: str | None) -> list[str]:
    project_root = None
    for candidate in [run_dir, *run_dir.parents]:
        if (candidate / "checkpoints").exists() and (candidate / "scripts").exists():
            project_root = candidate
            break
    if project_root is None:
        return []

    ckpt_root = project_root / "checkpoints" / "ropd"
    if experiment:
        ckpt_root = ckpt_root / experiment
    if not ckpt_root.exists():
        return []
    return [str(path) for path in sorted(ckpt_root.glob("global_step_*"))]


def last_relevant_lines(log_path: Path, limit: int = 40) -> list[str]:
    if not log_path.exists():
        return ["Log file was not found."]
    interesting = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_log_line(raw_line)
        if not line:
            continue
        lowered = line.lower()
        if (
            "error" in lowered
            or "traceback" in lowered
            or "exception" in lowered
            or "final validation" in lowered
            or "wandb sync" in lowered
            or "summary written" in lowered
            or "full log written" in lowered
        ):
            interesting.append(line)
    if interesting:
        return interesting[-limit:]
    lines = [strip_log_line(line) for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    return [line for line in lines if line][-limit:]


def diagnostics(steps: list[dict[str, Any]], status: str) -> list[str]:
    notes: list[str] = []
    if not steps:
        notes.append("No `step:N` metric lines were found in the log.")
        return notes
    last = steps[-1]
    judge_errors = metric_series(steps, "reward/judge_error_rate")
    fallback = metric_series(steps, "reward/fallback_rate")
    rewards = metric_series(steps, "train/reward_mean")
    grad_norm = metric_series(steps, "actor/grad_norm")
    if judge_errors and min(judge_errors) >= 1.0:
        notes.append("Rubricator/verifier judging failed for every recorded step (`reward/judge_error_rate=1.0`). The run completed, but the reward signal was entirely fallback/zero.")
    if fallback and min(fallback) >= 1.0:
        notes.append("Fallback reward was used for every recorded step (`reward/fallback_rate=1.0`).")
    if rewards and max(abs(value) for value in rewards) == 0:
        notes.append("Training reward stayed at 0.0 across all recorded steps.")
    if grad_norm and max(abs(value) for value in grad_norm) == 0:
        notes.append("Actor grad norm stayed at 0.0, consistent with zero rewards/advantages.")
    if isinstance(last.get("response_length/clip_ratio"), int | float) and float(last["response_length/clip_ratio"]) > 0:
        notes.append(f"Final step clipped {fmt(last['response_length/clip_ratio'])} of responses at the max response length.")
    if status != "succeeded":
        notes.append("Run did not finish successfully; inspect the log excerpt and full log path below.")
    if not notes:
        notes.append("No obvious runtime errors were detected in the selected metrics.")
    return notes


def write_markdown(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve()
    log_path = Path(args.log_path).resolve() if args.log_path else run_dir / "train.log"
    command, env = read_header_sections(log_path)
    steps = parse_step_metrics(log_path)
    created_at = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output).resolve() if args.output else run_dir / f"summary_{created_at}.md"
    status = args.status or ("succeeded" if str(args.exit_code) == "0" else "failed")
    experiment = env.get("EXPERIMENT")

    lines: list[str] = []
    lines.append(f"# ROPD Run Report ({created_at})")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Status: **{status}**")
    lines.append(f"- Exit code: `{args.exit_code}`")
    lines.append(f"- Run ID: `{env.get('RUN_ID', run_dir.name)}`")
    lines.append(f"- Experiment: `{experiment or '-'}`")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Full log: `{log_path}`")
    lines.append(f"- Report file: `{output}`")
    lines.append("")
    lines.append("## Key Takeaways")
    lines.append("")
    for note in diagnostics(steps, status):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    config_rows = [
        ("Student model", env.get("ROPD_MODEL_PATH")),
        ("Teacher index", env.get("ROPD_TEACHER_INDEX_PATH")),
        ("Teacher profile/model", f"{env.get('ROPD_TEACHER_PROFILE', '-') or '-'} / {env.get('ROPD_TEACHER_MODEL', '-') or '-'}"),
        ("Rubricator", f"{env.get('ROPD_RUBRICATOR_PROVIDER', '-') or '-'} / {env.get('ROPD_RUBRICATOR_MODEL', '-') or '-'}"),
        ("Verifier", f"{env.get('ROPD_VERIFIER_PROVIDER', '-') or '-'} / {env.get('ROPD_VERIFIER_MODEL', '-') or '-'}"),
        ("Train task", env.get("ROPD_TRAIN_TASK")),
        ("Train parquet", env.get("TRAIN_PARQUET")),
        ("Validation task", env.get("ROPD_VAL_TASK")),
        ("Train samples", env.get("TRAIN_SAMPLES")),
        ("Train batch size", env.get("TRAIN_BATCH_SIZE")),
        ("PPO mini batch size", env.get("PPO_MINI_BATCH_SIZE")),
        ("Training steps", env.get("TRAINING_STEPS")),
        ("Student rollout n", env.get("STUDENT_ROLLOUT_N")),
        ("Max response length", env.get("MAX_RESPONSE_LENGTH")),
        ("vLLM GPU memory utilization", env.get("VLLM_GPU_MEMORY_UTILIZATION")),
        ("vLLM max batched tokens", env.get("MAX_NUM_BATCHED_TOKENS")),
        ("AdamW foreach", env.get("ADAMW_FOREACH")),
    ]
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    for key, value in config_rows:
        lines.append(f"| {key} | `{value or '-'}` |")
    lines.append("")
    lines.append("## Step Metrics")
    lines.append("")
    if steps:
        lines.append("| step | " + " | ".join(label for label, _ in TABLE_METRICS) + " |")
        lines.append("|---:" + "|---:" * len(TABLE_METRICS) + "|")
        for metrics in steps:
            row = [str(metrics["step"])]
            row.extend(fmt(metrics.get(key)) for _, key in TABLE_METRICS)
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("No step metrics were found.")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | final | mean | max |")
    lines.append("|---|---:|---:|---:|")
    for key in KEY_METRICS:
        final, avg, max_value = summarize_metric(steps, key)
        lines.append(f"| `{key}` | {final} | {avg} | {max_value} |")
    lines.append("")
    ckpts = checkpoints(run_dir, experiment)
    lines.append("## Checkpoints")
    lines.append("")
    if ckpts:
        for path in ckpts:
            lines.append(f"- `{path}`")
    else:
        lines.append("No checkpoint directories were found for this experiment.")
    lines.append("")
    lines.append("## Command")
    lines.append("")
    lines.append("```bash")
    lines.append(command or "Command was not captured.")
    lines.append("```")
    lines.append("")
    lines.append("## Relevant Log Excerpt")
    lines.append("")
    lines.append("```text")
    lines.extend(last_relevant_lines(log_path))
    lines.append("```")
    lines.append("")
    lines.append("## Full Log")
    lines.append("")
    lines.append(f"See `{log_path}`.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest = run_dir / "summary_latest.md"
    latest.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a readable Markdown report from a ROPD train.log file.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing train.log.")
    parser.add_argument("--log-path", default=None, help="Explicit train.log path. Defaults to RUN_DIR/train.log.")
    parser.add_argument("--output", default=None, help="Output Markdown path. Defaults to RUN_DIR/summary_<timestamp>.md.")
    parser.add_argument("--timestamp", default=None, help="Timestamp string used in the report title/output filename.")
    parser.add_argument("--status", default=None, choices=["succeeded", "failed"], help="Run status.")
    parser.add_argument("--exit-code", default="0", help="Training process exit code.")
    args = parser.parse_args()
    output = write_markdown(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
