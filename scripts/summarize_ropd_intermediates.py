#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _truncate_text(value: Any, max_chars: int) -> str:
    if value is None:
        return "(missing)"
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + f"\n\n... [truncated, {len(value) - max_chars} chars omitted]"


def _fenced_text(value: Any, *, max_chars: int, language: str = "text") -> list[str]:
    return [f"```{language}", _truncate_text(value, max_chars), "```"]


def _select_records(rows: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_uid: set[str] = set()

    for row in rows:
        uid = str(row.get("uid") or "")
        if uid and uid in seen_uid:
            continue
        if uid:
            seen_uid.add(uid)
        selected.append(row)
        if len(selected) >= sample_count:
            break

    return selected


def build_report(
    *,
    debug_output_dir: Path,
    output: Path,
    sample_count: int,
    timestamp: str,
    max_answer_chars: int,
    max_rubric_chars: int,
) -> Path:
    artifacts_dir = debug_output_dir / "artifacts"
    trace_path = artifacts_dir / "pair_traces.jsonl"
    rubrics_dir = artifacts_dir / "rubrics"
    rows = _read_jsonl(trace_path)
    selected = _select_records(rows, sample_count)

    lines: list[str] = [
        f"# ROPD Intermediate Samples ({timestamp})",
        "",
        "This report is generated from ROPD text artifacts after training.",
        "",
        "## Source",
        "",
        f"- Debug output: `{debug_output_dir}`",
        f"- Pair trace: `{trace_path}`",
        f"- Total pair traces: {len(rows)}",
        f"- Samples shown: {len(selected)}",
        "",
    ]

    if not selected:
        lines.extend(
            [
                "## No Samples Found",
                "",
                "No pair traces were available. Make sure the run enables:",
                "",
                "```text",
                "+reward_model.reward_kwargs.ropd.debug.include_text_artifacts=true",
                "+reward_model.reward_kwargs.ropd.debug.text_artifact_mode=all_pairs",
                "```",
                "",
            ]
        )
    else:
        for idx, row in enumerate(selected, start=1):
            rubric_hash = row.get("rubric_hash")
            rubric_payload = None
            if rubric_hash:
                rubric_payload = _load_json(rubrics_dir / f"{rubric_hash}.json")

            lines.extend(
                [
                    f"## Sample {idx}",
                    "",
                    f"- UID: `{row.get('uid', '(missing)')}`",
                    f"- Pair index: `{row.get('pair_index', '(missing)')}`",
                    f"- Student score: `{row.get('student_score', '(missing)')}`",
                    f"- Teacher score: `{row.get('teacher_score', '(missing)')}`",
                    f"- Reward gap: `{row.get('reward_gap', '(missing)')}`",
                    f"- Student win: `{row.get('student_win', '(missing)')}`",
                    f"- Fallback used: `{row.get('fallback_used', '(missing)')}`",
                    f"- Judge error: `{row.get('judge_error', '(missing)')}`",
                    f"- Rubric hash: `{rubric_hash or '(missing)'}`",
                    "",
                    "### Student Answer",
                    "",
                    *_fenced_text(row.get("student_answer"), max_chars=max_answer_chars),
                    "",
                    "### Teacher Answer",
                    "",
                    *_fenced_text(row.get("teacher_answer"), max_chars=max_answer_chars),
                    "",
                    "### Rubric",
                    "",
                    *_fenced_text(rubric_payload or "(rubric json missing)", max_chars=max_rubric_chars, language="json"),
                    "",
                    "### Verifier Judgement",
                    "",
                    *_fenced_text(
                        {
                            "student_verifier_judgement": row.get("student_verifier_judgement"),
                            "teacher_verifier_judgement": row.get("teacher_verifier_judgement"),
                        },
                        max_chars=max_rubric_chars,
                        language="json",
                    ),
                    "",
                ]
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    latest = output.parent / "intermediate_samples_latest.md"
    latest.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ROPD student/rubric intermediate artifacts.")
    parser.add_argument("--debug-output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--max-answer-chars", type=int, default=3000)
    parser.add_argument("--max-rubric-chars", type=int, default=4000)
    args = parser.parse_args()

    output = Path(args.output)
    timestamp = args.timestamp or output.stem
    output_path = build_report(
        debug_output_dir=Path(args.debug_output_dir),
        output=output,
        sample_count=max(args.sample_count, 0),
        timestamp=timestamp,
        max_answer_chars=max(args.max_answer_chars, 256),
        max_rubric_chars=max(args.max_rubric_chars, 256),
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
