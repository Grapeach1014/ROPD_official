#!/usr/bin/env python
from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from algo.ropd.client import build_ropd_judge_config, coerce_ropd_teacher_answer_count
from algo.ropd_clients import (
    ONLINE_ROLE_PROVIDERS,
    PROMPT_TEMPLATE_VERSION,
    AnthropicCompatibleProvider,
    BlackOPDClientError,
    OpenAICompatibleProvider,
    OpenAITeacherClient,
    _prepare_repo_environment,
)
from algo.ropd_pipeline import normalize_raw_prompt
from algo.ropd_teacher_index import (
    OFFLINE_TEACHER_INDEX_SCHEMA_VERSION,
    OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
    OfflineTeacherIndex,
    build_teacher_fingerprint_payload,
    hash_canonical_raw_prompt,
)

logger = logging.getLogger("ropd.build_teacher_index")


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    row_number: int
    uid: str
    raw_prompt: Any
    raw_prompt_hash: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.uid, self.raw_prompt_hash)


@dataclass(frozen=True, slots=True)
class DatasetScan:
    total_rows: int
    unique_keys: tuple[tuple[str, str], ...]
    duplicate_key_count: int
    duplicate_prompt_count: int


@dataclass(frozen=True, slots=True)
class BuildResult:
    generated_count: int
    failed_count: int


class ProgressBar:
    def __init__(self, *, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.enabled = enabled
        self.count = 0
        self._last_render = 0.0

    def update(self, increment: int = 1) -> None:
        self.count += increment
        if not self.enabled:
            return
        now = time.monotonic()
        if self.count < self.total and now - self._last_render < 0.1:
            return
        self._last_render = now
        self._render(final=self.count >= self.total)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._render(final=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, *, final: bool) -> None:
        if self.total <= 0:
            text = f"\r{self.label}: {self.count}"
        else:
            width = 28
            ratio = min(1.0, self.count / self.total)
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            text = f"\r{self.label}: [{bar}] {self.count}/{self.total} ({ratio * 100:5.1f}%)"
        sys.stderr.write(text)
        if final:
            sys.stderr.write(" " * 8)
        sys.stderr.flush()


def _positive_int(value: str) -> int:
    resolved_value = int(value)
    if resolved_value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return resolved_value


def _resolve_repo_path(path: str | Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return REPO_ROOT / resolved_path


def _default_input_path() -> Path:
    data_root = os.getenv("DATA_ROOT", "datasets/unified")
    train_task = os.getenv("ROPD_TRAIN_TASK", "math/dapo-math-17k")
    configured_path = _resolve_repo_path(Path(data_root) / train_task / "train.parquet")
    if configured_path.exists():
        return configured_path

    bundled_path = REPO_ROOT / "training" / "data" / "DAPO-Math-17k" / "data" / "dapo-math-17k.parquet"
    if data_root == "datasets/unified" and train_task == "math/dapo-math-17k" and bundled_path.exists():
        return bundled_path
    return configured_path


def _default_output_path() -> Path:
    configured_path = os.getenv("ROPD_TEACHER_INDEX_PATH")
    if configured_path is not None and configured_path.strip():
        return _resolve_repo_path(configured_path)

    data_root = os.getenv("DATA_ROOT", "datasets/unified")
    train_task = os.getenv("ROPD_TRAIN_TASK", "math/dapo-math-17k")
    return _resolve_repo_path(
        Path(data_root) / train_task / "artifacts" / "teacher_index" / "shared-teacher-index.jsonl"
    )


def _load_ropd_reward_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"ROPD config not found: {config_path}")
    raw_config = OmegaConf.load(config_path)
    ropd_config = OmegaConf.select(raw_config, "reward_model.reward_kwargs.ropd")
    if ropd_config is None:
        raise ValueError(f"{config_path} does not define reward_model.reward_kwargs.ropd")
    resolved_config = OmegaConf.to_container(ropd_config, resolve=True)
    if not isinstance(resolved_config, Mapping):
        raise ValueError("reward_model.reward_kwargs.ropd must resolve to a mapping")
    return dict(resolved_config)


def _build_script_ropd_config(args: argparse.Namespace) -> dict[str, Any]:
    ropd_config = _load_ropd_reward_config(_resolve_repo_path(args.config))
    provider_resolution = dict(ropd_config.get("provider_resolution") or {})
    provider_resolution["entrypoint"] = "build_teacher_index"
    provider_resolution["spec_path"] = str(
        _resolve_repo_path(provider_resolution.get("spec_path") or "verl/trainer/config/judge_providers.yaml")
    )
    ropd_config["provider_resolution"] = provider_resolution

    teacher_answer_count = coerce_ropd_teacher_answer_count(
        args.teacher_answer_count if args.teacher_answer_count is not None else ropd_config.get("teacher_answer_count"),
        default=4,
        field_name="--teacher-answer-count",
    )
    ropd_config["teacher_answer_count"] = teacher_answer_count
    return ropd_config


def _build_teacher_fingerprint(role_config: Any) -> dict[str, Any]:
    return build_teacher_fingerprint_payload(
        provider=role_config.provider,
        model=role_config.model,
        base_url=role_config.base_url,
        reasoning_effort=role_config.reasoning_effort,
        max_output_tokens=role_config.max_output_tokens,
        temperature=role_config.temperature,
        top_p=role_config.top_p,
        timeout_seconds=role_config.timeout_seconds,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )


def _resolve_uid(row: Mapping[str, Any]) -> str:
    for key in ("uid", "index"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)

    extra_info = row.get("extra_info")
    if isinstance(extra_info, Mapping):
        value = extra_info.get("index")
        if value is not None and str(value).strip():
            return str(value)

    raise ValueError("dataset row is missing a stable uid; expected uid, index, or extra_info.index")


def _resolve_raw_prompt(row: Mapping[str, Any], *, prompt_key: str) -> Any:
    if "raw_prompt" in row and row["raw_prompt"] is not None:
        return row["raw_prompt"]
    if prompt_key in row and row[prompt_key] is not None:
        return row[prompt_key]
    raise ValueError(f"dataset row is missing prompt field {prompt_key!r}")


def _iter_dataset_records(input_path: Path, *, batch_size: int, prompt_key: str) -> Any:
    parquet_file = pq.ParquetFile(input_path)
    row_number = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            row_number += 1
            if not isinstance(row, Mapping):
                raise TypeError(f"dataset row {row_number} is not a mapping")
            raw_prompt = normalize_raw_prompt(_resolve_raw_prompt(row, prompt_key=prompt_key))
            yield DatasetRecord(
                row_number=row_number,
                uid=_resolve_uid(row),
                raw_prompt=raw_prompt,
                raw_prompt_hash=hash_canonical_raw_prompt(raw_prompt),
            )


def _parquet_row_count(input_path: Path) -> int:
    return int(pq.ParquetFile(input_path).metadata.num_rows)


def _scan_dataset(input_path: Path, *, batch_size: int, prompt_key: str, progress: bool) -> DatasetScan:
    uid_to_prompt_hash: dict[str, str] = {}
    prompt_hash_to_uid: dict[str, str] = {}
    duplicate_prompt_hashes: set[str] = set()
    key_order: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_key_count = 0

    total_rows = _parquet_row_count(input_path)
    progress_bar = ProgressBar(total=total_rows, label="Scanning", enabled=progress)
    try:
        for record in _iter_dataset_records(input_path, batch_size=batch_size, prompt_key=prompt_key):
            existing_prompt_hash = uid_to_prompt_hash.get(record.uid)
            if existing_prompt_hash is not None and existing_prompt_hash != record.raw_prompt_hash:
                raise ValueError(
                    f"Found multiple raw_prompt values for uid={record.uid!r}: "
                    f"{existing_prompt_hash} and {record.raw_prompt_hash}."
                )
            uid_to_prompt_hash.setdefault(record.uid, record.raw_prompt_hash)

            first_uid_for_prompt = prompt_hash_to_uid.get(record.raw_prompt_hash)
            if first_uid_for_prompt is None:
                prompt_hash_to_uid[record.raw_prompt_hash] = record.uid
            elif first_uid_for_prompt != record.uid:
                duplicate_prompt_hashes.add(record.raw_prompt_hash)

            if record.key in seen_keys:
                duplicate_key_count += 1
            else:
                seen_keys.add(record.key)
                key_order.append(record.key)
            progress_bar.update()
    finally:
        progress_bar.finish()

    return DatasetScan(
        total_rows=total_rows,
        unique_keys=tuple(key_order),
        duplicate_key_count=duplicate_key_count,
        duplicate_prompt_count=len(duplicate_prompt_hashes),
    )


def _teacher_answer_count_from_row(row: Mapping[str, Any]) -> int:
    schema_version = row.get("schema_version")
    if schema_version == OFFLINE_TEACHER_INDEX_SCHEMA_VERSION:
        return 1
    raw_teacher_answers = row.get("teacher_answers")
    if isinstance(raw_teacher_answers, list):
        return len(raw_teacher_answers)
    return 0


def _load_completed_keys(
    output_path: Path,
    *,
    expected_fingerprint: Mapping[str, Any],
    min_answer_count: int,
) -> set[tuple[str, str]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    OfflineTeacherIndex.load(index_path=output_path, expected_fingerprint=expected_fingerprint)
    completed_keys: set[tuple[str, str]] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            answer_count = _teacher_answer_count_from_row(row)
            if answer_count < min_answer_count:
                raise ValueError(
                    f"Existing teacher index row {output_path}:{line_number} has {answer_count} teacher answers; "
                    f"requested at least {min_answer_count}. Use --overwrite to rebuild it."
                )
            completed_keys.add((str(row["uid"]), str(row["raw_prompt_hash"])))
    return completed_keys


def _prepare_output_path(output_path: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if output_path.exists() and output_path.stat().st_size > 0 and not resume and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --resume or --overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output_path.write_text("", encoding="utf-8")
    elif not output_path.exists():
        output_path.touch()


def _jsonl_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _build_index_row(
    record: DatasetRecord,
    *,
    teacher_client: OpenAITeacherClient,
    teacher_answer_count: int,
    teacher_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    raw_answers = teacher_client.generate_many(record.raw_prompt, uid=record.uid, count=teacher_answer_count)
    teacher_answers = tuple(str(answer).strip() for answer in raw_answers)
    if len(teacher_answers) != teacher_answer_count:
        raise ValueError(
            f"teacher.generate_many returned {len(teacher_answers)} answers for uid={record.uid!r}; "
            f"expected {teacher_answer_count}."
        )
    if any(not answer for answer in teacher_answers):
        raise ValueError(f"teacher.generate_many returned an empty answer for uid={record.uid!r}.")
    return {
        "schema_version": OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
        "uid": record.uid,
        "raw_prompt_hash": record.raw_prompt_hash,
        "teacher_answers": list(teacher_answers),
        "teacher_fingerprint": dict(teacher_fingerprint),
    }


def _failure_payload(record: DatasetRecord, exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "ropd.teacher_index.failure.v1",
        "uid": record.uid,
        "raw_prompt_hash": record.raw_prompt_hash,
        "row_number": record.row_number,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, BlackOPDClientError):
        payload.update(
            {
                "stage": exc.stage,
                "client_error_type": exc.error_type,
                "retriable": exc.retriable,
                "status_code": exc.status_code,
                "details": exc.details,
            }
        )
    return {key: value for key, value in payload.items() if value is not None}


def _process_record_batch(
    records: list[DatasetRecord],
    *,
    executor: ThreadPoolExecutor,
    teacher_client: OpenAITeacherClient,
    teacher_answer_count: int,
    teacher_fingerprint: Mapping[str, Any],
    output_handle: Any,
    failure_handle: Any,
    progress_bar: ProgressBar,
) -> BuildResult:
    futures = {
        executor.submit(
            _build_index_row,
            record,
            teacher_client=teacher_client,
            teacher_answer_count=teacher_answer_count,
            teacher_fingerprint=teacher_fingerprint,
        ): record
        for record in records
    }
    generated_count = 0
    failed_count = 0
    for future in as_completed(futures):
        record = futures[future]
        try:
            row = future.result()
        except Exception as exc:
            failure_handle.write(_jsonl_dumps(_failure_payload(record, exc)) + "\n")
            failure_handle.flush()
            failed_count += 1
            logger.error("Teacher generation failed for uid=%s: %s", record.uid, exc)
        else:
            output_handle.write(_jsonl_dumps(row) + "\n")
            output_handle.flush()
            generated_count += 1
        progress_bar.update()
    return BuildResult(generated_count=generated_count, failed_count=failed_count)


def _generate_missing_records(
    input_path: Path,
    *,
    prompt_key: str,
    batch_size: int,
    max_workers: int,
    pending_keys: set[tuple[str, str]],
    teacher_client: OpenAITeacherClient,
    teacher_answer_count: int,
    teacher_fingerprint: Mapping[str, Any],
    output_path: Path,
    failure_path: Path,
    progress: bool,
) -> BuildResult:
    generated_count = 0
    failed_count = 0
    submitted_keys: set[tuple[str, str]] = set()
    records: list[DatasetRecord] = []
    progress_bar = ProgressBar(total=len(pending_keys), label="Generating", enabled=progress)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        with output_path.open("a", encoding="utf-8") as output_handle, failure_path.open(
            "a", encoding="utf-8"
        ) as failure_handle:
            for record in _iter_dataset_records(input_path, batch_size=batch_size, prompt_key=prompt_key):
                if record.key not in pending_keys or record.key in submitted_keys:
                    continue
                submitted_keys.add(record.key)
                records.append(record)
                if len(records) < batch_size:
                    continue
                result = _process_record_batch(
                    records,
                    executor=executor,
                    teacher_client=teacher_client,
                    teacher_answer_count=teacher_answer_count,
                    teacher_fingerprint=teacher_fingerprint,
                    output_handle=output_handle,
                    failure_handle=failure_handle,
                    progress_bar=progress_bar,
                )
                generated_count += result.generated_count
                failed_count += result.failed_count
                records = []

            if records:
                result = _process_record_batch(
                    records,
                    executor=executor,
                    teacher_client=teacher_client,
                    teacher_answer_count=teacher_answer_count,
                    teacher_fingerprint=teacher_fingerprint,
                    output_handle=output_handle,
                    failure_handle=failure_handle,
                    progress_bar=progress_bar,
                )
                generated_count += result.generated_count
                failed_count += result.failed_count
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        progress_bar.finish()

    return BuildResult(generated_count=generated_count, failed_count=failed_count)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline ROPD teacher-response index.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input parquet dataset. Defaults to DATA_ROOT/ROPD_TRAIN_TASK/train.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to ROPD_TEACHER_INDEX_PATH.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Truncate an existing output JSONL before writing.")
    parser.add_argument(
        "--teacher-answer-count",
        type=_positive_int,
        default=None,
        help="Teacher answers to generate per prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=64,
        help="Dataset rows to scan or submit at a time.",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=4,
        help="Maximum concurrent dataset records to generate.",
    )
    parser.add_argument("--prompt-key", default="prompt", help="Dataset prompt column name.")
    parser.add_argument(
        "--config",
        default="verl/trainer/config/ropd.yaml",
        help="ROPD trainer config used for reward/provider defaults.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    progress_enabled = not args.no_progress

    _prepare_repo_environment()
    input_path = _resolve_repo_path(args.input) if args.input is not None else _default_input_path()
    output_path = _resolve_repo_path(args.output) if args.output is not None else _default_output_path()
    failure_path = output_path.with_suffix(output_path.suffix + ".failures.jsonl")

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    ropd_config = _build_script_ropd_config(args)
    judge_config = build_ropd_judge_config(ropd_config)
    teacher_answer_count = coerce_ropd_teacher_answer_count(
        judge_config.teacher_answer_count,
        default=4,
        field_name="--teacher-answer-count",
    )
    if judge_config.teacher.provider not in ONLINE_ROLE_PROVIDERS:
        raise ValueError(
            "scripts/build_teacher_index.py requires an online teacher provider; "
            f"resolved {judge_config.teacher.provider!r}."
        )

    teacher_fingerprint = _build_teacher_fingerprint(judge_config.teacher)
    logger.info("Input dataset: %s", input_path)
    logger.info("Output index: %s", output_path)
    logger.info("Teacher model: %s", judge_config.teacher.model)
    logger.info("Teacher answer count: %s", teacher_answer_count)

    if output_path.exists() and output_path.stat().st_size > 0 and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --resume or --overwrite.")

    completed_keys: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        completed_keys = _load_completed_keys(
            output_path,
            expected_fingerprint=teacher_fingerprint,
            min_answer_count=teacher_answer_count,
        )
        logger.info("Loaded %d completed index rows for resume.", len(completed_keys))

    scan = _scan_dataset(input_path, batch_size=args.batch_size, prompt_key=args.prompt_key, progress=progress_enabled)
    logger.info(
        "Dataset scan complete: unique_prompts=%d duplicate_keys=%d duplicate_prompts=%d",
        len(scan.unique_keys),
        scan.duplicate_key_count,
        scan.duplicate_prompt_count,
    )
    if scan.duplicate_prompt_count:
        logger.warning(
            "Detected %d prompt hash(es) attached to multiple uids; each uid will still receive its own index row.",
            scan.duplicate_prompt_count,
        )

    dataset_keys = set(scan.unique_keys)
    stale_completed_keys = completed_keys - dataset_keys
    if stale_completed_keys:
        logger.warning("Resume index contains %d key(s) not present in this dataset.", len(stale_completed_keys))
    pending_keys = dataset_keys - completed_keys

    _prepare_output_path(output_path, resume=args.resume, overwrite=args.overwrite)
    if args.overwrite and failure_path.exists():
        failure_path.write_text("", encoding="utf-8")
    elif not failure_path.exists():
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.touch()

    if not pending_keys:
        logger.info("No missing records to generate.")
        _load_completed_keys(
            output_path,
            expected_fingerprint=teacher_fingerprint,
            min_answer_count=teacher_answer_count,
        )
        return 0

    provider_cls = (
        AnthropicCompatibleProvider
        if judge_config.teacher.provider == "anthropic"
        else OpenAICompatibleProvider
    )
    provider = provider_cls(
        judge_config.transport,
        provider_limits=judge_config.provider_limits,
        request_scheduler_config=judge_config.request_scheduler,
    )
    teacher_client = OpenAITeacherClient(provider=provider, role_config=judge_config.teacher)
    try:
        result = _generate_missing_records(
            input_path,
            prompt_key=args.prompt_key,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            pending_keys=pending_keys,
            teacher_client=teacher_client,
            teacher_answer_count=teacher_answer_count,
            teacher_fingerprint=teacher_fingerprint,
            output_path=output_path,
            failure_path=failure_path,
            progress=progress_enabled,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted. Completed rows already flushed to %s; rerun with --resume.", output_path)
        return 130
    finally:
        provider.close()

    logger.info("Generated %d rows; %d rows failed.", result.generated_count, result.failed_count)
    _load_completed_keys(output_path, expected_fingerprint=teacher_fingerprint, min_answer_count=teacher_answer_count)
    if result.failed_count:
        logger.error("Some rows failed. See %s and rerun with --resume after fixing the issue.", failure_path)
        return 1
    logger.info("Teacher index validated successfully with OfflineTeacherIndex.load().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
