#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from algo.ropd_pipeline import normalize_raw_prompt
from algo.ropd_teacher_index import hash_canonical_raw_prompt


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
    raise ValueError("dataset row is missing uid/index/extra_info.index")


def _resolve_raw_prompt(row: Mapping[str, Any], *, prompt_key: str) -> Any:
    if row.get("raw_prompt") is not None:
        return row["raw_prompt"]
    if row.get(prompt_key) is not None:
        return row[prompt_key]
    raise ValueError(f"dataset row is missing prompt field {prompt_key!r}")


def _load_index_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        try:
            keys.add((str(row["uid"]), str(row["raw_prompt_hash"])))
        except KeyError as exc:
            raise ValueError(f"teacher index row {line_number} is missing {exc.args[0]!r}") from exc
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a parquet file to rows present in an offline teacher index.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--teacher-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--min-rows", type=int, default=1)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    index_path = args.teacher_index.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index_keys = _load_index_keys(index_path)
    table = pq.read_table(input_path)
    rows = table.to_pylist()
    keep_indices: list[int] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"parquet row {row_index + 1} is not a mapping")
        uid = _resolve_uid(row)
        raw_prompt = normalize_raw_prompt(_resolve_raw_prompt(row, prompt_key=args.prompt_key))
        key = (uid, hash_canonical_raw_prompt(raw_prompt))
        if key in index_keys:
            keep_indices.append(row_index)

    if len(keep_indices) < args.min_rows:
        raise ValueError(
            f"only {len(keep_indices)} rows in {input_path} are covered by {index_path}; "
            f"need at least {args.min_rows}"
        )

    filtered = table.take(pa.array(keep_indices, type=pa.int64()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, output_path)
    print(f"Wrote {len(keep_indices)} aligned rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
