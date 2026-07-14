#!/usr/bin/env python3
"""Build UID-disjoint math train/validation/test splits aligned to a Teacher Index."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from algo.ropd_teacher_index import hash_canonical_raw_prompt


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_index = [json.loads(line) for line in args.teacher_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    index_by_hash: dict[str, dict[str, Any]] = {}
    for record in raw_index:
        raw_hash = str(record["raw_prompt_hash"])
        if raw_hash in index_by_hash:
            raise ValueError(f"Duplicate raw_prompt_hash in Teacher Index: {raw_hash}")
        index_by_hash[raw_hash] = record

    input_rows = pq.read_table(args.input_parquet).to_pylist()
    unique_rows: dict[str, dict[str, Any]] = {}
    remapped_source_uids = 0
    for row_number, row in enumerate(input_rows):
        raw_hash = hash_canonical_raw_prompt(row["prompt"])
        teacher_record = index_by_hash.get(raw_hash)
        if teacher_record is None:
            raise ValueError(f"Parquet row {row_number} has no matching Teacher Index raw_prompt_hash")

        teacher_uid = str(teacher_record["uid"])
        if teacher_uid in unique_rows:
            continue

        extra_info = dict(row.get("extra_info") or {})
        source_uid = str(extra_info.get("index", ""))
        if source_uid != teacher_uid:
            remapped_source_uids += 1
            extra_info["source_index"] = source_uid
        # verl promotes extra_info.index to the batch uid. It must equal the
        # offline Teacher Index uid for lookup and GRPO grouping to be correct.
        extra_info["index"] = teacher_uid

        selected = dict(row)
        selected["extra_info"] = extra_info
        unique_rows[teacher_uid] = selected

    usable_uids = sorted(unique_rows)
    shuffled_uids = usable_uids.copy()
    random.Random(args.seed).shuffle(shuffled_uids)
    total = len(shuffled_uids)
    train_count = round(total * 0.80)
    val_count = round(total * 0.10)
    split_uids = {
        "train": shuffled_uids[:train_count],
        "val": shuffled_uids[train_count : train_count + val_count],
        "test": shuffled_uids[train_count + val_count :],
    }
    if sum(len(values) for values in split_uids.values()) != total:
        raise AssertionError("Split sizes do not add up")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_uid_sets = {name: set(values) for name, values in split_uids.items()}
    if split_uid_sets["train"] & split_uid_sets["val"] or split_uid_sets["train"] & split_uid_sets["test"] or split_uid_sets["val"] & split_uid_sets["test"]:
        raise AssertionError("UID leakage across splits")

    for name, uids in split_uids.items():
        rows = [unique_rows[uid] for uid in uids]
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / f"{name}.parquet", compression="zstd")

        records = [index_by_hash[hash_canonical_raw_prompt(unique_rows[uid]["prompt"])] for uid in uids]
        if [str(record["uid"]) for record in records] != uids:
            raise AssertionError(f"Teacher Index UID ordering mismatch for {name}")
        with (args.output_dir / f"teacher_index_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "ropd.math_split.v1",
        "seed": args.seed,
        "input_parquet": str(args.input_parquet),
        "teacher_index": str(args.teacher_index),
        "input_rows": len(input_rows),
        "teacher_index_records": len(raw_index),
        "usable_unique_uids": total,
        "duplicate_rows_removed": len(input_rows) - total,
        "teacher_index_uids_without_parquet": len(raw_index) - total,
        "source_uid_remaps": remapped_source_uids,
        "splits": {
            name: {"count": len(uids), "uid_sha256": sha256_lines(uids), "uids": uids}
            for name, uids in split_uids.items()
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
