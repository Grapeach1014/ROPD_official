#!/usr/bin/env python3
"""Create UID- and prompt-disjoint math splits aligned to an offline Teacher Index."""

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


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index_records = [json.loads(line) for line in args.teacher_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    index_by_uid = {str(record["uid"]): record for record in index_records}
    if len(index_by_uid) != len(index_records):
        raise ValueError("Teacher Index contains duplicate UIDs")

    input_rows = pq.read_table(args.input_parquet).to_pylist()
    rows_by_uid: dict[str, dict[str, Any]] = {}
    prompt_hash_by_uid: dict[str, str] = {}
    for row_number, row in enumerate(input_rows):
        extra_info = dict(row.get("extra_info") or {})
        uid = str(extra_info.get("index", ""))
        record = index_by_uid.get(uid)
        if record is None:
            raise ValueError(f"Row {row_number}: uid={uid!r} missing from Teacher Index")
        prompt_hash = hash_canonical_raw_prompt(row["prompt"])
        if prompt_hash != str(record["raw_prompt_hash"]):
            raise ValueError(f"Row {row_number}: uid={uid!r} prompt hash mismatches Teacher Index")
        if uid not in rows_by_uid:
            extra_info["index"] = uid
            selected = dict(row)
            selected["extra_info"] = extra_info
            rows_by_uid[uid] = selected
            prompt_hash_by_uid[uid] = prompt_hash

    all_uids = sorted(rows_by_uid)
    targets = {"train": round(len(all_uids) * 0.80), "val": round(len(all_uids) * 0.10), "test": 0}
    targets["test"] = len(all_uids) - targets["train"] - targets["val"]

    # The index has a few identical prompts with distinct UIDs. Keep each such
    # prompt group wholly in one split to prevent prompt-level leakage.
    uid_groups: dict[str, list[str]] = {}
    for uid in all_uids:
        uid_groups.setdefault(prompt_hash_by_uid[uid], []).append(uid)
    groups = list(uid_groups.values())
    random.Random(args.seed).shuffle(groups)
    split_uids = {name: [] for name in targets}
    for group in groups:
        eligible = [name for name in targets if len(split_uids[name]) + len(group) <= targets[name]]
        if not eligible:
            raise AssertionError("Cannot allocate prompt group under requested split ratios")
        name = max(eligible, key=lambda item: targets[item] - len(split_uids[item]))
        split_uids[name].extend(group)
    if {name: len(values) for name, values in split_uids.items()} != targets:
        raise AssertionError("Final split counts do not match targets")

    uid_sets = {name: set(values) for name, values in split_uids.items()}
    if any(uid_sets[a] & uid_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("UID leakage detected")
    for group in uid_groups.values():
        memberships = {name for name, values in uid_sets.items() if set(group) & values}
        if len(memberships) != 1:
            raise AssertionError("Prompt leakage detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, uids in split_uids.items():
        pq.write_table(pa.Table.from_pylist([rows_by_uid[uid] for uid in uids]), args.output_dir / f"{name}.parquet", compression="zstd")
        with (args.output_dir / f"teacher_index_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for uid in uids:
                handle.write(json.dumps(index_by_uid[uid], ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "ropd.math_split.v1",
        "seed": args.seed,
        "input_parquet": str(args.input_parquet),
        "teacher_index": str(args.teacher_index),
        "input_rows": len(input_rows),
        "teacher_index_records": len(index_records),
        "usable_unique_uids": len(all_uids),
        "duplicate_rows_removed": len(input_rows) - len(all_uids),
        "teacher_index_uids_without_parquet": len(index_records) - len(all_uids),
        "unique_prompt_hashes": len(uid_groups),
        "duplicate_prompt_hash_groups": sum(len(group) > 1 for group in uid_groups.values()),
        "identical_prompt_cross_split": False,
        "splits": {name: {"count": len(uids), "uid_sha256": digest(uids), "uids": uids} for name, uids in split_uids.items()},
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"usable_unique_uids": len(all_uids), "splits": {name: len(uids) for name, uids in split_uids.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
