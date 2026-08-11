#!/usr/bin/env python3
"""Build or verify the ordered 100-step Cold Start + GRPO dataset.

The existing Stage123 contract has 2,560 + 1,280 + 2,560 rows.  At a
64-prompt global batch this is exactly 100 optimizer steps.  This script
preserves the frozen stage order and adds one global ``grpo_order`` column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SHARDS = (("stage1", 2560), ("stage2", 1280), ("stage3", 2560))
TOTAL_ROWS = sum(rows for _, rows in SHARDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(dataset_root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    expected_columns: list[str] | None = None
    source_indices: list[int] = []
    for name, expected_rows in SHARDS:
        path = dataset_root / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if len(frame) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} rows, found {len(frame)}")
        columns = list(frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"{path}: column/order mismatch")
        if "stage123_source_index" not in frame:
            raise ValueError(f"{path}: missing stage123_source_index")
        source_indices.extend(int(value) for value in frame["stage123_source_index"])
        frames.append(frame)
        sources.append({"name": name, "path": str(path), "rows": len(frame), "sha256": sha256(path)})

    if len(set(source_indices)) != TOTAL_ROWS:
        raise ValueError("stage1/stage2/stage3 source indices overlap")
    merged = pd.concat(frames, ignore_index=True)
    merged.insert(0, "grpo_order", range(TOTAL_ROWS))
    return merged, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output or args.dataset_root / "cold_start_grpo_stage1_stage2_stage3.parquet"
    expected, sources = build(args.dataset_root)

    if args.verify_only:
        if not output.is_file():
            raise FileNotFoundError(output)
        actual = pd.read_parquet(output)
        if not actual.equals(expected):
            raise ValueError(f"{output}: content/order mismatch")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        expected.to_parquet(output, index=False)

    receipt = {
        "schema_version": 1,
        "output": str(output),
        "rows": len(expected),
        "prompt_batch_size": 64,
        "supported_steps": len(expected) // 64,
        "composition": [name for name, _ in SHARDS],
        "sources": sources,
        "sha256": sha256(output),
        "status": "verified",
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
