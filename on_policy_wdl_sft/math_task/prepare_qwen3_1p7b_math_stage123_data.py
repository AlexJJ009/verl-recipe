#!/usr/bin/env python3
"""Create immutable, disjoint MATH shards for Qwen3-1.7B cold-start and Stage123."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path("/data-1/dataset/math/train_rl_format.parquet")
DEFAULT_OUTPUT_ROOT = Path("/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719")
SPLIT_SIZES = {
    "cold_start": 1100,
    "stage1": 2560,
    "stage2": 1280,
    "stage3": 2560,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def verify_receipt(output_root: Path, *, expected_source: Path | None = None, expected_seed: int | None = None) -> dict:
    receipt_path = output_root / "dataset_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt["schema_version"] != 1:
        raise ValueError("unsupported dataset receipt schema")
    source_path = Path(receipt["source"])
    if expected_source is not None and source_path.resolve() != expected_source.resolve():
        raise ValueError(f"receipt source mismatch: {source_path} != {expected_source}")
    if expected_seed is not None and receipt["seed"] != expected_seed:
        raise ValueError(f"receipt seed mismatch: {receipt['seed']} != {expected_seed}")
    if not source_path.is_file() or sha256(source_path) != receipt["source_sha256"]:
        raise ValueError("source dataset identity mismatch")
    if receipt["source_rows"] != sum(SPLIT_SIZES.values()):
        raise ValueError("source row count does not match the frozen split contract")
    seen_indices: set[int] = set()
    primary_names = ("cold_start", "stage1", "stage2", "stage3")
    for name in primary_names:
        info = receipt["shards"][name]
        if info["rows"] != SPLIT_SIZES[name]:
            raise ValueError(f"shard row count does not match frozen contract: {name}")
        path = Path(info["path"])
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError(f"shard identity mismatch: {name}")
        frame = pd.read_parquet(path)
        indices = set(int(value) for value in frame["stage123_source_index"].tolist())
        if len(indices) != info["rows"] or seen_indices.intersection(indices):
            raise ValueError(f"shard overlap or duplicate source index: {name}")
        seen_indices.update(indices)
    if len(seen_indices) != receipt["source_rows"]:
        raise ValueError("shards do not cover the source dataset exactly once")
    stage2_indices = pd.read_parquet(receipt["shards"]["stage2"]["path"])["stage123_source_index"].tolist()
    stage3_indices = pd.read_parquet(receipt["shards"]["stage3"]["path"])["stage123_source_index"].tolist()
    control_info = receipt["shards"]["stage1_control"]
    if control_info["rows"] != SPLIT_SIZES["stage2"] + SPLIT_SIZES["stage3"]:
        raise ValueError("stage1_control row count does not match frozen contract")
    control_path = Path(control_info["path"])
    if not control_path.is_file() or sha256(control_path) != control_info["sha256"]:
        raise ValueError("stage1_control identity mismatch")
    control_indices = pd.read_parquet(control_path)["stage123_source_index"].tolist()
    if control_indices != stage2_indices + stage3_indices:
        raise ValueError("stage1_control must exactly concatenate stage2 then stage3")
    return receipt


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(
            json.dumps(
                verify_receipt(args.output_root, expected_source=args.source, expected_seed=args.seed),
                indent=2,
                sort_keys=True,
            )
        )
        return

    receipt_path = args.output_root / "dataset_receipt.json"
    if receipt_path.exists():
        print(
            json.dumps(
                verify_receipt(args.output_root, expected_source=args.source, expected_seed=args.seed),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial dataset directory: {args.output_root}")

    source = pd.read_parquet(args.source).reset_index(drop=True)
    if len(source) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected {sum(SPLIT_SIZES.values())} source rows, found {len(source)}")

    permutation = np.random.default_rng(args.seed).permutation(len(source)).tolist()
    args.output_root.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict] = {}
    offset = 0
    for name, size in SPLIT_SIZES.items():
        source_indices = permutation[offset : offset + size]
        offset += size
        shard = source.iloc[source_indices].copy().reset_index(drop=True)
        shard["stage123_source_index"] = source_indices
        shard["stage123_split"] = name
        shard["stage123_order"] = list(range(size))
        path = args.output_root / f"{name}.parquet"
        shard.to_parquet(path, index=False)
        shards[name] = {"path": str(path), "rows": len(shard), "sha256": sha256(path)}

    control = pd.concat(
        [pd.read_parquet(shards["stage2"]["path"]), pd.read_parquet(shards["stage3"]["path"])],
        ignore_index=True,
    )
    control_path = args.output_root / "stage1_control_stage2_then_stage3.parquet"
    control.to_parquet(control_path, index=False)
    shards["stage1_control"] = {
        "path": str(control_path),
        "rows": len(control),
        "sha256": sha256(control_path),
        "composition": ["stage2", "stage3"],
    }

    receipt = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "source_rows": len(source),
        "seed": args.seed,
        "ordering": "numpy.default_rng(seed).permutation; preserved within every shard",
        "shards": shards,
        "overlap_policy": {
            "cold_start_vs_stage1_stage2_stage3": "disjoint",
            "stage1_vs_stage2_vs_stage3": "pairwise_disjoint",
            "stage1_control": "intentional ordered reuse of stage2 then stage3 for matched control",
        },
    }
    (args.output_root / "dataset_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify_receipt(args.output_root, expected_source=args.source, expected_seed=args.seed)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
