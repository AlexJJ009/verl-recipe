#!/usr/bin/env python3
"""Create and verify fixed-order, disjoint KodCode shards for Code Stage123."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(
    "/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet"
)
DEFAULT_COLD_START = Path(
    "/data-1/dataset/code/format_cold_start_cotmask_v3_author_signature_v2/kodcode_light_sft_messages.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data-1/dataset/code/verl_rl/qwen3_1p7b_code_stage123_author_signature_v2_seed20260706"
)
SPLIT_SIZES = {
    "stage1": 2560,
    "stage2": 1280,
    "stage3": 2560,
}
DEFAULT_COLD_START_STEPS = 20
DEFAULT_COLD_START_BATCH_SIZE = 64


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
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--cold-start-file", type=Path, default=DEFAULT_COLD_START)
    parser.add_argument("--cold-start-steps", type=int, default=DEFAULT_COLD_START_STEPS)
    parser.add_argument("--cold-start-batch-size", type=int, default=DEFAULT_COLD_START_BATCH_SIZE)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def cold_start_source_indices(path: Path) -> set[int]:
    frame = pd.read_parquet(path, columns=["extra_info"])
    indices = {int(extra["source_index"]) for extra in frame["extra_info"]}
    if len(indices) != len(frame):
        raise ValueError("cold-start source indices are missing or duplicated")
    return indices


def consumed_cold_start_source_indices(path: Path, *, steps: int, batch_size: int) -> set[int]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("cold-start steps and batch size must be positive")
    frame = pd.read_parquet(path, columns=["extra_info"])
    consumed_rows = steps * batch_size
    if consumed_rows > len(frame):
        raise ValueError(
            f"cold-start consumed rows exceed dataset: {consumed_rows} > {len(frame)}"
        )
    indices = [int(extra["source_index"]) for extra in frame.iloc[:consumed_rows]["extra_info"]]
    if len(indices) != consumed_rows or len(set(indices)) != consumed_rows:
        raise ValueError("consumed cold-start source indices are missing or duplicated")
    return set(indices)


def verify_receipt(
    output_root: Path,
    *,
    expected_source: Path | None = None,
    expected_seed: int | None = None,
    expected_cold_start: Path = DEFAULT_COLD_START,
    expected_cold_start_steps: int = DEFAULT_COLD_START_STEPS,
    expected_cold_start_batch_size: int = DEFAULT_COLD_START_BATCH_SIZE,
) -> dict:
    receipt_path = output_root / "dataset_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != 2:
        raise ValueError("unsupported dataset receipt schema")
    source_path = Path(receipt["source"])
    if expected_source is not None and source_path.resolve() != expected_source.resolve():
        raise ValueError(f"receipt source mismatch: {source_path} != {expected_source}")
    if expected_seed is not None and receipt["seed"] != expected_seed:
        raise ValueError(f"receipt seed mismatch: {receipt['seed']} != {expected_seed}")
    if not source_path.is_file() or sha256(source_path) != receipt["source_sha256"]:
        raise ValueError("source dataset identity mismatch")
    if receipt["source_rows"] < sum(SPLIT_SIZES.values()):
        raise ValueError("source dataset is too small for the frozen split contract")
    if Path(receipt["cold_start_file"]).resolve() != expected_cold_start.resolve():
        raise ValueError("cold-start dataset path mismatch")
    if sha256(expected_cold_start) != receipt["cold_start_sha256"]:
        raise ValueError("cold-start dataset identity mismatch")
    if receipt.get("cold_start_steps") != expected_cold_start_steps:
        raise ValueError("cold-start step count mismatch")
    if receipt.get("cold_start_batch_size") != expected_cold_start_batch_size:
        raise ValueError("cold-start batch size mismatch")
    cold_start_indices = consumed_cold_start_source_indices(
        expected_cold_start,
        steps=expected_cold_start_steps,
        batch_size=expected_cold_start_batch_size,
    )
    if receipt.get("cold_start_rows_consumed") != len(cold_start_indices):
        raise ValueError("cold-start consumed-row count mismatch")

    seen_indices: set[int] = set()
    for name, expected_rows in SPLIT_SIZES.items():
        info = receipt["shards"][name]
        path = Path(info["path"])
        if info["rows"] != expected_rows or not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError(f"shard identity mismatch: {name}")
        frame = pd.read_parquet(path)
        indices = [int(value) for value in frame["stage123_source_index"].tolist()]
        if len(indices) != expected_rows or len(set(indices)) != expected_rows or seen_indices.intersection(indices):
            raise ValueError(f"shard overlap or duplicate source index: {name}")
        if cold_start_indices.intersection(indices):
            raise ValueError(f"cold-start overlap detected in shard: {name}")
        if frame["stage123_order"].tolist() != list(range(expected_rows)):
            raise ValueError(f"shard order drift: {name}")
        seen_indices.update(indices)

    stage2_indices = pd.read_parquet(receipt["shards"]["stage2"]["path"])["stage123_source_index"].tolist()
    stage3_indices = pd.read_parquet(receipt["shards"]["stage3"]["path"])["stage123_source_index"].tolist()
    control_info = receipt["shards"]["stage1_control"]
    control_path = Path(control_info["path"])
    if control_info["rows"] != SPLIT_SIZES["stage2"] + SPLIT_SIZES["stage3"]:
        raise ValueError("stage1_control row count mismatch")
    if not control_path.is_file() or sha256(control_path) != control_info["sha256"]:
        raise ValueError("stage1_control identity mismatch")
    if pd.read_parquet(control_path)["stage123_source_index"].tolist() != stage2_indices + stage3_indices:
        raise ValueError("stage1_control must concatenate stage2 then stage3")
    return receipt


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_receipt(args.output_root, expected_source=args.source, expected_seed=args.seed, expected_cold_start=args.cold_start_file, expected_cold_start_steps=args.cold_start_steps, expected_cold_start_batch_size=args.cold_start_batch_size), indent=2, sort_keys=True))
        return
    receipt_path = args.output_root / "dataset_receipt.json"
    if receipt_path.exists():
        print(json.dumps(verify_receipt(args.output_root, expected_source=args.source, expected_seed=args.seed, expected_cold_start=args.cold_start_file, expected_cold_start_steps=args.cold_start_steps, expected_cold_start_batch_size=args.cold_start_batch_size), indent=2, sort_keys=True))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial dataset directory: {args.output_root}")

    source = pd.read_parquet(args.source).reset_index(drop=True)
    if len(source) < sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected at least {sum(SPLIT_SIZES.values())} rows, found {len(source)}")
    cold_start_indices = consumed_cold_start_source_indices(
        args.cold_start_file,
        steps=args.cold_start_steps,
        batch_size=args.cold_start_batch_size,
    )
    eligible_indices = [index for index in range(len(source)) if index not in cold_start_indices]
    if len(eligible_indices) < sum(SPLIT_SIZES.values()):
        raise ValueError("not enough rows remain after excluding cold-start data")
    permutation = np.random.default_rng(args.seed).permutation(eligible_indices).tolist()
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
        "schema_version": 2,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "source_rows": len(source),
        "seed": args.seed,
        "cold_start_file": str(args.cold_start_file),
        "cold_start_sha256": sha256(args.cold_start_file),
        "cold_start_rows": len(cold_start_indices),
        "cold_start_steps": args.cold_start_steps,
        "cold_start_batch_size": args.cold_start_batch_size,
        "cold_start_rows_consumed": len(cold_start_indices),
        "cold_start_consumption_policy": "first steps*batch_size rows; SFT data shuffle disabled",
        "ordering": "numpy.default_rng(seed).permutation; preserved within every shard; runtime shuffle disabled",
        "shards": shards,
        "unused_source_rows": len(source) - sum(SPLIT_SIZES.values()),
        "overlap_policy": {
            "stage1_vs_stage2_vs_stage3": "pairwise_disjoint",
            "cold_start_vs_stage1_stage2_stage3": "pairwise_disjoint",
            "stage1_control": "intentional ordered reuse of stage2 then stage3 for matched control",
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify_receipt(
        args.output_root,
        expected_source=args.source,
        expected_seed=args.seed,
        expected_cold_start=args.cold_start_file,
        expected_cold_start_steps=args.cold_start_steps,
        expected_cold_start_batch_size=args.cold_start_batch_size,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
