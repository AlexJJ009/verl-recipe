#!/usr/bin/env python3
"""Create schema-aligned Math-7 validation parquet files for online validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import datasets
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("/data-1/dataset/math/qwen3_1p7b_math7_validation_v1")
DEFAULT_SOURCES = (
    Path("/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet"),
    Path("/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet"),
    Path("/data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet"),
    Path("/data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet"),
    Path("/data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet"),
    Path("/data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet"),
    Path("/data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet"),
)
REQUIRED_COLUMNS = ("data_source", "ability", "reward_model", "prompt", "split", "extra_info")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(source: Path) -> str:
    return source.name.removesuffix(".parquet") + "_schema_aligned.parquet"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns and column != "split"]
    if missing:
        raise ValueError(f"validation dataset is missing required columns: {missing}")
    normalized = frame.copy()
    if "split" not in normalized:
        normalized["split"] = "test"
    normalized["extra_info"] = normalized["extra_info"].map(
        lambda value: json.dumps(
            value if isinstance(value, dict) else {},
            sort_keys=True,
            ensure_ascii=False,
            default=json_default,
        )
    )
    for column in ("data_source", "ability", "split"):
        normalized[column] = normalized[column].astype(str)
    return normalized.loc[:, REQUIRED_COLUMNS]


def verify_receipt(output_root: Path, sources: tuple[Path, ...] = DEFAULT_SOURCES) -> dict:
    receipt_path = output_root / "dataset_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing Math-7 validation receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    expected_sources = [str(path) for path in sources]
    if receipt.get("sources") != expected_sources:
        raise ValueError("Math-7 validation receipt source order mismatch")
    if len(receipt.get("outputs", [])) != len(sources):
        raise ValueError("Math-7 validation receipt output count mismatch")
    output_paths = []
    for source, item in zip(sources, receipt["outputs"], strict=True):
        output = Path(item["path"])
        expected_output = output_root / normalized_name(source)
        if output != expected_output:
            raise ValueError(f"Math-7 validation output path mismatch: expected {expected_output}, got {output}")
        if item["source"] != str(source) or item["source_sha256"] != sha256(source):
            raise ValueError(f"Math-7 validation source identity mismatch: {source}")
        if not output.is_file() or item["sha256"] != sha256(output):
            raise ValueError(f"Math-7 validation output identity mismatch: {output}")
        frame = pd.read_parquet(output)
        if tuple(frame.columns) != REQUIRED_COLUMNS or frame["extra_info"].map(type).ne(str).any():
            raise ValueError(f"Math-7 validation output schema mismatch: {output}")
        if len(frame) != item["rows"]:
            raise ValueError(f"Math-7 validation row count mismatch: {output}")
        output_paths.append(str(output))
    loaded = [datasets.load_dataset("parquet", data_files=path, split="train") for path in output_paths]
    combined = datasets.concatenate_datasets(loaded)
    if len(combined) != receipt["total_rows"]:
        raise ValueError("Math-7 validation concatenation row count mismatch")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_receipt(args.output_root), indent=2, sort_keys=True))
        return
    receipt_path = args.output_root / "dataset_receipt.json"
    if receipt_path.exists():
        print(json.dumps(verify_receipt(args.output_root), indent=2, sort_keys=True))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial Math-7 validation directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    total_rows = 0
    for source in DEFAULT_SOURCES:
        frame = normalize_frame(pd.read_parquet(source))
        output = args.output_root / normalized_name(source)
        frame.to_parquet(output, index=False)
        outputs.append(
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "path": str(output),
                "sha256": sha256(output),
                "rows": len(frame),
            }
        )
        total_rows += len(frame)
    receipt = {
        "schema_version": 1,
        "normalization": "extra_info canonical JSON string; fixed required column order",
        "sources": [str(path) for path in DEFAULT_SOURCES],
        "outputs": outputs,
        "total_rows": total_rows,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify_receipt(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
