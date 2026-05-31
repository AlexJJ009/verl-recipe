#!/usr/bin/env python3
"""Create and verify the Stage 2 non-overlap shard for staged v1.

The shard follows the contract in
docs/joint_training/plans/active/stage2_model2_rollout_fused_loss_fast_validation.md:
filter prompts as Stage 1 did, apply a torch randperm with the Stage 1 seed,
skip the first 150 * 64 eligible prompts, and write the next 75 * 64 prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.dataset.rl_dataset import RLHFDataset


DEFAULT_SOURCE = Path("/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet")
DEFAULT_OUTPUT = Path(
    "/data-1/dataset/EnsembleLLM-data-processed/staged_v1/"
    "stage2_after_s1_150steps_seed20260528_75steps.parquet"
)
DEFAULT_MODEL = Path(
    "/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/"
    "snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--offset", type=int, default=150 * 64)
    parser.add_argument("--length", type=int, default=75 * 64)
    parser.add_argument("--max-prompt-length", type=int, default=500)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--truncation", default="left")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_dataset(args: argparse.Namespace) -> RLHFDataset:
    if not args.source.is_file():
        raise FileNotFoundError(f"source parquet not found: {args.source}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"tokenizer/model path not found: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    config = OmegaConf.create(
        {
            "cache_dir": "~/.cache/verl/rlhf",
            "prompt_key": args.prompt_key,
            "max_prompt_length": args.max_prompt_length,
            "truncation": args.truncation,
            "filter_overlong_prompts": True,
            "shuffle": True,
            "seed": args.seed,
            # Keep this deterministic and avoid worker-order ambiguity.
            "filter_overlong_prompts_workers": 1,
        }
    )
    return RLHFDataset(
        data_files=str(args.source),
        tokenizer=tokenizer,
        config=config,
        processor=None,
        max_samples=-1,
    )


def deterministic_indices(eligible_count: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(eligible_count, generator=generator).tolist()


def extra_info_index(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    return str(extra_info.get("index"))


def make_manifest(
    args: argparse.Namespace,
    dataset: RLHFDataset,
    raw_source_row_count: int,
    selected_positions: list[int],
    selected_indices: list[str],
    output_sha256: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "source_path": str(args.source),
        "output_path": str(args.output),
        "raw_source_row_count": raw_source_row_count,
        "eligible_row_count": len(dataset.dataframe),
        "tokenizer_model_path": str(args.model_path),
        "filter_settings": {
            "prompt_key": args.prompt_key,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "truncation": args.truncation,
            "filter_overlong_prompts_workers": 1,
        },
        "permutation": {
            "implementation": "torch.randperm",
            "generator": "torch.Generator().manual_seed(seed)",
            "seed": args.seed,
            "offset": args.offset,
            "length": args.length,
            "stage1_consumed_prompts": args.offset,
            "stage2_start_offset": args.offset,
            "stage2_end_offset": args.offset + args.length,
        },
        "selected_row_count": len(selected_positions),
        "selected_positions_first10": selected_positions[:10],
        "selected_positions_last10": selected_positions[-10:],
        "selected_extra_info_index_first10": selected_indices[:10],
        "selected_extra_info_index_last10": selected_indices[-10:],
        "output_sha256": output_sha256,
        "command": " ".join(sys.argv),
        "repo_commit": get_git_commit(repo_root),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.output.is_file():
        raise FileNotFoundError(f"missing shard: {args.output}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    dataset = build_dataset(args)
    order = deterministic_indices(len(dataset.dataframe), args.seed)
    prefix_positions = order[: args.offset]
    selected_positions = order[args.offset : args.offset + args.length]

    if len(selected_positions) != args.length:
        raise ValueError(f"selected length mismatch: {len(selected_positions)} != {args.length}")

    prefix_indices = {extra_info_index(dataset.dataframe[int(pos)]) for pos in prefix_positions}
    selected_indices = [extra_info_index(dataset.dataframe[int(pos)]) for pos in selected_positions]
    overlap = sorted(prefix_indices.intersection(selected_indices))
    if overlap:
        raise ValueError(f"Stage 2 shard overlaps Stage 1 consumed prefix: {overlap[:10]}")

    import pyarrow.parquet as pq

    row_count = pq.ParquetFile(args.output).metadata.num_rows
    if row_count != args.length:
        raise ValueError(f"shard row count mismatch: {row_count} != {args.length}")

    if manifest.get("selected_extra_info_index_first10") != selected_indices[:10]:
        raise ValueError("manifest first10 extra_info.index does not match regenerated permutation")
    if manifest.get("selected_extra_info_index_last10") != selected_indices[-10:]:
        raise ValueError("manifest last10 extra_info.index does not match regenerated permutation")

    actual_sha = sha256_file(args.output)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError(f"sha256 mismatch: manifest={manifest.get('output_sha256')} actual={actual_sha}")

    return {
        "status": "PASS",
        "output": str(args.output),
        "manifest": str(manifest_path),
        "row_count": row_count,
        "eligible_row_count": len(dataset.dataframe),
        "overlap_count": 0,
        "first10": selected_indices[:10],
        "last10": selected_indices[-10:],
        "sha256": actual_sha,
    }


def create(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq

    raw_source_row_count = pq.ParquetFile(args.source).metadata.num_rows
    dataset = build_dataset(args)
    order = deterministic_indices(len(dataset.dataframe), args.seed)
    selected_positions = order[args.offset : args.offset + args.length]
    if len(selected_positions) != args.length:
        raise ValueError(
            f"not enough eligible rows: need end offset {args.offset + args.length}, "
            f"got {len(dataset.dataframe)}"
        )

    selected_dataset = dataset.dataframe.select(selected_positions)
    selected_indices = [extra_info_index(row) for row in selected_dataset]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_dataset.to_parquet(str(args.output))
    output_sha = sha256_file(args.output)
    manifest = make_manifest(args, dataset, raw_source_row_count, selected_positions, selected_indices, output_sha)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify(args)


def main() -> int:
    args = parse_args()
    result = verify(args) if args.verify_only else create(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
