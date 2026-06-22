#!/usr/bin/env python3
"""Create or verify a Stage2 non-overlap code train shard.

The shard reproduces the PPO train sampler used by Stage1:
``data.shuffle=True`` creates a torchdata RandomSampler backed by
``torch.randperm(len(dataset), generator.manual_seed(seed))``.  The Stage2 shard
then skips the prompts consumed before the chosen handoff step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer


DEFAULT_SOURCE = Path("/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet")
DEFAULT_OUTPUT = Path(
    "/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_handoff.parquet"
)
DEFAULT_MODEL = Path(
    "/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/"
    "snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--input", dest="source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--stage1-steps", type=int, default=110)
    parser.add_argument("--stage1-train-batch-size", type=int, default=64)
    parser.add_argument("--stage1-consumed-rows", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=40)
    parser.add_argument("--stage2-train-batch-size", type=int, default=64)
    parser.add_argument("--stage2-rows", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
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


def normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [dict(message) for message in prompt]


def prompt_len(tokenizer: Any, prompt: Any) -> int:
    ids = tokenizer.apply_chat_template(normalize_prompt(prompt), add_generation_prompt=True)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        ids = list(ids)
        if ids and isinstance(ids[0], Sequence) and not isinstance(ids[0], (str, bytes)):
            if len(ids) != 1:
                raise ValueError("expected a single sequence of chat template token ids")
            ids = list(ids[0])
    else:
        raise TypeError(f"unsupported chat template output: {type(ids)!r}")
    return len(ids)


def build_filtered_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if not args.source.is_file():
        raise FileNotFoundError(f"source parquet not found: {args.source}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"tokenizer/model path not found: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    df = pd.read_parquet(args.source).reset_index(drop=True)
    keep = [prompt_len(tokenizer, prompt) <= args.max_prompt_length for prompt in df[args.prompt_key]]
    return df.loc[keep].reset_index(drop=True)


def sampler_order(eligible_count: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(eligible_count, generator=generator).tolist()


def extra_info_uid(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    return str(extra_info.get("uid", extra_info.get("index")))


def resolve_counts(args: argparse.Namespace) -> tuple[int, int]:
    consumed = args.stage1_consumed_rows
    if consumed is None:
        consumed = args.stage1_steps * args.stage1_train_batch_size
    rows = args.stage2_rows
    if rows is None:
        rows = args.stage2_steps * args.stage2_train_batch_size
    return consumed, rows


def manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def make_manifest(
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    raw_source_row_count: int,
    consumed_rows: int,
    requested_rows: int,
    selected_positions: list[int],
    selected_uids: list[str],
    output_sha256: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_path": str(args.source),
        "source_sha256": sha256_file(args.source),
        "output_path": str(args.output),
        "output_sha256": output_sha256,
        "raw_source_row_count": raw_source_row_count,
        "eligible_row_count": len(dataframe),
        "selected_row_count": len(selected_positions),
        "stage1": {
            "seed": args.seed,
            "handoff_step": args.stage1_steps,
            "train_batch_size": args.stage1_train_batch_size,
            "consumed_rows": consumed_rows,
        },
        "stage2": {
            "train_batch_size": args.stage2_train_batch_size,
            "steps": args.stage2_steps,
            "rows_requested": requested_rows,
        },
        "filter_settings": {
            "prompt_key": args.prompt_key,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "truncation": args.truncation,
            "filter_overlong_prompts_workers": 1,
        },
        "sampler": {
            "implementation": "torchdata.stateful_dataloader.sampler.RandomSampler",
            "order": "torch.randperm(eligible_count, generator=torch.Generator().manual_seed(seed))",
            "seed": args.seed,
            "offset": consumed_rows,
            "length": len(selected_positions),
        },
        "selected_positions_first10": selected_positions[:10],
        "selected_positions_last10": selected_positions[-10:],
        "selected_extra_info_uid_first10": selected_uids[:10],
        "selected_extra_info_uid_last10": selected_uids[-10:],
        "prompt_template_version": "code-think-answer-python-v1",
        "command": " ".join(sys.argv),
        "repo_commit": get_git_commit(repo_root),
    }


def expected_selection(args: argparse.Namespace) -> tuple[pd.DataFrame, int, int, list[int]]:
    dataframe = build_filtered_dataframe(args)
    consumed_rows, requested_rows = resolve_counts(args)
    order = sampler_order(len(dataframe), args.seed)
    selected_positions = order[consumed_rows : consumed_rows + requested_rows]
    if len(selected_positions) != requested_rows:
        raise ValueError(
            f"not enough eligible rows after Stage1 prefix: need {requested_rows} rows after "
            f"offset {consumed_rows}, got {len(selected_positions)} of {len(dataframe)}"
        )
    return dataframe, consumed_rows, requested_rows, selected_positions


def verify(args: argparse.Namespace) -> dict[str, Any]:
    mpath = manifest_path(args.output)
    if not args.output.is_file():
        raise FileNotFoundError(f"missing shard: {args.output}")
    if not mpath.is_file():
        raise FileNotFoundError(f"missing manifest: {mpath}")

    manifest = json.loads(mpath.read_text())
    dataframe, consumed_rows, requested_rows, selected_positions = expected_selection(args)
    consumed_positions = sampler_order(len(dataframe), args.seed)[:consumed_rows]

    consumed_uids = {extra_info_uid(dataframe.iloc[int(pos)].to_dict()) for pos in consumed_positions}
    selected_uids = [extra_info_uid(dataframe.iloc[int(pos)].to_dict()) for pos in selected_positions]
    overlap = sorted(consumed_uids.intersection(selected_uids))
    if overlap:
        raise ValueError(f"Stage2 shard overlaps Stage1 consumed prefix: {overlap[:10]}")

    row_count = pq.ParquetFile(args.output).metadata.num_rows
    if row_count != requested_rows:
        raise ValueError(f"shard row count mismatch: {row_count} != {requested_rows}")
    if manifest.get("selected_extra_info_uid_first10") != selected_uids[:10]:
        raise ValueError("manifest first10 uid does not match regenerated sampler order")
    if manifest.get("selected_extra_info_uid_last10") != selected_uids[-10:]:
        raise ValueError("manifest last10 uid does not match regenerated sampler order")
    if manifest.get("stage1", {}).get("consumed_rows") != consumed_rows:
        raise ValueError("manifest stage1 consumed rows mismatch")
    actual_sha = sha256_file(args.output)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError(f"sha256 mismatch: manifest={manifest.get('output_sha256')} actual={actual_sha}")

    return {
        "status": "PASS",
        "output": str(args.output),
        "manifest": str(mpath),
        "row_count": row_count,
        "eligible_row_count": len(dataframe),
        "stage1_consumed_rows": consumed_rows,
        "stage2_rows": requested_rows,
        "overlap_count": 0,
        "first10": selected_uids[:10],
        "last10": selected_uids[-10:],
        "sha256": actual_sha,
    }


def create(args: argparse.Namespace) -> dict[str, Any]:
    raw_source_row_count = pq.ParquetFile(args.source).metadata.num_rows
    dataframe, consumed_rows, requested_rows, selected_positions = expected_selection(args)
    selected_dataset = dataframe.iloc[selected_positions].reset_index(drop=True)
    selected_uids = [extra_info_uid(row) for row in selected_dataset.to_dict(orient="records")]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_dataset.to_parquet(str(args.output), index=False)
    output_sha = sha256_file(args.output)
    mpath = manifest_path(args.output)
    manifest = make_manifest(
        args=args,
        dataframe=dataframe,
        raw_source_row_count=raw_source_row_count,
        consumed_rows=consumed_rows,
        requested_rows=requested_rows,
        selected_positions=selected_positions,
        selected_uids=selected_uids,
        output_sha256=output_sha,
    )
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify(args)


def main() -> int:
    args = parse_args()
    result = verify(args) if args.verify_only else create(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
