#!/usr/bin/env python3
"""Prepare KodCode multiturn SFT data for format cold start."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def normalize_messages(prompt: Any) -> list[dict[str, str]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        raise ValueError(f"prompt must be a list of chat messages, got {type(prompt).__name__}")
    messages: list[dict[str, str]] = []
    for item in prompt:
        if not isinstance(item, dict):
            raise ValueError(f"prompt item must be dict, got {type(item).__name__}")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unexpected prompt role: {role!r}")
        messages.append({"role": role, "content": content})
    return messages


def build_target(solution: str) -> str:
    return (
        "<think>We need provide executable Python code matching the requested function and tests.</think>\n"
        "<answer>\n"
        "```python\n"
        f"{solution.rstrip()}\n"
        "```\n"
        "</answer>"
    )


def convert(df: pd.DataFrame, seed: int, max_samples: int) -> pd.DataFrame:
    required = {"prompt", "extra_info"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input missing required columns: {missing}")

    if max_samples >= 0 and len(df) > max_samples:
        indices = list(range(len(df)))
        random.Random(seed).shuffle(indices)
        df = df.iloc[indices[:max_samples]].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        extra = row["extra_info"]
        if not isinstance(extra, dict):
            raise ValueError(f"row {idx}: extra_info must be dict")
        solution = extra.get("original_solution")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"row {idx}: missing extra_info.original_solution")

        messages = normalize_messages(row["prompt"])
        prompt_text = "\n".join(message["content"] for message in messages)
        if "\\boxed" in prompt_text or "boxed{" in prompt_text:
            raise ValueError(f"row {idx}: code prompt contains math boxed target")

        assistant = build_target(solution)
        if "<answer>" not in assistant or "```python" not in assistant:
            raise ValueError(f"row {idx}: assistant target failed code format check")
        rows.append(
            {
                "messages": [*messages, {"role": "assistant", "content": assistant}],
                "data_source": row.get("data_source", "kodcode_light_rl_10k"),
                "split": row.get("split", "train"),
                "extra_info": {
                    "source_index": int(idx),
                    "source_dataset": extra.get("source_dataset", "KodCode/KodCode-Light-RL-10K"),
                    "entry_point": extra.get("entry_point"),
                    "uid": extra.get("uid"),
                    "format_cold_start": "code-python-answer-v1",
                },
            }
        )
    return pd.DataFrame(rows)


def verify(df: pd.DataFrame) -> dict[str, Any]:
    if "messages" not in df.columns:
        raise ValueError("output must contain messages column")
    for idx, messages in enumerate(df["messages"].tolist()):
        messages = normalize_messages(messages)
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(f"row {idx}: last message must be assistant")
        assistant = messages[-1]["content"]
        if "<answer>" not in assistant or "```python" not in assistant:
            raise ValueError(f"row {idx}: assistant target must contain <answer> and ```python")
        joined = "\n".join(message["content"] for message in messages)
        if "\\boxed" in joined or "boxed{" in joined:
            raise ValueError(f"row {idx}: code SFT data contains boxed math notation")
    return {"rows": int(len(df)), "columns": list(df.columns)}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"ERROR: input parquet not found: {args.input}")

    if args.verify_only:
        if not args.output.is_file():
            raise SystemExit(f"ERROR: --verify-only output parquet not found: {args.output}")
        out_df = pd.read_parquet(args.output)
        stats = verify(out_df)
    else:
        src_df = pd.read_parquet(args.input)
        out_df = convert(src_df, args.seed, args.max_samples)
        stats = verify(out_df)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(args.output, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).as_posix(),
        "input": args.input.as_posix(),
        "output": args.output.as_posix(),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "verify_only": args.verify_only,
        "format": "multiturn_sft_messages",
        **stats,
    }
    write_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
