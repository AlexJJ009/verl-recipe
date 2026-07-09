#!/usr/bin/env python3
"""Prepare MATH multiturn SFT data for format cold start."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("/data-1/dataset/math/train_rl_format.parquet")


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


def clean_ground_truth(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def build_target(solution: str, ground_truth: str) -> str:
    return f"<think>\n{solution.strip()}\n</think>\n<answer>\\boxed{{{ground_truth}}}</answer>"


def convert(df: pd.DataFrame, seed: int, max_samples: int) -> pd.DataFrame:
    required = {"prompt", "extra_info", "reward_model"}
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
        reward_model = row["reward_model"]
        if not isinstance(extra, dict):
            raise ValueError(f"row {idx}: extra_info must be dict")
        if not isinstance(reward_model, dict):
            raise ValueError(f"row {idx}: reward_model must be dict")
        solution = extra.get("solution")
        ground_truth = clean_ground_truth(reward_model.get("ground_truth", ""))
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"row {idx}: missing extra_info.solution")
        if not ground_truth:
            raise ValueError(f"row {idx}: empty reward_model.ground_truth")

        assistant = build_target(solution, ground_truth)
        if "```python" in assistant:
            raise ValueError(f"row {idx}: math target must not contain fenced python")
        messages = normalize_messages(row["prompt"])
        rows.append(
            {
                "messages": [*messages, {"role": "assistant", "content": assistant}],
                "data_source": row.get("data_source", "ck46/hendrycks_math"),
                "split": row.get("split", "train"),
                "extra_info": {
                    "source_index": int(idx),
                    "source": extra.get("source"),
                    "subject": extra.get("subject"),
                    "level": extra.get("level"),
                    "format_cold_start": "math-boxed-answer-v1",
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
        if "<think>" not in assistant or "</think>" not in assistant or "<answer>\\boxed{" not in assistant:
            raise ValueError(f"row {idx}: assistant target must contain think and boxed answer")
        if "```python" in assistant:
            raise ValueError(f"row {idx}: math assistant target contains fenced python")
    return {"rows": int(len(df)), "columns": list(df.columns)}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(
            "ERROR: MATH train RL-format parquet not found: "
            f"{args.input}\n"
            "Fallback hint: first build /data-1/dataset/math/train_rl_format.parquet from the curated MATH "
            "train source; this script intentionally does not synthesize math SFT data from EnsembleLLM."
        )

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
