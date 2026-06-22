#!/usr/bin/env python3
"""Filter DeepCoder RL parquet files by Docker-runtime prompt token length."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from verl.utils.chat_template import normalize_chat_template_token_ids


def json_load_if_string(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    series = pd.Series(values)
    return {
        "min": int(series.min()),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": int(series.max()),
    }


def prompt_len(tokenizer: Any, prompt_value: Any) -> int:
    prompt = json_load_if_string(prompt_value)
    ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    return len(normalize_chat_template_token_ids(ids))


def source_counts(df: pd.DataFrame) -> dict[str, int]:
    if "data_source" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["data_source"].value_counts().sort_index().items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--prompt-key", default="prompt")
    args = parser.parse_args()

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    df = pd.read_parquet(args.input)
    lengths = [prompt_len(tokenizer, value) for value in df[args.prompt_key]]
    keep_mask = [length <= args.max_prompt_length for length in lengths]
    kept = df.loc[keep_mask].copy()
    dropped = df.loc[[not keep for keep in keep_mask]].copy()
    kept_lengths = [length for length, keep in zip(lengths, keep_mask, strict=True) if keep]
    dropped_lengths = [length for length, keep in zip(lengths, keep_mask, strict=True) if not keep]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(args.output, index=False)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(args.input),
        "output": str(args.output),
        "tokenizer_path": args.tokenizer_path,
        "max_prompt_length": args.max_prompt_length,
        "prompt_key": args.prompt_key,
        "input_rows": int(len(df)),
        "kept_rows": int(len(kept)),
        "dropped_rows": int(len(dropped)),
        "input_sha256": sha256_file(args.input),
        "output_sha256": sha256_file(args.output),
        "source_counts_input": source_counts(df),
        "source_counts_kept": source_counts(kept),
        "source_counts_dropped": source_counts(dropped),
        "prompt_tokens_input": quantiles(lengths),
        "prompt_tokens_kept": quantiles(kept_lengths),
        "prompt_tokens_dropped": quantiles(dropped_lengths),
        "dropped_examples_top20": [
            {
                "row_index": int(idx),
                "prompt_tokens": int(lengths[idx]),
                "data_source": str(df.iloc[idx].get("data_source", "")),
            }
            for idx in sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:20]
            if lengths[idx] > args.max_prompt_length
        ],
        "elapsed_sec": round(time.time() - started, 3),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
