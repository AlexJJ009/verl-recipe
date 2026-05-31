#!/usr/bin/env python3
"""Create a boxed-prompt EnsembleLLM train parquet for staged v1.

The source train set already has the same system prompt as validation, but most
training user prompts do not ask for a boxed final answer. This script appends
the validation-style boxed instruction to user prompts that do not already
contain a boxed-answer request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset


DEFAULT_SOURCE = Path("/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet")
DEFAULT_OUTPUT = Path(
    "/data-1/dataset/EnsembleLLM-data-processed/staged_v1/"
    "train_rl_format_boxed_prompt.parquet"
)
BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suffix", default=BOXED_SUFFIX)
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


def user_text(prompt: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in prompt if m.get("role") == "user")


def has_boxed_instruction(text: str) -> bool:
    return "\\boxed" in text


def append_suffix(text: str, suffix: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return suffix
    return stripped + "\n\n" + suffix


def transform_row(row: dict[str, Any], suffix: str) -> dict[str, Any]:
    prompt = []
    changed = False
    for message in row["prompt"]:
        item = dict(message)
        if item.get("role") == "user" and not has_boxed_instruction(str(item.get("content") or "")):
            item["content"] = append_suffix(str(item.get("content") or ""), suffix)
            changed = True
        prompt.append(item)
    row["prompt"] = prompt
    row["_boxed_prompt_appended"] = changed
    return row


def summarize(dataset: Dataset) -> dict[str, Any]:
    total = len(dataset)
    user_boxed = 0
    exact_suffix = 0
    system_think_answer = 0
    appended = 0
    for row in dataset:
        prompt = row["prompt"]
        text = user_text(prompt)
        user_boxed += int("\\boxed" in text)
        exact_suffix += int(BOXED_SUFFIX in text)
        appended += int(bool(row.get("_boxed_prompt_appended")))
        system = "\n".join(str(m.get("content") or "") for m in prompt if m.get("role") == "system")
        system_think_answer += int("<think>" in system and "<answer>" in system)
    return {
        "row_count": total,
        "user_boxed_count": user_boxed,
        "user_boxed_fraction": user_boxed / max(total, 1),
        "exact_suffix_count": exact_suffix,
        "exact_suffix_fraction": exact_suffix / max(total, 1),
        "system_think_answer_count": system_think_answer,
        "system_think_answer_fraction": system_think_answer / max(total, 1),
        # This is only populated before the temporary audit column is removed.
        "appended_count": appended,
        "appended_fraction": appended / max(total, 1),
    }


def write_manifest(args: argparse.Namespace, source_rows: int, summary: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    output_sha = sha256_file(args.output)
    manifest = {
        "source_path": str(args.source),
        "output_path": str(args.output),
        "source_row_count": source_rows,
        "output_sha256": output_sha,
        "boxed_suffix": args.suffix,
        "summary": summary,
        "command": " ".join(sys.argv),
        "repo_commit": get_git_commit(repo_root),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_dataset(path: Path) -> Dataset:
    return Dataset.from_parquet(str(path))


def create(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source.is_file():
        raise FileNotFoundError(f"source parquet not found: {args.source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source_rows = pq.ParquetFile(args.source).metadata.num_rows
    dataset = load_dataset(args.source)
    dataset = dataset.map(lambda row: transform_row(row, args.suffix), desc="append boxed prompt")
    summary = summarize(dataset)
    dataset = dataset.remove_columns(["_boxed_prompt_appended"])
    dataset.to_parquet(str(args.output))
    manifest = write_manifest(args, source_rows, summary)
    return verify(args, expected_manifest=manifest)


def verify(args: argparse.Namespace, expected_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.output.is_file():
        raise FileNotFoundError(f"missing boxed parquet: {args.output}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = expected_manifest or json.loads(manifest_path.read_text())
    actual_sha = sha256_file(args.output)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError(f"sha256 mismatch: manifest={manifest.get('output_sha256')} actual={actual_sha}")

    dataset = load_dataset(args.output)
    summary = summarize(dataset)
    if summary["user_boxed_count"] != summary["row_count"]:
        raise ValueError(f"user boxed coverage is not 100%: {summary}")
    source_rows = pq.ParquetFile(args.source).metadata.num_rows
    if pq.ParquetFile(args.output).metadata.num_rows != source_rows:
        raise ValueError("output row count differs from source row count")
    return {
        "status": "PASS",
        "output": str(args.output),
        "manifest": str(manifest_path),
        "sha256": actual_sha,
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    result = verify(args) if args.verify_only else create(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
