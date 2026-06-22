#!/usr/bin/env python3
"""Convert code-train JSONL to verl RL parquet with code prompt contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROMPT_TEMPLATE_VERSION = "code-think-answer-python-v1"
SYSTEM_PROMPT = (
    "You are a code generation assistant. Think through the problem, then return executable Python code only in "
    "<answer> fenced with ```python ... ```."
)
CONTRACT_SUFFIX = (
    "\n\nUse this exact response format:\n"
    "<think>your concise reasoning</think>\n"
    "<answer>\n```python\n# executable solution\n```\n</answer>\n"
    "Do not use boxed final-answer notation."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_test_case(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def schema_name(test_case: Any) -> str:
    if isinstance(test_case, dict):
        if "inputs" in test_case and "outputs" in test_case:
            return "dict_inputs_outputs"
        if "entry_point" in test_case and "testcase" in test_case:
            return "dict_entry_point_testcase"
        if "ground_truth" in test_case and "style" in test_case:
            return "dict_prime"
        return "dict_other:" + ",".join(sorted(test_case.keys())[:5])
    if isinstance(test_case, list):
        return "list"
    return type(test_case).__name__


def first_user_content(prompt: list[dict[str, str]]) -> str:
    for msg in prompt:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def build_prompt(user_content: str) -> list[dict[str, str]]:
    # Strip old formatting instruction so the new contract is the only one.
    user_content = user_content.replace("Please reason step by step, and put your final solution code in a Python code block (```python ... ```).", "").strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content + CONTRACT_SUFFIX},
    ]


def convert(input_path: Path, output_path: Path, val_smoke_path: Path, humaneval_smoke_path: Path, max_val_rows: int) -> dict[str, Any]:
    records = []
    schema_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    raw_count = 0
    for idx, line in enumerate(input_path.open(encoding="utf-8")):
        if not line.strip():
            continue
        raw_count += 1
        obj = json.loads(line)
        tc = parse_test_case(obj["test_case"])
        schema_counts[schema_name(tc)] += 1
        source_counts[obj.get("source", "unknown")] += 1
        prompt = build_prompt(first_user_content(obj.get("prompt", [])))
        uid = hashlib.sha256(f"{idx}:{obj.get('source')}:{first_user_content(obj.get('prompt', []))}".encode()).hexdigest()[:24]
        records.append(
            {
                "data_source": "code_train",
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(tc, ensure_ascii=False)},
                "prompt": prompt,
                "split": "train",
                "extra_info": {
                    "uid": uid,
                    "raw_index": idx,
                    "source": obj.get("source"),
                    "reference_answer": obj.get("reference_answer", ""),
                    "test_case": json.dumps(tc, ensure_ascii=False),
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                },
            }
        )

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    smoke = df.head(max_val_rows).copy()
    smoke["data_source"] = "code_val_smoke"
    smoke["split"] = "validation"
    smoke.to_parquet(val_smoke_path, index=False)

    humaneval = smoke.head(min(16, len(smoke))).copy()
    humaneval["data_source"] = "HumanEval"
    humaneval.to_parquet(humaneval_smoke_path, index=False)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_path": str(input_path),
        "source_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_count": len(df),
        "raw_row_count": raw_count,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "test_case_schema_counts": dict(schema_counts),
        "source_counts": dict(source_counts),
        "validation_smoke_path": str(val_smoke_path),
        "humaneval_smoke_path": str(humaneval_smoke_path),
        "contract": "<think>...</think><answer>```python ... ```</answer>",
        "boxed_prompt": False,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/data-1/dataset/code/code-train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/data-1/dataset/code/verl_rl/code_train_rl_format.parquet"))
    parser.add_argument("--val-smoke-output", type=Path, default=Path("/data-1/dataset/code/verl_rl/code_val_smoke.parquet"))
    parser.add_argument("--humaneval-smoke-output", type=Path, default=Path("/data-1/dataset/code/verl_rl/humaneval_val_smoke.parquet"))
    parser.add_argument("--max-val-rows", type=int, default=64)
    args = parser.parse_args()
    manifest = convert(args.input, args.output, args.val_smoke_output, args.humaneval_smoke_output, args.max_val_rows)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
