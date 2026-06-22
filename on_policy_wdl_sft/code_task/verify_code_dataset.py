#!/usr/bin/env python3
"""Verify code-task RL parquet schema, prompt contract, and manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def verify_prompt(prompt: Any) -> tuple[bool, str]:
    prompt = parse_json(prompt)
    if not isinstance(prompt, list):
        return False, "prompt_not_list"
    roles = [m.get("role") for m in prompt if isinstance(m, dict)]
    if roles != ["system", "user"]:
        return False, f"roles={roles}"
    joined = "\n".join(m.get("content", "") for m in prompt)
    if "\\boxed" in joined or "boxed{" in joined:
        return False, "boxed_prompt"
    if "<think>" not in joined or "<answer>" not in joined or "```python" not in joined:
        return False, "missing_code_contract"
    return True, "ok"


def uid_of_extra(extra: Any) -> str:
    extra = parse_json(extra)
    return str(extra["uid"])


def verify_file(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    required_cols = {"data_source", "ability", "reward_model", "prompt", "split", "extra_info"}
    missing_cols = sorted(required_cols - set(df.columns))
    prompt_failures = []
    schema_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for idx, row in df.iterrows():
        ok, reason = verify_prompt(row["prompt"])
        if not ok and len(prompt_failures) < 20:
            prompt_failures.append({"row": int(idx), "reason": reason})
        rm = parse_json(row["reward_model"])
        gt = parse_json(rm["ground_truth"])
        schema_counts[schema_name(gt)] += 1
        extra = parse_json(row["extra_info"])
        source_counts[str(extra.get("source", "unknown"))] += 1
    return {
        "path": str(path),
        "row_count": len(df),
        "columns_missing": missing_cols,
        "prompt_failures": prompt_failures,
        "test_case_schema_counts": dict(schema_counts),
        "source_counts": dict(source_counts),
        "uids": set(df["extra_info"].map(uid_of_extra)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("/data-1/dataset/code/verl_rl/code_train_rl_format.parquet"))
    parser.add_argument("--stage2", type=Path, default=Path("/data-1/dataset/code/verl_rl/code_stage2_after_s1_seed20260528.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("/data-1/dataset/code/code-train.jsonl"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    train = verify_file(args.train)
    stage2 = verify_file(args.stage2) if args.stage2.exists() else None
    raw_count = sum(1 for line in args.raw.open(encoding="utf-8") if line.strip())
    overlap = len(train["uids"] & stage2["uids"]) if stage2 else None
    stage2_manifest = args.stage2.with_suffix(".manifest.json")
    train_manifest = args.train.with_suffix(".manifest.json")
    report = {
        "raw_path": str(args.raw),
        "raw_row_count": raw_count,
        "train": {k: v for k, v in train.items() if k != "uids"},
        "stage2": {k: v for k, v in stage2.items() if k != "uids"} if stage2 else None,
        "train_manifest_exists": train_manifest.exists(),
        "stage2_manifest_exists": stage2_manifest.exists(),
        "stage2_overlap_with_train_full": overlap,
        "prompt_template_version": "code-think-answer-python-v1",
    }
    ok = (
        raw_count == 19457
        and train["row_count"] == raw_count
        and not train["columns_missing"]
        and not train["prompt_failures"]
        and (stage2 is None or (not stage2["columns_missing"] and not stage2["prompt_failures"]))
        and train_manifest.exists()
        and (stage2 is None or stage2_manifest.exists())
    )
    # Full-train overlap is expected because Stage2 is a shard from the same train file;
    # non-overlap against consumed Stage1 rows is recorded in the Stage2 manifest.
    if stage2_manifest.exists():
        manifest = json.loads(stage2_manifest.read_text(encoding="utf-8"))
        report["stage2_manifest_overlap_count"] = manifest.get("overlap_count")
        ok = ok and manifest.get("overlap_count") == 0
    report["ok"] = ok
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
