#!/usr/bin/env python3
"""Merge independently generated pass@k shards with exact-coverage checks.

Math shards are ``eval_details.parquet`` files produced by offline_eval.py.
Code shards are raw JSONL files produced by code_task/eval_code_vllm.py.
Every row must carry a stable prompt_id and a global sample_index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identify one dataset row even when a benchmark contains duplicate prompts."""
    if "dataset_path" in row and "dataset_row_index" in row:
        return (str(row["dataset_path"]), int(row["dataset_row_index"]), str(row["prompt_id"]))
    return (str(row["prompt_id"]),)


def _validate_coverage(rows: list[dict[str, Any]], expected_n: int) -> dict[str, Any]:
    required = {"prompt_id", "sample_index", "data_source"}
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for position, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {position} missing fields: {sorted(missing)}")
        grouped[_prompt_key(row)].append(row)

    expected = list(range(expected_n))
    failures = []
    for prompt_key, prompt_rows in grouped.items():
        indices = sorted(int(row["sample_index"]) for row in prompt_rows)
        if indices != expected:
            failures.append(
                {
                    "prompt_key": prompt_key,
                    "count": len(indices),
                    "missing": sorted(set(expected) - set(indices))[:20],
                    "duplicates": sorted({index for index in indices if indices.count(index) > 1})[:20],
                    "out_of_range": [index for index in indices if index < 0 or index >= expected_n][:20],
                }
            )
    if failures:
        raise ValueError(f"pass@k shard coverage failed for {len(failures)} prompts: {failures[:3]}")
    return {"prompt_count": len(grouped), "response_count": len(rows), "expected_n": expected_n}


def _response_diversity(rows: list[dict[str, Any]], text_key: str) -> dict[str, float]:
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows:
        text = str(row.get(text_key, "")).strip()
        grouped[_prompt_key(row)].append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    counts = [len(set(values)) for values in grouped.values()]
    n_values = [len(values) for values in grouped.values()]
    return {
        "mean_unique_response_count": sum(counts) / max(len(counts), 1),
        "mean_distinct_response_rate": sum(c / max(n, 1) for c, n in zip(counts, n_values)) / max(len(counts), 1),
    }


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    params = payload.get("generation_params", {})
    thinking = params.get("thinking_canary") or {}
    if params.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        raise ValueError(f"thinking was not explicitly enabled in {path}")
    if thinking.get("template_effect") is not True:
        raise ValueError(f"thinking template canary did not pass in {path}")
    ignored = {"n", "n_default", "n_per_dataset", "seed", "sample_offset"}
    return {
        "model": payload.get("model", payload.get("model_path")),
        "validation": payload.get("validation_parquet", params.get("n_per_dataset")),
        "params": {key: value for key, value in params.items() if key not in ignored},
    }


def _validate_contracts(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one --contract-summary is required")
    contracts = [_load_contract(path) for path in paths]
    first = contracts[0]
    for path, contract in zip(paths[1:], contracts[1:]):
        if contract != first:
            raise ValueError(f"generation contract mismatch in {path}: {contract} != {first}")
    return first


def merge_code(inputs: list[Path], output: Path, expected_n: int) -> dict[str, Any]:
    rows = [row for path in inputs for row in _jsonl(path)]
    coverage = _validate_coverage(rows, expected_n)
    rows.sort(key=lambda row: (int(row.get("task_index", 0)), int(row["sample_index"])))
    _write_jsonl(output, rows)
    return {**coverage, **_response_diversity(rows, "solution_str"), "output": str(output)}


def merge_math(inputs: list[Path], output: Path, expected_n: int) -> dict[str, Any]:
    import pandas as pd

    from recipe.joint_training.offline_eval import (
        compute_metrics_for_k,
        compute_shared_metrics,
        get_k_values,
    )

    frame = pd.concat([pd.read_parquet(path) for path in inputs], ignore_index=True)
    rows = frame.to_dict("records")
    coverage = _validate_coverage(rows, expected_n)
    prompt_columns = ["dataset_path", "dataset_row_index", "prompt_id"]
    frame = frame.sort_values(["data_source", *prompt_columns, "sample_index"], kind="stable")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    metrics: dict[str, Any] = {}
    for data_source, source_frame in frame.groupby("data_source", sort=True):
        entries = []
        for _, prompt_frame in source_frame.groupby(prompt_columns, sort=True):
            results = prompt_frame.to_dict("records")
            entries.append({"results": results})
        source_metrics = compute_shared_metrics(entries, expected_n)
        source_metrics["n_used"] = expected_n
        source_metrics["k_values"] = get_k_values(expected_n)
        for k in source_metrics["k_values"]:
            source_metrics.update(compute_metrics_for_k(entries, k))
        pred_counts = [
            len({str(value) for value in prompt_frame["pred"].dropna() if str(value) not in {"", "[NO_BOXED]"}})
            for _, prompt_frame in source_frame.groupby(prompt_columns, sort=True)
        ]
        source_metrics["mean_unique_prediction_count"] = sum(pred_counts) / max(len(pred_counts), 1)
        source_metrics[f"oracle_uplift_pass{expected_n}_minus_pass1"] = (
            source_metrics[f"pass@{expected_n}"] - source_metrics["pass@1"]
        )
        metrics[str(data_source)] = source_metrics

    metrics_path = output.with_name("eval_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "n": expected_n,
                "k_values": get_k_values(expected_n),
                "metrics": metrics,
                "coverage": coverage,
                "diversity": _response_diversity(rows, "response_text"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        **coverage,
        **_response_diversity(rows, "response_text"),
        "output": str(output),
        "metrics": str(metrics_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["math", "code"], required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--contract-summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=256)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if len(args.input) != len(args.contract_summary):
        parser.error("--input and --contract-summary must have the same number of paths")
    contract = _validate_contracts(args.contract_summary)
    if args.kind == "math":
        result = merge_math(args.input, args.output, args.expected_n)
    else:
        result = merge_code(args.input, args.output, args.expected_n)
    payload = {"kind": args.kind, "contract": contract, **result, "status": "pass"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
