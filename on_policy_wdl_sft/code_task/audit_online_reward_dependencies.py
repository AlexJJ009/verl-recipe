#!/usr/bin/env python3
"""Audit online reward dependencies on train-smoke and official validation rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from recipe.on_policy_wdl_sft.code_task.code_extraction import wrap_reference_answer
from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import compute_score_code_official_aligned


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if limit > 0:
        df = df.head(limit)
    return df.to_dict("records")


def reference_solution(row: dict[str, Any]) -> str:
    gt = row["reward_model"]["ground_truth"]
    if isinstance(gt, str):
        gt = json.loads(gt)
    data_source = str(row["data_source"])
    if data_source in {"code_train", "code_val_smoke"}:
        return row.get("extra_info", {}).get("reference_answer", "")
    if data_source in {"HumanEval", "HumanEval+", "MBPP", "MBPP+"}:
        return gt["canonical_solution"]
    if data_source == "BigCodeBench":
        return f"def {gt['entry_point']}(*args, **kwargs):\n    return None"
    if data_source == "LiveCodeBench":
        return "print('definitely wrong')"
    raise ValueError(f"unsupported data_source: {data_source}")


def expected_reference_score(row: dict[str, Any]) -> float | None:
    data_source = str(row["data_source"])
    if data_source in {"code_train", "code_val_smoke", "HumanEval", "HumanEval+", "MBPP", "MBPP+"}:
        return 1.0
    if data_source in {"BigCodeBench", "LiveCodeBench"}:
        return 0.0
    return None


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    code = reference_solution(row)
    payload = compute_score_code_official_aligned(
        data_source=str(row["data_source"]),
        solution_str=wrap_reference_answer(code),
        ground_truth=row["reward_model"]["ground_truth"],
        extra_info=row.get("extra_info") or {},
    )
    payload["data_source"] = str(row["data_source"])
    payload["expected_score"] = expected_reference_score(row)
    payload["expected_score_match"] = payload["expected_score"] is None or payload["score"] == payload["expected_score"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("/data-1/dataset/code/verl_rl/reports/online_reward_dependency_audit.json"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "/data-1/dataset/code/verl_rl/code_val_smoke.parquet",
            "/data-1/dataset/code/verl_rl/official_humaneval_plus_val.parquet",
            "/data-1/dataset/code/verl_rl/official_mbpp_plus_val.parquet",
            "/data-1/dataset/code/verl_rl/official_bigcodebench_val.parquet",
            "/data-1/dataset/code/verl_rl/official_livecodebench_val.parquet",
        ],
    )
    args = parser.parse_args()

    results = []
    errors = []
    for dataset in args.datasets:
        path = Path(dataset)
        for idx, row in enumerate(load_rows(path, args.limit)):
            try:
                result = score_row(row)
                result["dataset_path"] = str(path)
                result["row_index"] = idx
                results.append(result)
            except Exception as exc:
                errors.append({"dataset_path": str(path), "row_index": idx, "error": repr(exc)})

    by_source: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["data_source"]].append(result)
    for source, rows in grouped.items():
        by_source[source] = {
            "count": len(rows),
            "statuses": dict(Counter(row["code_reward_status"] for row in rows)),
            "methods": dict(Counter(row["verification_method"] for row in rows)),
            "dependency_errors": sum(row.get("code_reward_dependency_error", 0) for row in rows),
            "runtime_errors": sum(row.get("code_reward_runtime_error", 0) for row in rows),
            "compile_errors": sum(row.get("code_reward_compile_error", 0) for row in rows),
            "timeouts": sum(row.get("code_reward_timeout", 0) for row in rows),
            "extraction_fails": sum(row.get("code_reward_extraction_fail", 0) for row in rows),
            "expected_score_mismatches": sum(not row["expected_score_match"] for row in rows),
        }

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": args.datasets,
        "limit": args.limit,
        "results_count": len(results),
        "errors": errors,
        "by_source": by_source,
        "ok": not errors
        and all(item["dependency_errors"] == 0 for item in by_source.values())
        and all(item["runtime_errors"] == 0 for item in by_source.values())
        and all(item["extraction_fails"] == 0 for item in by_source.values())
        and all(item["expected_score_mismatches"] == 0 for item in by_source.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
