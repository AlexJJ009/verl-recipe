#!/usr/bin/env python3
"""Verify official-only code reward paths with reference and wrong answers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import compute_score_code_official_aligned


REPORT = Path("/data-1/dataset/code/verl_rl/reports/official_only_reward_report.json")


def first_row(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise RuntimeError(f"empty parquet: {path}")
    return df.iloc[0].to_dict()


def score(row: dict[str, Any], code: str) -> dict[str, Any]:
    return compute_score_code_official_aligned(
        data_source=row["data_source"],
        solution_str=f"<answer>\n```python\n{code}\n```\n</answer>",
        ground_truth=row["reward_model"]["ground_truth"],
        extra_info=row.get("extra_info") or {},
    )


def main() -> int:
    rows = {
        "humaneval_plus": first_row(Path("/data-1/dataset/code/verl_rl/official_humaneval_plus_val.parquet")),
        "mbpp_plus": first_row(Path("/data-1/dataset/code/verl_rl/official_mbpp_plus_val.parquet")),
        "bigcodebench": first_row(Path("/data-1/dataset/code/verl_rl/official_bigcodebench_val.parquet")),
        "livecodebench": first_row(Path("/data-1/dataset/code/verl_rl/official_livecodebench_val.parquet")),
    }
    he_gt = json.loads(rows["humaneval_plus"]["reward_model"]["ground_truth"])
    mbpp_gt = json.loads(rows["mbpp_plus"]["reward_model"]["ground_truth"])
    bcb_gt = json.loads(rows["bigcodebench"]["reward_model"]["ground_truth"])

    results = {
        "humaneval_ref": score(rows["humaneval_plus"], he_gt["canonical_solution"]),
        "humaneval_wrong": score(rows["humaneval_plus"], f"def {he_gt['entry_point']}(*args, **kwargs):\n    return None"),
        "mbpp_ref": score(rows["mbpp_plus"], mbpp_gt["canonical_solution"]),
        "mbpp_wrong": score(rows["mbpp_plus"], f"def {mbpp_gt['entry_point']}(*args, **kwargs):\n    return None"),
    }
    # BigCodeBench canonical solution is not copied into validation ground_truth
    # to avoid training-time answer leakage; verify wrong-path and evaluator import.
    results["bigcodebench_wrong"] = score(rows["bigcodebench"], f"def {bcb_gt['entry_point']}(*args, **kwargs):\n    return None")
    results["livecodebench_wrong"] = score(rows["livecodebench"], "print('definitely wrong')")

    ok = (
        results["humaneval_ref"]["score"] == 1.0
        and results["humaneval_wrong"]["score"] == 0.0
        and results["humaneval_ref"]["verification_method"] == "evalplus"
        and results["mbpp_ref"]["score"] == 1.0
        and results["mbpp_wrong"]["score"] == 0.0
        and results["mbpp_ref"]["verification_method"] == "evalplus"
        and results["bigcodebench_wrong"]["score"] == 0.0
        and results["bigcodebench_wrong"]["verification_method"] == "bigcodebench"
        and results["livecodebench_wrong"]["score"] == 0.0
        and results["livecodebench_wrong"]["verification_method"] == "livecodebench"
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ok": ok,
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
