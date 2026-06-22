#!/usr/bin/env python3
"""Run code reward/extraction verifier on sampled train rows."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from recipe.on_policy_wdl_sft.code_task.code_extraction import extraction_self_check, wrap_reference_answer
from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import compute_score_code_official_aligned


REQUIRED_KEYS = {
    "score",
    "acc",
    "code_reward_status",
    "code_reward_extraction_fail",
    "code_reward_compile_error",
    "code_reward_runtime_error",
    "code_reward_timeout",
    "code_reward_dependency_error",
    "code_reward_num_tests",
    "code_reward_num_passed",
    "code_reward_stderr_excerpt",
    "code_reward_sandbox",
    "pred",
    "verification_method",
    "official_aligned",
}


def load_rows(path: Path, sample_size: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    rows = [(idx, json.loads(line)) for idx, line in enumerate(path.open(encoding="utf-8")) if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:sample_size]


def first_user_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") or []
    for msg in prompt:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/data-1/dataset/code/code-train.jsonl"))
    parser.add_argument("--env-report", type=Path, default=Path("/data-1/dataset/code/verl_rl/reports/code_eval_deps_report.json"))
    parser.add_argument("--report", type=Path, default=Path("/data-1/dataset/code/verl_rl/reports/code_reward_env_report.json"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    extraction = extraction_self_check()
    extraction_ok = (
        extraction["answer_fenced_python"]["ok"]
        and extraction["fenced_python"]["ok"]
        and extraction["raw_python"]["ok"]
        and not extraction["missing_code"]["ok"]
        and not extraction["malformed_repeated"]["ok"]
    )

    rows = load_rows(args.input, args.sample_size, args.seed)
    reference_results = []
    wrong_results = []
    malformed_results = []
    missing_key_rows = []

    for idx, row in rows:
        gt = row.get("test_case")
        extra = {"prompt": first_user_prompt(row), "source_index": idx, "source": row.get("source")}
        ref = compute_score_code_official_aligned("code_val_smoke", wrap_reference_answer(row.get("reference_answer", "")), gt, extra)
        wrong = compute_score_code_official_aligned("code_val_smoke", "<answer>```python\ndef definitely_wrong(*args, **kwargs):\n    return None\n```</answer>", gt, extra)
        malformed = compute_score_code_official_aligned("code_val_smoke", "<answer>no executable code</answer>", gt, extra)
        for payload in (ref, wrong, malformed):
            missing = sorted(REQUIRED_KEYS - set(payload))
            if missing:
                missing_key_rows.append({"index": idx, "missing": missing, "payload": payload})
        reference_results.append({"index": idx, "source": row.get("source"), "status": ref["code_reward_status"], "score": ref["score"], "num_tests": ref["code_reward_num_tests"]})
        wrong_results.append({"index": idx, "status": wrong["code_reward_status"], "score": wrong["score"]})
        malformed_results.append({"index": idx, "status": malformed["code_reward_status"], "score": malformed["score"]})

    ref_pass = sum(1 for item in reference_results if item["score"] >= 1.0)
    wrong_fail = sum(1 for item in wrong_results if item["score"] <= 0.0)
    malformed_fail = sum(1 for item in malformed_results if item["status"] == "extraction_fail")
    sample_count = len(rows)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(args.input),
        "env_report": str(args.env_report),
        "env_report_exists": args.env_report.exists(),
        "sandbox_fusion_url_configured": bool(os.environ.get("SANDBOX_FUSION_URL")),
        "local_fallback_label": "smoke_only" if not os.environ.get("SANDBOX_FUSION_URL") else "",
        "extraction": extraction,
        "extraction_ok": extraction_ok,
        "required_metadata_keys": sorted(REQUIRED_KEYS),
        "missing_key_rows": missing_key_rows,
        "reference_results": reference_results,
        "wrong_results": wrong_results,
        "malformed_results": malformed_results,
        "reference_pass_rate": ref_pass / sample_count if sample_count else 0.0,
        "wrong_fail_rate": wrong_fail / sample_count if sample_count else 0.0,
        "malformed_extraction_fail_rate": malformed_fail / sample_count if sample_count else 0.0,
    }
    report["ok"] = (
        extraction_ok
        and not missing_key_rows
        and report["reference_pass_rate"] >= 0.90
        and report["wrong_fail_rate"] >= 0.90
        and report["malformed_extraction_fail_rate"] >= 1.0
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
