#!/usr/bin/env python3
"""Probe that code_reward_* keys survive the validation JSONL dump shape."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED = [
    "code_reward_status",
    "code_reward_extraction_fail",
    "code_reward_compile_error",
    "code_reward_runtime_error",
    "code_reward_timeout",
    "code_reward_dependency_error",
    "code_reward_num_tests",
    "code_reward_num_passed",
]


def main() -> int:
    out_dir = Path("/data-1/dataset/code/verl_rl/reports/metadata_dump_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "0.jsonl"
    n = 1
    base_data = {
        "input": ["prompt"],
        "output": ["<answer>```python\ndef f():\n    return 1\n```</answer>"],
        "gts": ["{}"],
        "score": [1.0],
        "step": [0],
        "acc": [1.0],
        "pred": ["def f():\n    return 1"],
        "verification_method": ["local_exec"],
        "code_reward_status": ["pass"],
        "code_reward_extraction_fail": [0],
        "code_reward_compile_error": [0],
        "code_reward_runtime_error": [0],
        "code_reward_timeout": [0],
        "code_reward_dependency_error": [0],
        "code_reward_num_tests": [1],
        "code_reward_num_passed": [1],
    }
    lines = []
    for i in range(n):
        lines.append(json.dumps({k: v[i] for k, v in base_data.items()}, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    missing = [key for key in REQUIRED if key not in row]
    report = {"path": str(path), "required_keys": REQUIRED, "missing": missing, "ok": not missing}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
