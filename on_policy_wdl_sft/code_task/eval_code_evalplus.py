#!/usr/bin/env python3
"""Deprecated metadata probe; use eval_code_official.py for official scores."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tiny-smoke", action="store_true")
    args = parser.parse_args()
    official = importlib.util.find_spec("evalplus") is not None
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark": args.benchmark,
        "harness": "EvalPlus",
        "harness_status": "official-installed" if official else "official-unavailable",
        "harness_path": "python:evalplus" if official else "",
        "samples": str(args.samples) if args.samples else "",
        "tiny_smoke": args.tiny_smoke,
        "simplified_test_plus_jsonl_used_as_official": False,
        "pass_at_1": None,
        "num_tasks": 0,
        "extraction_fail_rate": 0.0,
        "compile_error_rate": 0.0,
        "runtime_error_rate": 0.0,
        "timeout_rate": 0.0,
        "dependency_error_rate": 0.0,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if official or args.tiny_smoke else 1


if __name__ == "__main__":
    raise SystemExit(main())
