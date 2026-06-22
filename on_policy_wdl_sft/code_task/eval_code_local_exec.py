#!/usr/bin/env python3
"""Local diagnostic runner for BigCodeBench/LiveCodeBench when official harness is unavailable."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["bigcodebench", "livecodebench"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=5)
    args = parser.parse_args()
    rows = []
    if args.dataset.exists():
        with args.dataset.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= args.max_tasks:
                    break
                if line.strip():
                    rows.append(json.loads(line))
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark": args.benchmark,
        "harness_status": "local-runner",
        "harness_path": str(Path(__file__).resolve()),
        "dataset": str(args.dataset),
        "num_tasks": len(rows),
        "pass_at_1": None,
        "extraction_fail_rate": None,
        "compile_error_rate": None,
        "runtime_error_rate": None,
        "timeout_rate": None,
        "dependency_error_rate": None,
        "label_warning": "Diagnostic only; not official benchmark score.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
