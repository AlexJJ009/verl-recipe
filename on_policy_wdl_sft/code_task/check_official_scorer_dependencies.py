#!/usr/bin/env python3
"""Validate the official code evaluators before calibration allocates GPUs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


REQUIRED_IMPORTS = (
    "evalplus.evaluate",
    "evalplus.gen.util",
    "evalplus.eval._special_oracle",
    "lcb_runner.benchmarks.code_generation",
    "lcb_runner.evaluation.compute_code_generation_metrics",
)


def validate(pythonpath: str) -> dict[str, object]:
    entries = [os.path.abspath(item) for item in pythonpath.split(os.pathsep) if item]
    required_paths = [
        "/workspace/verl",
        "/data-1/code_eval_envs/official_site",
        "/data-1/code_eval_envs/LiveCodeBench",
    ]
    missing_paths = [path for path in required_paths if path not in entries or not Path(path).exists()]
    if missing_paths:
        raise RuntimeError(f"official scorer PYTHONPATH is incomplete: {missing_paths}")
    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    imported = []
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
        imported.append(module_name)
    index = Path(os.environ.get("LCB_INPUT_OUTPUT_INDEX", ""))
    if not index.is_file():
        raise RuntimeError(f"LiveCodeBench input/output index missing: {index}")
    return {"ok": True, "pythonpath": entries, "imports": imported, "lcb_index": str(index.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH", ""))
    args = parser.parse_args()
    try:
        result = validate(args.pythonpath)
    except Exception as exc:
        print(json.dumps({"ok": False, "failure_class": "dependency_failure", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
