#!/usr/bin/env python3
"""Probe official code evaluator imports in the current Python environment."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys


def main() -> int:
    mods = [
        "evalplus",
        "bigcodebench",
        "lcb_runner",
        "lcb_runner.runner.custom_evaluator",
        "lcb_runner.evaluation.compute_code_generation_metrics",
    ]
    out = {"python": sys.executable, "pythonpath": os.environ.get("PYTHONPATH", ""), "modules": {}}
    for mod in mods:
        spec = importlib.util.find_spec(mod)
        out["modules"][mod] = {"available": spec is not None, "origin": getattr(spec, "origin", "") if spec else ""}
    probes = []
    for name, code in {
        "evalplus": "import evalplus; print(getattr(evalplus, '__version__', 'unknown'))",
        "bigcodebench": "import bigcodebench; print('ok')",
        "lcb": "import lcb_runner; print('ok')",
        "lcb_custom": "import lcb_runner.runner.custom_evaluator; print('ok')",
    }.items():
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        probes.append(
            {
                "name": name,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-500:],
                "stderr": proc.stderr[-2000:],
            }
        )
    out["probes"] = probes
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if all(p["returncode"] == 0 for p in probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
