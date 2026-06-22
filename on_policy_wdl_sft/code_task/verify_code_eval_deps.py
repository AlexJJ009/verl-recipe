#!/usr/bin/env python3
"""Verify code-eval dependencies and record official harness status."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path("/data-1/dataset/code/verl_rl/reports/code_eval_deps_report.json")
LOCAL_DATA_ROOT = Path("/data-1/dataset/EnsembleLLM-data-processed")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def run_probe(name: str, code: str, timeout: float = 3.0) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "name": name,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - start, 3),
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-1000:],
    }


def timeout_probe() -> dict[str, Any]:
    start = time.time()
    try:
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": "timeout",
            "ok": True,
            "returncode": "timeout",
            "elapsed_sec": round(time.time() - start, 3),
            "stdout": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
        }
    return {"name": "timeout", "ok": False, "returncode": 0, "elapsed_sec": round(time.time() - start, 3)}


def binary_probe(name: str, args: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    info: dict[str, Any] = {"name": name, "path": path or "", "ok": bool(path)}
    if not path:
        return info
    proc = subprocess.run([path, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    info.update(
        {
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
        }
    )
    return info


def attempt_official_import(label: str, modules: list[str], packages: list[str]) -> dict[str, Any]:
    module_hits = {name: module_available(name) for name in modules}
    versions = {pkg: package_version(pkg) for pkg in packages}
    installed = any(module_hits.values())
    status = "official-installed" if installed else "official-unavailable"
    return {
        "label": label,
        "status": status,
        "modules": module_hits,
        "versions": versions,
        "attempted_command": "python import/package metadata probe",
        "source": packages,
        "exit_code": 0 if installed else 1,
        "stderr_excerpt": "" if installed else f"None of modules importable: {', '.join(modules)}",
    }


def evalplus_probe() -> dict[str, Any]:
    info = attempt_official_import("EvalPlus", ["evalplus"], ["evalplus"])
    info["never_official_from"] = [
        str(LOCAL_DATA_ROOT / "HumanEval/test_plus.jsonl"),
        str(LOCAL_DATA_ROOT / "MBPP/test_plus.jsonl"),
    ]
    if info["status"] == "official-installed":
        code = "import evalplus; print(getattr(evalplus, '__version__', 'unknown'))"
        info["import_probe"] = run_probe("evalplus-import", code)
    return info


def dataset_visibility() -> dict[str, Any]:
    paths = {
        "ensemble_train": LOCAL_DATA_ROOT / "train_rl_format.parquet",
        "humaneval_local": LOCAL_DATA_ROOT / "HumanEval/test.jsonl",
        "humaneval_plus_simplified_not_official": LOCAL_DATA_ROOT / "HumanEval/test_plus.jsonl",
        "mbpp_local": LOCAL_DATA_ROOT / "MBPP/test.jsonl",
        "mbpp_plus_simplified_not_official": LOCAL_DATA_ROOT / "MBPP/test_plus.jsonl",
        "bigcodebench_local": LOCAL_DATA_ROOT / "BigCodeBench/test.jsonl",
        "livecodebench_local": LOCAL_DATA_ROOT / "LiveCodeBench/test.jsonl",
        "code_train": Path("/data-1/dataset/code/code-train.jsonl"),
    }
    return {key: {"path": str(path), "exists": path.exists()} for key, path in paths.items()}


def write_report(report: Path, payload: dict[str, Any]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    tmp = report.with_suffix(report.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-evalplus", action="store_true", default=True)
    parser.add_argument("--no-require-evalplus", action="store_false", dest="require_evalplus")
    args = parser.parse_args()

    probes = [
        run_probe("pass", "assert 1 + 1 == 2; print('pass')"),
        run_probe("failing_assertion", "assert 1 + 1 == 3"),
        run_probe("import_failure", "import definitely_missing_code_eval_probe"),
        timeout_probe(),
    ]
    firejail_probe = binary_probe("firejail", ["--version"])
    system_pytest_path = Path("/usr/bin/python3")
    if system_pytest_path.is_file():
        proc = subprocess.run(
            [str(system_pytest_path), "-m", "pytest", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        system_pytest_probe = {
            "name": "system-pytest",
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
        }
    else:
        system_pytest_probe = {"name": "system-pytest", "ok": False, "path": str(system_pytest_path)}
    expected = {
        "pass": True,
        "failing_assertion": False,
        "import_failure": False,
        "timeout": True,
    }
    execution_ok = all((p["ok"] == expected[p["name"]]) for p in probes)

    harnesses = {
        "evalplus": evalplus_probe(),
        "bigcodebench": attempt_official_import(
            "BigCodeBench",
            ["bigcodebench", "bigcodebench.evaluate"],
            ["bigcodebench"],
        ),
        "livecodebench": attempt_official_import(
            "LiveCodeBench",
            ["lcb_runner", "livecodebench"],
            ["livecodebench", "lcb_runner"],
        ),
    }

    if harnesses["bigcodebench"]["status"] == "official-unavailable":
        harnesses["bigcodebench"]["fallback_status"] = "none-official-required"
    if harnesses["livecodebench"]["status"] == "official-unavailable":
        harnesses["livecodebench"]["fallback_status"] = "none-official-required"

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "install_command": "python3 -m pip install --cache-dir ${PIP_CACHE_DIR:-/data-1/pip_cache/code_eval} -r requirements-code-eval.txt",
        "pip_cache_dir": os.environ.get("PIP_CACHE_DIR", "/data-1/pip_cache/code_eval"),
        "sandbox_fusion_url_configured": bool(os.environ.get("SANDBOX_FUSION_URL")),
        "sandbox_fusion_url": os.environ.get("SANDBOX_FUSION_URL", ""),
        "harnesses": harnesses,
        "datasets": dataset_visibility(),
        "execution_smoke": probes,
        "execution_smoke_ok": execution_ok,
        "kodcode_official_sandbox": {
            "firejail": firejail_probe,
            "pytest_import": system_pytest_probe,
            "status": "official-ready" if firejail_probe["ok"] and system_pytest_probe["ok"] else "official-unavailable",
            "required": True,
        },
        "official_plus_boundary": {
            "simplified_test_plus_jsonl_is_official": False,
            "reason": "Local test_plus.jsonl files are never accepted as official EvalPlus+ scores.",
        },
    }
    payload["ok"] = (
        execution_ok
        and firejail_probe["ok"]
        and system_pytest_probe["ok"]
        and (not args.require_evalplus or harnesses["evalplus"]["status"] == "official-installed")
    )
    write_report(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
