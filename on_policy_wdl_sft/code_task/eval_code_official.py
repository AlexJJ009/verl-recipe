#!/usr/bin/env python3
"""Run official code benchmark evaluators on existing generated samples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def require_module(module: str, hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"Required official evaluator module is not importable: {module}. {hint}")


def copy_input(src: Path, output_dir: Path, suffix: str, overwrite: bool = False) -> Path:
    if not src.is_file():
        raise FileNotFoundError(f"official eval input file not found: {src}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{src.stem}.{suffix}{src.suffix}"
    if dst.resolve() != src.resolve():
        if overwrite and dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
    return dst


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": cmd,
        "cwd": str(cwd) if cwd else "",
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_excerpt": proc.stdout[-4000:],
        "stderr_excerpt": proc.stderr[-4000:],
    }


def run_command_to_log(
    cmd: list[str],
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run scorer without PIPE capture so child processes cannot hold fd EOF."""
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    excerpt = ""
    try:
        excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        excerpt = ""
    return {
        "command": cmd,
        "cwd": str(cwd) if cwd else "",
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - started, 3),
        "log_path": str(log_path),
        "log_excerpt": excerpt,
    }


def official_env() -> dict[str, str]:
    """Use the project-owned cache roots for official scorers."""
    env = os.environ.copy()
    project_cache_root = env.get("PROJECT_CACHE_ROOT", "/data-1/.cache")
    hf_home = env.get("HF_HOME", str(Path(project_cache_root) / "huggingface"))
    env["PROJECT_CACHE_ROOT"] = project_cache_root
    env["HF_HOME"] = hf_home
    env["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    env["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
    env["TRANSFORMERS_CACHE"] = hf_home
    env["XDG_CACHE_HOME"] = project_cache_root
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


def derived_jsonl_result(path: Path) -> Path:
    if path.name.endswith(".jsonl"):
        return path.with_name(path.name[:-6] + "_eval_results.json")
    return path.with_name(path.name + "_eval_results.json")


def summarize_evalplus(result_path: Path) -> dict[str, Any]:
    if not result_path.is_file():
        return {"result_path": str(result_path), "result_exists": False}
    data = json.loads(result_path.read_text())
    eval_data = data.get("eval", {})
    total_samples = 0
    base_sample_pass = 0
    plus_sample_pass = 0
    total_tasks = len(eval_data)
    base_task_pass = 0
    plus_task_pass = 0
    for rows in eval_data.values():
        base_task_pass += int(any(row.get("base_status") == "pass" for row in rows))
        plus_task_pass += int(any(row.get("plus_status") == "pass" for row in rows))
        for row in rows:
            total_samples += 1
            base_sample_pass += int(row.get("base_status") == "pass")
            plus_sample_pass += int(row.get("plus_status") == "pass")
    return {
        "result_path": str(result_path),
        "result_exists": True,
        "num_tasks": total_tasks,
        "num_samples": total_samples,
        "base_pass_rate": base_sample_pass / total_samples if total_samples else None,
        "plus_pass_rate": plus_sample_pass / total_samples if total_samples else None,
        "base_pass_at_n": base_task_pass / total_tasks if total_tasks else None,
        "plus_pass_at_n": plus_task_pass / total_tasks if total_tasks else None,
    }


def summarize_bcb(result_path: Path) -> dict[str, Any]:
    pass_at_k_path = result_path.with_name(result_path.name.replace("eval_results.json", "pass_at_k.json"))
    out: dict[str, Any] = {"result_path": str(result_path), "result_exists": result_path.is_file()}
    if pass_at_k_path.is_file():
        out["pass_at_k_path"] = str(pass_at_k_path)
        out["pass_at_k"] = json.loads(pass_at_k_path.read_text())
    return out


def summarize_lcb(custom_output: Path, scenario: str) -> dict[str, Any]:
    output_path = custom_output.with_name(custom_output.name[:-5] + f"_{scenario}_output.json")
    eval_path = output_path.with_name(output_path.name.replace(".json", "_eval.json"))
    out: dict[str, Any] = {
        "output_path": str(output_path),
        "eval_path": str(eval_path),
        "eval_exists": eval_path.is_file(),
    }
    if eval_path.is_file():
        out["metrics"] = json.loads(eval_path.read_text())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "bigcodebench", "livecodebench"], required=True)
    parser.add_argument("--samples", type=Path, help="Official samples JSONL for EvalPlus/BigCodeBench")
    parser.add_argument("--custom-output", type=Path, help="LiveCodeBench custom_output_file JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parallel", default=os.environ.get("CODE_OFFICIAL_EVAL_PARALLEL", "1"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bcb-split", default=os.environ.get("BCB_SPLIT", "complete"))
    parser.add_argument("--bcb-subset", default=os.environ.get("BCB_SUBSET", "full"))
    parser.add_argument("--bcb-calibrated", action=argparse.BooleanOptionalAction, default=os.environ.get("BCB_CALIBRATED", "0").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--bcb-override-path", type=Path, default=Path(os.environ.get("BIGCODEBENCH_OVERRIDE_PATH", "")))
    parser.add_argument("--lcb-repo", type=Path, default=Path(os.environ.get("LCB_REPO_DIR", "/data-1/code_eval_envs/LiveCodeBench")))
    parser.add_argument("--lcb-python", default=os.environ.get("LCB_PYTHON") or sys.executable)
    parser.add_argument("--lcb-release-version", default=os.environ.get("LCB_RELEASE_VERSION", "release_v1"))
    parser.add_argument("--lcb-scenario", default=os.environ.get("LCB_SCENARIO", "codegeneration"))
    parser.add_argument("--lcb-timeout", default=os.environ.get("LCB_TIMEOUT", "6"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark": args.benchmark,
        "official_only": True,
        "output_dir": str(args.output_dir),
    }

    if args.benchmark in {"humaneval", "mbpp"}:
        require_module("evalplus.evaluate", "Set PYTHONPATH to include /data-1/code_eval_envs/official_site.")
        if args.samples is None:
            raise RuntimeError(f"{args.benchmark} official eval requires --samples JSONL")
        samples = copy_input(args.samples, args.output_dir, args.benchmark, overwrite=args.overwrite)
        result_path = derived_jsonl_result(samples)
        if args.overwrite and result_path.exists():
            result_path.unlink()
        cmd = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            args.benchmark,
            f"--samples={samples}",
            f"--parallel={args.parallel}",
            "--test_details=True",
        ]
        payload["harness"] = "EvalPlus"
        payload["run"] = run_command(cmd, env=official_env())
        payload["summary"] = summarize_evalplus(result_path)

    elif args.benchmark == "bigcodebench":
        require_module("bigcodebench.evaluate", "Set PYTHONPATH to include /data-1/code_eval_envs/official_site.")
        if args.samples is None:
            raise RuntimeError("BigCodeBench official eval requires --samples JSONL")
        if not str(args.bcb_override_path):
            raise RuntimeError(
                "BigCodeBench official eval requires BIGCODEBENCH_OVERRIDE_PATH pointing to the project-owned official JSONL. "
                "Do not rely on Hugging Face Hub or external local caches."
            )
        if not args.bcb_override_path.is_file():
            raise FileNotFoundError(f"BigCodeBench official JSONL not found: {args.bcb_override_path}")
        samples = copy_input(args.samples, args.output_dir, "bigcodebench", overwrite=args.overwrite)
        result_path = derived_jsonl_result(samples)
        pass_at_k_path = result_path.with_name(result_path.name.replace("eval_results.json", "pass_at_k.json"))
        if args.overwrite:
            for path in (result_path, pass_at_k_path):
                if path.exists():
                    path.unlink()
        cmd = [
            sys.executable,
            "-m",
            "bigcodebench.evaluate",
            args.bcb_split,
            args.bcb_subset,
            f"--samples={samples}",
            "--execution=local",
            f"--parallel={args.parallel}",
            "--pass_k=1,3",
            f"--calibrated={str(args.bcb_calibrated)}",
            "--save_pass_rate=True",
        ]
        payload["harness"] = "BigCodeBench"
        payload["calibrated"] = bool(args.bcb_calibrated)
        payload["sample_format"] = "completion" if args.bcb_calibrated else "full_source"
        env = official_env()
        env["BIGCODEBENCH_OVERRIDE_PATH"] = str(args.bcb_override_path)
        payload["run"] = run_command_to_log(cmd, args.output_dir / "bigcodebench_official.log", env=env)
        payload["summary"] = summarize_bcb(result_path)

    else:
        if args.custom_output is None:
            raise RuntimeError("LiveCodeBench official eval requires --custom-output JSON")
        if not args.lcb_repo.is_dir():
            raise FileNotFoundError(f"LiveCodeBench repo not found: {args.lcb_repo}")
        if not Path(args.lcb_python).is_file():
            raise FileNotFoundError(f"LiveCodeBench python not found: {args.lcb_python}")
        custom_output = copy_input(args.custom_output, args.output_dir, "livecodebench", overwrite=args.overwrite)
        for suffix in (f"_{args.lcb_scenario}_output.json", f"_{args.lcb_scenario}_output_eval.json", f"_{args.lcb_scenario}_output_eval_all.json"):
            path = custom_output.with_name(custom_output.name[:-5] + suffix)
            if args.overwrite and path.exists():
                path.unlink()
        env = official_env()
        env["PYTHONPATH"] = f"{args.lcb_repo}:{env.get('PYTHONPATH', '')}"
        cmd = [
            args.lcb_python,
            "-m",
            "lcb_runner.runner.custom_evaluator",
            "--scenario",
            args.lcb_scenario,
            "--release_version",
            args.lcb_release_version,
            "--custom_output_file",
            str(custom_output),
            "--num_process_evaluate",
            str(args.parallel),
            "--timeout",
            str(args.lcb_timeout),
        ]
        payload["harness"] = "LiveCodeBench"
        payload["run"] = run_command(cmd, cwd=args.lcb_repo, env=env)
        payload["summary"] = summarize_lcb(custom_output, args.lcb_scenario)

    payload["ok"] = bool(payload.get("run", {}).get("ok"))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
