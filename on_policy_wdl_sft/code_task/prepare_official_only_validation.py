#!/usr/bin/env python3
"""Prepare validation parquet files from official benchmark loaders."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from recipe.on_policy_wdl_sft.code_task.prepare_code_rl_dataset import PROMPT_TEMPLATE_VERSION, build_prompt


def configure_project_cache(project_cache_root: Path, hf_home: Path, official_source_root: Path) -> None:
    os.environ["PROJECT_CACHE_ROOT"] = str(project_cache_root)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["XDG_CACHE_HOME"] = str(project_cache_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = str(
        official_source_root / "bigcodebench" / "BigCodeBench-v0.1.4.jsonl"
    )


def _jsonable(obj: Any) -> Any:
    """Convert official-loader objects into JSON-serializable metadata."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return {"__tuple__": [_jsonable(v) for v in obj]}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, complex):
        return {"__complex__": [obj.real, obj.imag]}
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return obj


def _record(data_source: str, uid: str, prompt: str, ground_truth: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(_jsonable(ground_truth), ensure_ascii=False)},
        "prompt": build_prompt(prompt),
        "split": "validation",
        "extra_info": {
            "uid": uid,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            **extra,
        },
    }


def build_evalplus_records(limit: int) -> dict[str, list[dict[str, Any]]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    out: dict[str, list[dict[str, Any]]] = {"HumanEval+": [], "MBPP+": []}
    for idx, (task_id, problem) in enumerate(get_human_eval_plus().items()):
        if idx >= limit:
            break
        gt = {
            "benchmark": "humaneval",
            "task_id": task_id,
            "entry_point": problem["entry_point"],
            "prompt": problem["prompt"],
            "base_input": problem["base_input"],
            "plus_input": problem["plus_input"],
            "canonical_solution": problem["canonical_solution"],
            "atol": problem["atol"],
        }
        out["HumanEval+"].append(_record("HumanEval+", task_id, problem["prompt"], gt, {"benchmark": "HumanEval+"}))

    for idx, (task_id, problem) in enumerate(get_mbpp_plus().items()):
        if idx >= limit:
            break
        gt = {
            "benchmark": "mbpp",
            "task_id": task_id,
            "entry_point": problem["entry_point"],
            "prompt": problem["prompt"],
            "base_input": problem["base_input"],
            "plus_input": problem["plus_input"],
            "canonical_solution": problem["canonical_solution"],
            "atol": problem["atol"],
        }
        out["MBPP+"].append(_record("MBPP+", task_id, problem["prompt"], gt, {"benchmark": "MBPP+"}))
    return out


def build_bigcodebench_records(limit: int, subset: str) -> list[dict[str, Any]]:
    from bigcodebench.data import get_bigcodebench

    records = []
    for idx, (task_id, problem) in enumerate(get_bigcodebench(subset=subset).items()):
        if idx >= limit:
            break
        gt = {
            "benchmark": "bigcodebench",
            "task_id": task_id,
            "subset": subset,
            "entry_point": problem["entry_point"],
            "complete_prompt": problem["complete_prompt"],
            "code_prompt": problem.get("code_prompt", ""),
            "test": problem["test"],
        }
        records.append(
            _record(
                "BigCodeBench",
                task_id,
                problem.get("instruct_prompt") or problem["complete_prompt"],
                gt,
                {"benchmark": "BigCodeBench", "subset": subset},
            )
        )
    return records


def load_livecodebench_problems(
    limit: int,
    release_version: str,
    lcb_python: Path,
    lcb_repo: Path,
    include_io: bool,
) -> list[dict[str, Any]]:
    if not lcb_python.is_file():
        raise FileNotFoundError(f"LiveCodeBench python not found: {lcb_python}")
    if not lcb_repo.is_dir():
        raise FileNotFoundError(f"LiveCodeBench repo not found: {lcb_repo}")
    code = r"""
import json
import sys
from lcb_runner.benchmarks import load_code_generation_dataset

release_version = sys.argv[1]
limit = int(sys.argv[2])
include_io = sys.argv[3] == "1"
rows = []
for idx, problem in enumerate(load_code_generation_dataset(release_version=release_version)):
    if idx >= limit:
        break
    row = {
        "question_id": problem.question_id,
        "question_content": problem.question_content,
        "starter_code": problem.starter_code,
    }
    if include_io:
        sample = problem.get_evaluation_sample()
        row["input_output"] = sample["input_output"]
    rows.append(row)
print(json.dumps(rows, ensure_ascii=False))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{lcb_repo}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [str(lcb_python), "-c", code, release_version, str(limit), "1" if include_io else "0"],
        cwd=str(lcb_repo),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "LiveCodeBench official loader failed\n"
            f"cmd={lcb_python} -c <loader> {release_version} {limit}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-4000:]}"
        )
    json_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(json_line)


def build_livecodebench_records(
    limit: int,
    release_version: str,
    lcb_python: Path,
    lcb_repo: Path,
    include_io: bool,
) -> list[dict[str, Any]]:
    problems = load_livecodebench_problems(limit, release_version, lcb_python, lcb_repo, include_io)

    records = []
    for problem in problems:
        prompt = build_livecodebench_prompt(problem["question_content"], problem.get("starter_code") or "")
        gt = {
            "benchmark": "livecodebench",
            "question_id": problem["question_id"],
            "release_version": release_version,
        }
        if include_io:
            gt["input_output"] = problem["input_output"]
        records.append(
            _record(
                "LiveCodeBench",
                str(problem["question_id"]),
                prompt,
                gt,
                {"benchmark": "LiveCodeBench", "release_version": release_version},
            )
        )
    return records


def build_livecodebench_prompt(question_content: str, starter_code: str) -> str:
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += (
            "### Format:\n"
            "Use the following starter code. Preserve the provided class, function names, method signatures, and return types. "
            "Fill in the implementation and return one complete executable Python solution.\n"
            "```python\n"
            f"{starter_code}\n"
            "```\n"
        )
    else:
        prompt += (
            "### Format:\n"
            "Write a complete Python program. It must read all required input from stdin and write the answer to stdout. "
            "Do not hard-code sample inputs or outputs. Do not return only a helper function.\n"
            "```python\n"
            "# YOUR CODE HERE\n"
            "```\n"
        )
    return prompt


def write_parquet(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)
    return {"path": str(path), "row_count": len(records)}


def existing_official_datasets(output_root: Path) -> dict[str, dict[str, Any]]:
    expected = {
        "HumanEval+": output_root / "official_humaneval_plus_val.parquet",
        "MBPP+": output_root / "official_mbpp_plus_val.parquet",
        "BigCodeBench": output_root / "official_bigcodebench_val.parquet",
        "LiveCodeBench": output_root / "official_livecodebench_val.parquet",
    }
    datasets = {}
    for name, path in expected.items():
        if path.is_file():
            datasets[name] = {"path": str(path), "row_count": int(len(pd.read_parquet(path)))}
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data-1/dataset/code/verl_rl"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--bcb-subset", default="full")
    parser.add_argument("--lcb-release-version", default="release_v1")
    parser.add_argument("--benchmarks", default="humaneval_plus,mbpp_plus,bigcodebench,livecodebench")
    parser.add_argument("--project-cache-root", type=Path, default=Path("/data-1/.cache"))
    parser.add_argument("--hf-home", type=Path, default=Path("/data-1/.cache/huggingface"))
    parser.add_argument("--official-source-root", type=Path, default=Path("/data-1/dataset/code/official_sources"))
    parser.add_argument("--lcb-python", type=Path, default=Path(os.environ.get("LCB_PYTHON") or sys.executable))
    parser.add_argument("--lcb-repo", type=Path, default=Path("/data-1/code_eval_envs/LiveCodeBench"))
    parser.add_argument("--lcb-include-io", action="store_true")
    args = parser.parse_args()
    configure_project_cache(args.project_cache_root, args.hf_home, args.official_source_root)

    datasets: dict[str, Any] = {}
    requested = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    if "humaneval_plus" in requested:
        evalplus = build_evalplus_records(args.limit)
        if "humaneval_plus" in requested:
            datasets["HumanEval+"] = write_parquet(args.output_root / "official_humaneval_plus_val.parquet", evalplus["HumanEval+"])
    if "mbpp_plus" in requested:
        evalplus = build_evalplus_records(args.limit)
        if "mbpp_plus" in requested:
            datasets["MBPP+"] = write_parquet(args.output_root / "official_mbpp_plus_val.parquet", evalplus["MBPP+"])
    if "bigcodebench" in requested:
        datasets["BigCodeBench"] = write_parquet(
            args.output_root / "official_bigcodebench_val.parquet", build_bigcodebench_records(args.limit, args.bcb_subset)
        )
    if "livecodebench" in requested:
        datasets["LiveCodeBench"] = write_parquet(
            args.output_root / "official_livecodebench_val.parquet",
            build_livecodebench_records(
                args.limit,
                args.lcb_release_version,
                args.lcb_python,
                args.lcb_repo,
                args.lcb_include_io,
            ),
        )
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "official_only": True,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "project_cache_root": os.environ["PROJECT_CACHE_ROOT"],
        "hf_home": os.environ["HF_HOME"],
        "hf_datasets_cache": os.environ["HF_DATASETS_CACHE"],
        "code_official_source_root": str(args.official_source_root),
        "bigcodebench_override_path": os.environ["BIGCODEBENCH_OVERRIDE_PATH"],
        "lcb_python": str(args.lcb_python),
        "lcb_repo": str(args.lcb_repo),
        "lcb_include_io": args.lcb_include_io,
        "datasets": {**existing_official_datasets(args.output_root), **datasets},
    }
    manifest_path = args.output_root / "official_only_validation.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
