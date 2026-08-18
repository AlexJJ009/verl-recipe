#!/usr/bin/env python3
"""Prepare KodCode-Light-RL-10K for code-task RL training.

The output follows the existing code-task prompt and reward contract:
system+user chat prompt, rule reward, and Python code inside
<answer>```python ...```</answer>.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "KodCode/KodCode-Light-RL-10K"
DATASET_FILE = "data/train-00000-of-00001.parquet"
AUTHOR_PREPROCESSING_REPOSITORY = "https://github.com/KodCode-AI/code-r1"
AUTHOR_PREPROCESSING_COMMIT = "c348f894a803d0eff3c4d529dbf82af6e1262ae1"
AUTHOR_PREPROCESSING_PATH = "examples/data_preprocess/kodcode.py"
AUTHOR_FUNCTION_DECLARATION_TEMPLATE = (
    "Note that the function declaration is `{function_declaration}`. "
    "Preserve this exact function name and parameter signature. "
    "Your code should be wrapped in a markdown code block."
)
PROMPT_TEMPLATE_VERSION = "code-think-answer-python-v2-kodcode-author-signature"
SYSTEM_PROMPT = (
    "You are a code generation assistant. Think through the problem, then return executable Python code only in "
    "<answer> fenced with ```python ... ```."
)
CONTRACT_SUFFIX = (
    "\n\nUse this exact response format:\n"
    "<think>your concise reasoning</think>\n"
    "<answer>\n```python\n# executable solution\n```\n</answer>\n"
    "Do not use boxed final-answer notation."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalize_problem(text: str) -> str:
    text = (text or "").strip()
    text = text.replace(
        "Please reason step by step, and put your final solution code in a Python code block (```python ... ```).",
        "",
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_function_contract(row: dict[str, Any]) -> dict[str, str]:
    test_info = parse_jsonish(row.get("test_info"), default=None)
    if not isinstance(test_info, list) or len(test_info) != 1 or not isinstance(test_info[0], dict):
        raise ValueError("KodCode-Light row must contain exactly one test_info function contract")
    info = test_info[0]
    contract = {
        key: str(info.get(key) or "").strip()
        for key in ("function_declaration", "function_name", "parameter_list")
    }
    if not contract["function_declaration"] or not contract["function_name"]:
        raise ValueError("KodCode-Light test_info must contain function_declaration and function_name")
    try:
        declaration_module = ast.parse(f"{contract['function_declaration']}\n    pass\n")
    except SyntaxError as exc:
        raise ValueError("KodCode-Light function_declaration is not valid Python") from exc
    declaration = declaration_module.body[0] if declaration_module.body else None
    if not isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef)) or declaration.name != contract["function_name"]:
        raise ValueError("KodCode-Light function_declaration does not match function_name")
    return contract


def build_prompt(problem: str, function_declaration: str) -> list[dict[str, str]]:
    user_content = normalize_problem(problem)
    user_content += "\n\n" + AUTHOR_FUNCTION_DECLARATION_TEMPLATE.format(
        function_declaration=function_declaration
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content + CONTRACT_SUFFIX},
    ]


def extract_function_name(row: dict[str, Any]) -> str | None:
    try:
        return extract_function_contract(row)["function_name"]
    except ValueError:
        pass
    for code_key in ("solution", "r1_solution"):
        code = row.get(code_key) or ""
        match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(code), re.M)
        if match:
            return match.group(1)
    return None


def build_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    test_info = parse_jsonish(row.get("test_info"), default=None)
    verified_solution = row.get("r1_solution") or row.get("solution") or ""
    gt = {
        "test": row.get("test") or "",
        "test_info": test_info,
        "entry_point": extract_function_name(row),
        "reference_solution": verified_solution,
        "original_solution": row.get("solution") or "",
        "r1_solution": row.get("r1_solution") or "",
        "style": row.get("style"),
        "source": "KodCode-Light-RL-10K",
        "verification_method": "kodcode_exec",
    }
    return gt


def difficulty_proxy(row: dict[str, Any], prompt_len_chars: int, test_len_chars: int) -> str:
    subset = str(row.get("subset") or row.get("category") or row.get("domain") or "").lower()
    text = f"{row.get('problem') or ''}\n{row.get('instruction') or ''}".lower()
    if any(k in subset or k in text for k in ("competitive", "dynamic programming", "graph", "tree", "hard")):
        return "hard_proxy"
    if prompt_len_chars > 3500 or test_len_chars > 3500:
        return "hard_proxy"
    if prompt_len_chars > 1800 or test_len_chars > 1600:
        return "medium_proxy"
    return "easy_proxy"


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "min": int(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "mean": float(arr.mean()),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": int(arr.max()),
    }


def load_tokenizer(model_path: str | None) -> Any | None:
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required for --tokenizer-path length audit") from exc
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def prompt_token_len(tokenizer: Any, prompt: list[dict[str, str]]) -> int:
    text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def maybe_download(raw_path: Path, force_download: bool = False) -> Path:
    if raw_path.exists() and not force_download:
        return raw_path
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download KodCode-Light-RL-10K") from exc

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=DATASET_FILE,
            local_dir=str(raw_path.parent),
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
    )
    if downloaded != raw_path:
        raw_path.write_bytes(downloaded.read_bytes())
    return raw_path


def convert(raw_path: Path, output_path: Path, report_path: Path, tokenizer_path: str | None = None) -> dict[str, Any]:
    raw_df = pd.read_parquet(raw_path)
    tokenizer = load_tokenizer(tokenizer_path)
    records: list[dict[str, Any]] = []
    prompt_chars: list[int] = []
    prompt_tokens: list[int] = []
    user_chars: list[int] = []
    solution_chars: list[int] = []
    test_chars: list[int] = []
    source_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    gpt_difficulty_counts: Counter[str] = Counter()
    r1_correctness_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    gpt_pass_percentages: list[float] = []
    gpt_pass_trial_nums: list[int] = []
    r1_pass_trial_nums: list[int] = []
    missing_tests = 0
    missing_entry_point = 0
    missing_function_declaration = 0

    for idx, row_obj in raw_df.iterrows():
        row = row_obj.to_dict()
        problem = row.get("question") or row.get("problem") or row.get("instruction") or row.get("prompt") or ""
        try:
            function_contract = extract_function_contract(row)
        except ValueError:
            missing_function_declaration += 1
            raise
        prompt = build_prompt(str(problem), function_contract["function_declaration"])
        if tokenizer is not None:
            prompt_tokens.append(prompt_token_len(tokenizer, prompt))
        user_content = prompt[1]["content"]
        gt = build_ground_truth(row)
        if not gt["test"]:
            missing_tests += 1
        if not gt["entry_point"]:
            missing_entry_point += 1
        uid = hashlib.sha256(f"kodcode-light-rl-10k:{idx}:{normalize_problem(str(problem))}".encode()).hexdigest()[:24]
        test_len = len(str(gt["test"]))
        diff = difficulty_proxy(row, len(user_content), test_len)
        difficulty_counts[diff] += 1
        source_counts[str(row.get("source") or row.get("subset") or row.get("category") or "unknown")] += 1
        style_counts[str(row.get("style") or "unknown")] += 1
        gpt_difficulty_counts[str(row.get("gpt_difficulty") or "unknown")] += 1
        r1_correctness_counts[str(row.get("r1_correctness") or "unknown")] += 1
        for value, target, cast in (
            (row.get("gpt_pass_percentage"), gpt_pass_percentages, float),
            (row.get("gpt_pass_trial_num"), gpt_pass_trial_nums, int),
            (row.get("r1_pass_trial_num"), r1_pass_trial_nums, int),
        ):
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                target.append(cast(value))
        prompt_chars.append(len(prompt[0]["content"]) + len(user_content))
        user_chars.append(len(user_content))
        solution_chars.append(len(str(row.get("solution") or "")))
        test_chars.append(test_len)
        records.append(
            {
                "data_source": "kodcode_light_rl_10k",
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
                "prompt": prompt,
                "split": "train",
                "extra_info": {
                    "uid": uid,
                    "raw_index": int(idx),
                    "source_dataset": DATASET_ID,
                    "source_file": DATASET_FILE,
                    "source": row.get("source"),
                    "style": row.get("style"),
                    "test_info": json.dumps(parse_jsonish(row.get("test_info"), default=None), ensure_ascii=False),
                    "reference_answer": gt["reference_solution"],
                    "original_solution": row.get("solution") or "",
                    "r1_solution": row.get("r1_solution") or "",
                    "entry_point": gt["entry_point"],
                    "function_declaration": function_contract["function_declaration"],
                    "function_name": function_contract["function_name"],
                    "parameter_list": function_contract["parameter_list"],
                    "difficulty_proxy": diff,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                },
            }
        )

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_id": DATASET_ID,
        "dataset_file": DATASET_FILE,
        "license": "CC-BY-NC-4.0",
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_count": int(len(out_df)),
        "raw_columns": list(raw_df.columns),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "contract": "<think>...</think><answer>```python ...```</answer>",
        "data_source": "kodcode_light_rl_10k",
        "missing_tests": missing_tests,
        "missing_entry_point": missing_entry_point,
        "missing_function_declaration": missing_function_declaration,
        "prompt_function_declaration_count": int(len(out_df)),
        "author_preprocessing_reference": {
            "repository": AUTHOR_PREPROCESSING_REPOSITORY,
            "commit": AUTHOR_PREPROCESSING_COMMIT,
            "path": AUTHOR_PREPROCESSING_PATH,
            "function_declaration_template": AUTHOR_FUNCTION_DECLARATION_TEMPLATE,
        },
        "source_counts_top20": dict(source_counts.most_common(20)),
        "style_counts": dict(style_counts),
        "gpt_difficulty_counts": dict(gpt_difficulty_counts),
        "r1_correctness_counts": dict(r1_correctness_counts),
        "gpt_pass_percentage": quantiles([int(round(v * 1000)) for v in gpt_pass_percentages]),
        "gpt_pass_trial_num": quantiles(gpt_pass_trial_nums),
        "r1_pass_trial_num": quantiles(r1_pass_trial_nums),
        "difficulty_proxy_counts": dict(difficulty_counts),
        "length_chars": {
            "prompt_total": quantiles(prompt_chars),
            "user_prompt": quantiles(user_chars),
            "reference_solution": quantiles(solution_chars),
            "test": quantiles(test_chars),
        },
        "length_tokens": {
            "tokenizer_path": tokenizer_path,
            "prompt_total": quantiles(prompt_tokens),
            "over_max_prompt_length_1024": int(sum(v > 1024 for v in prompt_tokens)),
            "over_max_prompt_length_2048": int(sum(v > 2048 for v in prompt_tokens)),
        }
        if tokenizer is not None
        else None,
        "step_coverage": {
            "train_prompt_bsz_64": {
                "100_steps_unique_prompts": 6400,
                "100_steps_fraction_of_dataset": 6400 / max(1, len(out_df)),
                "150_steps_unique_prompts": 9600,
                "150_steps_fraction_of_dataset": 9600 / max(1, len(out_df)),
                "steps_for_one_epoch": int(np.ceil(len(out_df) / 64)),
            },
            "train_prompt_bsz_32": {
                "100_steps_unique_prompts": 3200,
                "100_steps_fraction_of_dataset": 3200 / max(1, len(out_df)),
                "150_steps_unique_prompts": 4800,
                "150_steps_fraction_of_dataset": 4800 / max(1, len(out_df)),
                "steps_for_one_epoch": int(np.ceil(len(out_df) / 32)),
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=Path("/data-1/dataset/KodCode-Light-RL-10K/data/train-00000-of-00001.parquet"))
    parser.add_argument("--output", type=Path, default=Path("/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet"))
    parser.add_argument("--report", type=Path, default=Path("/data-1/dataset/KodCode-Light-RL-10K/reports/kodcode_light_rl_10k_audit.json"))
    parser.add_argument(
        "--tokenizer-path",
        default="/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539",
    )
    parser.add_argument("--download", action="store_true", help="Download the raw parquet from Hugging Face if missing.")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    raw_path = args.raw_path
    if args.download or args.force_download or not raw_path.exists():
        raw_path = maybe_download(raw_path, force_download=args.force_download)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw KodCode parquet not found: {raw_path}")
    report = convert(raw_path, args.output, args.report, tokenizer_path=args.tokenizer_path)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
