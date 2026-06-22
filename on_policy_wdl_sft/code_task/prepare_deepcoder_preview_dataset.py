#!/usr/bin/env python3
"""Prepare DeepCoder-Preview-Dataset for code-task RL training.

The raw Hugging Face dataset is preserved under /data-1/dataset, while the
training outputs follow the current code-task prompt and reward contract:
system+user chat prompt, Python code inside <answer>```python ...```</answer>,
and stdin/stdout test cases for reward execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "agentica-org/DeepCoder-Preview-Dataset"
PROMPT_TEMPLATE_VERSION = "code-think-answer-python-v1"
SYSTEM_PROMPT = (
    "You are a code generation assistant for stdin/stdout programming problems. Think through the problem, then "
    "return one complete executable Python program only in <answer> fenced with ```python ... ```."
)
CONTRACT_SUFFIX = (
    "\n\nUse this exact response format:\n"
    "<think>your concise reasoning</think>\n"
    "<answer>\n```python\n# executable solution\n```\n</answer>\n"
    "The submitted program must read all inputs from stdin and print the required outputs to stdout.\n"
    "Do not write a function-only answer unless the problem explicitly asks for one and also provides stdin/stdout.\n"
    "Do not use boxed final-answer notation."
)


TRAIN_CONFIGS = (("taco", "train"), ("primeintellect", "train"), ("lcbv5", "train"))
TEST_CONFIGS = (("lcbv5", "test"), ("codeforces", "test"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def build_prompt(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_problem(problem) + CONTRACT_SUFFIX},
    ]


def normalize_stdio_tests(tests: Any) -> tuple[list[dict[str, str]], str | None]:
    tests = parse_jsonish(tests, default=[])
    cases: list[dict[str, str]] = []
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        if tests.get("fn_name"):
            return [], "function_call_tests"
        for inp, out in zip(tests.get("inputs") or [], tests.get("outputs") or []):
            cases.append({"input": str(inp), "output": str(out)})
        return cases, None if cases else "missing_tests"
    if isinstance(tests, list):
        case_types = {
            str(item.get("testtype") or item.get("type") or "")
            for item in tests
            if isinstance(item, dict)
        }
        unsupported = case_types - {"stdin", "stdin_stdout", ""}
        if unsupported:
            return [], "unsupported_testtype:" + ",".join(sorted(unsupported))
        for item in tests:
            if not isinstance(item, dict):
                continue
            if "input" in item and "output" in item:
                cases.append(
                    {
                        "input": str(item.get("input") or ""),
                        "output": str(item.get("output") or ""),
                        "testtype": str(item.get("testtype") or item.get("type") or ""),
                    }
                )
        return cases, None if cases else "missing_tests"
    return cases, "missing_tests"


def normalize_solutions(value: Any) -> list[str]:
    value = parse_jsonish(value, default=[])
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def normalize_reference_solution(value: str) -> str:
    text = str(value or "").strip()
    match = re.match(r"^```(?:python|py)?\s*\n(?P<code>.*?)```\s*$", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group("code").strip()
    return text


def difficulty_proxy(source_config: str, problem: str, num_tests: int, prompt_len_chars: int) -> str:
    text = problem.lower()
    hard_terms = ("dynamic programming", "graph", "tree", "segment tree", "dfs", "bfs", "codeforces")
    if source_config in {"taco", "lcbv5", "codeforces"}:
        if prompt_len_chars > 3500 or num_tests >= 20 or any(term in text for term in hard_terms):
            return "hard_proxy"
        return "medium_proxy"
    if prompt_len_chars > 2500 or num_tests >= 12:
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


def download_snapshot(snapshot_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download DeepCoder-Preview-Dataset") from exc
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(snapshot_dir),
        allow_patterns=["*.parquet", "README.md", ".gitattributes"],
    )
    return snapshot_dir


def load_hf_split(config: str, split: str, snapshot_dir: Path | None = None) -> pd.DataFrame:
    if snapshot_dir is not None:
        files = sorted((snapshot_dir / config).glob(f"{split}-*.parquet"))
        if files:
            return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required to download DeepCoder-Preview-Dataset") from exc
    ds = load_dataset(DATASET_ID, config, split=split)
    return ds.to_pandas()


def row_uid(config: str, split: str, idx: int, problem: str) -> str:
    return hashlib.sha256(f"{DATASET_ID}:{config}:{split}:{idx}:{normalize_problem(problem)}".encode()).hexdigest()[:24]


def convert_row(row: dict[str, Any], config: str, split_name: str, idx: int) -> dict[str, Any] | None:
    problem = str(row.get("problem") or "")
    tests, skip_reason = normalize_stdio_tests(row.get("tests"))
    if not problem.strip() or not tests:
        return None
    prompt = build_prompt(problem)
    solutions = [normalize_reference_solution(solution) for solution in normalize_solutions(row.get("solutions"))]
    metadata = parse_jsonish(row.get("metadata"), default={})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    starter_code = str(row.get("starter_code") or "")
    gt = {
        "tests": tests,
        "verification_method": "stdin_stdout_exec",
        "source_dataset": DATASET_ID,
        "source_config": config,
        "source_split": split_name,
        "starter_code": starter_code,
        "metadata": metadata,
    }
    user_content = prompt[1]["content"]
    diff = difficulty_proxy(config, problem, len(tests), len(user_content))
    uid = row_uid(config, split_name, idx, problem)
    return {
        "data_source": f"deepcoder_preview_{config}",
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
        "prompt": prompt,
        "split": split_name,
        "extra_info": {
            "uid": uid,
            "raw_index": int(idx),
            "source_dataset": DATASET_ID,
            "source_config": config,
            "source_split": split_name,
            "source": config,
            "reference_answer": solutions[0] if solutions else "",
            "num_reference_solutions": len(solutions),
            "starter_code": starter_code,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "difficulty_proxy": diff,
            "num_tests": len(tests),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "verification_method": "stdin_stdout_exec",
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    # Large DeepCoder rows contain very long nested test metadata. Keeping the
    # non-tensor columns as JSON strings avoids pyarrow chunked nested read bugs
    # while RLHFDataset restores them before training.
    for column in ("reward_model", "prompt", "extra_info"):
        df[column] = df[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    df.to_parquet(path, index=False)


def collect_records(
    configs: tuple[tuple[str, str], ...],
    split_label: str,
    snapshot_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    for config, hf_split in configs:
        df = load_hf_split(config, hf_split, snapshot_dir=snapshot_dir)
        raw_counts[f"{config}/{hf_split}"] = int(len(df))
        skipped = 0
        for idx, row_obj in df.iterrows():
            record = convert_row(row_obj.to_dict(), config, split_label, int(idx))
            if record is None:
                skipped += 1
                continue
            records.append(record)
        skipped_counts[f"{config}/{hf_split}"] = skipped
    stats = {"raw_counts": raw_counts, "skipped_counts": skipped_counts}
    return records, stats


def manifest_for(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    official_test_records: list[dict[str, Any]],
    output_root: Path,
    tokenizer_path: str | None,
    tokenizer: Any | None,
    stats: dict[str, Any],
) -> dict[str, Any]:
    all_records = train_records + dev_records + official_test_records
    prompt_chars: list[int] = []
    prompt_tokens: list[int] = []
    test_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    diff_counts: Counter[str] = Counter()
    data_source_counts: Counter[str] = Counter()
    for record in all_records:
        prompt = record["prompt"]
        prompt_chars.append(sum(len(m.get("content", "")) for m in prompt))
        if tokenizer is not None:
            prompt_tokens.append(prompt_token_len(tokenizer, prompt))
        extra = record["extra_info"]
        test_counts.append(int(extra.get("num_tests") or 0))
        source_counts[str(extra.get("source_config"))] += 1
        split_counts[str(record.get("split"))] += 1
        diff_counts[str(extra.get("difficulty_proxy"))] += 1
        data_source_counts[str(record.get("data_source"))] += 1

    paths = {
        "hf_snapshot": output_root / "hf_snapshot",
        "raw_jsonl_train": output_root / "data" / "train.jsonl",
        "raw_jsonl_dev": output_root / "data" / "dev.jsonl",
        "raw_jsonl_official_test": output_root / "data" / "official_test.jsonl",
        "rl_train_parquet": Path("/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet"),
        "rl_dev_parquet": Path("/data-1/dataset/code/verl_rl/deepcoder_preview_dev_rl_format.parquet"),
        "rl_official_test_parquet": Path("/data-1/dataset/code/verl_rl/deepcoder_preview_official_test_rl_format.parquet"),
    }
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_id": DATASET_ID,
        "license": "MIT",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "contract": "<think>...</think><answer>```python ...```</answer>",
        "verification_method": "stdin_stdout_exec",
        "output_root": str(output_root),
        "paths": {key: str(value) for key, value in paths.items()},
        "row_count": {
            "train": len(train_records),
            "dev": len(dev_records),
            "official_test": len(official_test_records),
            "all": len(all_records),
        },
        "hf_stats": stats,
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "data_source_counts": dict(data_source_counts),
        "difficulty_proxy_counts": dict(diff_counts),
        "length_chars": {"prompt_total": quantiles(prompt_chars)},
        "num_tests": quantiles(test_counts),
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
                "100_steps_unique_prompts": min(6400, len(train_records)),
                "100_steps_fraction_of_dataset": min(6400, len(train_records)) / max(1, len(train_records)),
                "150_steps_unique_prompts": min(9600, len(train_records)),
                "150_steps_fraction_of_dataset": min(9600, len(train_records)) / max(1, len(train_records)),
                "steps_for_one_epoch": int(np.ceil(len(train_records) / 64)),
            },
        },
    }
    return report


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root: Path = args.output_root
    snapshot_dir = args.snapshot_dir or (output_root / "hf_snapshot")
    if not args.skip_snapshot_download:
        download_snapshot(snapshot_dir)
    tokenizer = load_tokenizer(args.tokenizer_path)
    train_pool, train_stats = collect_records(TRAIN_CONFIGS, "train", snapshot_dir=snapshot_dir)
    official_test_records, test_stats = collect_records(TEST_CONFIGS, "official_test", snapshot_dir=snapshot_dir)
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    dev_records = train_pool[: args.dev_size]
    train_records = train_pool[args.dev_size :]
    for record in dev_records:
        record["split"] = "dev"
    stats = {"train_sources": train_stats, "official_test_sources": test_stats, "seed": args.seed, "dev_size": args.dev_size}

    raw_train_jsonl = output_root / "data" / "train.jsonl"
    raw_dev_jsonl = output_root / "data" / "dev.jsonl"
    raw_official_test_jsonl = output_root / "data" / "official_test.jsonl"
    rl_train = Path("/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet")
    rl_dev = Path("/data-1/dataset/code/verl_rl/deepcoder_preview_dev_rl_format.parquet")
    rl_official_test = Path("/data-1/dataset/code/verl_rl/deepcoder_preview_official_test_rl_format.parquet")

    write_jsonl(raw_train_jsonl, train_records)
    write_jsonl(raw_dev_jsonl, dev_records)
    write_jsonl(raw_official_test_jsonl, official_test_records)
    write_parquet(rl_train, train_records)
    write_parquet(rl_dev, dev_records)
    write_parquet(rl_official_test, official_test_records)

    report = manifest_for(train_records, dev_records, official_test_records, output_root, args.tokenizer_path, tokenizer, stats)
    report["sha256"] = {
        "raw_jsonl_train": sha256_file(raw_train_jsonl),
        "raw_jsonl_dev": sha256_file(raw_dev_jsonl),
        "raw_jsonl_official_test": sha256_file(raw_official_test_jsonl),
        "rl_train_parquet": sha256_file(rl_train),
        "rl_dev_parquet": sha256_file(rl_dev),
        "rl_official_test_parquet": sha256_file(rl_official_test),
    }
    report["source_signature"] = sha256_text(json.dumps(report["hf_stats"], sort_keys=True))

    report_path = output_root / "reports" / "deepcoder_preview_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for path in (rl_train, rl_dev, rl_official_test):
        path.with_suffix(".manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data-1/dataset/DeepCoder-Preview-Dataset"))
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--skip-snapshot-download", action="store_true")
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--tokenizer-path",
        default="/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539",
    )
    args = parser.parse_args()
    if args.dev_size < 0:
        raise ValueError("--dev-size must be non-negative")
    report = prepare(args)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
