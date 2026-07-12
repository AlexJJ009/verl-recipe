#!/usr/bin/env python3
"""Compute deterministic validation eligibility with the native RLHFDataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_data_config(max_prompt_length: int) -> dict:
    return {
        "apply_chat_template_kwargs": {},
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers_semantics": "parallel_order_preserving",
        "max_prompt_length": max_prompt_length,
        "need_tools_kwargs": False,
        "prompt_key": "prompt",
        "require_source_uid": True,
        "tool_config_path": None,
        "truncation": "left",
    }


def compute(*, repo_root: Path, tokenizer_path: Path, datasets: list[tuple[str, Path]], max_prompt_length: int) -> dict:
    import sys
    from omegaconf import OmegaConf

    sys.path.insert(0, str(repo_root))
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import hf_tokenizer

    config_doc = normalized_data_config(max_prompt_length)
    config = OmegaConf.create(config_doc)
    tokenizer = hf_tokenizer(str(tokenizer_path), trust_remote_code=True)
    dataset_rows = []
    submitted = 0
    for source_index, (name, path) in enumerate(datasets):
        dataset = RLHFDataset([str(path)], tokenizer, config, processor=None, max_samples=-1)
        uids = []
        for extra in dataset.dataframe["extra_info"]:
            uid = extra.get("uid") if isinstance(extra, dict) else None
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"{name}: missing stable UID after eligibility filtering")
            uids.append(uid)
        if len(uids) != len(set(uids)):
            raise ValueError(f"{name}: duplicate eligible UID")
        submitted += len(uids)
        dataset_rows.append({"name": name, "source_index": source_index, "ordered_uids": uids})

    eligibility_tool = Path(__file__).resolve()
    rl_dataset = repo_root / "verl/utils/dataset/rl_dataset.py"
    filter_evidence = {
        "eligibility_tool_sha256": file_sha256(eligibility_tool),
        "normalized_manifest_data_config": config_doc,
        "rl_dataset_sha256": file_sha256(rl_dataset),
    }
    uid_doc = {"schema_version": SCHEMA_VERSION, "datasets": dataset_rows}
    return {
        "max_prompt_length": max_prompt_length,
        "filter_enabled": True,
        "filter_implementation_sha256": hashlib.sha256(canonical_json(filter_evidence)).hexdigest(),
        "ordered_eligible_uid_sha256": hashlib.sha256(canonical_json(uid_doc)).hexdigest(),
        "per_dataset_eligible_counts": {row["name"]: len(row["ordered_uids"]) for row in dataset_rows},
        "submitted_prompt_count": submitted,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--humaneval-plus", type=Path, required=True)
    parser.add_argument("--mbpp-plus", type=Path, required=True)
    parser.add_argument("--livecodebench", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(
        repo_root=args.repo_root,
        tokenizer_path=args.tokenizer,
        datasets=[("HumanEval+", args.humaneval_plus), ("MBPP+", args.mbpp_plus), ("LiveCodeBench", args.livecodebench)],
        max_prompt_length=args.max_prompt_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
