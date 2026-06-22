#!/usr/bin/env python3
"""Generate code benchmark samples with vLLM.

The output JSONL is intentionally raw: one row per generated completion with
task_index/sample_index metadata. `convert_official_outputs.py` then applies the
same code extractor used by online reward before official benchmark scoring.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _messages(prompt_value: Any) -> list[dict[str, str]]:
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt_value]


def _task_id(row: dict[str, Any]) -> str:
    ground_truth = row["reward_model"]["ground_truth"]
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    return str(ground_truth.get("task_id") or ground_truth.get("question_id") or row.get("extra_info", {}).get("uid"))


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-thinking", type=parse_bool, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.validation_parquet)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    rows = df.to_dict("records")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    chat_template_kwargs: dict[str, Any] = {}
    if args.enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = args.enable_thinking
    prompts = [
        tokenizer.apply_chat_template(
            _messages(row["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for row in rows
    ]

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    started = time.time()
    outputs = llm.generate(prompts, sampling_params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    num_outputs = 0
    with args.output.open("w", encoding="utf-8") as f:
        for task_index, request_output in enumerate(outputs):
            row = rows[task_index]
            for sample_index, completion in enumerate(request_output.outputs):
                num_outputs += 1
                f.write(
                    json.dumps(
                        {
                            "task_index": task_index,
                            "sample_index": sample_index,
                            "task_id": _task_id(row),
                            "data_source": row["data_source"],
                            "solution_str": completion.text,
                            "finish_reason": getattr(completion, "finish_reason", None),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "validation_parquet": str(args.validation_parquet),
        "output": str(args.output),
        "num_tasks": len(rows),
        "num_outputs": num_outputs,
        "generation_params": {
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "tensor_parallel": args.tensor_parallel,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "chat_template_kwargs": chat_template_kwargs,
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
