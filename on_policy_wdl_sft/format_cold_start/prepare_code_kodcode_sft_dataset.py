#!/usr/bin/env python3
"""Prepare KodCode multiturn SFT data for format cold start."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path(
    "/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet"
)
DEFAULT_RAW_SOURCE = Path("/data-1/dataset/KodCode-Light-RL-10K/data/train-00000-of-00001.parquet")
DEFAULT_TOKENIZER = Path("/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
MAX_REPEAT_8GRAM_FRACTION = 0.40
MAX_REPEAT_16GRAM_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-source", type=Path, default=DEFAULT_RAW_SOURCE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--max-sequence-length", type=int, default=9216)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def normalize_messages(prompt: Any) -> list[dict[str, str]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        raise ValueError(f"prompt must be a list of chat messages, got {type(prompt).__name__}")
    messages: list[dict[str, str]] = []
    for item in prompt:
        if not isinstance(item, dict):
            raise ValueError(f"prompt item must be dict, got {type(item).__name__}")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unexpected prompt role: {role!r}")
        messages.append({"role": role, "content": content})
    return messages


def extract_reasoning(conversations: Any) -> str:
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    if not isinstance(conversations, list):
        raise ValueError("raw conversations must be a list")
    assistant_contents = [
        str(message.get("value", ""))
        for message in conversations
        if isinstance(message, dict) and message.get("from") in {"gpt", "assistant"}
    ]
    if not assistant_contents:
        raise ValueError("raw conversations contain no assistant response")
    match = re.search(r"<think>(.*?)</think>", assistant_contents[-1], flags=re.DOTALL | re.IGNORECASE)
    if match is None or not match.group(1).strip():
        raise ValueError("raw assistant response contains no non-empty closed <think> block")
    return match.group(1).strip()


def repeated_ngram_fraction(reasoning: str, n: int) -> float:
    words = re.findall(r"[a-z0-9_]+", reasoning.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    return sum(count for count in counts.values() if count > 1) / len(ngrams)


def reasoning_quality(reasoning: str) -> dict[str, float | bool]:
    repeat_8gram_fraction = repeated_ngram_fraction(reasoning, 8)
    repeat_16gram_fraction = repeated_ngram_fraction(reasoning, 16)
    accepted = (
        repeat_8gram_fraction <= MAX_REPEAT_8GRAM_FRACTION
        and repeat_16gram_fraction <= MAX_REPEAT_16GRAM_FRACTION
    )
    return {
        "repeat_8gram_fraction": repeat_8gram_fraction,
        "repeat_16gram_fraction": repeat_16gram_fraction,
        "accepted": accepted,
    }


def build_target(reasoning: str, solution: str) -> str:
    return (
        "<think>\n"
        f"{reasoning.strip()}\n"
        "</think>\n"
        "<answer>\n"
        "```python\n"
        f"{solution.rstrip()}\n"
        "```\n"
        "</answer>"
    )


def normalize_problem(text: str) -> str:
    text = (text or "").strip()
    text = text.replace(
        "Please reason step by step, and put your final solution code in a Python code block (```python ... ```).",
        "",
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def expected_uid(raw_index: int, raw_row: pd.Series) -> str:
    problem = raw_row.get("question") or raw_row.get("problem") or raw_row.get("instruction") or raw_row.get("prompt") or ""
    payload = f"kodcode-light-rl-10k:{raw_index}:{normalize_problem(str(problem))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def raw_problem(raw_row: pd.Series) -> str:
    problem = raw_row.get("question") or raw_row.get("problem") or raw_row.get("instruction") or raw_row.get("prompt")
    if problem:
        return str(problem)
    conversations = raw_row.get("conversations")
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, dict) and message.get("from") in {"human", "user"}:
                return str(message.get("value") or "")
    return ""


def convert(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    seed: int,
    max_samples: int,
    tokenizer: Any = None,
    max_response_length: int = 8192,
) -> pd.DataFrame:
    required = {"prompt", "extra_info"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input missing required columns: {missing}")

    candidates: list[tuple[int, int, pd.Series, str, dict[str, float | bool]]] = []
    for source_index, row in df.iterrows():
        extra = row["extra_info"]
        if not isinstance(extra, dict):
            raise ValueError(f"row {source_index}: extra_info must be dict")
        raw_index = extra.get("raw_index")
        if not isinstance(raw_index, int) or raw_index < 0 or raw_index >= len(raw_df):
            raise ValueError(f"row {source_index}: invalid extra_info.raw_index={raw_index!r}")
        reasoning = extract_reasoning(raw_df.iloc[raw_index]["conversations"])
        quality = reasoning_quality(reasoning)
        if not quality["accepted"]:
            continue
        messages = normalize_messages(row["prompt"])
        prompt_text = "\n".join(message["content"] for message in messages)
        raw_problem_text = raw_problem(raw_df.iloc[raw_index])
        if normalize_problem(raw_problem_text) not in prompt_text:
            raise ValueError(f"row {source_index}: raw_index does not match the model-visible problem")
        uid = extra.get("uid")
        if isinstance(uid, str) and re.fullmatch(r"[0-9a-f]{24}", uid) and uid != expected_uid(raw_index, raw_df.iloc[raw_index]):
            raise ValueError(f"row {source_index}: raw_index does not match extra_info.uid")
        function_declaration = extra.get("function_declaration")
        entry_point = extra.get("entry_point")
        if not isinstance(function_declaration, str) or not function_declaration.strip():
            raise ValueError(f"row {source_index}: missing extra_info.function_declaration")
        if not isinstance(entry_point, str) or not entry_point.strip():
            raise ValueError(f"row {source_index}: missing extra_info.entry_point")
        if function_declaration not in prompt_text or entry_point not in prompt_text:
            raise ValueError(f"row {source_index}: model-visible prompt omits the KodCode function contract")
        if "\\boxed" in prompt_text or "boxed{" in prompt_text:
            continue
        solution = extra.get("original_solution")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"row {source_index}: missing extra_info.original_solution")
        assistant = build_target(reasoning, solution)
        response_tokens = None
        if tokenizer is not None:
            response_tokens = len(tokenizer.encode(assistant, add_special_tokens=False))
            if response_tokens > max_response_length:
                continue
        quality["response_tokens"] = response_tokens
        raw_index = int(raw_index)
        candidates.append((int(source_index), raw_index, row, reasoning, quality))

    if max_samples >= 0 and len(candidates) > max_samples:
        random.Random(seed).shuffle(candidates)
        candidates = candidates[:max_samples]
    if max_samples >= 0 and len(candidates) < max_samples:
        raise ValueError(f"only {len(candidates)} CoT-quality rows available, requested {max_samples}")

    rows: list[dict[str, Any]] = []
    for source_index, raw_index, row, reasoning, quality in candidates:
        extra = row["extra_info"]
        function_declaration = extra["function_declaration"]
        entry_point = extra["entry_point"]
        solution = extra.get("original_solution")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"row {raw_index}: missing extra_info.original_solution")

        messages = normalize_messages(row["prompt"])
        prompt_text = "\n".join(message["content"] for message in messages)
        if "\\boxed" in prompt_text or "boxed{" in prompt_text:
            raise ValueError(f"row {raw_index}: code prompt contains math boxed target")
        assistant = build_target(reasoning, solution)
        if "<answer>" not in assistant or "```python" not in assistant:
            raise ValueError(f"row {raw_index}: assistant target failed code format check")
        rows.append(
            {
                "messages": [*messages, {"role": "assistant", "content": assistant}],
                "data_source": row.get("data_source", "kodcode_light_rl_10k"),
                "split": row.get("split", "train"),
                "extra_info": {
                    "source_index": raw_index,
                    "rl_source_index": source_index,
                    "source_dataset": extra.get("source_dataset", "KodCode/KodCode-Light-RL-10K"),
                    "entry_point": entry_point,
                    "function_declaration": function_declaration,
                    "prompt_template_version": extra.get("prompt_template_version"),
                    "uid": extra.get("uid"),
                    "format_cold_start": "code-cot-python-answer-v3",
                    "reasoning_source": "KodCode.conversations.gpt.think",
                    "reasoning_repeat_8gram_fraction": quality["repeat_8gram_fraction"],
                    "reasoning_repeat_16gram_fraction": quality["repeat_16gram_fraction"],
                    "response_tokens": quality["response_tokens"],
                },
            }
        )
    return pd.DataFrame(rows)


def verify(
    df: pd.DataFrame,
    tokenizer: Any = None,
    max_response_length: int = 8192,
    max_sequence_length: int = 9216,
) -> dict[str, Any]:
    if "messages" not in df.columns:
        raise ValueError("output must contain messages column")
    for idx, messages in enumerate(df["messages"].tolist()):
        messages = normalize_messages(messages)
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(f"row {idx}: last message must be assistant")
        assistant = messages[-1]["content"]
        think_match = re.search(r"<think>(.*?)</think>", assistant, flags=re.DOTALL)
        if think_match is None or not think_match.group(1).strip():
            raise ValueError(f"row {idx}: assistant target must contain non-empty complete reasoning")
        quality = reasoning_quality(think_match.group(1))
        if not quality["accepted"]:
            raise ValueError(f"row {idx}: assistant reasoning failed repetition quality gate: {quality}")
        if "<answer>" not in assistant or "</answer>" not in assistant or "```python" not in assistant:
            raise ValueError(f"row {idx}: assistant target must contain complete <answer> and ```python")
        extra = df.iloc[idx]["extra_info"]
        function_declaration = extra.get("function_declaration") if isinstance(extra, dict) else None
        entry_point = extra.get("entry_point") if isinstance(extra, dict) else None
        joined = "\n".join(message["content"] for message in messages)
        if not function_declaration or not entry_point:
            raise ValueError(f"row {idx}: SFT metadata omits the KodCode function contract")
        if function_declaration not in joined or entry_point not in joined:
            raise ValueError(
                f"row {idx} source_index={extra.get('source_index')}: SFT prompt omits the model-visible "
                f"KodCode function contract; declaration={function_declaration!r} entry_point={entry_point!r} "
                f"prompt_excerpt={joined[:500]!r}"
            )
        if tokenizer is not None:
            response_tokens = len(tokenizer.encode(assistant, add_special_tokens=False))
            if response_tokens > max_response_length:
                raise ValueError(
                    f"row {idx}: response_tokens={response_tokens} exceeds max_response_length={max_response_length}"
                )
            sequence_tokens = len(
                tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
            )
            if sequence_tokens > max_sequence_length:
                raise ValueError(
                    f"row {idx}: sequence_tokens={sequence_tokens} exceeds max_sequence_length={max_sequence_length}"
                )
        answer_body = assistant.split("<answer>", 1)[1].split("</answer>", 1)[0]
        if "\\boxed" in answer_body or "boxed{" in answer_body:
            raise ValueError(f"row {idx}: code answer contains boxed math notation")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "format_cold_start": "code-cot-python-answer-v3",
        "max_repeat_8gram_fraction": MAX_REPEAT_8GRAM_FRACTION,
        "max_repeat_16gram_fraction": MAX_REPEAT_16GRAM_FRACTION,
        "max_response_length": max_response_length,
        "max_sequence_length": max_sequence_length,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"ERROR: input parquet not found: {args.input}")
    if args.max_response_length <= 0 or args.max_sequence_length <= 0:
        raise ValueError("--max-response-length and --max-sequence-length must be positive")
    if args.max_sequence_length <= args.max_response_length:
        raise ValueError("--max-sequence-length must leave room for prompt tokens")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.verify_only:
        if not args.output.is_file():
            raise SystemExit(f"ERROR: --verify-only output parquet not found: {args.output}")
        out_df = pd.read_parquet(args.output)
        stats = verify(out_df, tokenizer, args.max_response_length, args.max_sequence_length)
    else:
        if not args.raw_source.is_file():
            raise SystemExit(f"ERROR: raw KodCode parquet not found: {args.raw_source}")
        src_df = pd.read_parquet(args.input)
        raw_df = pd.read_parquet(args.raw_source)
        if "conversations" not in raw_df.columns:
            raise ValueError("raw KodCode source must contain conversations column")
        out_df = convert(
            src_df,
            raw_df,
            args.seed,
            args.max_samples,
            tokenizer=tokenizer,
            max_response_length=args.max_response_length,
        )
        stats = verify(out_df, tokenizer, args.max_response_length, args.max_sequence_length)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(args.output, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).as_posix(),
        "input": args.input.as_posix(),
        "raw_source": args.raw_source.as_posix(),
        "tokenizer": args.tokenizer.as_posix(),
        "max_response_length": args.max_response_length,
        "max_sequence_length": args.max_sequence_length,
        "output": args.output.as_posix(),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "verify_only": args.verify_only,
        "format": "multiturn_sft_messages",
        **stats,
    }
    write_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
