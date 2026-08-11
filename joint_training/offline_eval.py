"""Offline vLLM inference + evaluation on math benchmarks.

The script always computes a single mean@n using all n responses per prompt.
When n is a power of 2, it additionally computes pass@k and maj@k for
all k in {1, 2, 4, ..., n}. Otherwise it computes pass@n and maj@n only.

Per-dataset n is supported via --n_per_dataset: specify dataset_path:n pairs
so different benchmarks can use different sample counts in a single run.

Usage:
    # Uniform n for all datasets:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python recipe/joint_training/offline_eval.py \
        --model_path /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2 \
        --tensor_parallel 4 \
        --n 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --max_tokens 4096 \
        --output_dir /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2/inference_n8

    # Per-dataset n (MATH-500 uses n=8, AIME-2025 uses n=16):
    CUDA_VISIBLE_DEVICES=4,5,6,7 python recipe/joint_training/offline_eval.py \
        --model_path /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2 \
        --tensor_parallel 4 \
        --n 8 \
        --n_per_dataset /data-1/dataset/AIME-2025/aime-2025.parquet:16 \
        --output_dir /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2/inference
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Compatibility patch: transformers 5.x TokenizersBackend lacks
# all_special_tokens_extended, which vLLM accesses.
# ---------------------------------------------------------------------------
try:
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
    if not hasattr(TokenizersBackend, "all_special_tokens_extended"):
        TokenizersBackend.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except ImportError:
    pass

import numpy as np
import pandas as pd


def parse_optional_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def normalize_messages(prompt_value) -> list[dict[str, str]]:
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt_value]


def render_chat_prompt(tokenizer, messages, enable_thinking: bool | None) -> str:
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        normalize_messages(messages),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def stable_prompt_id(data_source: str, messages, ground_truth) -> str:
    payload = json.dumps(
        {
            "data_source": data_source,
            "prompt": normalize_messages(messages),
            "ground_truth": ground_truth,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_k_values(n: int) -> list[int]:
    """Return the list of k values to compute metrics for.
    If n is a power of 2, returns [1, 2, 4, ..., n].
    Otherwise returns [1, n] (always includes k=1 for pass@1)."""
    if is_power_of_two(n):
        return [2**i for i in range(int(math.log2(n)) + 1)]
    if n == 1:
        return [1]
    return [1, n]


def comb_estimator(n: int, c: int, k: int) -> float:
    """pass@k: probability that at least 1 of k randomly chosen samples is correct.
    n = total samples, c = number correct, k = how many we pick."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def majority_vote(predictions: list[dict]) -> float:
    """Majority vote: pick the most common prediction, return its acc value."""
    valid = [p for p in predictions if p["pred"] not in (None, "", "[NO_BOXED]")]
    if not valid:
        return 0.0
    counts = Counter(p["pred"] for p in valid)
    most_common_pred = counts.most_common(1)[0][0]
    for p in valid:
        if p["pred"] == most_common_pred:
            return float(p["acc"])
    return 0.0


def bootstrap_majority(predictions: list[dict], k: int, n_bootstrap: int = 200, seed: int = 42) -> float:
    """Bootstrap maj@k: sample k predictions, majority vote, repeat."""
    rng = np.random.RandomState(seed)
    n = len(predictions)
    if n <= k:
        return majority_vote(predictions)
    results = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=k, replace=False)
        subset = [predictions[i] for i in indices]
        results.append(majority_vote(subset))
    return float(np.mean(results))


def compute_shared_metrics(prompt_entries: list[dict], n_for_mean: int) -> dict:
    """Compute accuracy and output-format metrics once for the full run."""
    mean_vals = []
    telemetry_totals = defaultdict(int)
    total_responses = 0

    for entry in prompt_entries:
        results = entry["results"]
        accs = [r["acc"] for r in results]

        mean_vals.append(float(np.mean(accs)))

        for r in results:
            total_responses += 1
            for key in (
                "think_complete",
                "answer_complete",
                "boxed_extraction_success",
                "reward_grader_success",
                "format_contract_success",
                "has_eos",
                "truncated",
            ):
                telemetry_totals[key] += int(bool(r.get(key, False)))

    return {
        f"mean@{n_for_mean}": float(np.mean(mean_vals)),
        "n_prompts": len(prompt_entries),
        "think_complete_rate": telemetry_totals["think_complete"] / max(total_responses, 1),
        "answer_complete_rate": telemetry_totals["answer_complete"] / max(total_responses, 1),
        "boxed_extraction_success_rate": telemetry_totals["boxed_extraction_success"] / max(total_responses, 1),
        "reward_grader_success_rate": telemetry_totals["reward_grader_success"] / max(total_responses, 1),
        "format_contract_success_rate": telemetry_totals["format_contract_success"] / max(total_responses, 1),
        "eos_rate": telemetry_totals["has_eos"] / max(total_responses, 1),
        "truncation_rate": telemetry_totals["truncated"] / max(total_responses, 1),
        "extraction_fail": 1.0 - telemetry_totals["boxed_extraction_success"] / max(total_responses, 1),
    }


def compute_metrics_for_k(prompt_entries: list[dict], k: int) -> dict:
    """Compute pass@k and maj@k for a given k value."""
    pass_vals, maj_vals = [], []

    for entry in prompt_entries:
        results = entry["results"]
        n = len(results)
        accs = [r["acc"] for r in results]
        n_correct = sum(accs)

        pass_vals.append(comb_estimator(n, n_correct, min(k, n)))
        maj_vals.append(bootstrap_majority(results, min(k, n)))

    return {
        f"pass@{k}": float(np.mean(pass_vals)),
        f"maj@{k}": float(np.mean(maj_vals)),
    }


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def load_reward_function():
    """Import compute_score from custom_reward_function_latex_verify."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from custom_reward_function_latex_verify import compute_score_latex_verify
    return compute_score_latex_verify


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline vLLM inference + evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--n", type=int, default=8, help="Default number of responses per prompt (power of 2 enables multi-k pass/maj)")
    parser.add_argument("--n_per_dataset", type=str, nargs="*", default=[],
                        help="Per-dataset n overrides as path:n pairs, e.g. /data-1/dataset/AIME-2025/aime-2025.parquet:16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_files", type=str, nargs="+", default=[
        "/data-1/dataset/MATH-500/math500-test.parquet",
        "/data-1/dataset/AIME-2025/aime-2025.parquet",
    ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        type=parse_optional_bool,
        default=None,
        help="Explicit Qwen chat-template thinking switch. Pass true for the Qwen3 diversity contract.",
    )
    parser.add_argument(
        "--require-explicit-thinking",
        action="store_true",
        help="Fail unless --enable-thinking was explicitly provided and changes the rendered chat template.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Global sample-index offset for merging independent n-sample shards.",
    )
    args = parser.parse_args()

    if args.require_explicit_thinking and args.enable_thinking is None:
        parser.error("--require-explicit-thinking requires --enable-thinking true|false")
    if args.sample_offset < 0:
        parser.error("--sample-offset must be non-negative")

    # Parse per-dataset n overrides into a dict {filepath: n}
    n_overrides = {}
    for entry in args.n_per_dataset:
        if ":" not in entry:
            parser.error(f"--n_per_dataset entry must be path:n, got: {entry}")
        path_part, n_part = entry.rsplit(":", 1)
        n_overrides[path_part] = int(n_part)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load datasets and resolve per-dataset n ----
    print("=" * 60)
    print("Loading datasets ...")

    # samples_by_n groups samples by their effective n value
    # {n_value: [{"data_source", "prompt", "ground_truth", "file_path"}, ...]}
    samples_by_n: dict[int, list[dict]] = defaultdict(list)
    total_samples = 0

    for fpath in args.test_files:
        effective_n = n_overrides.get(fpath, args.n)
        df = pd.read_parquet(fpath)
        for dataset_row_index, (_, row) in enumerate(df.iterrows()):
            prompt_id = stable_prompt_id(row["data_source"], row["prompt"], row["reward_model"]["ground_truth"])
            samples_by_n[effective_n].append({
                "data_source": row["data_source"],
                "prompt": row["prompt"],  # list of dicts (chat format)
                "ground_truth": row["reward_model"]["ground_truth"],
                "dataset_path": fpath,
                "dataset_row_index": dataset_row_index,
                "prompt_id": prompt_id,
            })
        total_samples += len(df)
        n_label = f"n={effective_n}" + (" (override)" if fpath in n_overrides else " (default)")
        print(f"  {fpath}: {len(df)} samples, {n_label}")

    n_values_used = sorted(samples_by_n.keys())
    print(f"  Total: {total_samples} samples across n values: {n_values_used}")

    # ---- Build prompts ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    all_prompts_flat = []
    for n_val in n_values_used:
        for sample in samples_by_n[n_val]:
            text = render_chat_prompt(tokenizer, sample["prompt"], args.enable_thinking)
            all_prompts_flat.append(text)

    thinking_canary = None
    if all_prompts_flat:
        first_sample = samples_by_n[n_values_used[0]][0]
        rendered_true = render_chat_prompt(tokenizer, first_sample["prompt"], True)
        rendered_false = render_chat_prompt(tokenizer, first_sample["prompt"], False)
        thinking_canary = {
            "enabled_requested": args.enable_thinking,
            "template_effect": rendered_true != rendered_false,
            "rendered_true_sha256": hashlib.sha256(rendered_true.encode("utf-8")).hexdigest(),
            "rendered_false_sha256": hashlib.sha256(rendered_false.encode("utf-8")).hexdigest(),
        }
        if args.require_explicit_thinking and not thinking_canary["template_effect"]:
            raise RuntimeError("enable_thinking true/false produced identical prompts; refusing thinking-required eval")

    # ---- Initialize vLLM (once) ----
    print("=" * 60)
    print(f"Initializing vLLM (tp={args.tensor_parallel}, mem={args.gpu_memory_utilization}) ...")

    from vllm import LLM, SamplingParams

    max_prompt_tokens = max(len(tokenizer.encode(p)) for p in all_prompts_flat)
    max_model_len = min(max_prompt_tokens + args.max_tokens + 64, 32768)
    print(f"  Max prompt tokens: {max_prompt_tokens}, max_model_len: {max_model_len}")

    engine_kwargs = {}
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        max_model_len=max_model_len,
        seed=args.seed,
        **engine_kwargs,
    )

    # ---- vLLM inference (one pass per distinct n value) ----
    # outputs_by_n[n_val] = list of vLLM outputs, aligned with samples_by_n[n_val]
    outputs_by_n: dict[int, list] = {}
    total_gen_time = 0.0

    for n_val in n_values_used:
        samples = samples_by_n[n_val]
        prompts = []
        for sample in samples:
            text = render_chat_prompt(tokenizer, sample["prompt"], args.enable_thinking)
            prompts.append(text)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            n=n_val,
        )

        print("=" * 60)
        print(f"Generating n={n_val} responses per prompt for {len(prompts)} prompts ...")
        t0 = time.time()
        outputs_by_n[n_val] = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0
        total_gen_time += elapsed
        print(f"  Generation done in {elapsed:.1f}s")

    # ---- Score responses ----
    print("=" * 60)
    print("Scoring responses ...")
    compute_score = load_reward_function()

    # results_by_source[data_source] = {"n": int, "entries": [{"ground_truth", "results": [...]}]}
    results_by_source: dict[str, dict] = {}

    scored_count = 0
    for n_val in n_values_used:
        samples = samples_by_n[n_val]
        outputs = outputs_by_n[n_val]
        for sample, output in zip(samples, outputs):
            prompt_results = []
            for completion in output.outputs:
                response_text = completion.text
                result = compute_score(
                    data_source=sample["data_source"],
                    solution_str=response_text,
                    ground_truth=sample["ground_truth"],
                    extra_info={
                        "valid_response_length": len(completion.token_ids),
                        "max_resp_len": args.max_tokens,
                    },
                )
                if isinstance(result, (int, float)):
                    result = {"score": float(result), "acc": float(result) > 0, "pred": None}
                prompt_results.append({
                    "acc": bool(result.get("acc", result.get("score", 0) > 0)),
                    "score": float(result.get("score", 0)),
                    "pred": result.get("pred"),
                    "verification_method": result.get("verification_method"),
                    "think_complete": bool(result.get("think_complete", False)),
                    "answer_complete": bool(result.get("answer_complete", False)),
                    "boxed_extraction_success": bool(result.get("boxed_extraction_success", False)),
                    "reward_grader_success": bool(result.get("reward_grader_success", False)),
                    "format_contract_success": bool(result.get("format_contract_success", False)),
                    "has_eos": bool(result.get("has_eos", completion.finish_reason != "length")),
                    "truncated": bool(result.get("truncated", completion.finish_reason == "length")),
                    "finish_reason": completion.finish_reason,
                    "response_text": response_text,
                })
            ds = sample["data_source"]
            if ds not in results_by_source:
                results_by_source[ds] = {"n": n_val, "entries": []}
            results_by_source[ds]["entries"].append({
                "prompt_id": sample["prompt_id"],
                "dataset_path": sample["dataset_path"],
                "dataset_row_index": sample["dataset_row_index"],
                "ground_truth": sample["ground_truth"],
                "results": prompt_results,
            })
            scored_count += 1
            if scored_count % 200 == 0 or scored_count == total_samples:
                print(f"  Scored {scored_count}/{total_samples} prompts ...", flush=True)

    # ---- Compute metrics per data source (each with its own n) ----
    print("=" * 60)
    print("Computing metrics ...")

    # all_metrics[data_source] = {n_used, mean@n, pass@k, maj@k, ...}
    all_metrics = {}
    for data_source, info in results_by_source.items():
        ds_n = info["n"]
        ds_k_values = get_k_values(ds_n)
        metrics = compute_shared_metrics(info["entries"], ds_n)
        metrics["n_used"] = ds_n
        metrics["k_values"] = ds_k_values
        for k in ds_k_values:
            metrics.update(compute_metrics_for_k(info["entries"], k))
        all_metrics[data_source] = metrics

    macro_sources = sorted(all_metrics)
    macro_n_values = {all_metrics[source]["n_used"] for source in macro_sources}
    macro_mean_key = f"mean@{macro_n_values.pop()}" if len(macro_n_values) == 1 else "mean@configured_n"
    macro_metrics = {
        "dataset_count": len(macro_sources),
        "data_sources": macro_sources,
        macro_mean_key: float(
            np.mean(
                [all_metrics[source][f"mean@{all_metrics[source]['n_used']}"] for source in macro_sources]
            )
        ),
    }
    for key in (
        "think_complete_rate",
        "answer_complete_rate",
        "boxed_extraction_success_rate",
        "reward_grader_success_rate",
        "format_contract_success_rate",
        "eos_rate",
        "truncation_rate",
    ):
        macro_metrics[key] = float(np.mean([all_metrics[source][key] for source in macro_sources]))

    total_responses = sum(
        len(entry["results"])
        for info in results_by_source.values()
        for entry in info["entries"]
    )
    micro_metrics = {"response_count": total_responses}
    for key in (
        "think_complete",
        "answer_complete",
        "boxed_extraction_success",
        "reward_grader_success",
        "format_contract_success",
        "has_eos",
        "truncated",
    ):
        count = sum(
            int(bool(result[key]))
            for info in results_by_source.values()
            for entry in info["entries"]
            for result in entry["results"]
        )
        rate_name = "eos_rate" if key == "has_eos" else f"{key}_rate"
        micro_metrics[rate_name] = count / max(total_responses, 1)

    # ---- Print results ----
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for data_source, metrics in all_metrics.items():
        ds_n = metrics["n_used"]
        ds_k_values = metrics["k_values"]
        mean_key = f"mean@{ds_n}"
        n_prompts = metrics["n_prompts"]
        ext_fail = metrics["extraction_fail"]
        mean_v = metrics[mean_key]
        print(f"\n  [{data_source}] ({n_prompts} prompts, n={ds_n}, extraction_fail={ext_fail:.4f} ({ext_fail*100:.1f}%))")
        print(f"    {mean_key}: {mean_v:.4f} ({mean_v*100:.1f}%)")
        for k in ds_k_values:
            pass_v = metrics[f"pass@{k}"]
            maj_v = metrics[f"maj@{k}"]
            print(f"    pass@{k}: {pass_v:.4f} ({pass_v*100:.1f}%)  "
                  f"maj@{k}: {maj_v:.4f} ({maj_v*100:.1f}%)")

    # ---- Print tabular summary (for easy copy-paste) ----
    # Group data sources by n for cleaner display
    sources_by_n: dict[int, list[str]] = defaultdict(list)
    for ds, m in all_metrics.items():
        sources_by_n[m["n_used"]].append(ds)

    for ds_n in sorted(sources_by_n.keys()):
        ds_k_values = get_k_values(ds_n)
        sources = sorted(sources_by_n[ds_n])

        print("\n" + "=" * 60)
        if len(ds_k_values) > 1:
            print(f"TABULAR SUMMARY n={ds_n} (mean@{ds_n} + multi-k pass/maj)")
        else:
            print(f"TABULAR SUMMARY n={ds_n}")
        print("=" * 60)

        header = f"{'Benchmark':<25} {'Samples':>7} {'mean@'+str(ds_n):>8}"
        for k in ds_k_values:
            header += f" {'pass@'+str(k):>8} {'maj@'+str(k):>8}"
        header += f" {'ext_fail':>8}"
        print(header)
        print("-" * len(header))

        for data_source in sources:
            metrics = all_metrics[data_source]
            n_prompts = metrics["n_prompts"]
            ext_fail = metrics["extraction_fail"]
            line = f"{data_source:<25} {n_prompts:>7} {metrics[f'mean@{ds_n}']:>7.1%}"
            for k in ds_k_values:
                line += f" {metrics[f'pass@{k}']:>7.1%} {metrics[f'maj@{k}']:>7.1%}"
            line += f" {ext_fail:>7.1%}"
            print(line)

    # ---- Save results ----
    n_config = {fpath: n_overrides.get(fpath, args.n) for fpath in args.test_files}

    save_data = {
        "model_path": args.model_path,
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "n_default": args.n,
            "n_per_dataset": n_config,
            "max_tokens": args.max_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
            "sample_offset": args.sample_offset,
            "chat_template_kwargs": (
                {"enable_thinking": args.enable_thinking} if args.enable_thinking is not None else {}
            ),
            "thinking_canary": thinking_canary,
        },
        "n_values_used": n_values_used,
        "generation_time_s": total_gen_time,
        "metrics": all_metrics,
        "macro_metrics": macro_metrics,
        "micro_metrics": micro_metrics,
    }
    metrics_file = output_dir / "eval_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save per-prompt details (raw data for re-computation)
    detail_rows = []
    total_generations = 0
    for data_source, info in results_by_source.items():
        for entry in info["entries"]:
            for local_sample_index, r in enumerate(entry["results"]):
                detail_rows.append({
                    "data_source": data_source,
                    "dataset_path": entry["dataset_path"],
                    "dataset_row_index": entry["dataset_row_index"],
                    "prompt_id": entry["prompt_id"],
                    "sample_index": args.sample_offset + local_sample_index,
                    "ground_truth": entry["ground_truth"],
                    "acc": r["acc"],
                    "score": r["score"],
                    "pred": r["pred"],
                    "verification_method": r["verification_method"],
                    "think_complete": r["think_complete"],
                    "answer_complete": r["answer_complete"],
                    "boxed_extraction_success": r["boxed_extraction_success"],
                    "reward_grader_success": r["reward_grader_success"],
                    "format_contract_success": r["format_contract_success"],
                    "has_eos": r["has_eos"],
                    "truncated": r["truncated"],
                    "finish_reason": r["finish_reason"],
                    "response_text": r["response_text"],
                    "n": info["n"],
                })
                total_generations += 1
    details_file = output_dir / "eval_details.parquet"
    pd.DataFrame(detail_rows).to_parquet(str(details_file))
    print(f"Details saved to: {details_file}")
    print(f"\nTotal: {total_samples} prompts, {total_generations} generations, completed in {total_gen_time:.0f}s")


if __name__ == "__main__":
    main()
