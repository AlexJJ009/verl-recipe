"""Offline vLLM inference + evaluation on math benchmarks.

The script always computes a single mean@n using all n responses per prompt.
When n is a power of 2, it additionally computes pass@k and maj@k for
all k in {1, 2, 4, ..., n}. Otherwise it computes pass@n and maj@n only.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python recipe/joint_training/offline_eval.py \
        --model_path /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2 \
        --tensor_parallel 4 \
        --n 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --max_tokens 4096 \
        --output_dir /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2/inference_n8
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Compatibility patch: transformers 5.x TokenizersBackend lacks
# all_special_tokens_extended, which vLLM 0.8.5 accesses.
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_k_values(n: int) -> list[int]:
    """Return the list of k values to compute metrics for.
    If n is a power of 2, returns [1, 2, 4, ..., n].
    Otherwise returns [n]."""
    if is_power_of_two(n):
        return [2**i for i in range(int(math.log2(n)) + 1)]
    return [n]


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
    """Compute mean@n and extraction-failure rate once for the full run."""
    mean_vals = []
    extraction_failures = 0
    total_responses = 0

    for entry in prompt_entries:
        results = entry["results"]
        accs = [r["acc"] for r in results]

        mean_vals.append(float(np.mean(accs)))

        for r in results:
            total_responses += 1
            if r["pred"] in (None, "", "[NO_BOXED]"):
                extraction_failures += 1

    return {
        f"mean@{n_for_mean}": float(np.mean(mean_vals)),
        "n_prompts": len(prompt_entries),
        "extraction_fail": extraction_failures / max(total_responses, 1),
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
    parser.add_argument("--n", type=int, default=8, help="Number of responses per prompt (power of 2 enables multi-k pass/maj)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_files", type=str, nargs="+", default=[
        "/data-1/dataset/MATH-500/math500-test.parquet",
        "/data-1/dataset/AIME-2025/aime-2025.parquet",
    ])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = get_k_values(args.n)
    print(f"n={args.n}, k_values={k_values} ({'multi-k (power of 2)' if len(k_values) > 1 else 'single-k'})")

    # ---- Load datasets ----
    print("=" * 60)
    print("Loading datasets ...")
    all_samples = []
    for fpath in args.test_files:
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            all_samples.append({
                "data_source": row["data_source"],
                "prompt": row["prompt"],  # list of dicts (chat format)
                "ground_truth": row["reward_model"]["ground_truth"],
            })
        print(f"  {fpath}: {len(df)} samples")
    print(f"  Total: {len(all_samples)} samples")

    # ---- Build prompts ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts = []
    for sample in all_samples:
        text = tokenizer.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    # ---- vLLM inference ----
    print("=" * 60)
    print(f"Initializing vLLM (tp={args.tensor_parallel}, mem={args.gpu_memory_utilization}) ...")

    from vllm import LLM, SamplingParams

    max_prompt_tokens = max(len(tokenizer.encode(p)) for p in prompts)
    max_model_len = min(max_prompt_tokens + args.max_tokens + 64, 32768)
    print(f"  Max prompt tokens: {max_prompt_tokens}, max_model_len: {max_model_len}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=max_model_len,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    print(f"Generating {args.n} responses per prompt for {len(prompts)} prompts ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"  Generation done in {gen_time:.1f}s")

    # ---- Score responses ----
    print("=" * 60)
    print("Scoring responses ...")
    compute_score = load_reward_function()

    results_by_source = defaultdict(list)

    for idx, (sample, output) in enumerate(zip(all_samples, outputs)):
        prompt_results = []
        for completion in output.outputs:
            response_text = completion.text
            result = compute_score(
                data_source=sample["data_source"],
                solution_str=response_text,
                ground_truth=sample["ground_truth"],
            )
            if isinstance(result, (int, float)):
                result = {"score": float(result), "acc": float(result) > 0, "pred": None}
            prompt_results.append({
                "acc": bool(result.get("acc", result.get("score", 0) > 0)),
                "score": float(result.get("score", 0)),
                "pred": result.get("pred"),
                "verification_method": result.get("verification_method"),
                "response_text": response_text,
            })
        results_by_source[sample["data_source"]].append({
            "ground_truth": sample["ground_truth"],
            "results": prompt_results,
        })

    # ---- Compute metrics for all k values ----
    print("=" * 60)
    print(f"Computing metrics for k={k_values} ...")

    # all_metrics[data_source] = {mean@n, pass@k, maj@k, ...}
    all_metrics = {}
    for data_source, prompt_entries in results_by_source.items():
        all_metrics[data_source] = compute_shared_metrics(prompt_entries, args.n)
        for k in k_values:
            all_metrics[data_source].update(compute_metrics_for_k(prompt_entries, k))

    # ---- Print results ----
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for data_source in all_metrics:
        mean_key = f"mean@{args.n}"
        n_prompts = all_metrics[data_source]["n_prompts"]
        ext_fail = all_metrics[data_source]["extraction_fail"]
        mean_v = all_metrics[data_source][mean_key]
        print(f"\n  [{data_source}] ({n_prompts} prompts, extraction_fail={ext_fail:.4f} ({ext_fail*100:.1f}%))")
        print(f"    {mean_key}: {mean_v:.4f} ({mean_v*100:.1f}%)")
        for k in k_values:
            metrics = all_metrics[data_source]
            pass_v = metrics[f"pass@{k}"]
            maj_v = metrics[f"maj@{k}"]
            print(f"    pass@{k}: {pass_v:.4f} ({pass_v*100:.1f}%)  "
                  f"maj@{k}: {maj_v:.4f} ({maj_v*100:.1f}%)")

    # ---- Print tabular summary (for easy copy-paste) ----
    if len(k_values) > 1:
        print("\n" + "=" * 60)
        print("TABULAR SUMMARY (mean@n + multi-k pass/maj)")
        print("=" * 60)
        # Header
        header = f"{'Benchmark':<25} {'Samples':>7} {'mean@'+str(args.n):>8}"
        for k in k_values:
            header += f" {'pass@'+str(k):>8} {'maj@'+str(k):>8}"
        header += f" {'ext_fail':>8}"
        print(header)
        print("-" * len(header))

        for data_source in sorted(all_metrics.keys()):
            n_prompts = all_metrics[data_source]["n_prompts"]
            ext_fail = all_metrics[data_source]["extraction_fail"]
            line = f"{data_source:<25} {n_prompts:>7} {all_metrics[data_source][f'mean@{args.n}']:>7.1%}"
            for k in k_values:
                metrics = all_metrics[data_source]
                line += f" {metrics[f'pass@{k}']:>7.1%} {metrics[f'maj@{k}']:>7.1%}"
            line += f" {ext_fail:>7.1%}"
            print(line)

    # ---- Save results ----

    save_data = {
        "model_path": args.model_path,
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "k_values": k_values,
        "generation_time_s": gen_time,
        "metrics": all_metrics,
    }
    metrics_file = output_dir / "eval_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save per-prompt details (raw data for re-computation)
    detail_rows = []
    for data_source, prompt_entries in results_by_source.items():
        for entry in prompt_entries:
            for r in entry["results"]:
                detail_rows.append({
                    "data_source": data_source,
                    "ground_truth": entry["ground_truth"],
                    "acc": r["acc"],
                    "score": r["score"],
                    "pred": r["pred"],
                    "verification_method": r["verification_method"],
                    "response_text": r["response_text"],
                })
    details_file = output_dir / "eval_details.parquet"
    pd.DataFrame(detail_rows).to_parquet(str(details_file))
    print(f"Details saved to: {details_file}")
    print(f"\nTotal: {len(all_samples)} prompts × {args.n} = {len(detail_rows)} generations, completed in {gen_time:.0f}s")


if __name__ == "__main__":
    main()
