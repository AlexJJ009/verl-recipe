"""Offline vLLM inference + evaluation on MATH-500 and AIME-2025.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python recipe/joint_training/offline_eval.py \
        --model_path /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2 \
        --tensor_parallel 4 \
        --n 3 \
        --temperature 1.0 \
        --top_p 0.95 \
        --max_tokens 4096 \
        --output_dir /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2/inference
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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
# Metric helpers (inlined to avoid import issues with recipe paths)
# ---------------------------------------------------------------------------

def comb_estimator(n: int, c: int, k: int) -> float:
    """pass@k: probability that at least 1 of k randomly chosen samples is correct.
    n = total samples, c = number correct, k = how many we pick."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def majority_vote(predictions: list[dict]) -> float:
    """Majority vote: pick the most common prediction, return its acc value."""
    from collections import Counter
    # Filter out non-answers
    valid = [p for p in predictions if p["pred"] not in (None, "", "[NO_BOXED]")]
    if not valid:
        return 0.0
    counts = Counter(p["pred"] for p in valid)
    most_common_pred = counts.most_common(1)[0][0]
    # Return the acc of any sample with the most common prediction
    for p in valid:
        if p["pred"] == most_common_pred:
            return float(p["acc"])
    return 0.0


def bootstrap_majority(predictions: list[dict], k: int, n_bootstrap: int = 200, seed: int = 42) -> tuple[float, float]:
    """Bootstrap maj@k: sample k predictions, majority vote, repeat."""
    rng = np.random.RandomState(seed)
    n = len(predictions)
    if n < k:
        return majority_vote(predictions), 0.0
    results = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=k, replace=False)
        subset = [predictions[i] for i in indices]
        results.append(majority_vote(subset))
    return float(np.mean(results)), float(np.std(results))


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
    parser.add_argument("--n", type=int, default=3, help="Number of responses per prompt")
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

    # results[data_source] = list of per-prompt dicts
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
            # Normalize result to dict
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

    # ---- Compute metrics ----
    print("=" * 60)
    print("Computing metrics ...")

    all_metrics = {}
    for data_source, prompt_entries in results_by_source.items():
        n_prompts = len(prompt_entries)
        acc_values = []  # per-prompt mean acc
        pass_at_k_values = []
        maj_at_k_values = []
        extraction_failures = 0
        verification_methods = defaultdict(int)
        total_responses = 0

        for entry in prompt_entries:
            results = entry["results"]
            n_resp = len(results)
            accs = [r["acc"] for r in results]
            n_correct = sum(accs)

            acc_values.append(np.mean(accs))
            pass_at_k_values.append(comb_estimator(n_resp, n_correct, min(args.n, n_resp)))

            # Majority vote
            maj_val = majority_vote(results)
            maj_at_k_values.append(maj_val)

            for r in results:
                total_responses += 1
                if r["pred"] in (None, "", "[NO_BOXED]"):
                    extraction_failures += 1
                if r["verification_method"]:
                    verification_methods[r["verification_method"]] += 1

        metrics = {
            f"mean@{args.n}": float(np.mean(acc_values)),
            f"pass@{args.n}": float(np.mean(pass_at_k_values)),
            f"maj@{args.n}": float(np.mean(maj_at_k_values)),
            "n_prompts": n_prompts,
            "answer_extraction_failure_rate": extraction_failures / max(total_responses, 1),
        }

        # Verification method distribution
        for method, count in verification_methods.items():
            metrics[f"verification/{method}"] = count / max(total_responses, 1)

        all_metrics[data_source] = metrics

    # ---- Print results ----
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for data_source, metrics in all_metrics.items():
        print(f"\n  [{data_source}] ({metrics['n_prompts']} prompts)")
        for k, v in metrics.items():
            if k == "n_prompts":
                continue
            if isinstance(v, float):
                print(f"    {k}: {v:.4f} ({v*100:.1f}%)")
            else:
                print(f"    {k}: {v}")

    # ---- Save results ----
    save_data = {
        "model_path": args.model_path,
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.max_tokens,
        },
        "generation_time_s": gen_time,
        "metrics": all_metrics,
    }
    metrics_file = output_dir / "eval_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save per-prompt details
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


if __name__ == "__main__":
    main()
