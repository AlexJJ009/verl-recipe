"""Compute mean@n plus pass@k/maj@k from eval_details.parquet."""

import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def comb_estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def majority_vote(preds_and_accs: list[tuple]) -> float:
    valid = [(p, a) for p, a in preds_and_accs if p not in (None, "", "[NO_BOXED]")]
    if not valid:
        return 0.0
    counts = Counter(p for p, a in valid)
    most_common = counts.most_common(1)[0][0]
    for p, a in valid:
        if p == most_common:
            return float(a)
    return 0.0


def bootstrap_majority(preds_and_accs: list[tuple], k: int, n_bootstrap: int = 200, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    n = len(preds_and_accs)
    if n <= k:
        return majority_vote(preds_and_accs)
    results = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=k, replace=False)
        subset = [preds_and_accs[i] for i in indices]
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
        mean_vals.append(np.mean(accs))

        for r in results:
            total_responses += 1
            if r["pred"] in (None, "", "[NO_BOXED]"):
                extraction_failures += 1

    return {
        f"mean@{n_for_mean}": float(np.mean(mean_vals)),
        "extraction_fail": extraction_failures / max(total_responses, 1),
    }


def compute_metrics_for_k(prompt_entries: list[dict], k: int) -> dict:
    """Compute pass@k and maj@k for a given k."""
    pass_vals, maj_vals = [], []

    for entry in prompt_entries:
        results = entry["results"]
        n = len(results)
        accs = [r["acc"] for r in results]
        n_correct = sum(accs)

        pass_vals.append(comb_estimator(n, n_correct, min(k, n)))
        preds_and_accs = [(r["pred"], r["acc"]) for r in results]
        maj_vals.append(bootstrap_majority(preds_and_accs, min(k, n)))

    return {
        f"pass@{k}": float(np.mean(pass_vals)),
        f"maj@{k}": float(np.mean(maj_vals)),
    }


def main():
    parquet_path = sys.argv[1]
    k_values = [int(x) for x in sys.argv[2].split(",")]

    df = pd.read_parquet(parquet_path)

    n_per_prompt = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    # Group by data_source, then split into chunks of n_per_prompt
    grouped = defaultdict(list)
    for data_source, source_df in df.groupby("data_source", sort=False):
        rows = list(source_df.itertuples(index=False))
        for i in range(0, len(rows), n_per_prompt):
            chunk = rows[i:i + n_per_prompt]
            results = [{"acc": bool(r.acc), "pred": r.pred} for r in chunk]
            grouped[data_source].append({"results": results})

    # Compute and print metrics
    print(f"{'Benchmark':<25} {'Samples':>7} {'mean@'+str(n_per_prompt):>8}", end="")
    for k in k_values:
        print(f" {'pass@'+str(k):>8} {'maj@'+str(k):>8}", end="")
    print(f" {'ext_fail':>8}")
    print("-" * (25 + 7 + 9 + len(k_values) * 18 + 9))

    for data_source in sorted(grouped.keys()):
        entries = grouped[data_source]
        metrics = compute_shared_metrics(entries, n_per_prompt)
        print(f"{data_source:<25} {len(entries):>7} {metrics[f'mean@{n_per_prompt}']:>7.1%}", end="")
        for k in k_values:
            metrics.update(compute_metrics_for_k(entries, k))
            print(f" {metrics[f'pass@{k}']:>7.1%} {metrics[f'maj@{k}']:>7.1%}", end="")
        print(f" {metrics['extraction_fail']:>7.1%}")


if __name__ == "__main__":
    main()
