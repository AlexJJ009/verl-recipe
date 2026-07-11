#!/usr/bin/env python3
"""Render deterministic Stage1/2/3 preflight benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


PHASES = ("stage1", "stage2", "stage3")
DATASETS = ("HumanEval+", "MBPP+", "LiveCodeBench")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def aggregate(phase: dict) -> dict:
    repetitions = [item for item in phase["repetitions"] if not item.get("warmup")]
    assert len(repetitions) == 3
    submitted = sum(item["submitted"] for item in repetitions)
    completed = sum(item["completed"] for item in repetitions)
    valid = sum(item["valid_scores"] for item in repetitions)
    timeouts = sum(item["timeouts"] for item in repetitions)
    elapsed = [float(item["validation_elapsed_seconds"]) for item in repetitions]
    scorer_latencies = [float(value) for item in repetitions for value in item["scorer_latencies_seconds"]]
    distribution: dict[str, int] = {}
    for item in repetitions:
        for key, value in item["score_distribution"].items():
            distribution[key] = distribution.get(key, 0) + int(value)
    return {
        "rollout_latency_seconds": statistics.median(float(item["rollout_latency_seconds"]) for item in repetitions),
        "rollout_tokens_per_second": statistics.median(float(item["rollout_tokens_per_second"]) for item in repetitions),
        "scorer_latency_p50_seconds": percentile(scorer_latencies, 0.50),
        "scorer_latency_p95_seconds": percentile(scorer_latencies, 0.95),
        "timeout_rate": round(timeouts / submitted, 6),
        "invalid_score_rate": round((completed - valid) / submitted, 6),
        "valid_score_rate": round(valid / submitted, 6),
        "valid_scores_per_minute": round(valid / (sum(elapsed) / 60), 6),
        "score_distribution": dict(sorted(distribution.items())),
        "peak_rss_gib": max(float(item["peak_rss_gib"]) for item in repetitions),
        "gpu_wait_fraction": statistics.median(float(item["gpu_wait_fraction"]) for item in repetitions),
        "validation_elapsed_seconds": statistics.median(elapsed),
        "complete_validation_metrics": all(bool(item["complete_validation_metrics"]) for item in repetitions),
    }


def render(fixture: Path) -> dict:
    source = json.loads((fixture / "benchmark_input.json").read_text())
    contract = source["contract"]
    assert contract["max_response_length"] == 8192
    assert tuple(contract["validation_datasets"]) == DATASETS
    output_phases = []
    for expected, phase in zip(PHASES, source["phases"], strict=True):
        assert phase["phase"] == expected
        assert {item["name"] for item in phase["datasets"]} == set(DATASETS)
        if expected == "stage2":
            assert phase["model_topology"] == "fixed_model2_rollout_joint_fused_loss"
            assert phase.get("fixed_model2_source")
        output_phases.append({**phase, "metrics": aggregate(phase)})
    semantic = {
        "contract": contract,
        "phases": [{"phase": item["phase"], "model_topology": item["model_topology"], "datasets": item["datasets"], "profile_hash": item["profile_hash"]} for item in output_phases],
    }
    return {
        "schema_version": 1,
        "evidence_class": "infrastructure_preflight",
        "contract": contract,
        "semantic_hash": hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "phases": output_phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "output": str(args.output), "semantic_hash": report["semantic_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
