#!/usr/bin/env python3
"""Fail closed on missing or contradictory GON-37 classifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=SCRIPT_DIR / "job_130_baseline.json")
    parser.add_argument("--audit", type=Path, default=SCRIPT_DIR / "scheduler_audit.json")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["verdict"] != "adapter_contract_ready_runtime_qualification_deferred":
        fail("unexpected scheduler audit verdict")
    if audit["submission_gate"]["full_gpu_submission_allowed"] is not False:
        fail("GON-36 must not authorize a full GPU submission")
    if audit["historical_lane"]["config_only_directly_rerun"] is not False:
        fail("unavailable historical lane was promoted without evidence")

    common = audit["common_boundary"]
    entry = baseline["entry"]
    if common["entry_path"] != entry["path"] or common["entry_git_blob"] != entry["git_blob"]:
        fail("common experiment entry drifted")
    if common["common_path"] != entry["common_path"] or common["common_git_blob"] != entry["common_git_blob"]:
        fail("common launcher drifted")
    if common["required_invocations"] != 1:
        fail("common entry must be invoked exactly once")

    config_keys = audit["p0"]["config_only_keys"]
    if len(config_keys) != len(set(config_keys)):
        fail("duplicate P0 config-only classification")
    required_canaries = {
        "learning_rate",
        "total_training_steps",
        "total_epochs",
        "actor_seed",
        "rollout_seed",
        "data_seed",
        "init_model_path",
        "train_file",
        "train_prompt_bsz",
        "rollout_n",
        "ppo_mini_batch_size",
        "loss_mode",
        "loss_agg_mode",
        "rollout_is",
        "enable_thinking",
    }
    missing = sorted(required_canaries - set(config_keys))
    if missing:
        fail(f"P0 audit is missing canaries: {missing}")

    for classification in ("p1", "p2"):
        rows = audit[classification]
        fields = [row.get("field") for row in rows]
        if None in fields or len(fields) != len(set(fields)):
            fail(f"{classification.upper()} fields are missing or duplicated")
        for row in rows:
            for key in ("slurm", "pueue_a800", "decision"):
                if not str(row.get(key, "")).strip():
                    fail(f"unclassified {classification.upper()} value for {row.get('field')}: {key}")
    print("ok: scheduler audit has complete P0/P1/P2 decisions and remains GPU fail-closed")


if __name__ == "__main__":
    main()
