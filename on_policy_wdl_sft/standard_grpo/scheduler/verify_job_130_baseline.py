#!/usr/bin/env python3
"""Verify the immutable, source-backed parts of the GON-40 baseline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_ROOT = SCRIPT_DIR.parents[2]
MANIFEST = SCRIPT_DIR / "job_130_baseline.json"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=RECIPE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        fail(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_only(manifest: dict[str, object]) -> dict[str, str]:
    launch = manifest["effective_launch"]
    assert isinstance(launch, dict)
    required_environment = launch["required_environment"]
    assert isinstance(required_environment, dict)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "GRPO_CONFIG_ONLY": "1",
        "STAGE1_MODEL_PATH": str(required_environment["STAGE1_MODEL_PATH"]),
        "TOTAL_TRAINING_STEPS": str(required_environment["TOTAL_TRAINING_STEPS"]),
        "TOTAL_EPOCHS": str(required_environment["TOTAL_EPOCHS"]),
    }
    entry = manifest["entry"]
    assert isinstance(entry, dict)
    result = subprocess.run(
        ["bash", str(RECIPE_ROOT / str(entry["path"]))],
        cwd=RECIPE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        fail(f"GRPO_CONFIG_ONLY failed: {result.stderr.strip()}")
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            fail(f"unclassified config-only line: {line!r}")
        key, value = line.split("=", 1)
        if key in parsed:
            fail(f"duplicate config-only key: {key}")
        parsed[key] = value
    return parsed


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    entry = manifest["entry"]
    source = manifest["source_identity"]
    p0 = manifest["p0"]
    evidence = manifest["evidence"]
    if selection["job_id"] != 130 or selection["local_terminal_step"] != 160:
        fail("canonical Job 130 terminal identity drifted")
    if selection["effective_terminal_step"] != 200:
        fail("effective terminal step drifted")
    if evidence["historical_runtime_lane"]["available_in_this_checkout"] is not False:
        fail("historical runtime lane must not be promoted without direct evidence")

    recipe_commit = str(source["recipe_commit"])
    git("cat-file", "-e", f"{recipe_commit}^{{commit}}")
    for path_key, blob_key, sha_key in (
        ("path", "git_blob", "sha256"),
        ("common_path", "common_git_blob", "common_sha256"),
    ):
        path = str(entry[path_key])
        expected_blob = str(entry[blob_key])
        expected_sha = str(entry[sha_key])
        if git("rev-parse", f"{recipe_commit}:{path}") != expected_blob:
            fail(f"frozen blob mismatch for {path}")
        if git("rev-parse", f"HEAD:{path}") != expected_blob:
            fail(f"current candidate changed frozen experiment entry {path}")
        if sha256(RECIPE_ROOT / path) != expected_sha:
            fail(f"working-tree sha256 mismatch for {path}")

    output = config_only(manifest)
    expected = {
        "task": "math",
        "pipeline": "stage1_grpo",
        "learning_rate": str(p0["optimization"]["learning_rate"]),
        "total_training_steps": str(p0["optimization"]["total_training_steps"]),
        "total_epochs": str(p0["optimization"]["total_epochs"]),
        "train_prompt_bsz": str(p0["optimization"]["prompt_batch_size"]),
        "rollout_n": str(p0["optimization"]["rollout_n"]),
        "ppo_mini_batch_size": str(p0["optimization"]["ppo_mini_batch_size_prompt_groups"]),
        "actor_seed": str(p0["randomness"]["actor_seed"]),
        "rollout_seed": str(p0["randomness"]["rollout_seed"]),
        "data_seed": str(p0["randomness"]["data_seed"]),
        "train_file": str(p0["data"]["path"]),
        "init_model_path": str(p0["model"]["path"]),
        "loss_mode": str(p0["optimization"]["loss_mode"]),
        "loss_agg_mode": str(p0["optimization"]["loss_aggregation"]),
        "rollout_is": "null",
        "enable_thinking": "True",
        "resume_mode": "disable",
    }
    unknown = sorted(set(expected) - set(output))
    if unknown:
        fail(f"missing config-only keys: {unknown}")
    mismatches = {
        key: {"expected": value, "actual": output[key]} for key, value in expected.items() if output[key] != value
    }
    if mismatches:
        fail(f"config-only drift: {json.dumps(mismatches, sort_keys=True)}")
    print("ok: Job 130 baseline identity and config-only contract verified")


if __name__ == "__main__":
    main()
