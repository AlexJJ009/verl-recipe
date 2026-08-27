#!/usr/bin/env python3
"""Static fail-closed checks for the thin GON-41 Pueue adapter."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ENTRY = "recipe/on_policy_wdl_sft/standard_grpo/run_math_stage1_grpo.sh"
SCIENTIFIC_VARIABLES = {
    "ACTOR_GRAD_CLIP",
    "ACTOR_SHUFFLE",
    "CLIP_RATIO_HIGH",
    "CLIP_RATIO_LOW",
    "CUSTOM_REWARD_FN_NAME",
    "CUSTOM_REWARD_FN_PATH",
    "DATA_SEED",
    "DATA_SHUFFLE",
    "ENABLE_THINKING",
    "INIT_MODEL_PATH",
    "KL_LOSS_COEF",
    "KL_LOSS_TYPE",
    "LOSS_AGG_MODE",
    "LOSS_MODE",
    "LR",
    "LR_WARMUP_STEPS",
    "MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH",
    "NORM_ADV_BY_STD_IN_GRPO",
    "PPO_EPOCHS",
    "REF_FSDP_OFFLOAD",
    "ROLLOUT_GPU_MEMORY_UTILIZATION",
    "ROLLOUT_IS",
    "ROLLOUT_N",
    "ROLLOUT_SEED",
    "ROLLOUT_TP_SIZE",
    "STAGE1_MODEL_PATH",
    "TEMPERATURE",
    "TOP_K",
    "TOP_P",
    "TOTAL_EPOCHS",
    "TOTAL_TRAINING_STEPS",
    "TRAIN_FILE",
    "TRAINING_SEED",
    "TRAIN_PROMPT_BSZ",
    "TRAIN_PROMPT_MINI_BSZ",
    "USE_KL_IN_REWARD",
    "USE_KL_LOSS",
    "VAL_N",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    submit = args.adapter_dir / "submit_math_stage1_grpo.sh"
    worker = args.adapter_dir / "worker_math_stage1_grpo.sh"
    if not submit.is_file() or not worker.is_file():
        fail("adapter shell files are missing")
    sources = {path.name: path.read_text(encoding="utf-8") for path in (submit, worker)}
    combined = "\n".join(sources.values())
    if combined.count(ENTRY) != 1:
        fail("adapter must reach the exact experiment entry exactly once")
    if sources[worker.name].count("command -v verl-dev-run") != 1:
        fail("worker must check verl-dev-run exactly once")
    if sources[worker.name].count("exec verl-dev-run") != 1:
        fail("worker must invoke verl-dev-run exactly once")
    if "pueue add" not in sources[submit.name] or "--group gpu8" not in sources[submit.name]:
        fail("submitter must use Pueue group gpu8")
    if "printf -v task_command_shell '%q '" not in sources[submit.name]:
        fail("submitter must shell-quote the complete Pueue command")
    if '"$task_command_shell"' not in sources[submit.name]:
        fail("Pueue must receive the pre-quoted command as one argument")
    if "--print-task-id" not in sources[submit.name] or "PUEUE_TASK_ID" not in sources[worker.name]:
        fail("adapter must preserve the Pueue native task ID")
    immutable_worker_runner = 'git -C "$repo_root/recipe" show "${recipe_candidate}:${worker_path}" | bash -s -- "$@"'
    if immutable_worker_runner not in sources[submit.name]:
        fail("submitter must execute worker bytes from the admitted candidate Git object")
    required_worker_guards = (
        'actual_root_candidate=$(git -C "$repo_root" rev-parse HEAD)',
        '[[ "$actual_root_candidate" == "$expected_root_candidate" ]]',
        'git -C "$repo_root" status --porcelain=v1 --untracked-files=all',
        'actual_recipe_candidate=$(git -C "$repo_root/recipe" rev-parse HEAD)',
        '[[ "$actual_recipe_candidate" == "$expected_recipe_candidate" ]]',
        'git -C "$repo_root/recipe" status --porcelain=v1 --untracked-files=all',
        '[[ -z "$checkout_changes" ]]',
        'gitlink_candidate=$(git -C "$repo_root" ls-tree HEAD recipe',
        '[[ "$gitlink_candidate" == "$actual_recipe_candidate" ]]',
        'runtime_env_snapshot=$(mktemp "/tmp/gon36-runtime-${PUEUE_TASK_ID}.XXXXXX")',
        'source "$runtime_env_snapshot"',
    )
    for guard in required_worker_guards:
        if guard not in sources[worker.name]:
            fail(f"worker is missing fail-closed guard: {guard}")

    forbidden = {
        "sbatch": r"\bsbatch\b",
        "srun": r"\bsrun\b",
        "meituan": r"\bmeituan\b",
        "nested Docker": r"\bdocker\b",
        "L40S launcher": r"run_train\.sh",
        "tmux wrapper": r"\btmux\b",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, combined, re.IGNORECASE):
            fail(f"adapter contains forbidden {label} invocation")

    for name in sorted(SCIENTIFIC_VARIABLES):
        assignment = rf"(?m)^\s*(?:export\s+)?{re.escape(name)}\s*="
        if re.search(assignment, combined):
            fail(f"adapter assigns scientific variable {name}")
    print("ok: Pueue adapter is thin, native-ID preserving, and scientifically neutral")


if __name__ == "__main__":
    main()
