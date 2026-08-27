#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 7 ]] || {
    echo "ERROR: worker expects repo, output, receipt, runtime-env, root candidate, recipe candidate and digest" >&2
    exit 64
}
repo_root=$(realpath -m -- "$1")
output_root=$(realpath -m -- "$2")
receipt_root=$(realpath -m -- "$3")
runtime_env_file=$(realpath -m -- "$4")
expected_root_candidate=$5
expected_recipe_candidate=$6
expected_runtime_env_sha256=$7

: "${PUEUE_TASK_ID:?Pueue native task identity is required}"
[[ "$PUEUE_TASK_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid PUEUE_TASK_ID" >&2; exit 64; }
[[ -f "$runtime_env_file" ]] || { echo "ERROR: runtime environment file missing" >&2; exit 66; }
[[ "$expected_root_candidate" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid root candidate SHA" >&2; exit 64; }
[[ "$expected_recipe_candidate" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid recipe candidate SHA" >&2; exit 64; }
[[ "$expected_runtime_env_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: invalid runtime env SHA256" >&2; exit 64; }

actual_root_candidate=$(git -C "$repo_root" rev-parse HEAD)
[[ "$actual_root_candidate" == "$expected_root_candidate" ]] || {
    echo "ERROR: queued root checkout drifted from the admitted candidate" >&2
    exit 78
}
root_checkout_changes=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
[[ -z "$root_checkout_changes" ]] || {
    echo "ERROR: queued root checkout has tracked, staged or untracked changes" >&2
    exit 78
}
actual_recipe_candidate=$(git -C "$repo_root/recipe" rev-parse HEAD)
[[ "$actual_recipe_candidate" == "$expected_recipe_candidate" ]] || {
    echo "ERROR: queued recipe checkout drifted from the admitted candidate" >&2
    exit 78
}
checkout_changes=$(git -C "$repo_root/recipe" status --porcelain=v1 --untracked-files=all)
[[ -z "$checkout_changes" ]] || {
    echo "ERROR: queued recipe checkout has tracked, staged or untracked changes" >&2
    exit 78
}
gitlink_candidate=$(git -C "$repo_root" ls-tree HEAD recipe | awk '{print $3}')
[[ "$gitlink_candidate" == "$actual_recipe_candidate" ]] || {
    echo "ERROR: queued Recipe checkout no longer matches the root gitlink" >&2
    exit 78
}
umask 077
runtime_env_snapshot=$(mktemp "/tmp/gon36-runtime-${PUEUE_TASK_ID}.XXXXXX")
cleanup_runtime_env_snapshot() {
    rm -f -- "$runtime_env_snapshot"
}
trap cleanup_runtime_env_snapshot EXIT
cp -- "$runtime_env_file" "$runtime_env_snapshot"
chmod 0400 "$runtime_env_snapshot"
actual_runtime_env_sha256=$(python3 - "$runtime_env_snapshot" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)
[[ "$actual_runtime_env_sha256" == "$expected_runtime_env_sha256" ]] || {
    echo "ERROR: runtime environment changed after admission" >&2
    exit 78
}

set -a
# Source only the private task-local snapshot whose exact bytes were checked above.
# shellcheck disable=SC1090
source "$runtime_env_snapshot"
set +a
cleanup_runtime_env_snapshot
trap - EXIT

export PUEUE_GRPO_NATIVE_TASK_ID="$PUEUE_TASK_ID"
export GRPO_SCHEDULER_MANAGED=1
export GRPO_LAUNCH_ALLOWED=1
export BASE_CKPT_DIR="${output_root}/checkpoints"
export LOG_DIR="${output_root}/logs"
export WANDB_DIR="${output_root}/wandb"
export GRPO_ADMISSION_RECEIPT="${receipt_root}/runtime-${PUEUE_TASK_ID}.json"
mkdir -p "$BASE_CKPT_DIR" "$LOG_DIR" "$WANDB_DIR" "$receipt_root"

command -v verl-dev-run >/dev/null 2>&1 || { echo "ERROR: verl-dev-run is not installed" >&2; exit 69; }
exec verl-dev-run --repo "$repo_root" --a800-dev-profile -- \
    bash recipe/on_policy_wdl_sft/standard_grpo/run_math_stage1_grpo.sh
