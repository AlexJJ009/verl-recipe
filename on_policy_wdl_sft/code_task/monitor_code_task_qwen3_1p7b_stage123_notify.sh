#!/usr/bin/env bash
# Monitor and release the FRAC25/FRAC50 Qwen3-1.7B Stage123 chains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export STAGE123_MANIFEST=${STAGE123_MANIFEST:-${REPO_ROOT}/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
export STAGE123_MANIFEST_TOOL=${STAGE123_MANIFEST_TOOL:-${REPO_ROOT}/scripts/experiment_manifest.py}
manifest_json=$(mktemp)
trap 'rm -f "$manifest_json"' EXIT
python3 "$STAGE123_MANIFEST_TOOL" render "$STAGE123_MANIFEST" --format json > "$manifest_json"

export MONITOR_NAME=${MONITOR_NAME:-$(jq -r .monitor.name "$manifest_json")}
export QUEUE_TMUX=${QUEUE_TMUX:-$(jq -r .monitor.queue_tmux "$manifest_json")}
export CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
export METRICS_ROOT=${METRICS_ROOT:-${SCRIPT_DIR}/metrics}
export EXTRA_METRICS_ROOTS=${EXTRA_METRICS_ROOTS:-${REPO_ROOT}/recipe/on_policy_wdl_sft/staged_v1/metrics}
export WANDB_PROJECT=${WANDB_PROJECT:-$(jq -r .release.project "$manifest_json")}
export POLL_SEC=${POLL_SEC:-$(jq -r .monitor.poll_seconds "$manifest_json")}
export LOG_FILE=${LOG_FILE:-${SCRIPT_DIR}/monitor_code_task_qwen3_1p7b_stage123_notify.log}
export TRAINING_RELEASE_GATE_SCRIPT=${TRAINING_RELEASE_GATE_SCRIPT:-${REPO_ROOT}/scripts/training_result_release_gate.py}
export TRAINING_RELEASE_GATE_SHELL=${TRAINING_RELEASE_GATE_SHELL:-${REPO_ROOT}/scripts/training_release_gate_shell.sh}
export EXPERIMENT_NORMALIZED_MANIFEST="$manifest_json"
export EXPERIMENT_PREFLIGHT_RECEIPT=${EXPERIMENT_PREFLIGHT_RECEIPT:-}
export TRAINING_RELEASE_SUCCESS_HOOK=${TRAINING_RELEASE_SUCCESS_HOOK:-"REPO='${REPO_ROOT}' VERL_REPO_ROOT='${REPO_ROOT}' EXPERIMENT_NORMALIZED_MANIFEST='${manifest_json}' EXPERIMENT_PREFLIGHT_RECEIPT='${EXPERIMENT_PREFLIGHT_RECEIPT}' EXPERIMENT_REGISTRY_DB='/data-1/experiment_registry/experiment_registry.sqlite' REGISTRY_DB='/data-1/experiment_registry/experiment_registry.sqlite' WANDB_ROOT='/data-1/wandb_runs' bash '${REPO_ROOT}/scripts/stage123_manifest_release_dispatch.sh'"}

mapfile -t RUN_PREFIXES < <(jq -r '.runs[].run_prefix' "$manifest_json")
mapfile -t TMUX_NAMES < <(jq -r '.runs[].tmux_name' "$manifest_json")
mapfile -t FINAL_STEPS < <(jq -r '.runs[].final_step' "$manifest_json")
mapfile -t TRAIN_FILES < <(jq -r '.runs[].train_file' "$manifest_json")
export EXPERIMENT_MANIFEST_PATH="$STAGE123_MANIFEST"
export EXPERIMENT_MANIFEST_SHA256=$(jq -r .manifest_sha256 "$manifest_json")

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/training_queue_monitor.sh"
training_queue_monitor_main
