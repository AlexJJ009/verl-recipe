#!/usr/bin/env bash
# Approval-gated representative GPU smoke; never formal experiment evidence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PREFLIGHT_MANIFEST:?PREFLIGHT_MANIFEST is required}"
: "${PREFLIGHT_OUTPUT_ROOT:=/data-1/tmp/verl_agent_scratch/experiment_workflow/gpu_smoke}"
: "${PREFLIGHT_TMUX_NAME:=code_task_preflight_gpu_smoke}"

[ "${ALLOW_CODE_PREFLIGHT_GPU_SMOKE:-0}" = 1 ] || {
  echo "ERROR: GPU smoke requires ALLOW_CODE_PREFLIGHT_GPU_SMOKE=1" >&2
  exit 1
}
[ -f "$PREFLIGHT_MANIFEST" ] || { echo "ERROR: missing preflight manifest: $PREFLIGHT_MANIFEST" >&2; exit 1; }
case "$PREFLIGHT_OUTPUT_ROOT" in
  /data-1/tmp/verl_agent_scratch/*|/data-2/tmp/*) ;;
  *) echo "ERROR: GPU smoke output must use a declared scratch root" >&2; exit 1 ;;
esac
mkdir -p "$PREFLIGHT_OUTPUT_ROOT"

if [ "${PREFLIGHT_GPU_SMOKE_DRY_RUN:-0}" = 1 ]; then
  printf 'evidence_class=infrastructure_preflight\ntmux=%s\nmanifest=%s\noutput=%s\n' \
    "$PREFLIGHT_TMUX_NAME" "$PREFLIGHT_MANIFEST" "$PREFLIGHT_OUTPUT_ROOT"
  exit 0
fi

tmux has-session -t "$PREFLIGHT_TMUX_NAME" 2>/dev/null && {
  echo "ERROR: tmux session already exists: $PREFLIGHT_TMUX_NAME" >&2
  exit 1
}
tmux new-session -d -s "$PREFLIGHT_TMUX_NAME" \
  "cd '$SCRIPT_DIR/../../..' && exec bash '$SCRIPT_DIR/run_code_task_preflight_gpu_smoke_worker.sh' '$PREFLIGHT_MANIFEST' '$PREFLIGHT_OUTPUT_ROOT'"
echo "started infrastructure preflight tmux=$PREFLIGHT_TMUX_NAME"
