#!/usr/bin/env bash
# Sync staged v1 W&B offline runs after training finishes or is stopped cleanly.

set -euo pipefail

WANDB_PROJECT=${WANDB_PROJECT:-OnPolicySFT-Then-WDLSFT-StagedV1}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_SYNC_DIR=${WANDB_SYNC_DIR:-}
MARK_SYNCED=${MARK_SYNCED:-true}

usage() {
    cat <<'EOF'
Usage:
  WANDB_SYNC_DIR=/path/to/wandb/offline-run-* bash sync_wandb_offline.sh
  bash sync_wandb_offline.sh /path/to/wandb/offline-run-*

Environment:
  WANDB_PROJECT   default: OnPolicySFT-Then-WDLSFT-StagedV1
  WANDB_ENTITY    optional W&B entity/team
  MARK_SYNCED     default: true
  INCLUDE_SYNCED  default: false; set true to re-upload already synced runs
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ $# -gt 0 ]; then
    WANDB_SYNC_DIR="$1"
fi

if [ -z "$WANDB_SYNC_DIR" ]; then
    echo "ERROR: WANDB_SYNC_DIR is required." >&2
    usage >&2
    exit 2
fi
if [ ! -d "$WANDB_SYNC_DIR" ]; then
    echo "ERROR: W&B offline run dir not found: $WANDB_SYNC_DIR" >&2
    exit 2
fi
if ! command -v wandb >/dev/null 2>&1; then
    echo "ERROR: wandb CLI not found in PATH." >&2
    exit 2
fi

args=(sync "--project" "$WANDB_PROJECT")
if [ -n "$WANDB_ENTITY" ]; then
    args+=("--entity" "$WANDB_ENTITY")
fi
if [ "$MARK_SYNCED" = "true" ]; then
    args+=("--mark-synced")
fi
if [ "${INCLUDE_SYNCED:-false}" = "true" ]; then
    args+=("--include-synced")
fi
args+=("$WANDB_SYNC_DIR")

echo "[staged_v1/wandb] project=$WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then
    echo "[staged_v1/wandb] entity=$WANDB_ENTITY"
fi
echo "[staged_v1/wandb] sync_dir=$WANDB_SYNC_DIR"
exec wandb "${args[@]}"
