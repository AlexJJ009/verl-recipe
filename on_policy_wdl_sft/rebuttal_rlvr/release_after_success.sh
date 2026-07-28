#!/usr/bin/env bash
# Automatic release path for one terminally successful rebuttal RLVR worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${ARM:?ARM required}"
: "${INIT_PAIR:?INIT_PAIR required}"
: "${RLVR_SEED:?RLVR_SEED required}"
: "${INIT_MODEL_PATH:?INIT_MODEL_PATH required}"
: "${TRAIN_FILE:?TRAIN_FILE required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME required}"
: "${CKPTS_DIR:?CKPTS_DIR required}"
: "${RUN_LOG_DIR:?RUN_LOG_DIR required}"
: "${RUN_WANDB_DIR:?RUN_WANDB_DIR required}"
: "${ATTEMPT_ROOT:?ATTEMPT_ROOT required}"
: "${FINAL_STEP:=115}"

REGISTRY_ROOT=${REGISTRY_ROOT:-"${ROOT}/experiment_registry"}
REGISTRY_DB=${EXPERIMENT_REGISTRY_DB:-"${REGISTRY_ROOT}/experiment_registry.sqlite"}
GATE_STATE=${TRAINING_RELEASE_GATE_STATE:-"${REGISTRY_ROOT}/training_release_gate.jsonl"}
GATE_SCRIPT=${TRAINING_RELEASE_GATE_SCRIPT:-"${REPO_ROOT}/scripts/training_result_release_gate.py"}
IMPORT_SCRIPT=${REBUTTAL_REGISTRY_IMPORT_SCRIPT:-"${REPO_ROOT}/scripts/import_rebuttal_rlvr_registry.py"}
RELEASE_LOG=${RELEASE_LOG_FILE:-"${ATTEMPT_ROOT}/release.log"}
RELEASE_STATUS=${RELEASE_STATUS_FILE:-"${ATTEMPT_ROOT}/release_status.env"}
mkdir -p "$REGISTRY_ROOT" "$ATTEMPT_ROOT"

exec >>"$RELEASE_LOG" 2>&1
echo "[$(date -Iseconds)] automatic release start run=${WANDB_RUN_NAME}"

FINAL_CKPT="${CKPTS_DIR}/global_step_${FINAL_STEP}"
if [ ! -d "$FINAL_CKPT" ]; then
    echo "ERROR: final checkpoint missing: $FINAL_CKPT" >&2
    exit 1
fi

METRICS_PATH=$(python3 - "$RUN_LOG_DIR" "$FINAL_STEP" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
final_step = int(sys.argv[2])
matches = []
for path in sorted(root.rglob("*.jsonl")):
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        continue
    if any(int(row.get("step", -1)) == final_step for row in rows):
        matches.append(path)
if len(matches) != 1:
    raise SystemExit(f"expected one metrics JSONL containing step {final_step}, found {len(matches)}: {matches}")
print(matches[0])
PY
)

WANDB_SYNC_DIR=$(find "$RUN_WANDB_DIR" -type d -name 'offline-run-*' -print | sort | tail -1)
if [ -z "$WANDB_SYNC_DIR" ] || [ ! -d "$WANDB_SYNC_DIR" ]; then
    echo "ERROR: offline W&B run directory missing under $RUN_WANDB_DIR" >&2
    exit 1
fi

python3 "$GATE_SCRIPT" --state "$GATE_STATE" record \
    --run-name "$WANDB_RUN_NAME" \
    --family "rebuttal_rlvr_${ARM}" \
    --status success_complete \
    --source "rebuttal_rlvr:worker_terminal" \
    --checkpoint "$FINAL_CKPT" \
    --metrics "$METRICS_PATH" \
    --final-step "$FINAL_STEP" \
    --observed-step "$FINAL_STEP" \
    --notes "Training process exited zero and the fixed final checkpoint plus metrics exist."
python3 "$GATE_SCRIPT" --state "$GATE_STATE" check --run-name "$WANDB_RUN_NAME"

VERL_REPO_ROOT="$REPO_ROOT" EXPERIMENT_REGISTRY_DB="$REGISTRY_DB" TRAINING_RELEASE_GATE_STATE="$GATE_STATE" \
python3 "$IMPORT_SCRIPT" \
    --run-name "$WANDB_RUN_NAME" \
    --arm "$ARM" \
    --init-pair "$INIT_PAIR" \
    --rl-seed "$RLVR_SEED" \
    --init-model-path "$INIT_MODEL_PATH" \
    --checkpoint-dir "$CKPTS_DIR" \
    --metrics-path "$METRICS_PATH" \
    --train-file "$TRAIN_FILE" \
    --wandb-run "$WANDB_SYNC_DIR" \
    --final-step "$FINAL_STEP" \
    --db "$REGISTRY_DB"

python3 - "$REGISTRY_DB" "$WANDB_RUN_NAME" <<'PY'
import re
import sqlite3
import sys

db, run_name = sys.argv[1:]
key = "verl.rebuttal_rlvr.training." + re.sub(r"[^a-zA-Z0-9]+", "_", run_name).strip("_").lower()
row = sqlite3.connect(db).execute("select id from training_runs where training_run_key=?", (key,)).fetchone()
if row is None:
    raise SystemExit(f"registry verification failed: {key}")
print(f"registry verified training_run_id={row[0]}")
PY

sync_args=(wandb sync --project "${WANDB_PROJECT:-Rebuttal-RLVR-MATH}" --mark-synced)
if [ -n "${WANDB_ENTITY:-}" ]; then
    sync_args+=(--entity "$WANDB_ENTITY")
fi
sync_args+=("$WANDB_SYNC_DIR")
"${sync_args[@]}"

if ! find "$WANDB_SYNC_DIR" -maxdepth 1 -type f \( -name '*.wandb.synced' -o -name '.synced' \) -print -quit | grep -q .; then
    echo "ERROR: W&B sync marker missing after successful sync command" >&2
    exit 1
fi

printf '%s\n' \
    'release_status=complete' \
    "run_name=${WANDB_RUN_NAME}" \
    "checkpoint=${FINAL_CKPT}" \
    "metrics=${METRICS_PATH}" \
    "registry_db=${REGISTRY_DB}" \
    "wandb_sync_dir=${WANDB_SYNC_DIR}" \
    >"$RELEASE_STATUS"
echo "[$(date -Iseconds)] automatic release complete run=${WANDB_RUN_NAME}"
