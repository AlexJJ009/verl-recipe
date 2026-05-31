#!/usr/bin/env bash
# Resolve and merge the Stage 1 single-model checkpoint into a HF model dir.
# This file is sourced by Stage 2 wrappers before the joint model is prepared.

: "${BASE_CKPT_DIR:=/data-1/checkpoints}"
: "${STAGE1_RUN_PREFIX:=ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1}"
: "${STAGE1_STEP:=best}"
: "${STAGE1_MERGED_MODEL_ROOT:=/data-1/model_weights/staged_v1}"
: "${REQUIRE_MERGED_MODEL2_PROVENANCE:=False}"
: "${ALLOW_OVERWRITE_MERGED_MODEL2:=0}"

if [ -n "${MODEL2_PATH:-}" ]; then
    if [ ! -d "$MODEL2_PATH" ]; then
        echo "[staged_v1] ERROR: explicit MODEL2_PATH does not exist: $MODEL2_PATH" >&2
        exit 1
    fi
    echo "[staged_v1] Using explicit MODEL2_PATH=$MODEL2_PATH"
    return 0
fi

if [ -z "${STAGE1_CKPT_DIR:-}" ]; then
    STAGE1_CKPT_DIR=$(find "$BASE_CKPT_DIR" -maxdepth 1 -type d -name "${STAGE1_RUN_PREFIX}_*" 2>/dev/null | sort | tail -1)
fi

if [ -z "$STAGE1_CKPT_DIR" ] || [ ! -d "$STAGE1_CKPT_DIR" ]; then
    echo "[staged_v1] ERROR: Stage 1 checkpoint dir not found." >&2
    echo "[staged_v1]        Set STAGE1_CKPT_DIR or run Stage 1 first." >&2
    echo "[staged_v1]        BASE_CKPT_DIR=$BASE_CKPT_DIR STAGE1_RUN_PREFIX=$STAGE1_RUN_PREFIX" >&2
    exit 1
fi

if [ "$STAGE1_STEP" = "best" ]; then
    if [ -f "$STAGE1_CKPT_DIR/best_checkpoint.json" ]; then
        STAGE1_STEP=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$STAGE1_CKPT_DIR/best_checkpoint.json")
    elif [ -f "$STAGE1_CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
        STAGE1_STEP=$(tr -dc '0-9' < "$STAGE1_CKPT_DIR/latest_checkpointed_iteration.txt")
        echo "[staged_v1] WARNING: best_checkpoint.json missing; using latest step $STAGE1_STEP." >&2
    else
        echo "[staged_v1] ERROR: cannot resolve Stage 1 best/latest step in $STAGE1_CKPT_DIR" >&2
        exit 1
    fi
fi

FSDP_ACTOR_DIR="$STAGE1_CKPT_DIR/global_step_${STAGE1_STEP}/actor"
if [ ! -d "$FSDP_ACTOR_DIR" ]; then
    echo "[staged_v1] ERROR: Stage 1 actor checkpoint not found: $FSDP_ACTOR_DIR" >&2
    exit 1
fi

STAGE1_EXP_NAME="$(basename "$STAGE1_CKPT_DIR")"
MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-"${STAGE1_MERGED_MODEL_ROOT}/${STAGE1_EXP_NAME}/step_${STAGE1_STEP}"}
MERGED_MODEL2_PROVENANCE_FILE=${MERGED_MODEL2_PROVENANCE_FILE:-"${MERGED_MODEL2_DIR}/stage1_source.json"}

has_merged_weights() {
    [ -f "${MERGED_MODEL2_DIR}/model.safetensors" ] || [ -f "${MERGED_MODEL2_DIR}/model.safetensors.index.json" ]
}

write_merged_provenance() {
    mkdir -p "$MERGED_MODEL2_DIR"
    python3 - "$MERGED_MODEL2_PROVENANCE_FILE" "$STAGE1_RUN_PREFIX" "$STAGE1_CKPT_DIR" "$STAGE1_STEP" "$FSDP_ACTOR_DIR" "$MERGED_MODEL2_DIR" <<'PY'
import json
import sys
from pathlib import Path

path, run_prefix, ckpt_dir, step, actor_dir, merged_dir = sys.argv[1:]
payload = {
    "stage1_run_prefix": run_prefix,
    "stage1_ckpt_dir": ckpt_dir,
    "stage1_step": int(step),
    "fsdp_actor_dir": actor_dir,
    "merged_model2_dir": merged_dir,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

check_merged_provenance() {
    python3 - "$MERGED_MODEL2_PROVENANCE_FILE" "$STAGE1_RUN_PREFIX" "$STAGE1_CKPT_DIR" "$STAGE1_STEP" "$FSDP_ACTOR_DIR" "$MERGED_MODEL2_DIR" <<'PY'
import json
import sys
from pathlib import Path

path, run_prefix, ckpt_dir, step, actor_dir, merged_dir = sys.argv[1:]
path = Path(path)
if not path.exists():
    raise SystemExit(f"missing provenance file: {path}")
actual = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "stage1_run_prefix": run_prefix,
    "stage1_ckpt_dir": ckpt_dir,
    "stage1_step": int(step),
    "fsdp_actor_dir": actor_dir,
    "merged_model2_dir": merged_dir,
}
if actual != expected:
    raise SystemExit(
        "merged Model2 provenance mismatch\n"
        f"expected={json.dumps(expected, sort_keys=True)}\n"
        f"actual={json.dumps(actual, sort_keys=True)}"
    )
PY
}

if has_merged_weights; then
    if [ "$REQUIRE_MERGED_MODEL2_PROVENANCE" = "True" ]; then
        if ! check_merged_provenance; then
            if [ "$ALLOW_OVERWRITE_MERGED_MODEL2" = "1" ]; then
                echo "[staged_v1] WARNING: removing stale merged Model2 dir: $MERGED_MODEL2_DIR" >&2
                rm -rf "$MERGED_MODEL2_DIR"
            else
                echo "[staged_v1] ERROR: existing merged Model2 dir is stale or untracked: $MERGED_MODEL2_DIR" >&2
                echo "[staged_v1]        Set ALLOW_OVERWRITE_MERGED_MODEL2=1 to rebuild it deliberately." >&2
                exit 1
            fi
        fi
    fi
fi

if has_merged_weights; then
    echo "[staged_v1] Reusing merged Stage 1 model: $MERGED_MODEL2_DIR"
else
    echo "[staged_v1] Merging Stage 1 FSDP checkpoint to HF model dir..."
    echo "[staged_v1]   FSDP_ACTOR_DIR=$FSDP_ACTOR_DIR"
    echo "[staged_v1]   MERGED_MODEL2_DIR=$MERGED_MODEL2_DIR"
    mkdir -p "$(dirname "$MERGED_MODEL2_DIR")"
    CUDA_VISIBLE_DEVICES=${MERGE_CUDA_VISIBLE_DEVICES:-0} python3 -u -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$FSDP_ACTOR_DIR" \
        --target_dir "$MERGED_MODEL2_DIR" \
        --trust-remote-code
fi

if [ "$REQUIRE_MERGED_MODEL2_PROVENANCE" = "True" ]; then
    write_merged_provenance
fi

export MODEL2_PATH="$MERGED_MODEL2_DIR"
echo "[staged_v1] MODEL2_PATH=$MODEL2_PATH"
