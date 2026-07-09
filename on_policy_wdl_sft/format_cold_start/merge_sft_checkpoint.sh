#!/usr/bin/env bash
# Merge a format-cold-start FSDP SFT checkpoint into HuggingFace format.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: merge_sft_checkpoint.sh --checkpoint-dir DIR --target-dir DIR [--overwrite]

If --target-dir is omitted, TARGET_DIR may be set in env. The checkpoint dir may
be either a run dir containing global_step_* or the global_step_* dir itself.
EOF
}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
TARGET_DIR=${TARGET_DIR:-}
OVERWRITE=${OVERWRITE:-0}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --target-dir) TARGET_DIR="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[merge-sft] ERROR: unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[ -n "$CHECKPOINT_DIR" ] || { echo "[merge-sft] ERROR: --checkpoint-dir required" >&2; exit 1; }
[ -n "$TARGET_DIR" ] || { echo "[merge-sft] ERROR: --target-dir or TARGET_DIR required" >&2; exit 1; }
[ -d "$CHECKPOINT_DIR" ] || { echo "[merge-sft] ERROR: checkpoint dir not found: $CHECKPOINT_DIR" >&2; exit 1; }

if [[ "$(basename "$CHECKPOINT_DIR")" != global_step_* ]]; then
    step_dir=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)
else
    step_dir="$CHECKPOINT_DIR"
fi
[ -n "${step_dir:-}" ] && [ -d "$step_dir" ] || { echo "[merge-sft] ERROR: no global_step_* under $CHECKPOINT_DIR" >&2; exit 1; }

LOCAL_DIR="$step_dir"
if [ -d "${step_dir}/actor" ]; then
    LOCAL_DIR="${step_dir}/actor"
fi
[ -d "${LOCAL_DIR}/huggingface" ] || { echo "[merge-sft] ERROR: missing huggingface config dir in $LOCAL_DIR" >&2; exit 1; }
find "$LOCAL_DIR" -maxdepth 1 -name 'model_world_size_*_rank_0.pt' | grep -q . || {
    echo "[merge-sft] ERROR: missing FSDP model shards in $LOCAL_DIR" >&2
    exit 1
}

if [ -e "$TARGET_DIR" ] && [ "$OVERWRITE" != "1" ]; then
    echo "[merge-sft] ERROR: target exists; pass --overwrite to replace: $TARGET_DIR" >&2
    exit 1
fi
if [ -e "$TARGET_DIR" ] && [ "$OVERWRITE" = "1" ]; then
    rm -rf "$TARGET_DIR"
fi
mkdir -p "$TARGET_DIR"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

if [ ! -f "${TARGET_DIR}/config.json" ]; then
    echo "[merge-sft] ERROR: merged config.json missing in $TARGET_DIR" >&2
    exit 1
fi
if ! find "$TARGET_DIR" -maxdepth 1 \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q .; then
    echo "[merge-sft] ERROR: merged model weights missing in $TARGET_DIR" >&2
    exit 1
fi
if ! find "$TARGET_DIR" -maxdepth 1 \( -name 'tokenizer.json' -o -name 'tokenizer.model' \) | grep -q .; then
    echo "[merge-sft] ERROR: tokenizer file missing in $TARGET_DIR" >&2
    exit 1
fi

cat >"${TARGET_DIR}/format_cold_start_source.json" <<EOF
{
  "checkpoint_dir": "${CHECKPOINT_DIR}",
  "global_step_dir": "${step_dir}",
  "local_dir": "${LOCAL_DIR}",
  "target_dir": "${TARGET_DIR}",
  "merged_at": "$(date -Iseconds)"
}
EOF

echo "[merge-sft] merged ${LOCAL_DIR} -> ${TARGET_DIR}"
