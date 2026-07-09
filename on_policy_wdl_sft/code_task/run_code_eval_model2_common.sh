#!/usr/bin/env bash
# Container-side common code eval runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR=${MODEL_DIR:-${MODEL2_PATH:-}}
: "${MODEL_DIR:?MODEL_DIR or MODEL2_PATH must be set}"
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task}
BENCHMARKS=${BENCHMARKS:-"humaneval mbpp bigcodebench livecodebench"}
OFFICIAL_SITE=${OFFICIAL_SITE:-/data-1/code_eval_envs/official_site}
LCB_REPO_DIR=${LCB_REPO_DIR:-/data-1/code_eval_envs/LiveCodeBench}
LCB_PYTHON=${LCB_PYTHON:-/opt/venv/bin/python}
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-release_v5}
export PYTHONPATH="${OFFICIAL_SITE}:${LCB_REPO_DIR}:${PYTHONPATH:-}"

if [ "${DRY_RUN:-0}" != "1" ]; then
    [ -f "$MODEL_DIR/tokenizer_config.json" ] || { echo "ERROR tokenizer_config.json missing in $MODEL_DIR" >&2; exit 1; }
    [ -f "$MODEL_DIR/chat_template.jinja" ] || { echo "ERROR chat_template.jinja missing in $MODEL_DIR" >&2; exit 1; }
    ls "$MODEL_DIR"/*.safetensors "$MODEL_DIR"/model.safetensors.index.json >/dev/null 2>&1 || { echo "ERROR safetensors missing in $MODEL_DIR" >&2; exit 1; }
fi

mkdir -p "$OUTPUT_ROOT"
echo "[code-eval] MODEL_DIR=$MODEL_DIR OUTPUT_ROOT=$OUTPUT_ROOT BENCHMARKS=$BENCHMARKS OFFICIAL_SITE=$OFFICIAL_SITE LCB_REPO_DIR=$LCB_REPO_DIR LCB_PYTHON=$LCB_PYTHON LCB_RELEASE_VERSION=$LCB_RELEASE_VERSION"
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

require_file() {
    local var_name="$1" path="${!1:-}"
    if [ -z "$path" ] || [ ! -f "$path" ]; then
        echo "[code-eval] ERROR: $var_name must point to an existing official-eval input file; got '${path}'" >&2
        exit 1
    fi
}

for bench in $BENCHMARKS; do
    case "$bench" in
        humaneval|mbpp)
            var_name="$(printf '%s_SAMPLES' "$bench" | tr '[:lower:]' '[:upper:]')"
            require_file "$var_name"
            python3 "$SCRIPT_DIR/eval_code_official.py" \
                --benchmark "$bench" \
                --samples "${!var_name}" \
                --output-dir "$OUTPUT_ROOT/$bench" \
                --summary "$OUTPUT_ROOT/${bench}_official_summary.json" \
                --overwrite
            ;;
        bigcodebench)
            require_file BIGCODEBENCH_SAMPLES
            python3 "$SCRIPT_DIR/eval_code_official.py" \
                --benchmark bigcodebench \
                --samples "$BIGCODEBENCH_SAMPLES" \
                --output-dir "$OUTPUT_ROOT/bigcodebench" \
                --summary "$OUTPUT_ROOT/bigcodebench_official_summary.json" \
                --overwrite
            ;;
        livecodebench)
            require_file LIVECODEBENCH_CUSTOM_OUTPUT
            python3 "$SCRIPT_DIR/eval_code_official.py" \
                --benchmark livecodebench \
                --custom-output "$LIVECODEBENCH_CUSTOM_OUTPUT" \
                --output-dir "$OUTPUT_ROOT/livecodebench" \
                --summary "$OUTPUT_ROOT/livecodebench_official_summary.json" \
                --lcb-python "$LCB_PYTHON" \
                --lcb-release-version "$LCB_RELEASE_VERSION" \
                --overwrite
            ;;
        *)
            echo "[code-eval] ERROR: unsupported benchmark '$bench'" >&2
            exit 1
            ;;
    esac
done
