#!/usr/bin/env bash
# Download a catalogued initialization into the conventional HF cache layout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT=${1:-${EXPERIMENT:-}}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/model_paths.env"

case "$EXPERIMENT" in
    R02)
        model_id=$WDL_4B_MODEL_ID
        revision=$WDL_4B_REVISION
        ;;
    R03)
        : "${WDL_8B_MODEL_ID:?R03 has no verified public model ID}"
        : "${WDL_8B_REVISION:?R03 has no verified public revision}"
        model_id=$WDL_8B_MODEL_ID
        revision=$WDL_8B_REVISION
        ;;
    *)
        echo "ERROR: downloadable experiment ID must be R02 or R03" >&2
        exit 2
        ;;
esac

DOWNLOAD_PROXY_URL=${DOWNLOAD_PROXY_URL-http://127.0.0.1:7890}
HF_RUNTIME_IMAGE=${HF_RUNTIME_IMAGE:-verl-harness}

proxy_env=()
if [ -n "$DOWNLOAD_PROXY_URL" ]; then
    proxy_env=(
        "HTTP_PROXY=$DOWNLOAD_PROXY_URL"
        "HTTPS_PROXY=$DOWNLOAD_PROXY_URL"
        "ALL_PROXY=$DOWNLOAD_PROXY_URL"
    )
fi

if command -v hf >/dev/null 2>&1; then
    exec env "${proxy_env[@]}" HF_HUB_DISABLE_XET=1 \
        hf download "$model_id" --revision "$revision" --cache-dir "$HF_MODEL_CACHE_ROOT"
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: neither hf CLI nor docker is available" >&2
    exit 2
fi

docker_env=(-e HF_HUB_DISABLE_XET=1)
for item in "${proxy_env[@]}"; do
    docker_env+=(-e "$item")
done

exec docker run --rm --network host \
    -v "$ROOT:$ROOT" \
    -v "$HF_MODEL_CACHE_ROOT:$HF_MODEL_CACHE_ROOT" \
    "${docker_env[@]}" \
    "$HF_RUNTIME_IMAGE" \
    hf download "$model_id" --revision "$revision" --cache-dir "$HF_MODEL_CACHE_ROOT"
