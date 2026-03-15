#!/usr/bin/env bash
# Download Qwen3-4B-Base model via HTTP proxy (bypass SOCKS)
set -euo pipefail

unset ALL_PROXY all_proxy
export HTTP_PROXY=http://127.0.0.1:7891
export HTTPS_PROXY=http://127.0.0.1:7891

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-4B-Base',
    local_dir='/data-1/.cache/huggingface/Qwen3-4B-Base',
)
print('Download complete!')
"
