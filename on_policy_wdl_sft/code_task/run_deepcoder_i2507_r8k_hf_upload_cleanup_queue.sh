#!/usr/bin/env bash
# Upload DeepCoder Instruct-2507 R8K merged HF actors, verify Hub files, then delete source FSDP checkpoints.
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
LOG_FILE=${LOG_FILE:-${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/run_deepcoder_i2507_r8k_hf_upload_cleanup_queue.log}
STATUS_FILE=${STATUS_FILE:-${LOG_FILE%.log}_status.tsv}
HF_NAMESPACE=${HF_NAMESPACE:-AlexGeek}
PRIVATE=${PRIVATE:-1}
UPLOAD_REVISION=${UPLOAD_REVISION:-main}
DELETE_AFTER_VERIFY=${DELETE_AFTER_VERIFY:-1}
ALLOW_DELETE_DEEPCODER_CKPTS=${ALLOW_DELETE_DEEPCODER_CKPTS:-0}
USE_HOST_NETWORK=${USE_HOST_NETWORK:-1}
HTTP_PROXY_VALUE=${HTTP_PROXY_VALUE:-http://127.0.0.1:7890}

LABELS=(
  deepcoder_i2507_r8k_beta0_step120
  deepcoder_i2507_r8k_beta0_step150
  deepcoder_i2507_r8k_beta01_step115
  deepcoder_i2507_r8k_beta01_step150
)
LOCAL_DIRS=(
  /data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta0_step120/actor_step120
  /data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta0_step150/actor_step150
  /data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta01_step115/actor_step115
  /data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta01_step150/actor_step150
)
REPO_NAMES=(
  qwen3-4b-instruct2507-code-deepcoder-r8k-beta0-step120
  qwen3-4b-instruct2507-code-deepcoder-r8k-beta0-step150
  qwen3-4b-instruct2507-code-deepcoder-r8k-beta01-step115
  qwen3-4b-instruct2507-code-deepcoder-r8k-beta01-step150
)
DELETE_CKPTS=(
  /data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-DEEPCODER-R8K-S1-BETA0-V1_1782133285
  /data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-DEEPCODER-R8K-S1-BETA01-V1_1782232869
)

log() { mkdir -p "$(dirname "$LOG_FILE")"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
record() { mkdir -p "$(dirname "$STATUS_FILE")"; printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" "$3" >> "$STATUS_FILE"; }

docker_py() {
  local network_args=()
  if [ "$USE_HOST_NETWORK" = "1" ]; then network_args+=(--network=host); fi
  docker run --rm -i "${network_args[@]}" \
    -e HTTPS_PROXY="$HTTP_PROXY_VALUE" -e HTTP_PROXY="$HTTP_PROXY_VALUE" \
    -e https_proxy="$HTTP_PROXY_VALUE" -e http_proxy="$HTTP_PROXY_VALUE" \
    -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    -e LOCAL_DIR="${LOCAL_DIR:-}" -e REPO_ID="${REPO_ID:-}" -e PRIVATE="${PRIVATE:-1}" -e UPLOAD_REVISION="${UPLOAD_REVISION:-main}" \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -v /data-1:/data-1 \
    -v "$REPO_HOST":/workspace/verl \
    -w /workspace/verl \
    "$DOCKER_IMAGE" python3 "$@"
}

preflight() {
  log "preflight: HF namespace=${HF_NAMESPACE} private=${PRIVATE} delete_after_verify=${DELETE_AFTER_VERIFY} allow_delete=${ALLOW_DELETE_DEEPCODER_CKPTS}"
  for dir in "${LOCAL_DIRS[@]}"; do
    if [ ! -f "$dir/model.safetensors.index.json" ] && [ ! -f "$dir/model.safetensors" ]; then
      log "ERROR missing merged HF model: $dir"
      record preflight missing "$dir"
      exit 2
    fi
  done
  docker_py - <<'PY'
from huggingface_hub import HfApi, HfFolder
api = HfApi()
info = api.whoami()
auth = info.get("auth") or {}
tok = auth.get("accessToken") or {}
print("account", info.get("name"), "token_display", tok.get("displayName"), "token_role", tok.get("role"), flush=True)
role = tok.get("role")
if role not in {"write", "admin"}:
    raise SystemExit(f"HF token is not write-capable: role={role!r}")
PY
}

upload_and_verify_one() {
  local idx="$1"
  local label="${LABELS[$idx]}"
  local dir="${LOCAL_DIRS[$idx]}"
  local repo="${HF_NAMESPACE}/${REPO_NAMES[$idx]}"
  log "upload start: ${label} -> ${repo} from ${dir}"
  record "$label" upload_start "$repo"
  LOCAL_DIR="$dir" REPO_ID="$repo" PRIVATE="$PRIVATE" UPLOAD_REVISION="$UPLOAD_REVISION" docker_py - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, list_repo_files, get_hf_file_metadata, hf_hub_url
repo_id = os.environ["REPO_ID"]
folder = Path(os.environ["LOCAL_DIR"])
private = os.environ.get("PRIVATE", "1") == "1"
revision = os.environ.get("UPLOAD_REVISION", "main")
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model", revision=revision)
remote = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision))
missing = []
size_mismatch = []
for p in folder.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(folder).as_posix()
    if rel not in remote:
        missing.append(rel)
        continue
    meta = get_hf_file_metadata(hf_hub_url(repo_id, rel, repo_type="model", revision=revision))
    local_size = p.stat().st_size
    remote_size = getattr(meta, "size", None)
    if remote_size is not None and remote_size != local_size:
        size_mismatch.append((rel, local_size, remote_size))
if missing or size_mismatch:
    raise SystemExit({"missing": missing, "size_mismatch": size_mismatch})
print("verified", repo_id, "files", len(remote), flush=True)
PY
  log "upload verified: ${label} -> ${repo}"
  record "$label" upload_verified "$repo"
}

delete_ckpts() {
  if [ "$DELETE_AFTER_VERIFY" != "1" ]; then
    log "skip delete: DELETE_AFTER_VERIFY=${DELETE_AFTER_VERIFY}"
    return
  fi
  if [ "$ALLOW_DELETE_DEEPCODER_CKPTS" != "1" ]; then
    log "ERROR refusing delete without ALLOW_DELETE_DEEPCODER_CKPTS=1"
    record cleanup blocked "ALLOW_DELETE_DEEPCODER_CKPTS!=1"
    exit 3
  fi
  for d in "${DELETE_CKPTS[@]}"; do
    case "$d" in
      /data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-DEEPCODER-R8K-S1-BETA*-V1_*) ;;
      *) log "ERROR unsafe delete path: $d"; exit 4 ;;
    esac
    if [ -d "$d" ]; then
      size=$(du -sh "$d" | awk '{print $1}')
      log "delete checkpoint: $d size=$size"
      rm -rf --one-file-system "$d"
      record cleanup deleted "$d size=$size"
    else
      log "delete skip missing: $d"
      record cleanup missing "$d"
    fi
  done
  log "post-delete disk: $(df -h /data-1 | tail -1)"
}

main() {
  preflight
  for i in "${!LABELS[@]}"; do upload_and_verify_one "$i"; done
  delete_ckpts
  log "complete"
}
main "$@"
