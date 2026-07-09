#!/usr/bin/env bash
# Monitor the DeepCoder Instruct-2507 R8K unified N=3 official code offline eval queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
LOG_DIR=${LOG_DIR:-"${REPO_HOST}/recipe/on_policy_wdl_sft/code_task/eval_logs"}
QUEUE_TMUX=${QUEUE_TMUX:-deepcoder_i2507_r8k_offline_n3_queue}
QUEUE_LOG=${QUEUE_LOG:-"${LOG_DIR}/run_code_deepcoder_instruct2507_r8k_offline_n3_queue.log"}
STATUS_FILE=${STATUS_FILE:-"${LOG_DIR}/run_code_deepcoder_instruct2507_r8k_offline_n3_status.tsv"}
MONITOR_LOG=${MONITOR_LOG:-"${LOG_DIR}/monitor_code_deepcoder_instruct2507_r8k_offline_n3.log"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-1/eval_outputs/code_task/deepcoder_instruct2507_r8k_unified_n3}
MODEL_WEIGHT_ROOT=${MODEL_WEIGHT_ROOT:-/data-1/model_weights/code_task/offline_eval}
SUMMARY_JSON=${SUMMARY_JSON:-"${OUTPUT_ROOT}/summary_deepcoder_instruct2507_r8k_unified_n3.json"}
SUMMARY_MD=${SUMMARY_MD:-"${OUTPUT_ROOT}/summary_deepcoder_instruct2507_r8k_unified_n3.md"}
POLL_SEC=${POLL_SEC:-300}
WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-1}
MONITOR_EXIT_ON_COMPLETE=${MONITOR_EXIT_ON_COMPLETE:-1}

LABELS=(
    "deepcoder_i2507_r8k_beta0_step120"
    "deepcoder_i2507_r8k_beta0_step150"
    "deepcoder_i2507_r8k_beta01_step115"
    "deepcoder_i2507_r8k_beta01_step150"
)
MERGED_DIRS=(
    "${MODEL_WEIGHT_ROOT}/deepcoder_i2507_r8k_beta0_step120/actor_step120"
    "${MODEL_WEIGHT_ROOT}/deepcoder_i2507_r8k_beta0_step150/actor_step150"
    "${MODEL_WEIGHT_ROOT}/deepcoder_i2507_r8k_beta01_step115/actor_step115"
    "${MODEL_WEIGHT_ROOT}/deepcoder_i2507_r8k_beta01_step150/actor_step150"
)
BENCHMARKS=(humaneval mbpp bigcodebench livecodebench)

log() {
    mkdir -p "${LOG_DIR}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MONITOR_LOG}"
}

notify() {
    local title="$1" body="$2"
    if [ "${WXPUSHER_NOTIFY}" = "1" ]; then
        python3 /root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py \
            --title "$title" \
            --body "$body" || true
    fi
}

free_gb() {
    df -Pk /data-1 | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

merged_ready() {
    local merged="$1"
    [ -f "${merged}/model.safetensors.index.json" ] || [ -f "${merged}/model.safetensors" ]
}

summary_ok() {
    local path="$1"
    python3 - "$path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("ok") is True else 1)
PY
}

case_metric_line() {
    local path="$1"
    python3 - "$path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print("missing")
    raise SystemExit
payload = json.loads(path.read_text())
bench = payload.get("benchmark")
summary = payload.get("summary", {})
if bench in {"humaneval", "mbpp"}:
    print(f"mean@3={summary.get('plus_pass_rate')} pass@3={summary.get('plus_any_at_n', summary.get('plus_pass_at_n'))}")
elif bench == "bigcodebench":
    pass_at_k = summary.get("pass_at_k") or {}
    print(f"pass@3={pass_at_k.get('pass@3', pass_at_k.get('pass_at_3'))} result_exists={summary.get('result_exists')}")
elif bench == "livecodebench":
    metrics = summary.get("metrics")
    metric = metrics[0] if isinstance(metrics, list) and metrics else (metrics if isinstance(metrics, dict) else {})
    detail = metric.get("detail", {}) if isinstance(metric, dict) else {}
    vals = detail.get("pass@1", {}) if isinstance(detail, dict) else {}
    if isinstance(vals, dict) and vals:
        nums = [float(v) for v in vals.values()]
        mean = sum(nums) / len(nums)
        pass3 = sum(1 for v in nums if v > 0) / len(nums)
        print(f"mean@3={mean} pass@3={pass3} tasks={len(nums)}")
    else:
        print("no-metrics")
else:
    print("unknown")
PY
}

summary_table_excerpt() {
    if [ ! -s "${SUMMARY_MD}" ]; then
        echo "aggregate summary missing: ${SUMMARY_MD}"
        return
    fi
    sed -n '1,28p' "${SUMMARY_MD}"
}

monitor_once() {
    local completed=0 total=0 failed=0 merged_count=0
    local queue_state="missing"
    if tmux has-session -t "${QUEUE_TMUX}" 2>/dev/null; then
        queue_state="tmux-alive"
    fi
    local free
    free=$(free_gb)
    log "status: queue=${queue_state} free=${free}G output=${OUTPUT_ROOT}"
    log "gpu: $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd ';' - || true)"
    log "docker: $(docker ps --format '{{.Names}}:{{.Status}}:{{.Image}}' 2>/dev/null | paste -sd ';' - || true)"

    for i in "${!LABELS[@]}"; do
        if merged_ready "${MERGED_DIRS[$i]}"; then
            merged_count=$((merged_count + 1))
            log "merge ${LABELS[$i]}: present ${MERGED_DIRS[$i]}"
        else
            log "merge ${LABELS[$i]}: missing ${MERGED_DIRS[$i]}"
        fi
        for benchmark in "${BENCHMARKS[@]}"; do
            total=$((total + 1))
            local summary="${OUTPUT_ROOT}/${LABELS[$i]}/${benchmark}/official_summary.json"
            if summary_ok "${summary}"; then
                completed=$((completed + 1))
                log "case ${LABELS[$i]}/${benchmark}: complete $(case_metric_line "${summary}")"
            else
                log "case ${LABELS[$i]}/${benchmark}: pending summary=${summary}"
            fi
        done
    done

    if [ -f "${STATUS_FILE}" ]; then
        log "status-file tail:"
        tail -20 "${STATUS_FILE}" | tee -a "${MONITOR_LOG}"
        if awk -F '\t' '$6 == "failed" {found=1} END {exit found ? 0 : 1}' "${STATUS_FILE}"; then
            failed=1
        fi
    else
        log "status-file missing: ${STATUS_FILE}"
    fi
    if [ -f "${QUEUE_LOG}" ]; then
        log "queue-log tail:"
        tail -20 "${QUEUE_LOG}" | tee -a "${MONITOR_LOG}"
    fi

    log "progress: completed=${completed}/${total} merged=${merged_count}/${#LABELS[@]} failed=${failed}"
    if [ "${failed}" = "1" ]; then
        notify "DeepCoder Instruct R8K offline eval monitor: failed" "Status: failed\nEvidence: ${STATUS_FILE}\nNext action: inspect ${QUEUE_LOG}."
        return 2
    fi
    if [ "${completed}" -eq "${total}" ]; then
        if [ ! -s "${SUMMARY_JSON}" ] || [ ! -s "${SUMMARY_MD}" ]; then
            log "all cases complete, waiting for aggregate summary: json=${SUMMARY_JSON} md=${SUMMARY_MD}"
            return 1
        fi
        local marker="${OUTPUT_ROOT}/.monitor_complete_notified"
        if [ ! -f "${marker}" ]; then
            notify "DeepCoder Instruct R8K offline eval monitor: complete" "Status: completed\nWhat happened: DeepCoder Instruct-2507 R8K unified N=3 official eval finished.\nEvidence: ${SUMMARY_MD}\n\n$(summary_table_excerpt)"
            mkdir -p "${OUTPUT_ROOT}"
            date '+%Y-%m-%d %H:%M:%S' >"${marker}"
        fi
        return 0
    fi
    return 1
}

main() {
    log "DeepCoder Instruct2507 R8K offline N=3 monitor start; queue=${QUEUE_TMUX}"
    while true; do
        if monitor_once; then
            if [ "${MONITOR_EXIT_ON_COMPLETE}" = "1" ]; then
                exit 0
            fi
        else
            rc=$?
            if [ "${rc}" = "2" ]; then
                exit 2
            fi
        fi
        sleep "${POLL_SEC}"
    done
}

main "$@"
