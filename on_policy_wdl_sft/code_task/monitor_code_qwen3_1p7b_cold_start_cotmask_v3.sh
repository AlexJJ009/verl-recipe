#!/usr/bin/env bash
# Monitor Code Cold Start candidate events and notify on every five-step validation.
set -euo pipefail

QUEUE_TMUX=${QUEUE_TMUX:-code_qwen3_1p7b_cold_start_author_signature_v2}
EVENT_LOG=${EVENT_LOG:-/data-1/eval_outputs/code_task/qwen3_1p7b_cold_start_cotmask_v3_author_signature_v2_steps/events.jsonl}
SELECTION_FILE=${SELECTION_FILE:-/data-1/model_weights/code_task/qwen3_1p7b_cold_start_cotmask_v3_author_signature_v2_steps/model1_selection.json}
STATE_FILE=${STATE_FILE:-/data-1/tmp/verl_agent_scratch/code_qwen3_1p7b_cold_start_author_signature_v2_monitor/last_line}
POLL_SEC=${POLL_SEC:-60}
WXPUSHER_SCRIPT=${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}
mkdir -p "$(dirname "$STATE_FILE")"
last_line=$(cat "$STATE_FILE" 2>/dev/null || echo 0)

notify() {
    python3 "$WXPUSHER_SCRIPT" --title "$1" --body "$2" >/dev/null 2>&1 || true
}

while tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; do
    if [ -f "$EVENT_LOG" ]; then
        total=$(wc -l <"$EVENT_LOG")
        if [ "$total" -gt "$last_line" ]; then
            while IFS= read -r event; do
                kind=$(jq -r '.event' <<<"$event")
                if [ "$kind" = candidate_evaluated ]; then
                    step=$(jq -r '.step' <<<"$event")
                    rate=$(jq -r '.format_contract_success_rate' <<<"$event")
                    passed=$(jq -r '.passed_format_gate' <<<"$event")
                    sources=$(jq -c '.per_source' <<<"$event")
                    notify "Code Cold Start step ${step} validated" "Status: completed\nFormat contract: ${rate}; threshold=0.85; passed=${passed}\nPer dataset: ${sources}\nEvidence: ${EVENT_LOG}"
                elif [ "$kind" = queue_failed ]; then
                    notify "Code Cold Start failed" "Status: failed\nEvidence: ${EVENT_LOG}\nNext action: inspect queue log."
                fi
            done < <(tail -n "+$((last_line + 1))" "$EVENT_LOG")
            last_line=$total
            printf '%s\n' "$last_line" >"$STATE_FILE"
        fi
    fi
    sleep "$POLL_SEC"
done

if [ -f "$SELECTION_FILE" ]; then
    selected=$(jq -c '.selected' "$SELECTION_FILE")
    notify "Code Cold Start queue complete" "Status: completed\nSelected Model1: ${selected}\nEvidence: ${SELECTION_FILE}"
else
    notify "Code Cold Start queue stopped" "Status: failed or interrupted\nSelection file missing: ${SELECTION_FILE}"
fi
