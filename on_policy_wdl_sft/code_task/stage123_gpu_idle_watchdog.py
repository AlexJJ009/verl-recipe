#!/usr/bin/env python3
"""Detect prolonged Stage123 GPU starvation without killing or restarting work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def snapshot() -> dict[str, object]:
    gpu = run(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in gpu.stdout.splitlines():
        index, utilization, memory = (part.strip() for part in line.split(","))
        rows.append({"index": int(index), "utilization": int(utilization), "memory_used_mib": int(memory)})
    tmux = run("tmux", "list-sessions", "-F", "#{session_name}")
    sessions = tmux.stdout.splitlines()
    stage_session = next((name for name in sessions if name.startswith("stage123_") and name != "stage123_gpu_idle_watchdog"), "")
    pane = run("tmux", "capture-pane", "-pt", stage_session, "-S", "-1000") if stage_session else None
    text = pane.stdout if pane else ""
    log_dir = Path(__file__).resolve().parent
    current_logs = sorted(log_dir.glob("CODE-S[23]-QWEN3-1P7B-STAGE123-*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if current_logs:
        with current_logs[0].open("rb") as handle:
            handle.seek(max(0, current_logs[0].stat().st_size - 256_000))
            text += handle.read().decode("utf-8", errors="replace")
    reason = "unknown_wait"
    if "Reward computation timed out" in text:
        reason = "scorer_timeout"
    elif "OutOfMemoryError" in text or "CUDA out of memory" in text:
        reason = "oom"
    elif "Killed due to the node running low on memory" in text:
        reason = "ray_memory_kill"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue_active": "code_task_qwen3_1p7b_stage123_queue" in sessions,
        "stage_session": stage_session,
        "gpus": rows,
        "all_gpu_idle": bool(rows) and all(row["utilization"] <= 2 for row in rows),
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--idle-minutes", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log", default="/data-2/experiment_registry/stage123_gpu_idle_watchdog.jsonl")
    args = parser.parse_args()
    idle_since: float | None = None
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    while True:
        report = snapshot()
        expected_active = bool(report["queue_active"] and report["stage_session"])
        if expected_active and report["all_gpu_idle"]:
            idle_since = idle_since or time.monotonic()
        else:
            idle_since = None
        idle_seconds = 0 if idle_since is None else int(time.monotonic() - idle_since)
        report["idle_seconds"] = idle_seconds
        report["alert"] = expected_active and idle_seconds >= args.idle_minutes * 60
        print(json.dumps(report, ensure_ascii=True), flush=True)
        if report["alert"]:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=True) + "\n")
        if args.once:
            return 2 if report["alert"] else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
