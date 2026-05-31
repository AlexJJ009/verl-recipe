#!/usr/bin/env python3
"""Notify the user after each staged-v1 Stage 1 beta run completes.

The monitor is intentionally narrow: it watches the known Stage 1 beta
checkpoint prefixes, writes a short Markdown notification to disk, then sends it
through the guarded WxPusher helper. It does not launch or modify training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGED_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_ROOT = Path("/data-1/checkpoints")
DEFAULT_METRICS_ROOT = STAGED_DIR / "metrics" / "OnPolicySFT-Then-WDLSFT-StagedV1"
DEFAULT_NOTIFY_ROOT = STAGED_DIR / "monitor_notifications"
DEFAULT_WXPUSHER = Path("/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py")
DEFAULT_ENV_FILES = [
    Path("/root/.agent-core/secrets/wxpusher.env"),
    Path("/root/buaa/local_data1/scheduler/.env.wxpusher.local"),
]

BETA_RUNS = [
    ("s1-beta-0", "0.0", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1"),
    ("s1-beta-01", "0.1", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1"),
    ("s1-beta-02", "0.2", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA02-V1"),
    ("s1-beta-03", "0.3", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA03-V1"),
    ("s1-beta-04", "0.4", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA04-V1"),
    ("s1-beta-05", "0.5", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA05-V1"),
    ("s1-beta-06", "0.6", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA06-V1"),
    ("s1-beta-07", "0.7", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA07-V1"),
    ("s1-beta-08", "0.8", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA08-V1"),
    ("s1-beta-09", "0.9", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA09-V1"),
    ("s1-beta-10", "1.0", "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA10-V1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--notify-root", type=Path, default=DEFAULT_NOTIFY_ROOT)
    parser.add_argument("--final-step", type=int, default=int(os.environ.get("FINAL_STEP", "150")))
    parser.add_argument("--poll-sec", type=int, default=int(os.environ.get("POLL_SEC", "120")))
    parser.add_argument(
        "--fallback-sec-per-step",
        type=float,
        default=float(os.environ.get("FALLBACK_SEC_PER_STEP", "105")),
        help="ETA fallback before enough beta-run metrics exist.",
    )
    parser.add_argument("--wxpusher-script", type=Path, default=DEFAULT_WXPUSHER)
    parser.add_argument("--env-file", action="append", type=Path, default=[])
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_cst() -> dt.datetime:
    return dt.datetime.now(ZoneInfo("Asia/Shanghai"))


def log(path: Path, message: str) -> None:
    stamp = now_cst().strftime("%Y-%m-%d %H:%M:%S %Z")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"notified": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"notified": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def latest_ckpt_dir(ckpt_root: Path, prefix: str) -> Path | None:
    candidates = sorted(ckpt_root.glob(f"{prefix}_*"))
    dirs = [path for path in candidates if path.is_dir()]
    return dirs[-1] if dirs else None


def latest_step(ckpt_dir: Path | None) -> int:
    if ckpt_dir is None:
        return 0
    marker = ckpt_dir / "latest_checkpointed_iteration.txt"
    if marker.exists():
        digits = "".join(ch for ch in marker.read_text(encoding="utf-8", errors="ignore") if ch.isdigit())
        if digits:
            return int(digits)
    steps = []
    for path in ckpt_dir.glob("global_step_*"):
        if path.is_dir():
            try:
                steps.append(int(path.name.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    return max(steps) if steps else 0


def metrics_path(metrics_root: Path, run_name: str) -> Path | None:
    exact = metrics_root / f"{run_name}.jsonl"
    if exact.exists():
        return exact
    matches = sorted(metrics_root.glob(f"{run_name}.jsonl"))
    return matches[-1] if matches else None


def metrics_summary(metrics_root: Path, run_name: str) -> dict:
    path = metrics_path(metrics_root, run_name)
    if path is None:
        return {"path": None, "step": 0, "avg_step_sec": None, "total_step_sec": None}
    steps = []
    total = 0.0
    last_step = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = int(row.get("step") or row.get("data", {}).get("training/global_step") or 0)
            if step:
                last_step = max(last_step, step)
            sec = row.get("data", {}).get("timing_s/step")
            if isinstance(sec, (int, float)) and sec > 0:
                steps.append(float(sec))
                total += float(sec)
    avg = total / len(steps) if steps else None
    return {"path": str(path), "step": last_step, "avg_step_sec": avg, "total_step_sec": total or None}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "未知"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}天{hours}小时"
    if hours:
        return f"{hours}小时{minutes}分"
    return f"{minutes}分"


def estimate_remaining(
    args: argparse.Namespace,
    run_states: list[dict],
    just_completed_index: int,
) -> float | None:
    completed_durations = [
        state["metrics"]["total_step_sec"]
        for state in run_states
        if state["complete"] and state["metrics"].get("total_step_sec")
    ]
    full_run_sec = (
        sum(completed_durations) / len(completed_durations)
        if completed_durations
        else args.final_step * args.fallback_sec_per_step
    )

    remaining = 0.0
    for index, state in enumerate(run_states):
        if index <= just_completed_index or state["complete"]:
            continue
        step = max(state["latest_step"], state["metrics"].get("step") or 0)
        avg_step = state["metrics"].get("avg_step_sec") or args.fallback_sec_per_step
        if step > 0:
            remaining += max(args.final_step - step, 0) * avg_step
        else:
            remaining += full_run_sec
    return remaining


def build_run_states(args: argparse.Namespace) -> list[dict]:
    states = []
    for label, beta, prefix in BETA_RUNS:
        ckpt_dir = latest_ckpt_dir(args.ckpt_root, prefix)
        run_name = ckpt_dir.name if ckpt_dir else ""
        step = latest_step(ckpt_dir)
        summary = metrics_summary(args.metrics_root, run_name) if run_name else {
            "path": None,
            "step": 0,
            "avg_step_sec": None,
            "total_step_sec": None,
        }
        complete = step >= args.final_step
        states.append(
            {
                "label": label,
                "beta": beta,
                "prefix": prefix,
                "ckpt_dir": str(ckpt_dir) if ckpt_dir else "",
                "run_name": run_name,
                "latest_step": step,
                "complete": complete,
                "metrics": summary,
            }
        )
    return states


def write_message(args: argparse.Namespace, state: dict, run_state: dict, completed_index: int, eta: str) -> Path:
    messages_dir = args.notify_root / "messages"
    messages_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_cst().strftime("%Y%m%d-%H%M%S")
    msg_path = messages_dir / f"{stamp}-{run_state['label']}.md"
    completed_count = completed_index + 1
    text = f"S1 beta {run_state['beta']} 完成（{completed_count}/11）；预计剩余 {eta}。"
    msg_path.write_text(text + "\n", encoding="utf-8")
    return msg_path


def send_message(args: argparse.Namespace, run_state: dict, message_path: Path) -> None:
    title = f"S1 beta {run_state['beta']} 完成"
    body = message_path.read_text(encoding="utf-8").strip()
    cmd = [
        sys.executable,
        str(args.wxpusher_script),
        "--title",
        title,
        "--summary",
        title,
        "--body",
        body,
    ]
    for env_file in [*DEFAULT_ENV_FILES, *args.env_file]:
        if env_file.exists():
            cmd.extend(["--env-file", str(env_file)])
    if args.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def scan_once(args: argparse.Namespace, state_path: Path, log_path: Path) -> bool:
    state = load_state(state_path)
    notified = state.setdefault("notified", {})
    run_states = build_run_states(args)
    sent_any = False
    for index, run_state in enumerate(run_states):
        if not run_state["complete"]:
            continue
        previous = notified.get(run_state["label"])
        if previous and previous.get("run_name") == run_state["run_name"]:
            continue
        eta = format_duration(estimate_remaining(args, run_states, index))
        message_path = write_message(args, state, run_state, index, eta)
        send_message(args, run_state, message_path)
        notified[run_state["label"]] = {
            "run_name": run_state["run_name"],
            "ckpt_dir": run_state["ckpt_dir"],
            "step": run_state["latest_step"],
            "message_path": str(message_path),
            "sent_at": now_cst().isoformat(),
            "eta_remaining": eta,
        }
        save_state(state_path, state)
        sent_any = True
        log(log_path, f"sent completion notification for {run_state['label']} ({run_state['run_name']}); eta={eta}")
    if not sent_any:
        active = [r for r in run_states if r["latest_step"] > 0 and not r["complete"]]
        if active:
            log(log_path, "no completion; active=" + ", ".join(f"{r['label']}@{r['latest_step']}" for r in active))
        else:
            log(log_path, "no Stage 1 beta run active or completed yet")
    return all(r["complete"] for r in run_states)


def main() -> int:
    args = parse_args()
    args.notify_root.mkdir(parents=True, exist_ok=True)
    state_path = args.notify_root / "state.json"
    log_path = args.notify_root / "monitor.log"
    lock_path = args.notify_root / "monitor.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another monitor is already running: {lock_path}")
        log(log_path, f"monitor start loop={args.loop} final_step={args.final_step} poll_sec={args.poll_sec}")
        while True:
            all_done = scan_once(args, state_path, log_path)
            if not args.loop or all_done:
                break
            time.sleep(args.poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
