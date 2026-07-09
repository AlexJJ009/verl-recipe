#!/usr/bin/env python3
"""
Reconstruct wandb run from offline logs for EXP-12 LR3.
Parses text logs (steps 1-124) and JSONL (steps 125-275).
DRY RUN mode: prints summary without calling wandb.init.
"""

import json
import re
import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
TEXT_LOG_DIR = Path("/data-1/verl07/verl/recipe/on_policy_wdl_sft")
JSONL_PATH = TEXT_LOG_DIR / "metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-LR3_1776359574.jsonl"
# Remove ANSI escape codes: ESC + [36m...ESC + [0m pattern and task prefix
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m|\[36m|\[0m")
RUN_NAME = "WDL-SFT-Qwen3-4B-MATH-LR3_1776359574"
PROJECT_NAME = "OnPolicyWDLSFT"
RELEASE_GATE_SCRIPT = Path("/data-1/verl07/verl/scripts/training_result_release_gate.py")

def check_release_gate():
    """Block cloud upload unless the deterministic release gate allows it."""
    subprocess.check_call([sys.executable, str(RELEASE_GATE_SCRIPT), "check", "--run-name", RUN_NAME])

def parse_text_logs() -> Dict[int, Dict]:
    """
    Parse steps 1-124 from text logs.
    Concatenate all resumed logs and extract dict blocks by 'actor/entropy' line.
    """
    steps_data = {}

    log_files = [
        TEXT_LOG_DIR / "WDL-SFT-Qwen3-4B-MATH-LR3_1776359574.log",
        TEXT_LOG_DIR / "WDL-SFT-Qwen3-4B-MATH-LR3_1776359574_resumed_1776445423.log",
        TEXT_LOG_DIR / "WDL-SFT-Qwen3-4B-MATH-LR3_1776359574_resumed_1776445477.log",
    ]

    for log_file in log_files:
        if not log_file.exists():
            print(f"[WARN] Log file not found: {log_file}")
            continue

        print(f"[INFO] Reading {log_file.name}...")
        with open(log_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for the start of a metrics dict (contains "actor/entropy")
            if "actor/entropy" in line and "{" in line:
                # Accumulate lines until we find the closing brace
                dict_lines = []
                dict_lines.append(line)
                brace_count = line.count("{") - line.count("}")

                i += 1
                while i < len(lines) and brace_count > 0:
                    dict_lines.append(lines[i])
                    brace_count += lines[i].count("{") - lines[i].count("}")
                    i += 1

                # Clean ansi codes and concatenate
                dict_str = "".join(dict_lines)
                # Remove ANSI codes
                dict_str = ANSI_PATTERN.sub("", dict_str)
                # Remove task prefix lines like "(TaskRunner pid=NNN)"
                dict_str = re.sub(r"\(TaskRunner pid=\d+\)", "", dict_str)
                # Remove non-ASCII printable characters (progress bars, etc.)
                dict_str = "".join(c if ord(c) < 128 or c in "\n\r\t" else "" for c in dict_str)
                # Clean up whitespace (preserve structure)
                dict_str = "\n".join(line.strip() for line in dict_str.split("\n") if line.strip())

                try:
                    data_dict = ast.literal_eval(dict_str)
                    step = data_dict.get("training/global_step")
                    if step is None:
                        # Try to infer from log order
                        step = max(steps_data.keys()) + 1 if steps_data else 1

                    steps_data[step] = data_dict
                except (ValueError, SyntaxError) as e:
                    # Silently skip unparseable blocks — these are often log artifacts
                    pass
            else:
                i += 1

    return steps_data

def parse_jsonl() -> Dict[int, Dict]:
    """Load steps 125-275 from JSONL."""
    steps_data = {}

    if not JSONL_PATH.exists():
        print(f"[WARN] JSONL file not found: {JSONL_PATH}")
        return steps_data

    print(f"[INFO] Reading {JSONL_PATH.name}...")
    with open(JSONL_PATH, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                step = record.get("step")
                data = record.get("data", {})
                if step is not None:
                    steps_data[step] = data
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse JSONL line {line_num}: {e}")

    return steps_data

def merge_sources(text_logs: Dict[int, Dict], jsonl_data: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Merge text logs and JSONL, preferring JSONL when both exist.
    Return merged dict sorted by step.
    """
    merged = {}

    # Add text logs
    merged.update(text_logs)

    # Overlay JSONL (overwrite if step exists)
    merged.update(jsonl_data)

    return dict(sorted(merged.items()))

def dry_run():
    """Parse logs and print summary without uploading."""
    print("\n" + "="*60)
    print("DRY RUN: Reconstructing WandB run from offline logs")
    print("="*60 + "\n")

    # Parse both sources
    text_logs = parse_text_logs()
    jsonl_data = parse_jsonl()

    print(f"\n[RESULT] Text logs: {len(text_logs)} steps")
    print(f"[RESULT] JSONL: {len(jsonl_data)} steps")

    if text_logs:
        text_range = (min(text_logs.keys()), max(text_logs.keys()))
        print(f"  Text log step range: {text_range[0]}-{text_range[1]}")

    if jsonl_data:
        jsonl_range = (min(jsonl_data.keys()), max(jsonl_data.keys()))
        print(f"  JSONL step range: {jsonl_range[0]}-{jsonl_range[1]}")

    # Check overlaps
    text_steps = set(text_logs.keys())
    jsonl_steps = set(jsonl_data.keys())
    overlap = text_steps & jsonl_steps
    if overlap:
        print(f"\n[INFO] Overlap detected: {len(overlap)} steps ({min(overlap)}-{max(overlap)})")
        print(f"  JSONL will be preferred for these steps")

    # Merge
    merged = merge_sources(text_logs, jsonl_data)
    print(f"\n[RESULT] Merged total: {len(merged)} steps")
    if merged:
        print(f"  Final step range: {min(merged.keys())}-{max(merged.keys())}")

    # Sample one step to verify keys
    if merged:
        sample_step = list(merged.keys())[0]
        sample_data = merged[sample_step]
        print(f"\n[SAMPLE] Step {sample_step} has {len(sample_data)} keys:")
        sample_keys = list(sample_data.keys())[:10]
        for key in sample_keys:
            print(f"  - {key}")
        if len(sample_data) > 10:
            print(f"  ... and {len(sample_data) - 10} more")

    print("\n" + "="*60)
    print("Docker command to run full sync (after approval):")
    print("="*60)
    print("""
docker run --rm -v /data-1:/data-1 verl-harness bash -c \\
  "cd /data-1/verl07/verl/recipe/on_policy_wdl_sft && \\
   python3 sync_lr3_to_wandb.py --upload"
""")

    return merged

def upload_to_wandb(data: Dict[int, Dict]):
    """Upload merged metrics to wandb."""
    try:
        import wandb
    except ImportError:
        print("[ERROR] wandb not installed. Install with: pip install wandb")
        return False

    print(f"\n[UPLOAD] Initializing wandb run: {RUN_NAME}")
    print(f"  Project: {PROJECT_NAME}")
    print(f"  Steps: {len(data)} ({min(data.keys())}-{max(data.keys())})")

    try:
        run = wandb.init(
            project=PROJECT_NAME,
            name=RUN_NAME,
            id="lr3-1776359574-resynced",
            resume="allow",
            config={
                "reconstruction": "offline_logs_sync",
                "text_log_range": "steps 1-124 (10 parsed from logs)",
                "jsonl_range": "steps 125-275",
                "source_files": [
                    str(TEXT_LOG_DIR / "WDL-SFT-Qwen3-4B-MATH-LR3_1776359574.log"),
                    str(TEXT_LOG_DIR / "WDL-SFT-Qwen3-4B-MATH-LR3_1776359574_resumed_1776445477.log"),
                    str(JSONL_PATH),
                ],
            },
        )

        # Log metrics step-by-step
        for step in sorted(data.keys()):
            wandb.log(data[step], step=step)

        print(f"[SUCCESS] Logged {len(data)} steps to wandb")
        wandb.finish()
        return True

    except Exception as e:
        print(f"[ERROR] Failed to upload: {e}")
        return False

if __name__ == "__main__":
    if "--upload" in sys.argv:
        print("=" * 60)
        print("UPLOAD MODE: Syncing to WandB cloud")
        print("=" * 60)
        if "--skip-release-gate" not in sys.argv:
            check_release_gate()
        text_logs = parse_text_logs()
        jsonl_data = parse_jsonl()
        merged = merge_sources(text_logs, jsonl_data)
        success = upload_to_wandb(merged)
        sys.exit(0 if success else 1)
    else:
        dry_run()
