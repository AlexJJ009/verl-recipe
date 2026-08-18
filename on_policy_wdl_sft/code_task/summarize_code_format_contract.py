#!/usr/bin/env python3
"""Summarize strict Code Cold Start format compliance from vLLM JSONL outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

try:
    from recipe.on_policy_wdl_sft.code_task.code_extraction import compute_format_telemetry, extract_code
except ModuleNotFoundError:
    from code_extraction import compute_format_telemetry, extract_code  # type: ignore


RATE_FIELDS = (
    "think_nonempty",
    "think_complete",
    "answer_complete",
    "format_ordered",
    "python_fence_success",
    "extraction_success",
    "has_eos",
    "truncated",
    "format_contract_success",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row["_source_file"] = str(path)
                row["_line_number"] = line_number
                rows.append(row)
    return rows


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("solution_str", ""))
    telemetry = compute_format_telemetry(text)
    extraction = extract_code(text, strict_answer=True)
    finish_reason = str(row.get("finish_reason") or "").lower()
    has_eos = finish_reason == "stop"
    truncated = finish_reason == "length"
    python_fence_success = extraction.ok and extraction.source == "answer:fenced_python"
    format_contract_success = (
        telemetry["think_nonempty"]
        and telemetry["think_complete"]
        and telemetry["answer_complete"]
        and telemetry["format_ordered"]
        and python_fence_success
        and extraction.ok
        and has_eos
        and not truncated
    )
    return {
        **telemetry,
        "python_fence_success": python_fence_success,
        "extraction_success": extraction.ok,
        "extraction_status": extraction.status,
        "extraction_source": extraction.source,
        "has_eos": has_eos,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "format_contract_success": format_contract_success,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = {field: sum(bool(row[field]) for row in rows) for field in RATE_FIELDS}
    return {
        "response_count": total,
        **{f"{field}_count": count for field, count in counts.items()},
        **{f"{field}_rate": (count / total if total else 0.0) for field, count in counts.items()},
    }


def summarize(paths: list[Path], threshold: float) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for raw in read_jsonl(path):
            result = evaluate_row(raw)
            source = str(raw.get("data_source") or path.parent.name)
            record = {
                "data_source": source,
                "task_index": raw.get("task_index"),
                "sample_index": raw.get("sample_index"),
                "task_id": raw.get("task_id"),
                "source_file": raw["_source_file"],
                "line_number": raw["_line_number"],
                **result,
            }
            evaluated.append(record)
            by_source[source].append(record)

    per_source = {source: aggregate(source_rows) for source, source_rows in sorted(by_source.items())}
    micro = aggregate(evaluated)
    macro_rate = (
        sum(metrics["format_contract_success_rate"] for metrics in per_source.values()) / len(per_source)
        if per_source
        else 0.0
    )
    return {
        "schema_version": 1,
        "threshold": threshold,
        "passed_format_gate": micro["format_contract_success_rate"] >= threshold,
        "gate_metric": "micro_metrics.format_contract_success_rate",
        "input_files": [str(path) for path in paths],
        "micro_metrics": micro,
        "macro_metrics": {"format_contract_success_rate": macro_rate},
        "per_source": per_source,
        "failures": [row for row in evaluated if not row["format_contract_success"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    payload = summarize(args.raw_output, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("passed_format_gate", "micro_metrics", "macro_metrics", "per_source")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
