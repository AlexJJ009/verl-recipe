#!/usr/bin/env python3
"""Convert raw model outputs into official evaluator input files.

The conversion deliberately reuses code_extraction.extract_code so offline
official eval and online reward validation apply the same answer extraction
contract before benchmark-specific scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from recipe.on_policy_wdl_sft.code_task.code_extraction import extract_code
except Exception:
    from code_extraction import extract_code  # type: ignore


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_task_metadata(validation_parquet: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(validation_parquet)
    out = []
    for row in df.to_dict("records"):
        gt = row["reward_model"]["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        out.append({"data_source": row["data_source"], "ground_truth": gt, "extra_info": row.get("extra_info") or {}})
    return out


def raw_text(row: dict[str, Any]) -> str:
    for key in ("solution_str", "response", "output", "generation", "raw_output", "text"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"raw output row has no recognized text field: {sorted(row)}")


def build_solution(data_source: str, gt: dict[str, Any], extracted_code: str) -> str:
    if data_source in {"HumanEval", "HumanEval+"}:
        if _defines_entry_point(extracted_code, gt["entry_point"]):
            return extracted_code
        return gt["prompt"].rstrip() + "\n" + _indent_body(extracted_code)
    if data_source in {"MBPP", "MBPP+"}:
        if _defines_entry_point(extracted_code, gt["entry_point"]):
            return extracted_code
        return gt["prompt"].rstrip() + "\n" + _indent_body(extracted_code)
    if data_source == "BigCodeBench":
        return extracted_code
    if data_source == "LiveCodeBench":
        return extracted_code
    raise ValueError(f"unsupported official data_source: {data_source}")


def _defines_entry_point(code: str, entry_point: str) -> bool:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point for node in tree.body)


def _indent_body(code: str) -> str:
    import inspect
    import textwrap

    return textwrap.indent(inspect.cleandoc(code), "    ") + "\n"


def convert(raw_outputs: Path, validation_parquet: Path, output: Path, benchmark: str, report: Path | None = None) -> dict[str, Any]:
    raw_rows = read_jsonl(raw_outputs)
    metadata = load_task_metadata(validation_parquet)
    if len(raw_rows) != len(metadata) and not all("task_index" in row for row in raw_rows):
        raise RuntimeError(
            f"raw output count {len(raw_rows)} != validation row count {len(metadata)} "
            "and rows do not contain task_index for n>1 conversion"
        )

    converted = []
    lcb_by_task: dict[int, dict[str, Any]] = {}
    extraction_rows = []
    residual_tag_rows = []
    for idx, raw in enumerate(raw_rows):
        task_index = int(raw.get("task_index", idx))
        if task_index < 0 or task_index >= len(metadata):
            raise RuntimeError(f"raw output row {idx} has out-of-range task_index={task_index}; metadata rows={len(metadata)}")
        meta = metadata[task_index]
        data_source = str(meta["data_source"])
        gt = meta["ground_truth"]
        text = raw_text(raw)
        extraction = extract_code(text)
        extraction_rows.append(
            {
                "index": idx,
                "task_index": task_index,
                "sample_index": int(raw.get("sample_index", 0)),
                "data_source": data_source,
                "status": extraction.status,
                "source": extraction.source,
            }
        )
        if not extraction.ok:
            code = ""
        else:
            code = build_solution(data_source, gt, extraction.code)
        if any(marker in code for marker in ("<answer", "</answer>", "<think", "</think>", "```")):
            residual_tag_rows.append(
                {
                    "index": idx,
                    "task_index": task_index,
                    "sample_index": int(raw.get("sample_index", 0)),
                    "data_source": data_source,
                    "source": extraction.source,
                }
            )

        if benchmark in {"humaneval", "mbpp", "bigcodebench"}:
            converted.append({"task_id": gt["task_id"], "solution": code})
        elif benchmark == "livecodebench":
            item = lcb_by_task.setdefault(task_index, {"question_id": gt["question_id"], "code_list": []})
            item["code_list"].append(code)
        else:
            raise ValueError(f"unsupported benchmark: {benchmark}")

    if benchmark in {"humaneval", "mbpp", "bigcodebench"}:
        write_jsonl(output, converted)
    else:
        converted = [lcb_by_task[idx] for idx in sorted(lcb_by_task)]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = {
        "raw_outputs": str(raw_outputs),
        "validation_parquet": str(validation_parquet),
        "output": str(output),
        "benchmark": benchmark,
        "num_rows": len(converted),
        "extraction": extraction_rows,
        "residual_tag_rows": residual_tag_rows,
        "ok": all(row["status"] == "ok" for row in extraction_rows) and not residual_tag_rows,
        "shared_extractor": "recipe.on_policy_wdl_sft.code_task.code_extraction.extract_code",
    }
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-outputs", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "bigcodebench", "livecodebench"], required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--print-full-report", action="store_true")
    parser.add_argument(
        "--allow-extraction-failures",
        action="store_true",
        help="Write failed extractions as empty solutions and return success so official scoring can count them as wrong.",
    )
    args = parser.parse_args()
    payload = convert(args.raw_outputs, args.validation_parquet, args.output, args.benchmark, args.report)
    if args.print_full_report:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for row in payload["extraction"]:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
        summary = {k: v for k, v in payload.items() if k != "extraction"}
        summary["extraction_status_counts"] = status_counts
        summary["extraction_source_counts"] = source_counts
        summary["residual_tag_count"] = len(payload["residual_tag_rows"])
        print(json.dumps(summary, indent=2, sort_keys=True))
    if payload["ok"]:
        return 0
    return 0 if args.allow_extraction_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
