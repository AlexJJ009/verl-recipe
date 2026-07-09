#!/usr/bin/env python3
"""Summarize unified N=3 official code offline eval outputs.

The official scorers expose different field names. This script normalizes them
into the two metrics used for code-task checkpoint selection: sample-level
mean@3 and task-level pass@3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BENCHMARKS = ("humaneval", "mbpp", "bigcodebench", "livecodebench")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def status_pass(row: dict[str, Any], key: str = "status") -> bool:
    return str(row.get(key, "")).lower() == "pass"


def summarize_evalplus(summary: dict[str, Any], plus: bool = True) -> dict[str, Any]:
    data = summary.get("summary", {})
    prefix = "plus" if plus else "base"
    return {
        "ok": bool(summary.get("ok")),
        "harness": summary.get("harness"),
        "scope": prefix,
        "num_tasks": data.get("num_tasks"),
        "num_samples": data.get("num_samples"),
        "mean@3": data.get(f"{prefix}_pass_rate"),
        "pass@3": data.get(f"{prefix}_any_at_n", data.get(f"{prefix}_pass_at_n")),
        "result_path": data.get("result_path"),
    }


def summarize_status_eval(result_path: Path, status_key: str = "status") -> dict[str, Any]:
    data = load_json(result_path)
    eval_rows = data.get("eval", {})
    num_tasks = len(eval_rows)
    num_samples = 0
    sample_pass = 0
    task_pass = 0
    for rows in eval_rows.values():
        row_passes = [status_pass(row, status_key) for row in rows]
        num_samples += len(row_passes)
        sample_pass += sum(row_passes)
        task_pass += int(any(row_passes))
    return {
        "num_tasks": num_tasks,
        "num_samples": num_samples,
        "mean@3": sample_pass / num_samples if num_samples else None,
        "pass@3": task_pass / num_tasks if num_tasks else None,
    }


def summarize_bigcodebench(summary: dict[str, Any]) -> dict[str, Any]:
    data = summary.get("summary", {})
    result_path = Path(data.get("result_path", ""))
    out = {
        "ok": bool(summary.get("ok")),
        "harness": summary.get("harness"),
        "scope": "full",
        "num_tasks": None,
        "num_samples": None,
        "mean@3": None,
        "pass@3": None,
        "result_path": str(result_path) if str(result_path) else None,
        "pass_at_k_path": data.get("pass_at_k_path"),
    }
    if result_path.is_file():
        out.update(summarize_status_eval(result_path))
    pass_at_k = data.get("pass_at_k") or {}
    if isinstance(pass_at_k, dict):
        # Keep official pass@3 when present, but do not let it mask a missing
        # mean@3 from the per-sample result file.
        for key in ("pass@3", "pass_at_3"):
            if key in pass_at_k:
                out["official_pass@3"] = pass_at_k[key]
                break
    return out


def summarize_livecodebench(summary: dict[str, Any]) -> dict[str, Any]:
    data = summary.get("summary", {})
    metrics = data.get("metrics")
    out = {
        "ok": bool(summary.get("ok")),
        "harness": summary.get("harness"),
        "scope": "codegeneration",
        "num_tasks": None,
        "num_samples": None,
        "mean@3": None,
        "pass@3": None,
        "result_path": data.get("eval_path"),
    }
    if isinstance(metrics, list) and metrics:
        metric = metrics[0]
    elif isinstance(metrics, dict):
        metric = metrics
    else:
        metric = {}
    detail = metric.get("detail", {}) if isinstance(metric, dict) else {}
    pass_detail = detail.get("pass@1", {}) if isinstance(detail, dict) else {}
    if isinstance(pass_detail, dict) and pass_detail:
        values = [float(v) for v in pass_detail.values()]
        out["num_tasks"] = len(values)
        out["num_samples"] = len(values) * 3
        out["mean@3"] = sum(values) / len(values)
        out["pass@3"] = sum(1 for v in values if v > 0) / len(values)
    if isinstance(metric, dict):
        for key in ("pass@3", "pass_at_3"):
            if key in metric:
                out["official_pass@3"] = metric[key]
    return out


def summarize_case(output_root: Path, label: str, benchmark: str) -> dict[str, Any]:
    summary_path = output_root / label / benchmark / "official_summary.json"
    if not summary_path.is_file():
        return {
            "label": label,
            "benchmark": benchmark,
            "ok": False,
            "missing": True,
            "summary_path": str(summary_path),
        }
    summary = load_json(summary_path)
    if benchmark in {"humaneval", "mbpp"}:
        normalized = summarize_evalplus(summary, plus=True)
        normalized["base"] = summarize_evalplus(summary, plus=False)
    elif benchmark == "bigcodebench":
        normalized = summarize_bigcodebench(summary)
    elif benchmark == "livecodebench":
        normalized = summarize_livecodebench(summary)
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    normalized.update({"label": label, "benchmark": benchmark, "summary_path": str(summary_path)})
    return normalized


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f}"


def render_markdown(rows: list[dict[str, Any]], title: str, setting_description: str) -> str:
    lines = [
        f"# {title}",
        "",
        setting_description,
        "",
        "| label | benchmark | mean@3 (%) | pass@3 (%) | tasks | samples | harness | ok |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['benchmark']} | {pct(row.get('mean@3'))} | "
            f"{pct(row.get('pass@3'))} | {row.get('num_tasks') or 'NA'} | "
            f"{row.get('num_samples') or 'NA'} | {row.get('harness') or 'NA'} | {row.get('ok')} |"
        )
    lines.append("")
    lines.append("For HumanEval and MBPP, the main row reports EvalPlus plus scope; base scope is retained in the JSON summary.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data-1/eval_outputs/code_task/v2_latest_unified_n3"))
    parser.add_argument("--labels", nargs="+", default=["v2_beta0_latest_step150", "v2_beta01_latest_step150"])
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--title", default="Code Task V2 Latest Unified N=3 Offline Eval")
    parser.add_argument(
        "--setting-description",
        default="Generation setting: n=3, temperature=1.0, top_p=0.95, max_tokens=4096, seed=42, enable_thinking=true.",
    )
    args = parser.parse_args()

    rows = [summarize_case(args.output_root, label, benchmark) for label in args.labels for benchmark in args.benchmarks]
    payload = {"output_root": str(args.output_root), "labels": args.labels, "benchmarks": args.benchmarks, "rows": rows}
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text(render_markdown(rows, args.title, args.setting_description), encoding="utf-8")
    print(render_markdown(rows, args.title, args.setting_description))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
