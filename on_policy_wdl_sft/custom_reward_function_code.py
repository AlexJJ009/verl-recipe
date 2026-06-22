#!/usr/bin/env python3
"""Custom code-execution reward for code-task On-Policy WDL-SFT."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

try:
    from recipe.on_policy_wdl_sft.code_task.code_extraction import extract_code
except Exception:
    from code_extraction import extract_code  # type: ignore


STATUS_KEYS = {
    "extraction_fail": "code_reward_extraction_fail",
    "compile_error": "code_reward_compile_error",
    "runtime_error": "code_reward_runtime_error",
    "timeout": "code_reward_timeout",
    "dependency_error": "code_reward_dependency_error",
}


def _base_payload(score: float, status: str, pred: str, method: str, num_tests: int = 0, num_passed: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": float(score),
        "acc": 1.0 if score >= 1.0 else 0.0,
        "code_reward_status": status,
        "code_reward_extraction_fail": 0,
        "code_reward_compile_error": 0,
        "code_reward_runtime_error": 0,
        "code_reward_timeout": 0,
        "code_reward_dependency_error": 0,
        "code_reward_num_tests": int(num_tests),
        "code_reward_num_passed": int(num_passed),
        "pred": pred[:4000],
        "verification_method": method,
    }
    if status in STATUS_KEYS:
        payload[STATUS_KEYS[status]] = 1
    return payload


def _parse_tests(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> Any:
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            pass
    if ground_truth not in (None, ""):
        return ground_truth
    if extra_info:
        for key in ("test_case", "tests"):
            if key in extra_info:
                return _parse_tests(extra_info[key])
    return None


def _test_count(test_case: Any) -> int:
    if isinstance(test_case, dict) and "inputs" in test_case and "outputs" in test_case:
        return min(len(test_case.get("inputs") or []), len(test_case.get("outputs") or []))
    if isinstance(test_case, list):
        return len(test_case)
    if isinstance(test_case, dict) and "testcase" in test_case:
        return len(test_case.get("testcase") or [])
    return 0


def _runner_script(code: str, test_case: Any) -> str:
    return textwrap.dedent(
        f"""
        import contextlib, io, json, sys, traceback
        ns = {{}}
        code = {code!r}
        tests = {json.dumps(test_case, ensure_ascii=False)!r}
        tests = json.loads(tests)
        exec(compile(code, '<candidate>', 'exec'), ns, ns)

        def callable_under_test():
            funcs = [v for v in ns.values() if callable(v) and getattr(v, '__name__', '').startswith('__') is False]
            funcs = [f for f in funcs if getattr(f, '__module__', None) in (None, 'builtins') or True]
            user_funcs = [v for k, v in ns.items() if callable(v) and not k.startswith('_')]
            if not user_funcs:
                raise RuntimeError('no callable function defined')
            return user_funcs[-1]

        def norm_expected(item):
            if isinstance(item, dict) and 'output' in item and len(item) == 1:
                return item['output']
            return item

        fn = callable_under_test()
        total = 0
        passed = 0
        if isinstance(tests, dict) and 'inputs' in tests and 'outputs' in tests:
            for inp, out in zip(tests['inputs'], tests['outputs']):
                total += 1
                expected = norm_expected(out)
                if isinstance(inp, dict):
                    got = fn(**inp)
                elif isinstance(inp, (list, tuple)):
                    got = fn(*inp)
                else:
                    got = fn(inp)
                if got == expected:
                    passed += 1
        elif isinstance(tests, list):
            for item in tests:
                total += 1
                if isinstance(item, str):
                    exec(item, ns, ns)
                    passed += 1
                elif isinstance(item, dict) and 'input' in item and 'output' in item:
                    inp = item['input']
                    expected = norm_expected(item['output'])
                    got = fn(**inp) if isinstance(inp, dict) else fn(*inp if isinstance(inp, list) else [inp])
                    if got == expected:
                        passed += 1
                else:
                    raise RuntimeError('unsupported list-style test item')
        else:
            raise RuntimeError('unsupported test_case schema')
        print(json.dumps({{'total': total, 'passed': passed}}))
        """
    )


def _classify_stderr(stderr: str) -> str:
    text = stderr.lower()
    if "modulenotfounderror" in text or "importerror" in text:
        return "dependency_error"
    if "syntaxerror" in text or "compile" in text:
        return "compile_error"
    return "runtime_error"


def _local_exec(code: str, test_case: Any, timeout: float) -> dict[str, Any]:
    try:
        ast.parse(code)
    except SyntaxError:
        return _base_payload(0.0, "compile_error", code, "local_exec", _test_count(test_case), 0)

    with tempfile.TemporaryDirectory(prefix="code_reward_") as td:
        runner = Path(td) / "runner.py"
        runner.write_text(_runner_script(code, test_case), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(runner)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _base_payload(0.0, "timeout", code, "local_exec", _test_count(test_case), 0)

    if proc.returncode != 0:
        status = _classify_stderr(proc.stderr)
        return _base_payload(0.0, status, code, "local_exec", _test_count(test_case), 0)
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        total = int(result.get("total", 0))
        passed = int(result.get("passed", 0))
    except Exception:
        return _base_payload(0.0, "runtime_error", code, "local_exec", _test_count(test_case), 0)
    score = 1.0 if total > 0 and passed == total else 0.0
    return _base_payload(score, "pass" if score == 1.0 else "wrong_answer", code, "local_exec", total, passed)


def compute_score_code(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    extraction = extract_code(solution_str)
    if not extraction.ok:
        return _base_payload(0.0, "extraction_fail", "[NO_CODE]", "local_exec", 0, 0)

    test_case = _parse_tests(ground_truth, extra_info)
    if test_case is None:
        return _base_payload(0.0, "runtime_error", extraction.code, "local_exec", 0, 0)

    # Sandbox Fusion integration is intentionally gated until a URL is configured.
    # Local execution is accepted for G1 smoke only and is labeled accordingly by verifiers.
    timeout = float(os.environ.get("CODE_REWARD_TIMEOUT", "5"))
    payload = _local_exec(extraction.code, test_case, timeout=timeout)
    payload["pred"] = extraction.code[:4000]
    return payload


compute_score = compute_score_code
