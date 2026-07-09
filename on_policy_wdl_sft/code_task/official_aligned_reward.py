#!/usr/bin/env python3
"""Official-only code reward helpers.

Benchmark data sources are scored with the corresponding official evaluator
package/repository. Smoke-only local execution remains available only for
non-official training data such as code_train/code_val_smoke.
"""

from __future__ import annotations

import ast
import base64
import functools
import inspect
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
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


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

OFFICIAL_METHODS = {
    "HumanEval": "evalplus",
    "HumanEval+": "evalplus",
    "MBPP": "evalplus",
    "MBPP+": "evalplus",
    "BigCodeBench": "bigcodebench",
    "LiveCodeBench": "livecodebench",
}

LCB_RELEASE_FILES = {
    "release_v1": ("test.jsonl",),
    "release_v2": ("test.jsonl", "test2.jsonl"),
    "release_v3": ("test.jsonl", "test2.jsonl", "test3.jsonl"),
    "release_v4": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"),
    "release_v5": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"),
    "release_v6": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"),
}


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _base_payload(
    score: float,
    status: str,
    pred: str,
    method: str,
    num_tests: int = 0,
    num_passed: int = 0,
    stderr_excerpt: str = "",
) -> dict[str, Any]:
    label_score = 1.0 if score >= 1.0 else float(os.environ.get("CODE_REWARD_FAILURE_SCORE", "-1.0"))
    payload: dict[str, Any] = {
        "score": label_score,
        "acc": 1.0 if score >= 1.0 else 0.0,
        "code_reward_status": status,
        "code_reward_extraction_fail": 0,
        "code_reward_compile_error": 0,
        "code_reward_runtime_error": 0,
        "code_reward_timeout": 0,
        "code_reward_dependency_error": 0,
        "code_reward_num_tests": int(num_tests),
        "code_reward_num_passed": int(num_passed),
        "code_reward_stderr_excerpt": stderr_excerpt[-1000:],
        "pred": pred[:4000],
        "verification_method": method,
        "official_aligned": method in {"evalplus", "bigcodebench", "livecodebench"},
        "code_reward_sandbox": "",
    }
    if status in STATUS_KEYS:
        payload[STATUS_KEYS[status]] = 1
    return payload


def _parse_tests(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> Any:
    if isinstance(ground_truth, str):
        try:
            return _restore_jsonable(json.loads(ground_truth))
        except json.JSONDecodeError:
            pass
    if ground_truth not in (None, ""):
        return _restore_jsonable(ground_truth)
    if extra_info:
        for key in ("test_case", "tests"):
            if key in extra_info:
                return _parse_tests(extra_info[key])
    return None


def _default_lcb_snapshot_dir() -> Path | None:
    if os.environ.get("LCB_JSONL_DIR"):
        path = Path(os.environ["LCB_JSONL_DIR"])
        return path if path.is_dir() else None
    hf_home = Path(os.environ.get("HF_HOME", "/data-1/.cache/huggingface"))
    snapshots = hf_home / "hub" / "datasets--livecodebench--code_generation_lite" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@functools.lru_cache(maxsize=8)
def _lcb_input_output_by_question_id(release_version: str) -> dict[str, Any]:
    snapshot_dir = _default_lcb_snapshot_dir()
    if snapshot_dir is None:
        raise FileNotFoundError(
            "LiveCodeBench JSONL snapshot not found; set LCB_JSONL_DIR or HF_HOME with code_generation_lite snapshot"
        )
    files = LCB_RELEASE_FILES.get(release_version)
    if files is None:
        raise ValueError(f"unsupported LiveCodeBench release_version={release_version}")
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    except Exception as exc:
        raise RuntimeError(f"LiveCodeBench loader unavailable. {_official_env_hint()}") from exc

    output_by_id: dict[str, Any] = {}
    for name in files:
        path = snapshot_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"LiveCodeBench local JSONL missing for {release_version}: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                problem = json.loads(line)
                sample = CodeGenerationProblem(**problem).get_evaluation_sample()
                output_by_id[str(problem["question_id"])] = _restore_jsonable(sample["input_output"])
    return output_by_id


def _resolve_livecodebench_input_output(test_case: dict[str, Any]) -> Any:
    if "input_output" in test_case:
        return test_case["input_output"]
    question_id = test_case.get("question_id")
    if question_id is None:
        raise KeyError("input_output")
    release_version = str(test_case.get("release_version") or os.environ.get("LCB_RELEASE_VERSION") or "release_v5")
    output_by_id = _lcb_input_output_by_question_id(release_version)
    key = str(question_id)
    if key not in output_by_id:
        raise KeyError(f"LiveCodeBench question_id not found in {release_version}: {key}")
    return output_by_id[key]


def _restore_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"__tuple__"} and isinstance(obj["__tuple__"], list):
            return tuple(_restore_jsonable(v) for v in obj["__tuple__"])
        if set(obj) == {"__complex__"} and isinstance(obj["__complex__"], list) and len(obj["__complex__"]) == 2:
            return complex(obj["__complex__"][0], obj["__complex__"][1])
        return {k: _restore_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_jsonable(v) for v in obj]
    return obj


def _test_count(test_case: Any) -> int:
    if isinstance(test_case, dict) and "inputs" in test_case and "outputs" in test_case:
        return min(len(test_case.get("inputs") or []), len(test_case.get("outputs") or []))
    if isinstance(test_case, dict) and "test" in test_case:
        return len([line for line in str(test_case.get("test") or "").splitlines() if line.lstrip().startswith("def test_")])
    if isinstance(test_case, list):
        return len(test_case)
    if isinstance(test_case, dict) and "testcase" in test_case:
        return len(test_case.get("testcase") or [])
    if isinstance(test_case, dict) and "test_code" in test_case:
        return 1
    return 0


def _classify_stderr(stderr: str) -> str:
    text = stderr.lower()
    if "modulenotfounderror" in text or "importerror" in text:
        return "dependency_error"
    if "syntaxerror" in text or "compile" in text:
        return "compile_error"
    if "assertionerror" in text:
        return "wrong_answer"
    return "runtime_error"


def _kill_processes_with_env_token(token: str) -> None:
    if os.name != "posix" or not token or not Path("/proc").is_dir():
        return
    token_bytes = token.encode()
    victims: set[int] = set()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == os.getpid():
            continue
        try:
            if token_bytes in (proc_dir / "environ").read_bytes():
                victims.add(pid)
        except Exception:
            continue
    pgids: set[int] = set()
    for pid in victims:
        try:
            pgids.add(os.getpgid(pid))
        except Exception:
            pass
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def _kill_processes_using_path(path: Path) -> None:
    if os.name != "posix" or not Path("/proc").is_dir():
        return
    path = path.resolve()
    path_bytes = str(path).encode()
    victims: set[int] = set()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == os.getpid():
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes()
        except Exception:
            cmdline = b""
        match_cmd = path_bytes in cmdline
        match_cwd = False
        try:
            cwd = Path(os.readlink(proc_dir / "cwd")).resolve()
            match_cwd = cwd == path or path in cwd.parents
        except Exception:
            pass
        if match_cmd or match_cwd:
            victims.add(pid)
    pgids: set[int] = set()
    for pid in victims:
        try:
            pgids.add(os.getpgid(pid))
        except Exception:
            pass
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def _run_subprocess(
    code: str,
    timeout: float,
    method: str,
    runner_script: str,
    num_tests: int,
    cwd: Path | None = None,
    command_prefix: list[str] | None = None,
    stdin_script: bool = False,
    python_executable: str | None = None,
    cleanup_paths: list[Path] | None = None,
) -> dict[str, Any]:
    try:
        ast.parse(code)
    except SyntaxError:
        return _base_payload(0.0, "compile_error", code, method, num_tests, 0)
    except (MemoryError, RecursionError):
        return _base_payload(0.0, "compile_error", code, method, num_tests, 0, "python parser failed on generated code")

    with tempfile.TemporaryDirectory(prefix=f"code_reward_{method}_") as td:
        runner = Path(td) / "runner.py"
        if not stdin_script:
            runner.write_text(runner_script, encoding="utf-8")
        py = python_executable or sys.executable
        cmd = [*(command_prefix or []), py, "-"] if stdin_script else [*(command_prefix or []), py, str(runner)]
        max_as_mb = _int_env("CODE_REWARD_EXEC_MAX_AS_MB", 4096)
        run_token = ""
        child_env = None
        if not command_prefix:
            run_token = f"code_reward_{method}_{os.getpid()}_{uuid.uuid4().hex}"
            child_env = os.environ.copy()
            child_env["CODE_REWARD_RUN_TOKEN"] = run_token

        def setup_child_process() -> None:
            if os.name == "posix":
                try:
                    os.setsid()
                except Exception:
                    pass
            if max_as_mb <= 0:
                return
            try:
                import resource

                max_as_bytes = max_as_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_as_bytes, max_as_bytes))
            except Exception:
                pass

        def kill_process_group(proc: subprocess.Popen[str] | None, pgid: int | None = None) -> None:
            if proc is None:
                return
            try:
                if os.name == "posix" and not command_prefix:
                    target_pgid = pgid if pgid is not None else os.getpgid(proc.pid)
                    os.killpg(target_pgid, signal.SIGKILL)
                elif proc.poll() is None:
                    proc.kill()
            except ProcessLookupError:
                pass
            except Exception:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        proc: subprocess.Popen[str] | None = None
        proc_pgid: int | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
                env=child_env,
                preexec_fn=setup_child_process if os.name == "posix" and not command_prefix else None,
            )
            if os.name == "posix" and not command_prefix:
                try:
                    proc_pgid = os.getpgid(proc.pid)
                except Exception:
                    proc_pgid = proc.pid
            stdout, stderr = proc.communicate(input=runner_script if stdin_script else None, timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_group(proc, proc_pgid)
            try:
                stdout, stderr = proc.communicate(timeout=1) if proc is not None else ("", "")
            except Exception:
                pass
            return _base_payload(0.0, "timeout", code, method, num_tests, 0)
        finally:
            kill_process_group(proc, proc_pgid)
            if run_token:
                _kill_processes_with_env_token(run_token)
            for cleanup_path in cleanup_paths or []:
                _kill_processes_using_path(cleanup_path)

    try:
        result = json.loads(stdout.strip().splitlines()[-1])
        total = int(result.get("total", 0))
        passed = int(result.get("passed", 0))
    except Exception:
        if proc.returncode != 0:
            status = _classify_stderr(stderr or stdout)
            return _base_payload(0.0, status, code, method, num_tests, 0, stderr or stdout)
        return _base_payload(0.0, "runtime_error", code, method, num_tests, 0, stderr or stdout)
    if proc.returncode != 0:
        failures = result.get("failures") if isinstance(result, dict) else None
        if isinstance(failures, list) and any(isinstance(item, dict) and item.get("status") == "timeout" for item in failures):
            status = "timeout"
        elif isinstance(failures, list) and failures and all(isinstance(item, dict) and item.get("status") == "wrong_answer" for item in failures):
            status = "wrong_answer"
        else:
            status = _classify_stderr(stderr or stdout)
        return _base_payload(0.0, status, code, method, total, passed, json.dumps(failures, ensure_ascii=False)[:1000] if failures else (stderr or stdout))
    score = 1.0 if total > 0 and passed == total else 0.0
    stderr_excerpt = ""
    if score != 1.0 and isinstance(result, dict) and result.get("failures"):
        stderr_excerpt = json.dumps(result.get("failures"), ensure_ascii=False)[:1000]
    return _base_payload(score, "pass" if score == 1.0 else "wrong_answer", code, method, total, passed, stderr_excerpt)


def _local_exec_runner_script(code: str, test_case: Any) -> str:
    return textwrap.dedent(
        f"""
        import json
        ns = {{}}
        code = {code!r}
        tests = json.loads({json.dumps(test_case, ensure_ascii=False)!r})
        exec(compile(code, '<candidate>', 'exec'), ns, ns)

        def callable_under_test():
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


def score_local_exec(code: str, test_case: Any, timeout: float) -> dict[str, Any]:
    return _run_subprocess(code, timeout, "local_exec", _local_exec_runner_script(code, test_case), _test_count(test_case))


def _normalize_stdout(text: Any) -> str:
    return "\n".join(str(text or "").replace("\r\n", "\n").strip().split())


def _stdin_stdout_cases(test_case: Any) -> list[dict[str, str]]:
    if isinstance(test_case, dict) and "tests" in test_case:
        return _stdin_stdout_cases(test_case.get("tests"))
    if isinstance(test_case, dict) and "inputs" in test_case and "outputs" in test_case:
        return [
            {"input": str(inp), "output": str(out)}
            for inp, out in zip(test_case.get("inputs") or [], test_case.get("outputs") or [])
        ]
    if isinstance(test_case, list):
        cases: list[dict[str, str]] = []
        for item in test_case:
            if not isinstance(item, dict):
                continue
            if "input" in item and "output" in item:
                cases.append({"input": str(item.get("input") or ""), "output": str(item.get("output") or "")})
        return cases
    return []


def _stdin_stdout_runner_script(code: str, cases: list[dict[str, str]], case_timeout: float, workdir: Path) -> str:
    encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    encoded_cases = base64.b64encode(json.dumps(cases, ensure_ascii=False).encode("utf-8")).decode("ascii")
    max_as_mb = _int_env("CODE_REWARD_EXEC_MAX_AS_MB", 4096)
    return textwrap.dedent(
        f"""
        import base64
        import json
        import os
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path

        code = base64.b64decode({encoded_code!r}).decode("utf-8")
        cases = json.loads(base64.b64decode({encoded_cases!r}).decode("utf-8"))

        def norm(text):
            return "\\n".join(str(text or "").replace("\\r\\n", "\\n").strip().split())

        def limit_child_resources():
            max_as_mb = {max_as_mb!r}
            if max_as_mb <= 0 or os.name != "posix":
                return
            try:
                import resource
                max_as_bytes = max_as_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_as_bytes, max_as_bytes))
            except Exception:
                pass

        def setup_child_process():
            if os.name == "posix":
                try:
                    os.setsid()
                except Exception:
                    pass
            limit_child_resources()

        def kill_process_group(proc, pgid=None):
            if proc is None:
                return
            try:
                if os.name == "posix":
                    target_pgid = pgid if pgid is not None else os.getpgid(proc.pid)
                    os.killpg(target_pgid, signal.SIGKILL)
                elif proc.poll() is None:
                    proc.kill()
            except ProcessLookupError:
                pass
            except Exception:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        def kill_candidate_residue(candidate, workdir):
            if os.name != "posix" or not Path("/proc").is_dir():
                return
            candidate_bytes = str(candidate).encode()
            workdir_str = str(workdir)
            victims = set()
            for proc_dir in Path("/proc").iterdir():
                if not proc_dir.name.isdigit():
                    continue
                pid = int(proc_dir.name)
                if pid == os.getpid():
                    continue
                try:
                    cmdline = (proc_dir / "cmdline").read_bytes()
                except Exception:
                    cmdline = b""
                match_cmd = candidate_bytes in cmdline
                match_cwd = False
                try:
                    match_cwd = os.path.realpath(os.readlink(proc_dir / "cwd")) == workdir_str
                except Exception:
                    pass
                if match_cmd or match_cwd:
                    victims.add(pid)
            pgids = set()
            for pid in victims:
                try:
                    pgids.add(os.getpgid(pid))
                except Exception:
                    pass
            for pgid in pgids:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            for pid in victims:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            if victims:
                time.sleep(0.05)

        td = {str(workdir)!r}
        try:
            Path(td).mkdir(parents=True, exist_ok=True)
            candidate = Path(td) / "solution.py"
            candidate.write_text(code, encoding="utf-8")
            total = len(cases)
            passed = 0
            failures = []
            for idx, case in enumerate(cases):
                proc = None
                proc_pgid = None
                try:
                    proc = subprocess.Popen(
                        [sys.executable, str(candidate)],
                        text=True,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=td,
                        preexec_fn=setup_child_process if os.name == "posix" else None,
                    )
                    if os.name == "posix":
                        try:
                            proc_pgid = os.getpgid(proc.pid)
                        except Exception:
                            proc_pgid = proc.pid
                    stdout, stderr = proc.communicate(input=str(case.get("input") or ""), timeout={case_timeout!r})
                except subprocess.TimeoutExpired:
                    if proc is not None:
                        kill_process_group(proc, proc_pgid)
                        try:
                            stdout, stderr = proc.communicate(timeout=1)
                        except Exception:
                            pass
                    kill_candidate_residue(candidate, Path(td))
                    failures.append({{"case": idx, "status": "timeout"}})
                    continue
                finally:
                    kill_process_group(proc, proc_pgid)
                    kill_candidate_residue(candidate, Path(td))
                if proc.returncode != 0:
                    failures.append({{"case": idx, "status": "runtime_error", "stderr": stderr[-500:]}})
                    continue
                if norm(stdout) == norm(case.get("output") or ""):
                    passed += 1
                else:
                    failures.append({{"case": idx, "status": "wrong_answer", "stdout": stdout[-500:]}})
            print(json.dumps({{"total": total, "passed": passed, "failures": failures[:5]}}))
            if total == 0 or passed != total:
                raise SystemExit(1)
        finally:
            kill_candidate_residue(Path(td) / "solution.py", Path(td))
        """
    )


def score_stdin_stdout_exec(code: str, test_case: Any, timeout: float) -> dict[str, Any]:
    cases = _stdin_stdout_cases(test_case)
    if not cases:
        return _base_payload(0.0, "runtime_error", code, "stdin_stdout_exec", 0, 0, "missing stdin/stdout tests")
    try:
        ast.parse(code)
    except SyntaxError:
        return _base_payload(0.0, "compile_error", code, "stdin_stdout_exec", len(cases), 0)
    except (MemoryError, RecursionError):
        return _base_payload(0.0, "compile_error", code, "stdin_stdout_exec", len(cases), 0, "python parser failed on generated code")
    per_case_timeout = float(os.environ.get("CODE_REWARD_STDIN_CASE_TIMEOUT", "2"))
    workdir = Path(tempfile.mkdtemp(prefix="deepcoder_stdio_"))
    try:
        return _run_subprocess(
            code,
            timeout,
            "stdin_stdout_exec",
            _stdin_stdout_runner_script(code, cases, per_case_timeout, workdir),
            len(cases),
            cleanup_paths=[workdir],
        )
    finally:
        _kill_processes_using_path(workdir)
        shutil.rmtree(workdir, ignore_errors=True)


def _kodcode_runner_script(code: str, test_case: dict[str, Any]) -> str:
    test_code = str(test_case.get("pytest") or test_case.get("test") or "")
    encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    encoded_test = base64.b64encode(test_code.encode("utf-8")).decode("ascii")
    return textwrap.dedent(
        f"""
        import base64
        import importlib.util
        import inspect
        import io
        import json
        import os
        import sys
        import types
        import contextlib
        from pathlib import Path

        code = base64.b64decode({encoded_code!r}).decode("utf-8")
        test_code = base64.b64decode({encoded_test!r}).decode("utf-8")

        class _Raises:
            def __init__(self, exc_type):
                self.exc_type = exc_type
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:
                    raise AssertionError(f"did not raise {{self.exc_type}}")
                return issubclass(exc_type, self.exc_type)

        class _Capture:
            def __init__(self):
                self.stdout = io.StringIO()
                self.stderr = io.StringIO()
                self._out_cm = contextlib.redirect_stdout(self.stdout)
                self._err_cm = contextlib.redirect_stderr(self.stderr)
            def __enter__(self):
                self._out_cm.__enter__()
                self._err_cm.__enter__()
                return self
            def __exit__(self, exc_type, exc, tb):
                self._err_cm.__exit__(exc_type, exc, tb)
                self._out_cm.__exit__(exc_type, exc, tb)
            def readouterr(self):
                class _CaptureResult:
                    def __init__(self, out, err):
                        self.out = out
                        self.err = err
                    def __iter__(self):
                        yield self.out
                        yield self.err
                return _CaptureResult(self.stdout.getvalue(), self.stderr.getvalue())

        pytest_stub = types.ModuleType("pytest")
        pytest_stub.raises = lambda exc_type, *args, **kwargs: _Raises(exc_type)
        pytest_stub.approx = lambda value, *args, **kwargs: value
        pytest_stub.main = lambda *args, **kwargs: 0
        sys.modules.setdefault("pytest", pytest_stub)

        workdir = Path.cwd()
        solution_path = workdir / "solution.py"
        solution_path.write_text(code, encoding="utf-8")

        solution_spec = importlib.util.spec_from_file_location("solution", str(solution_path))
        solution_module = importlib.util.module_from_spec(solution_spec)
        assert solution_spec and solution_spec.loader
        sys.modules["solution"] = solution_module
        solution_spec.loader.exec_module(solution_module)

        module = types.ModuleType("hidden_kodcode_tests")
        module.__dict__.update({{"__name__": "hidden_kodcode_tests", "__file__": "<hidden_kodcode_tests>"}})
        for name, obj in vars(solution_module).items():
            if not name.startswith("_"):
                module.__dict__.setdefault(name, obj)
        exec(compile(test_code, "<hidden_kodcode_tests>", "exec"), module.__dict__, module.__dict__)

        tests = [obj for name, obj in vars(module).items() if name.startswith("test_") and callable(obj)]
        total = len([line for line in str(test_code).splitlines() if line.lstrip().startswith("def test_")])
        if len(tests) > total:
            total = len(tests)
        passed = 0
        failures = []
        for fn in tests:
            try:
                params = list(inspect.signature(fn).parameters)
                if any(p not in {{"capsys", "capfd"}} for p in params):
                    failures.append(fn.__name__ + ": unsupported parameters")
                    continue
                if params:
                    with _Capture() as cap:
                        fn(**{{p: cap for p in params}})
                else:
                    fn()
                passed += 1
            except Exception as exc:
                failures.append(fn.__name__ + ": " + repr(exc))
        print(json.dumps({{"total": total, "passed": passed, "failures": failures[:5]}}))
        if total == 0 or passed != total:
            raise SystemExit(1)
        """
    )


def score_kodcode_exec(code: str, test_case: dict[str, Any], timeout: float) -> dict[str, Any]:
    test_code = test_case.get("pytest") or test_case.get("test") or ""
    if not test_code:
        return _base_payload(0.0, "runtime_error", code, "kodcode_exec", 0, 0, "missing KodCode pytest")
    pytest_code = str(test_code)
    if "from solution import" not in pytest_code:
        pytest_code = "from solution import *\n" + pytest_code.strip() + "\n"

    firejail = shutil.which(os.environ.get("KODCODE_EXEC", "firejail"))
    allow_unsandboxed = _bool_env("KODCODE_ALLOW_UNSANDBOXED", False)
    if not firejail and not allow_unsandboxed:
        return _base_payload(
            0.0,
            "dependency_error",
            code,
            "kodcode_exec",
            _test_count({**test_case, "test": pytest_code}),
            0,
            "KodCode official scorer requires firejail; set KODCODE_ALLOW_UNSANDBOXED=1 only for diagnostic smoke.",
        )
    sandbox = "firejail" if firejail else "diagnostic_unsandboxed_hidden_tests"
    command_prefix: list[str] = []
    if firejail:
        sandbox_python = os.environ.get("KODCODE_SANDBOX_PYTHON") or (
            "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else (shutil.which("python3") or sys.executable)
        )
        command_prefix = [
            firejail,
            "--quiet",
            "--seccomp=socket",
            "--profile=pip",
            "--private-dev",
            "--private-tmp",
            "--rlimit-nproc=32",
            "--rlimit-nofile=32",
            "--rlimit-fsize=2m",
            "--rlimit-as=4096m",
            f"--timeout=00:00:{int(timeout)}",
        ]
    with tempfile.TemporaryDirectory(prefix="kodcode_candidate_") as candidate_td:
        candidate_dir = Path(candidate_td)
        if firejail:
            command_prefix.insert(1, f"--private={candidate_dir}")
        runner_script = _kodcode_runner_script(code, {**test_case, "test": pytest_code})
        runner = _run_subprocess(
            code,
            timeout,
            "kodcode_exec",
            runner_script,
            _test_count({**test_case, "test": pytest_code}),
            cwd=candidate_dir,
            command_prefix=command_prefix,
            stdin_script=False,
            python_executable=sandbox_python if firejail else None,
        )
    runner["code_reward_sandbox"] = sandbox
    if runner["code_reward_status"] == "pass":
        runner["code_reward_num_passed"] = runner["code_reward_num_tests"]
    return runner


def _official_env_hint() -> str:
    return (
        "Set PYTHONPATH to include /data-1/code_eval_envs/official_site and, "
        "for LiveCodeBench, /data-1/code_eval_envs/LiveCodeBench."
    )


def _status_from_official(stat: str) -> str:
    stat = str(stat).lower()
    if stat == "pass":
        return "pass"
    if stat == "timeout":
        return "timeout"
    if stat == "fail":
        return "wrong_answer"
    return "runtime_error"


def _code_defines_entry_point(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, MemoryError, RecursionError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return True
    return False


def _build_evalplus_solution(prompt: str, code: str, entry_point: str) -> str:
    """Convert model output into the complete source expected by EvalPlus."""
    if _code_defines_entry_point(code, entry_point):
        return code
    body = inspect.cleandoc(code)
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"


def score_evalplus_official(code: str, test_case: dict[str, Any]) -> dict[str, Any]:
    try:
        from evalplus.evaluate import check_correctness
        from evalplus.gen.util import trusted_exec
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    except Exception as exc:
        raise RuntimeError(f"EvalPlus official evaluator is unavailable. {_official_env_hint()}") from exc

    dataset = str(test_case["benchmark"])
    entry_point = test_case["entry_point"]
    prompt = test_case["prompt"]
    canonical_solution = test_case["canonical_solution"]
    problem = {
        "task_id": test_case["task_id"],
        "prompt": prompt,
        "entry_point": entry_point,
        "base_input": test_case["base_input"],
        "plus_input": test_case["plus_input"],
        "atol": test_case.get("atol", 0),
    }
    output_not_none = entry_point in MBPP_OUTPUT_NOT_NONE_TASKS if dataset == "mbpp" else False
    expected_output = {
        "base": trusted_exec(prompt + canonical_solution, problem["base_input"], entry_point, output_not_none=output_not_none),
        "base_time": [0.1 for _ in problem["base_input"]],
        "plus": trusted_exec(prompt + canonical_solution, problem["plus_input"], entry_point, output_not_none=output_not_none),
        "plus_time": [0.1 for _ in problem["plus_input"]],
    }
    solution = _build_evalplus_solution(prompt, code, entry_point)
    result = check_correctness(
        dataset=dataset,
        completion_id=0,
        problem=problem,
        solution=solution,
        expected_output=expected_output,
        base_only=False,
        fast_check=False,
        identifier="training_reward",
        min_time_limit=float(os.environ.get("EVALPLUS_MIN_TIME_LIMIT", "1.0")),
        gt_time_limit_factor=float(os.environ.get("EVALPLUS_GT_TIME_LIMIT_FACTOR", "4.0")),
    )
    base_stat, base_details = result["base"]
    plus_stat, plus_details = result["plus"]
    passed = int(base_stat == "pass") + int(plus_stat == "pass")
    status = "pass" if passed == 2 else _status_from_official(plus_stat if base_stat == "pass" else base_stat)
    return _base_payload(1.0 if status == "pass" else 0.0, status, solution, "evalplus", 2, passed)


def score_bigcodebench_official(code: str, test_case: dict[str, Any]) -> dict[str, Any]:
    problem = {
        "task_id": test_case["task_id"],
        "entry_point": test_case["entry_point"],
        "test": test_case["test"],
    }
    solution = code
    calibrated = _bool_env("BIGCODEBENCH_CALIBRATED", False)
    if calibrated and test_case.get("code_prompt"):
        solution = test_case["code_prompt"] + "\n    pass\n" + solution

    # BigCodeBench's official reliability_guard changes RLIMIT_AS/RLIMIT_DATA in
    # the current process. Running it inside a Ray reward worker can break later
    # imports/mmap calls, so keep the official checker but isolate it.
    with tempfile.TemporaryDirectory(prefix="code_reward_bigcodebench_") as td:
        payload_path = Path(td) / "payload.json"
        runner_path = Path(td) / "runner.py"
        payload_path.write_text(
            json.dumps(
                {
                    "problem": problem,
                    "solution": solution,
                    "max_as_limit": int(os.environ.get("BIGCODEBENCH_MAX_AS_LIMIT", str(128 * 1024))),
                    "max_data_limit": int(os.environ.get("BIGCODEBENCH_MAX_DATA_LIMIT", str(128 * 1024))),
                    "max_stack_limit": int(os.environ.get("BIGCODEBENCH_MAX_STACK_LIMIT", "10")),
                    "min_time_limit": float(os.environ.get("BIGCODEBENCH_MIN_TIME_LIMIT", "1")),
                    "gt_time_limit": float(os.environ.get("BIGCODEBENCH_GT_TIME_LIMIT", "20")),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        runner_path.write_text(
            textwrap.dedent(
                """
                import json
                import sys
                from pathlib import Path

                try:
                    from bigcodebench.evaluate import check_correctness
                except Exception as exc:
                    raise RuntimeError("BigCodeBench official evaluator is unavailable") from exc

                payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
                result = check_correctness(
                    completion_id=0,
                    problem=payload["problem"],
                    solution=payload["solution"],
                    max_as_limit=payload["max_as_limit"],
                    max_data_limit=payload["max_data_limit"],
                    max_stack_limit=payload["max_stack_limit"],
                    identifier="training_reward",
                    min_time_limit=payload["min_time_limit"],
                    gt_time_limit=payload["gt_time_limit"],
                )
                print(json.dumps(result, ensure_ascii=False))
                """
            ),
            encoding="utf-8",
        )
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(payload_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(os.environ.get("BIGCODEBENCH_SUBPROCESS_TIMEOUT", "90")),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _base_payload(0.0, "timeout", code, "bigcodebench", 1, 0)

    if proc.returncode != 0:
        status = _classify_stderr(proc.stderr)
        return _base_payload(0.0, status, code, "bigcodebench", 1, 0, proc.stderr)
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return _base_payload(0.0, "runtime_error", code, "bigcodebench", 1, 0, proc.stderr or proc.stdout)
    stat, details = result["base"]
    status = _status_from_official(stat)
    return _base_payload(1.0 if status == "pass" else 0.0, status, code, "bigcodebench", 1, int(status == "pass"), json.dumps(details)[:1000])


def score_livecodebench_official(code: str, test_case: dict[str, Any]) -> dict[str, Any]:
    try:
        from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
    except Exception as exc:
        raise RuntimeError(f"LiveCodeBench official evaluator is unavailable. {_official_env_hint()}") from exc

    sample = {"input_output": _resolve_livecodebench_input_output(test_case)}
    result, metadata = check_correctness(
        sample,
        code,
        timeout=int(os.environ.get("LCB_TIMEOUT", "6")),
        debug=False,
    )
    passed_items = [x is True for x in result]
    passed = bool(result) and all(passed_items)
    status = "pass" if passed else "wrong_answer"
    return _base_payload(1.0 if passed else 0.0, status, code, "livecodebench", len(result), sum(passed_items), json.dumps(metadata)[:1000])


def compute_score_code_official_aligned(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    extraction = extract_code(solution_str)
    if not extraction.ok:
        method = OFFICIAL_METHODS.get(str(data_source), "local_exec")
        return _base_payload(0.0, "extraction_fail", "[NO_CODE]", method, 0, 0)

    test_case = _parse_tests(ground_truth, extra_info)
    if test_case is None:
        return _base_payload(0.0, "runtime_error", extraction.code, "local_exec", 0, 0, "missing ground_truth")

    data_source = str(data_source)

    if data_source in {"HumanEval", "HumanEval+", "MBPP", "MBPP+"}:
        return score_evalplus_official(extraction.code, test_case)
    if data_source == "BigCodeBench":
        return score_bigcodebench_official(extraction.code, test_case)
    if data_source == "LiveCodeBench":
        return score_livecodebench_official(extraction.code, test_case)
    timeout = float(os.environ.get("CODE_REWARD_TIMEOUT", "5"))
    if data_source == "kodcode_light_rl_10k" or (
        isinstance(test_case, dict) and test_case.get("verification_method") == "kodcode_exec"
    ):
        return score_kodcode_exec(extraction.code, test_case, timeout)
    if data_source.startswith("deepcoder_preview") or (
        isinstance(test_case, dict) and test_case.get("verification_method") == "stdin_stdout_exec"
    ):
        return score_stdin_stdout_exec(extraction.code, test_case, timeout)
    return score_local_exec(extraction.code, test_case, timeout)


compute_score = compute_score_code_official_aligned
