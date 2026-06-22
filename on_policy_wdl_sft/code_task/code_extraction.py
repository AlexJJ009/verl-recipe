#!/usr/bin/env python3
"""Extraction helpers for the code-task prompt contract."""

from __future__ import annotations

import re
from dataclasses import dataclass


ANSWER_RE = re.compile(r"<answer>(?P<body>.*?)</answer>", re.DOTALL | re.IGNORECASE)
FENCED_PY_RE = re.compile(r"```(?:python|py)\s*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)
FENCED_ANY_RE = re.compile(r"```\s*(?P<code>.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ExtractionResult:
    code: str
    status: str
    source: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _strip_code(code: str) -> str:
    return code.strip().replace("\r\n", "\n")


def _usable_fenced_code(code: str) -> str:
    code = _strip_code(code)
    first_line, sep, rest = code.partition("\n")
    if not sep and first_line.strip().lower() in {"python", "py"}:
        return ""
    if first_line.strip().lower() in {"python", "py"}:
        code = _strip_code(rest)
    return code


def extract_code(text: str, strict_answer: bool | None = None) -> ExtractionResult:
    """Extract executable Python from the code-task answer contract.

    Strict mode accepts only the first Python fenced block inside <answer>.
    It is the default for code-task training/eval so answer-tag mistakes are
    scored as extraction failures rather than silently salvaged from raw text.
    Set CODE_STRICT_ANSWER_EXTRACTION=0 only for diagnostic compatibility runs.
    """
    if text is None:
        return ExtractionResult("[NO_CODE]", "extraction_fail", "none")
    text = str(text)
    if strict_answer is None:
        import os

        strict_answer = os.environ.get("CODE_STRICT_ANSWER_EXTRACTION", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    answer_match = ANSWER_RE.search(text)
    if strict_answer and not answer_match:
        return ExtractionResult("[NO_CODE]", "extraction_fail", "strict:no_answer")

    search_area = answer_match.group("body") if answer_match else text
    source_prefix = "answer" if answer_match else "full"

    fenced_patterns = [(FENCED_PY_RE, "fenced_python")]
    if not strict_answer:
        fenced_patterns.append((FENCED_ANY_RE, "fenced"))

    for regex, suffix in fenced_patterns:
        match = regex.search(search_area)
        if match:
            code = _usable_fenced_code(match.group("code"))
            if code:
                return ExtractionResult(code, "ok", f"{source_prefix}:{suffix}")

    if strict_answer:
        return ExtractionResult("[NO_CODE]", "extraction_fail", f"{source_prefix}:strict_no_fenced_python")

    raw = _strip_code(search_area)
    if looks_like_python(raw):
        return ExtractionResult(raw, "ok", f"{source_prefix}:raw_python")

    return ExtractionResult("[NO_CODE]", "extraction_fail", source_prefix)


def looks_like_python(text: str) -> bool:
    if not text or len(text) < 8:
        return False
    markers = ("def ", "class ", "import ", "from ", "return ", "print(", "if __name__")
    return any(marker in text for marker in markers)


def wrap_reference_answer(code: str) -> str:
    return f"<think>Reference solution.</think>\n<answer>\n```python\n{code.strip()}\n```\n</answer>"


REQUIRED_EXTRACTION_CASES = {
    "answer_fenced_python": "<think>x</think><answer>\n```python\ndef f():\n    return 1\n```\n</answer>",
    "fenced_python": "```python\ndef f():\n    return 1\n```",
    "raw_python": "def f():\n    return 1\n",
    "missing_code": "<think>no executable answer</think><answer>forty two</answer>",
    "malformed_repeated": "<answer>```python\n```</answer><answer>not code</answer>",
}


def extraction_self_check() -> dict[str, dict[str, str | bool]]:
    out: dict[str, dict[str, str | bool]] = {}
    for name, text in REQUIRED_EXTRACTION_CASES.items():
        result = extract_code(text, strict_answer=name not in {"fenced_python", "raw_python"})
        out[name] = {
            "ok": result.ok,
            "status": result.status,
            "source": result.source,
            "code": result.code,
        }
    return out
