#!/usr/bin/env python3
"""Deprecated: use prepare_official_only_validation.py."""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "prepare_official_eval_validation.py is deprecated because it created official-shaped local rows. "
        "Use prepare_official_only_validation.py, which loads EvalPlus, BigCodeBench, and LiveCodeBench official data."
    )


if __name__ == "__main__":
    main()
