#!/usr/bin/env python3
"""Deprecated: use verify_official_only_reward.py."""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "verify_official_aligned_reward.py is deprecated. "
        "Use verify_official_only_reward.py, which verifies EvalPlus, BigCodeBench, and LiveCodeBench official scorer paths."
    )


if __name__ == "__main__":
    main()
