# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom reward function using LaTeX semantic verification.

This module uses the math_verify library for robust LaTeX answer extraction
and verification, following the approach used by the colleague's evaluation script.

Key differences from the original compute_score_strict_box:
1. Uses latex2sympy2_extended for LaTeX normalization
2. Uses math_verify for semantic equivalence checking
3. Extracts answers from the entire response, not just the last 100-300 characters
4. Supports multiple boxed expressions and various LaTeX formats
"""

from typing import Optional
import re

# Try to import math_verify dependencies
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not available. Install with: pip install math-verify latex2sympy2-extended")

# Fallback imports
from verl.utils.reward_score.math_dapo import normalize_final_answer, last_boxed_only_string, remove_boxed
from verl.utils.reward_score import math_verify as verl_math_verify


def verify_with_latex(pred: str, gold: str) -> Optional[bool]:
    """
    Verify answer using LaTeX semantic parsing.

    This follows the colleague's approach in eval_vllm_thinking_math.py:141-174.

    Args:
        pred: Model's prediction string (full response)
        gold: Ground truth answer

    Returns:
        True if correct, False if incorrect, None if verification failed
    """
    if not MATH_VERIFY_AVAILABLE:
        return None

    try:
        # Parse ground truth - wrap in $ for LaTeX parsing
        gold_str = '$' + gold.strip().strip('$') + '$'
        gold_parsed = parse(gold_str, extraction_mode="first_match")

        if len(gold_parsed) == 0:
            # Failed to parse gold, fall back to string matching
            return None

        # Parse prediction with rich configuration
        answer_parsed = parse(
            pred,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        # equations=True is deprecated - now handled by parser automatically
                        boxed="all",  # Extract all boxed content
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="all",
        )

        # Verify semantic equivalence
        return bool(verify(gold_parsed, answer_parsed))

    except Exception as e:
        # If math_verify fails, return None to trigger fallback
        return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last boxed answer from text.

    Args:
        text: Text containing \\boxed{} expression

    Returns:
        Content inside the last \\boxed{}, or None if not found
    """
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except AssertionError:
            return None
    return None


def compute_score_latex_verify(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> dict:
    """
    Compute score using LaTeX semantic verification.

    This function combines:
    1. EOS token checking (from original compute_score_strict_box)
    2. LaTeX semantic verification (from colleague's eval_vllm_thinking_math.py)
    3. Fallback to string matching if math_verify fails

    Design Philosophy:
    - No EOS token = Format Error = Wrong Answer → -1.0 reward
    - Has EOS + Wrong Answer → -1.0 reward
    - Has EOS + Correct Answer → +1.0 reward

    Args:
        data_source: Dataset source identifier
        solution_str: Model's solution string (already decoded with skip_special_tokens=True)
        ground_truth: Ground truth answer
        extra_info: Additional metadata containing 'valid_response_length' and 'max_resp_len'
        **kwargs: Additional arguments

    Returns:
        Dictionary containing score, acc, pred, has_eos, and verification details
    """
    # Check if EOS token was generated
    has_eos = True  # Default assumption

    if extra_info is not None:
        valid_length = extra_info.get('valid_response_length', None)
        max_resp_len = extra_info.get('max_resp_len', None)

        if valid_length is not None and max_resp_len is not None:
            # If valid_length reaches max_resp_len, the response was truncated (no EOS)
            has_eos = (valid_length < max_resp_len)

    # Extract predicted answer for logging
    pred = extract_boxed_answer(solution_str)
    if pred is None:
        pred = '[NO_BOXED]'

    # Try LaTeX semantic verification first
    latex_correct = verify_with_latex(solution_str, ground_truth)

    # If LaTeX verification succeeded, use its result
    if latex_correct is not None:
        correct = latex_correct
        verification_method = "latex_semantic"
    else:
        # Fallback 1: Try verl's math_verify
        try:
            verl_score = verl_math_verify.compute_score(solution_str, ground_truth)
            correct = (verl_score > 0)
            verification_method = "verl_math_verify"
        except Exception:
            correct = False
            verification_method = "verl_math_verify_error"

        # Fallback 2: If verl_math_verify said incorrect, try string matching
        # on extracted boxed answer as a safety net
        if not correct and pred != '[NO_BOXED]':
            normalized_pred = normalize_final_answer(pred)
            normalized_gt = normalize_final_answer(ground_truth)
            if normalized_pred == normalized_gt:
                correct = True
                verification_method = "string_match"

        # No answer extracted at all
        if not correct and verification_method == "verl_math_verify_error" and pred == '[NO_BOXED]':
            verification_method = "no_answer"

    # Reward Logic:
    # 1. No EOS → Always wrong (format error)
    # 2. Has EOS + Wrong Answer → Wrong
    # 3. Has EOS + Correct Answer → Correct
    if not has_eos:
        reward = -1.0
        final_correct = False
    elif correct:
        reward = 1.0
        final_correct = True
    else:
        reward = -1.0
        final_correct = False

    return {
        "score": reward,
        "acc": final_correct,
        "pred": pred,
        "has_eos": has_eos,
        "answer_correct": correct,
        "verification_method": verification_method,
    }


# Alias for backward compatibility
compute_score = compute_score_latex_verify
