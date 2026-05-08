# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import signal
import re
from functools import lru_cache
from importlib import import_module
from typing import Any, Optional

FORMAT_PENALTY = False
AIME24_EXTRACTION_WINDOW_CHARS = 1024
AIME24_MARKER_LOOKAHEAD_CHARS = 160
AIME24_INTEGER_PATTERN = re.compile(r"(?<!\d)\d{1,3}(?!\d)")
AIME24_TOKEN_PATTERN = re.compile(r"\\boxed\{[^{}]*\}|\$[^$]+\$|(?<!\d)\d{1,3}(?!\d)")
AIME24_ANSWER_MARKER_PATTERNS = (
    re.compile(r"(?i)\bfinal\s+(?:numeric\s+)?answer\s*(?:is|=|:)\s*"),
    re.compile(r"(?i)\b(?:the\s+)?answer\s*(?:is|=|:)\s*"),
    re.compile(r"(?i)\bfinal\s*(?:is|=|:)\s*"),
    re.compile(r"(?i)\b(?:submit|submitted|enter|entered)\s*(?:is|=|:)?\s*"),
)


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = r"\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return ""


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    pred = _prediction_window(pred, pause_tokens_index)

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred == gt, extracted_pred


def _prediction_window(solution_str: str, pause_tokens_index: Optional[list[int]] = None) -> str:
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        return solution_str[pause_tokens_index[-1] - 100 :]
    return solution_str[-100:]


@lru_cache(maxsize=1)
def _load_math_verify() -> tuple[Any, Any, tuple[Any, ...]]:
    try:
        math_verify_module = import_module("math_verify")
        parser_module = import_module("math_verify.parser")
    except ImportError as exc:
        raise RuntimeError(
            "math-verify is required for math scoring. Install `math-verify[antlr4_9_3]==0.8.0`."
        ) from exc

    extraction_config = (
        parser_module.LatexExtractionConfig(),
        parser_module.ExprExtractionConfig(),
    )
    return math_verify_module.parse, math_verify_module.verify, extraction_config


def _parse_math_verify_candidates(text: str) -> list[Any]:
    mv_parse, _, extraction_config = _load_math_verify()
    return list(mv_parse(text, extraction_config))


def _verify_math_candidates(pred_candidates: list[Any], gold_candidates: list[Any]) -> tuple[bool, Optional[str]]:
    if not pred_candidates:
        return False, None

    _, mv_verify, _ = _load_math_verify()
    fallback_pred = str(pred_candidates[0])

    try:
        with timeout(seconds=5):
            for pred_candidate in pred_candidates:
                for gold_candidate in gold_candidates:
                    if mv_verify(gold_candidate, pred_candidate):
                        return True, str(pred_candidate)
    except Exception:
        pass

    return False, fallback_pred


def _normalize_aime24_candidate_text(text: str) -> Optional[str]:
    candidate = text.strip()
    boxed_candidate = last_boxed_only_string(candidate)
    if boxed_candidate:
        candidate = remove_boxed(boxed_candidate) or candidate
    if candidate.startswith("$") and candidate.endswith("$") and len(candidate) >= 2:
        candidate = candidate[1:-1].strip()

    integer_match = AIME24_INTEGER_PATTERN.search(candidate)
    if integer_match is None:
        return None
    return str(int(integer_match.group(0)))


def _extract_aime24_fallback_candidate(solution_str: str) -> Optional[str]:
    tail_text = solution_str[-AIME24_EXTRACTION_WINDOW_CHARS:]
    ranked_candidates: list[tuple[int, str]] = []

    boxed_candidate = last_boxed_only_string(tail_text)
    if boxed_candidate:
        normalized_boxed = _normalize_aime24_candidate_text(boxed_candidate)
        if normalized_boxed is not None:
            ranked_candidates.append((tail_text.rfind(boxed_candidate), normalized_boxed))

    for marker_pattern in AIME24_ANSWER_MARKER_PATTERNS:
        for marker_match in marker_pattern.finditer(tail_text):
            candidate_window = tail_text[marker_match.end() : marker_match.end() + AIME24_MARKER_LOOKAHEAD_CHARS]
            token_matches = list(AIME24_TOKEN_PATTERN.finditer(candidate_window))
            if not token_matches:
                continue

            token_match = token_matches[-1]
            normalized_token = _normalize_aime24_candidate_text(token_match.group(0))
            if normalized_token is None:
                continue

            ranked_candidates.append((marker_match.end() + token_match.start(), normalized_token))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[0])
    return ranked_candidates[-1][1]


def _verify_aime24_fallback(solution_str: str, gold_candidates: list[Any]) -> tuple[bool, Optional[str]]:
    fallback_candidate = _extract_aime24_fallback_candidate(solution_str)
    if fallback_candidate is None:
        return False, None

    pred_candidates = _parse_math_verify_candidates(fallback_candidate)
    correct, pred = _verify_math_candidates(pred_candidates, gold_candidates)
    if pred is not None:
        return correct, pred
    return correct, fallback_candidate


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None, data_source: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (correct, extracted_prediction)
    """
    prediction_window = _prediction_window(solution_str, pause_tokens_index)
    gold_candidates = _parse_math_verify_candidates(answer)
    if not gold_candidates:
        gold_candidates = _parse_math_verify_candidates(rf"\boxed{{{answer}}}")

    pred_candidates = _parse_math_verify_candidates(prediction_window)
    correct, pred = _verify_math_candidates(pred_candidates, gold_candidates)
    if data_source == "aime24" and (pred is None or pred == ""):
        return _verify_aime24_fallback(solution_str, gold_candidates)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
) -> dict[str, object]:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, 0 for incorrect)
    """
    extra_info = extra_info or {}
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)
    data_source = extra_info.get("data_source")

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, pause_tokens_index, data_source=data_source)

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred is None or pred == ""
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    # Generate explicit feedback for format errors (analogous to code feedback)
    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        feedback = (
            "Your answer had the wrong format. Please provide a final answer that can be parsed as a "
            "mathematical expression."
        )
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
