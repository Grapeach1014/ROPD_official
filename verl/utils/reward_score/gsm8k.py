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

import re
from typing import Optional

_SOLUTION_CLIP_CHARS = 300
_STRICT_HASH_PATTERN = re.compile(r"#### ([+-]?[0-9\.,]+)")
_HASH_MARKER_PATTERN = re.compile(r"####(?P<content>.*?)(?=(?:####)|$)", re.DOTALL)
_NUMERIC_PATTERN = re.compile(r"(?:\$)?[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_BOXED_PATTERN = re.compile(r"\\boxed\{[^{}]+\}")


def _clip_solution(solution_str: str) -> str:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        return solution_str[-_SOLUTION_CLIP_CHARS:]
    return solution_str


def _normalize_number_token(token: str) -> str:
    return token.strip().replace(",", "").replace("$", "").lstrip("+")


def _extract_last_numeric_token(text: str) -> Optional[str]:
    matches = _NUMERIC_PATTERN.findall(text)
    if not matches:
        return None
    return _normalize_number_token(matches[-1])


def _extract_relaxed_hash_answer(hash_content: str) -> Optional[str]:
    for line in hash_content.splitlines():
        if not line.strip():
            continue
        answer = _extract_last_numeric_token(line)
        if answer is not None:
            return answer
    return _extract_last_numeric_token(hash_content)


def _extract_boxed_answer(solution_str: str) -> Optional[str]:
    boxed_matches = list(_BOXED_PATTERN.finditer(solution_str))
    if not boxed_matches:
        return None
    boxed_content = boxed_matches[-1].group(0)[len(r"\boxed{") : -1]
    return _extract_last_numeric_token(boxed_content)


def extract_solution_details(solution_str: str, method: str = "strict") -> dict[str, Optional[str]]:
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    solution_str = _clip_solution(solution_str)

    if method == "strict":
        solutions = _STRICT_HASH_PATTERN.findall(solution_str)
        if solutions:
            return {
                "answer": _normalize_number_token(solutions[-1]),
                "answer_marker_type": "hash",
                "parser_mode": "strict_hash",
            }

        hash_matches = list(_HASH_MARKER_PATTERN.finditer(solution_str))
        if hash_matches:
            final_answer = _extract_relaxed_hash_answer(hash_matches[-1].group("content"))
            return {
                "answer": final_answer,
                "answer_marker_type": "hash",
                "parser_mode": "relaxed_hash" if final_answer is not None else "none",
            }

        boxed_answer = _extract_boxed_answer(solution_str)
        if boxed_answer is not None:
            return {
                "answer": boxed_answer,
                "answer_marker_type": "boxed",
                "parser_mode": "boxed",
            }

        if _BOXED_PATTERN.search(solution_str):
            return {
                "answer": None,
                "answer_marker_type": "boxed",
                "parser_mode": "none",
            }

        return {
            "answer": None,
            "answer_marker_type": "none",
            "parser_mode": "none",
        }

    answer = re.findall(r"([+-]?[0-9\.,]+)", solution_str)
    final_answer = None
    if answer:
        invalid_str = ["", "."]
        # find the last number that is not '.'
        for final_answer in reversed(answer):
            if final_answer not in invalid_str:
                break
    return {
        "answer": final_answer,
        "answer_marker_type": "plain_number" if final_answer is not None else "none",
        "parser_mode": "flexible" if final_answer is not None else "none",
    }


def extract_solution(solution_str, method="strict"):
    return extract_solution_details(solution_str=solution_str, method=method)["answer"]


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
