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

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


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
        return {
            "score": 0.0,
            "acc": 0.0,
            "format_valid": False,
            "correct": False,
        }

    is_correct = answer == ground_truth
    return {
        "score": score if is_correct else format_score,
        "acc": 1.0 if is_correct else 0.0,
        "format_valid": True,
        "correct": is_correct,
    }


def compute_score_with_format_repair(solution_str, ground_truth, score=1.0):
    """Recover an untagged GSM8K answer without changing the original text.

    The strict parser remains the source of truth. Repair is only considered when
    the response has no ``####`` answer but the flexible parser can identify a
    final numeric answer. The repaired result is explicitly marked for ablation.
    """
    strict_result = compute_score(solution_str, ground_truth, score=score)
    if strict_result["format_valid"]:
        strict_result["format_repaired"] = False
        return strict_result

    repaired_answer = extract_solution(solution_str, method="flexible")
    if repaired_answer is None:
        strict_result["format_repaired"] = False
        return strict_result

    is_correct = repaired_answer == ground_truth
    return {
        "score": score if is_correct else 0.0,
        "acc": 1.0 if is_correct else 0.0,
        "format_valid": True,
        "correct": is_correct,
        "format_repaired": True,
    }
