import re
import random
import logging

logger = logging.getLogger(__name__)


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
NEIGHBOR_RE = re.compile(r"neighbo[u]?r", re.IGNORECASE)


def correct_format(output_str):
    """Accept either `<think></think><answer></answer>` or `<think></think>\n\nanswer` formats."""
    if not isinstance(output_str, str):
        return False

    think_match = None
    for match in THINK_RE.finditer(output_str):
        think_match = match
    if think_match is None:
        # Tolerate outputs that omit <think> but include <answer> tags.
        return ANSWER_TAG_RE.search(output_str) is not None

    if ANSWER_TAG_RE.search(output_str):
        return True

    tail = output_str[think_match.end():].strip()
    return bool(tail)


def extract_solution(solution_str):
    """Prefer content in <answer> tags; otherwise use the tail after </think>."""
    if not isinstance(solution_str, str):
        return ""

    tag_match = ANSWER_TAG_RE.search(solution_str)
    if tag_match:
        return tag_match.group(1).strip()

    think_match = None
    for match in THINK_RE.finditer(solution_str):
        think_match = match
    if think_match is None:
        return ""

    tail = solution_str[think_match.end():].strip()
    if not tail:
        return ""
    return tail.splitlines()[0].strip()


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    format_score=0.1,
    score=1.0,
):
    do_print = (random.randint(1, 64) == 1)

    if not correct_format(solution_str):
        if do_print:
            print("--------------------------------")
            print("Bad format")
            print(f"Solution string: {solution_str}")
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc_score": 0.0,
            "num_neighbor_words": 0,
        }

    solution = extract_solution(solution_str)
    num_neighbor_words = len(NEIGHBOR_RE.findall(solution_str)) if isinstance(solution_str, str) else 0

    if do_print:
        print("--------------------------------")
        print("Correct format")
        print(f"Target: {ground_truth}")
        print(f"Extracted solution: {solution}")
        print(f"Solution string: {solution_str}")
        print(f"Number of 'neighbor' in the solution string: {num_neighbor_words}")

    acc_score = 1.0 if str(solution) == str(ground_truth) else 0.0
    if do_print:
        print("Correct answer" if acc_score == 1.0 else "Incorrect answer")

    total = format_score + acc_score
    return {
        "score": total,
        "format_score": format_score,
        "acc_score": acc_score,
        "num_neighbor_words": num_neighbor_words,
    }
