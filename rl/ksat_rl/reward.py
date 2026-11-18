"""Rule-based reward for KSAT math (veRL custom_reward_function).

Phase-1 reward from ksat_tool_rl_v2/README.md §6, simplified for GRPO:
  1.0  final answer (last \\boxed{} after </think>) matches the official answer
  0.0  otherwise, including no </think> (truncated thinking) or no \\boxed{} (format failure)
Grading mirrors experiments/csat2026_baseline/code/solve.py::grade so train reward == eval metric:
mc accepts the choice number (circled digits normalised) or the exact choice value; sa needs the integer.
The official answer only reaches this function through ground_truth (grader-side), never the policy.
"""
import json
import re

CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}


def last_boxed(s: str):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k].strip()
    return None


def _norm(x: str) -> str:
    x = x.replace("$", "").replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    x = x.replace("\\text{", "{").replace("\\mathrm{", "{")
    return re.sub(r"\s+", "", x)


def grade(kind: str, answer: int, choices, content: str):
    b = last_boxed(content or "")
    if b is None:
        return None, False
    b2 = b
    for k, v in CIRCLED.items():
        b2 = b2.replace(k, v)
    b2 = re.sub(r"^\(?([1-5])\)?\.?$", r"\1", _norm(b2))
    b2 = re.sub(r"^\{([0-9]+)\}$", r"\1", b2)
    if kind == "mc":
        if b2 == str(answer):
            return b, True
        if choices and 1 <= answer <= len(choices) and _norm(b) == _norm(choices[answer - 1]):
            return b, True
        return b, False
    m = re.fullmatch(r"-?\d+", b2.replace(",", ""))
    return b, bool(m and int(m.group()) == int(answer))


def split_response(solution_str: str):
    """Returns (thinking_closed, content). Content = text after the last </think>."""
    if "</think>" not in solution_str:
        return False, ""
    content = solution_str.rsplit("</think>", 1)[1]
    content = content.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return True, content


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    closed, content = split_response(solution_str)
    pred, ok = (None, False)
    if closed:
        pred, ok = grade(gt["kind"], int(gt["answer"]), gt.get("choices"), content)
    return {
        "score": 1.0 if ok else 0.0,
        "acc": float(ok),
        "think_closed": float(closed),
        "has_boxed": float(pred is not None),
    }
