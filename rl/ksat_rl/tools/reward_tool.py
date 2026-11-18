"""Rule reward for multi-turn tool trajectories (veRL custom_reward_function: compute_score).

Same grading as ksat_rl/reward.py (official / consensus answer, mc choice number or sa integer), but applied
ONLY to the final assistant turn, because a tool trajectory's response string also contains tool outputs:

  response (decoded with skip_special_tokens=True, so <|im_start|>/<|im_end|> are gone):
    reasoning </think> ... <tool_call>...</tool_call> user <tool_response> OUT </tool_response> assistant <think> ...
  final turn = text after the last "</tool_response>" (tool outputs are defanged by KsatPythonTool, so code
               output can never contain a literal "</tool_response>", "</think>" or turn marker)
  score 1.0  iff the final turn closed its thinking, contains no unexecuted <tool_call> (budget exhausted or
             truncated mid-call = not finished, ksat_tool_rl_v2 §5) and the last \\boxed{} after </think> is right.
Metrics returned alongside the score (logged by veRL): acc, think_closed, has_boxed, unfinished_tool_call,
tool_calls, tool_errors, infra_fault.  Infra-faulted trajectories (sandbox_error) are also excluded from the
policy loss by KsatToolAgentLoop (response_mask zeroed).
"""
import json

from ksat_rl.reward import grade

_SPECIALS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def final_turn(solution_str: str) -> str:
    return solution_str.rsplit("</tool_response>", 1)[-1] if "</tool_response>" in solution_str else solution_str


def score_response(solution_str: str, kind: str, answer: int, choices=None) -> dict:
    turn = final_turn(solution_str)
    closed = "</think>" in turn
    content = turn.rsplit("</think>", 1)[1] if closed else ""
    for s in _SPECIALS:
        content = content.replace(s, "")
    unfinished = "<tool_call>" in content
    pred, ok = (None, False)
    if closed and not unfinished:
        pred, ok = grade(kind, int(answer), choices, content)
    return {"pred": pred, "ok": bool(ok), "closed": closed, "unfinished": unfinished, "content": content}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    r = score_response(solution_str, gt["kind"], gt["answer"], gt.get("choices"))
    ei = extra_info or {}

    def num(k):
        try:
            return float(ei.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "score": 1.0 if r["ok"] else 0.0,
        "acc": float(r["ok"]),
        "think_closed": float(r["closed"]),
        "has_boxed": float(r["pred"] is not None),
        "unfinished_tool_call": float(r["unfinished"]),
        "tool_calls": num("ksat_tool_calls"),
        "tool_errors": num("ksat_tool_error") + num("ksat_tool_timeout") + num("ksat_tool_memory") + num("ksat_tool_killed")
                       + num("ksat_tool_invalid"),
        "infra_fault": num("ksat_infra_fault"),
    }
