"""Tool spec + prompts for KSAT math tool-integrated reasoning (TIR).

One tool, `python(code)`, backed by the sandbox service (python_sandbox.py).  The same schema is used for
(1) teacher trajectory generation (OpenAI `tools=`), (2) Nanbeige chat-template rendering (`tools=`), and
(3) RL rollouts (SGLang `nanbeige` tool-call parser), so the policy sees one consistent spec.

Prompt layout for tool episodes (matches the no-tool RL prompt except for the system message + tools):
  system: TOOL_SYSTEM_PROMPT [+ optional few-shot demo block for the ICRL 2->1->0 curriculum]
  user:   make_parquet.instruction(p) + "\\n\\n" + p["text"]          (identical to the no-tool RL prompt)
"""
import json

TOOL_NAME = "python"
MAX_TOOL_CALLS = 6

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "격리된 Python 3 샌드박스에서 코드를 실행하고 표준 출력(실패 시 오류 메시지)을 돌려줍니다. "
            "정확한 산술(정수, fractions.Fraction), sympy 기호 계산(방정식 풀이, 미분, 적분, 인수분해, 극한), "
            "itertools 등을 이용한 경우의 수 열거, 손으로 구한 결과의 검산에 쓰세요. "
            "필요한 값은 print()로 출력하세요(마지막 줄이 식이면 그 값도 출력됩니다). "
            "호출마다 새 프로세스에서 실행되므로 변수와 import가 유지되지 않습니다. "
            "인터넷·파일 접근 불가, 호출당 10초·1GB 제한, 출력은 1000자까지만 보입니다.\n"
            "Runs Python 3 code in an isolated sandbox and returns stdout (or the error message). Use it for "
            "exact arithmetic (integers, fractions.Fraction), sympy (solve, diff, integrate, factor, limit), "
            "enumerating cases (itertools), and checking a result you derived. Print what you need with print() "
            "(a bare expression on the last line is printed too). Every call is a fresh process: variables and "
            "imports do not persist. No internet or files; 10 s and 1 GB per call; output is cut at 1000 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "실행할 Python 코드 (sympy, numpy, fractions, itertools 사용 가능) / "
                                   "Python source to run; sympy, numpy, fractions and itertools are available.",
                }
            },
            "required": ["code"],
        },
    },
}
TOOLS = [PYTHON_TOOL]

TOOL_SYSTEM_PROMPT = (
    "당신은 대학수학능력시험 수학 문제를 정확하게 푸는 수학 전문가입니다. python 도구로 코드를 실행할 수 있습니다.\n"
    "- 정확한 계산, 기호 계산(sympy), 경우의 수 열거, 조건 확인, 검산이 도움이 되는 지점에서는 python 도구를 적극적으로 "
    "사용하세요. 답을 print만 하는 호출은 의미가 없습니다.\n"
    "- 호출마다 새 프로세스에서 실행되므로 필요한 import와 정의를 매번 포함하고, 확인할 값을 print()로 출력하세요. "
    "한 번에 모든 것을 하려 하기보다 단계별로 확인하며 여러 번 호출해도 됩니다.\n"
    "- 도구 출력은 실제 실행 결과입니다. 출력된 값이 무엇을 뜻하는지 확인하고 다음 추론에 반영하세요. "
    "예상과 다르거나 오류가 나면 원인을 찾아 고치세요.\n"
    "- 수치 확인이나 일부 표본은 증명을 대신하지 않습니다. 문제의 모든 조건을 만족하는지 확인하세요.\n"
    f"- 도구는 한 문제에 최대 {MAX_TOOL_CALLS}번까지 호출할 수 있습니다.\n"
    "- 최종 응답에는 핵심 풀이를 한국어로 간결하게 정리하고, 마지막에 최종 답을 \\boxed{} 안에 넣으세요."
)


def tool_user_prompt(p: dict) -> str:
    """User turn = the no-tool RL prompt (ksat_rl.make_parquet.instruction) + problem text."""
    from ksat_rl.make_parquet import instruction
    return f"{instruction(p)}\n\n{p['text']}"


def tool_call_xml(name: str, args: dict) -> str:
    """Nanbeige/Qwen3-Coder XML tool-call block, byte-identical to the chat template's rendering."""
    s = f"<tool_call>\n<function={name}>\n"
    for k, v in args.items():
        v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        s += f"<parameter={k}>\n{v}\n</parameter>\n"
    return s + "</function>\n</tool_call>"


def render_demo(demo: dict, idx: int) -> str:
    """Plain-text rendering of one few-shot trajectory (fewshot_demos.json item) for the system prompt.
    Uses the exact XML call format and <tool_response> wrapping the policy must produce/receive, but no
    chat special tokens, so the demo cannot be confused with real turns."""
    lines = [f"### 예시 {idx}", "[문제]", demo["user"], "[풀이]"]
    for step in demo["steps"]:
        if step.get("think"):
            lines.append(step["think"].strip())
        if step.get("content"):
            lines.append(step["content"].strip())
        for call in step.get("tool_calls", []):
            lines.append(tool_call_xml(call["name"], call["arguments"]))
            lines.append(f"<tool_response>\n{call['output']}\n</tool_response>")
    return "\n".join(lines)


def system_prompt_with_demos(demos: list, k: int) -> str:
    """ICRL curriculum (ksat_tool_rl_v2 §4-B): k = 2 -> 1 -> 0 demos appended to the system prompt."""
    if k <= 0 or not demos:
        return TOOL_SYSTEM_PROMPT
    blocks = [render_demo(d, i + 1) for i, d in enumerate(demos[:k])]
    return (TOOL_SYSTEM_PROMPT + "\n\n## 도구 사용 예시 (다른 문제입니다. 형식만 참고하세요)\n\n" + "\n\n".join(blocks))


def build_messages(p: dict, demos: list | None = None, k_demos: int = 0) -> list:
    return [{"role": "system", "content": system_prompt_with_demos(demos or [], k_demos)},
            {"role": "user", "content": tool_user_prompt(p)}]
