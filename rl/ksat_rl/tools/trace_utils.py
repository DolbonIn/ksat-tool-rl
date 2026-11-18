"""Helpers for checking tool-integrated reasoning (TIR) trajectories.

Used by teacher-trajectory generation (data_pipeline/tool/gen_tir.py) and reusable for RL rollout analysis
(e.g. "was the tool output actually used?" statistics, ksat_tool_rl_v2 §5/§7).
"""
import re

# Strings that would corrupt the Nanbeige chat format or the SGLang nanbeige (Qwen3-Coder) tool-call parser
# if they appeared inside reasoning / content / code.
FORBIDDEN_MARKUP = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>",
                    "</tool_call>", "<tool_response>", "</tool_response>", "<function=", "</function>",
                    "<parameter=", "</parameter>")

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_TRIVIAL = {"0", "1", "-1", "2", "0.0", "1.0"}


def forbidden_markup(text: str) -> list:
    return [m for m in FORBIDDEN_MARKUP if m in (text or "")]


def salient_tokens(stdout: str, limit: int = 200) -> list:
    """Numbers (incl. fractions/decimals) and symbolic answers (e.g. 'sqrt(3)', 'True') in a tool output."""
    toks = []
    for t in _NUM.findall(stdout or "")[:limit]:
        if t not in _TRIVIAL and t not in toks:
            toks.append(t)
    for w in re.findall(r"\b(?:True|False)\b|sqrt\(\d+\)|\bpi\b", stdout or ""):
        if w not in toks:
            toks.append(w)
    return toks


def _norm_text(s: str) -> str:
    # unify LaTeX fractions / spacing so "\frac{3}{4}" matches "3/4" and "1,234" matches "1234"
    s = re.sub(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", s or "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s.replace("\\sqrt{", "sqrt(")


def output_used(stdout: str, following_text: str) -> list:
    """Salient tokens of `stdout` that re-appear in the assistant text that follows the tool response
    (next reasoning + content + later code).  Non-empty list => the output was referenced."""
    txt = _norm_text(following_text)
    hits = []
    for t in salient_tokens(stdout):
        if t in ("True", "False"):
            if re.search(rf"\b{t}\b|참|거짓|true|false|holds|맞|성립", txt):
                hits.append(t)
            continue
        if t.startswith("sqrt(") or t == "pi":
            if t.replace("sqrt(", "sqrt(") in txt or (t == "pi" and ("pi" in txt or "π" in txt)):
                hits.append(t)
            continue
        if re.search(rf"(?<![\d.]){re.escape(t)}(?![\d])", txt):
            hits.append(t)
    return hits


# Teacher reasoning that calls the tool unnecessary / used only because the prompt encourages it
# ("Not necessary but instructions encourage. We can verify with python.").  Consistent with the policy
# prompt, but a poor habit to distill; such trajectories are kept apart from the main TIR set.
# Explicit references to the prompt always count; generic phrases ("not necessary", "simple.") only count when
# a tool word is close by (they also occur in math, e.g. "simple root", "a table is not necessary").
_META_EXPLICIT = re.compile(r"encourag|maybe use (?:the )?tool|tool at least|system says|"
                            r"instructions? (?:says?|wants?) (?:to )?(?:use|actively)|"
                            r"instruct\w*.{0,40}?(?:tool|python)|(?:tool|python).{0,40}?instruct\w*", re.I)
_META_GENERIC = re.compile(r"not (?:really |strictly )?(?:necessary|needed)|unnecessary|\b[Ss]imple\.|"
                           r"\bcompl(?:y|iance)\b|적극적", re.I)
_TOOL_WORD = re.compile(r"tool|python|sympy|\bcode\b|\bcall\b|brute|enumerat", re.I)


def meta_tool_commentary(messages: list) -> list:
    """Snippets of assistant reasoning/content that call the tool unnecessary or prompt-driven (empty = clean)."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for fld in ("reasoning_content", "content"):
            t = m.get(fld) or ""
            for mt in _META_EXPLICIT.finditer(t):
                out.append(t[max(0, mt.start() - 40):mt.end() + 30].replace("\n", " "))
            for mt in _META_GENERIC.finditer(t):
                win = t[max(0, mt.start() - 80):mt.end() + 80]
                if _TOOL_WORD.search(win):
                    out.append(t[max(0, mt.start() - 40):mt.end() + 30].replace("\n", " "))
    return out


def defang_markup(text: str) -> str:
    """Neutralise chat/tool markup inside TOOL OUTPUT before it is rendered into the conversation.

    Tool output goes through the chat template, where a printed "<|im_start|>", "</think>" or "</tool_response>"
    would be tokenized as the real special/structural token, letting model-written code forge turns, close a
    tool response early or plant a "</think> ... \\boxed{}" (reward hacking).  The leading "<" of every marker in
    FORBIDDEN_MARKUP is replaced by "&lt;" (visible, never a special token)."""
    if not text or "<" not in text:
        return text
    for m in FORBIDDEN_MARKUP:
        if m in text:
            text = text.replace(m, "&lt;" + m[1:])
    return text
