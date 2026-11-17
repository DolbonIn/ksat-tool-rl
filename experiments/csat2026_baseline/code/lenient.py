"""Lenient answer extraction: \\boxed{} first, else explicit '정답' line, else the last circled/standalone choice at the end."""
import re
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[j+1:k]
    return None
def parse(content):
    b = last_boxed(content or "")
    if b is None: return None
    for k, v in {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}.items(): b = b.replace(k, v)
    m = re.search(r"[1-5]", b)
    return int(m.group()) if m else None
CIRC = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}
def lenient(content):
    p = parse(content)
    if p is not None: return p, "boxed"
    c = (content or "").strip()
    m = re.findall(r"(?:정답|최종\s*답|Final Answer|답)\s*[:：]?\s*\**\s*([①-⑤]|[1-5])(?![0-9])", c)
    if m: return CIRC.get(m[-1], None) or int(m[-1]), "정답표기"
    tail = c[-40:]
    m = re.findall(r"[①-⑤]|(?<![0-9.])[1-5](?![0-9.])", tail)
    if m: return CIRC.get(m[-1], None) or int(m[-1]), "끝부분"
    return None, "none"
