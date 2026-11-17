"""Build per-question prompts for 2026 CSAT 국어/영어 (홀수형) from PDF-extracted text."""
import re, json
D = "data/"
C = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}

def answer_key(fn, korean):
    t = open(D + fn).read().split('=== PAGE 2')[0]
    toks = [x.strip() for x in t.split('\n') if x.strip()]
    common, e1, e2 = {}, {}, {}
    for i in range(len(toks) - 2):
        a, b, c = toks[i:i+3]
        if a.isdigit() and b in C and c in ('1', '2', '3'):
            n = int(a)
            if not korean or n < 35: common[n] = (C[b], int(c))
            elif n not in e1: e1[n] = (C[b], int(c))
            else: e2[n] = (C[b], int(c))
    return common, e1, e2

def pages(fn, lo, hi):
    out, keep = [], False
    for line in open(D + fn).read().split('\n'):
        m = re.match(r'=== PAGE (\d+) ===', line)
        if m: keep = lo <= int(m.group(1)) <= hi; continue
        if keep: out.append(line)
    return out

DROP = re.compile(r'^(홀수형|2026학년도 대학수학능력시험 문제지|제\s?\d 교시|\((화법과 작문|언어와 매체)\)|이 문제지에 관한 저작권.*|- \d+ -|국어 영역|영어 영역)$')
def clean(lines):
    lines = [l.rstrip() for l in lines]
    out = []
    for i, l in enumerate(lines):
        s = l.strip()
        if not s or DROP.match(s): continue
        if re.fullmatch(r'\d{1,2}', s):  # page number / "k of N" footer next to headers
            nb = [lines[j].strip() for j in (i-1, i+1) if 0 <= j < len(lines)]
            if any(DROP.match(x) or re.fullmatch(r'(20|8)', x) for x in nb) or s in ('20', '8'): continue
        out.append(l)
    return out

def segment(lines, nums):
    """Return {n: (set_header_text_or_None, question_text)} for question numbers in nums."""
    starts, i = {}, 0
    for n in nums:
        while i < len(lines) and not re.match(rf'^{n}\.(\s|$)', lines[i].strip()): i += 1
        assert i < len(lines), f"question {n} not found"
        starts[n] = i; i += 1
    headers = [(k, int(m.group(1)), int(m.group(2))) for k, l in enumerate(lines)
               if (m := re.match(r'^\[(\d+)\s*～\s*(\d+)\]', l.strip()))]
    res = {}
    for idx, n in enumerate(nums):
        s = starts[n]
        nxt = [starts[nums[idx+1]]] if idx + 1 < len(nums) else [len(lines)]
        nxt += [k for k, a, b in headers if k > s]
        q = "\n".join(lines[s:min(nxt)]).strip()
        hdr = next(((k, a, b) for k, a, b in headers if a <= n <= b and k < s), None)
        passage = "\n".join(lines[hdr[0]:starts[hdr[1]]]).strip() if hdr else None
        res[n] = (passage, q)
    return res

units = []
def add(exam, section, n, passage, q, key, extra=None):
    a, p = key[n]
    parts = [x for x in (extra, passage, q) if x]
    units.append(dict(id=f"{exam}-{section}-{n}", exam=exam, section=section, no=n, points=p, answer=a,
                      text="\n\n".join(parts)))

# ---------------- 국어 (pages 1-20 = 홀수형) ----------------
kc, k_hw, k_em = answer_key("국어영역_정답표.txt", True)
L = clean(pages("국어영역_문제지.txt", 1, 20))
cut1 = next(i for i, l in enumerate(L) if '선택과목(화법과 작문)' in l)
cut2 = next(i for i, l in enumerate(L) if '선택과목(언어와 매체)' in l)
cut2b = max(i for i, l in enumerate(L) if '선택과목(언어와 매체)' in l)
for sec, lines, nums, key in [("common", L[:cut1], range(1, 35), kc),
                              ("hwajak", L[cut1+2:cut2], range(35, 46), k_hw),
                              ("eonmae", L[cut2b+2:], range(35, 46), k_em)]:
    for n, (ps, q) in segment(lines, list(nums)).items():
        add("korean", sec, n, ps, q, key)

# ---------------- 영어 (pages 1-8 = 홀수형) ----------------
ec, _, _ = answer_key("영어영역_정답표.txt", False)
E = clean(pages("영어영역_문제지.txt", 1, 8))
seg = segment(E, list(range(1, 46)))
S = clean(pages("영어영역듣기평가대본.txt", 1, 99))
S = [l for l in S if not re.match(r'^(2026학년도 대학수학능력시험|영어 영역 듣기평가 대본)$', l.strip())]
sstarts = {}
for n in range(1, 18):
    sstarts[n] = next(i for i, l in enumerate(S) if re.match(rf'^(\[16～17\] )?{n}\.(\s|$)', l.strip()) or (n == 16 and l.strip().startswith('[16～17]')))
script = {}
for n in range(1, 16):
    script[n] = "\n".join(S[sstarts[n]:sstarts[n+1]]).strip()
script[16] = script[17] = "\n".join(S[sstarts[16]:]).strip()
PICTURE_Q4 = None  # filled from figure description file if present
try: PICTURE_Q4 = open(D + "english_q4_figure.txt").read().strip()
except FileNotFoundError: pass
for n in range(1, 46):
    ps, q = seg[n]
    if n <= 17:
        extra = "[듣기 대본 — 실제 시험에서는 음성으로 들려주는 내용]\n" + script[n]
        if n == 4 and PICTURE_Q4: q = q + "\n\n[그림 설명]\n" + PICTURE_Q4
        add("english", "listening", n, ps, q, ec, extra)
    else:
        add("english", "reading", n, ps, q, ec)

from supplements_kor_eng import RULES
for prefix, pat, repl in RULES:
    hit = 0
    for u in units:
        if u["id"].startswith(prefix):
            f = repl if "\\1" in repl else (lambda m, r=repl: r)
            new, k = re.subn(pat, f, u["text"], flags=re.DOTALL)
            u["text"] = new; hit += k
    assert hit, f"rule did not match: {prefix} {pat[:40]!r}"
json.dump(units, open("units_kor_eng.json", "w"), ensure_ascii=False, indent=1)
from collections import Counter
print(len(units), Counter((u["exam"], u["section"]) for u in units))
for sec in ["common", "hwajak", "eonmae", "listening", "reading"]:
    print(sec, sum(u["points"] for u in units if u["section"] == sec))
