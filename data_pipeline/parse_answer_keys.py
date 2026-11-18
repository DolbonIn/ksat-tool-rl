"""Parse official KICE/교육청 math 정답표 PDFs (text layer) into JSON.

Table layout (all 2022+ papers): 5 column groups of (번호, 정답, 배점):
  공통 1-11 | 공통 12-22 | 확률과 통계 23-30 | 미적분 23-30 | 기하 23-30
Rows are read by y; within a row tokens are sorted by x and chunked into triples (group order = x order).
For 2-page keys (홀수형/짝수형) only the 홀수형 page is used.

Usage: python -I parse_answer_keys.py --jobs jobs.json --out answer_keys.json
Keys without a text layer are reported as missing (handled separately via page image reading).
"""
import argparse, json, re
import fitz

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
GROUPS = ["common", "common", "prob_stat", "calculus", "geometry"]


def parse_page(page):
    words = page.get_text("words")
    rows = {}
    for w in words:
        t = w[4].strip()
        if re.fullmatch(r"\d{1,3}|[①-⑤]", t):
            rows.setdefault(round((w[1] + w[3]) / 2 / 6), []).append((w[0], t))
    out = {}
    for k in sorted(rows):
        toks = [t for _, t in sorted(rows[k])]
        if len(toks) % 3 or len(toks) < 6:
            continue
        triples = [toks[i:i + 3] for i in range(0, len(toks), 3)]
        for gi, (num, ans, pts) in enumerate(triples):
            if not num.isdigit() or not pts.isdigit():
                break
            sec = GROUPS[gi]
            n = int(num)
            kind = "mc" if ans in CIRC else "sa"
            a = CIRC.get(ans, None)
            if a is None:
                a = int(ans)
            out[f"{sec}-{n}"] = dict(section=sec, number=n, answer=a, kind=kind, points=int(pts), raw=ans)
    return out


def check(key):
    errs = []
    exp = [f"common-{i}" for i in range(1, 23)] + [f"{s}-{i}" for s in ("prob_stat", "calculus", "geometry") for i in range(23, 31)]
    miss = [e for e in exp if e not in key]
    if miss:
        errs.append(f"missing {miss}")
    for s in ("prob_stat", "calculus", "geometry"):
        tot = sum(key[e]["points"] for e in exp if e in key and (e.startswith("common") or e.startswith(s)))
        if tot != 100:
            errs.append(f"{s} points sum {tot}")
    for e, v in key.items():
        if v["kind"] == "mc" and not 1 <= v["answer"] <= 5:
            errs.append(f"{e} mc answer {v['answer']}")
        if v["kind"] == "sa" and not 0 <= v["answer"] <= 999:
            errs.append(f"{e} sa answer {v['answer']}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for j in json.load(open(a.jobs)):
        doc = fitz.open(j["answer_key"])
        pages = [p for p in doc if "짝수" not in p.get_text()] or list(doc)
        key = parse_page(pages[0]) if pages[0].get_text().strip() else {}
        errs = check(key) if key else ["no text layer"]
        res[j["key"]] = dict(source_file=j["answer_key"], items=key, errors=errs)
        print(j["key"], len(key), "OK" if not errs else errs)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
