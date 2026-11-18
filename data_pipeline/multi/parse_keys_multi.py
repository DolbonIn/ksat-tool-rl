"""Parse official KICE 정답표 PDFs (text layer) for 국어/영어/과탐/사탐 into JSON.

Table layout: column groups of (문항번호, 정답, 배점), read row by row (y), tokens sorted by x and chunked into triples;
the triple's position in the row = column group. 국어: groups 0-1 = 공통(1-34), group 2 = 화법과 작문(35-45),
group 3 = 언어와 매체(35-45). 영어: 4 groups (1-45). 탐구: 4 groups (1-20). 2-page keys: the 홀수형 page.
Multi-subject key PDFs (2022 수능 탐구: one PDF): the page whose header names the subject.

Keys without a text layer are emitted with errors=["no text layer"] and a rendered PNG for image reading.
Usage: python -I parse_keys_multi.py --jobs /data/work2/jobs_kice.json --out /data/work2/answer_keys_kice.json --png-dir /data/work2/keypng
"""
import argparse, json, os, re, unicodedata
import fitz

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}


def norm(s):
    s = unicodedata.normalize("NFC", s).replace(" ", "").replace("Ⅰ", "1").replace("Ⅱ", "2").replace("·", "").replace("ㆍ", "").replace("․", "").replace("‧", "")
    s = re.sub(r"II(?![A-Za-z])", "2", s)
    return re.sub(r"I(?![A-Za-z])", "1", s)


def parse_page(page):
    words = page.get_text("words")
    toks = []
    for w in words:
        t = w[4].strip()
        if re.fullmatch(r"\d{1,2}|[①-⑤]", t):
            toks.append(((w[1] + w[3]) / 2, w[0], t))
    # "정답 없음" (voided item: no correct answer) is printed as two stacked words in the answer cell
    for w in words:
        if w[4].strip() == "없음":
            up = [v for v in words if v[4].strip() == "정답" and abs(v[0] - w[0]) < 3 and 5 < w[1] - v[1] < 16]
            if up:
                toks.append(((up[0][1] + w[3]) / 2, w[0], "X"))
    toks.sort()
    merged, last = {}, None
    for yc, x, t in toks:          # cluster by y-centre (gap > 4pt starts a new row)
        if last is None or yc - last > 4:
            merged[yc] = []
            key = yc
        merged[key].append((x, t)); last = yc
    out = []
    for k in sorted(merged):
        toks = [t for _, t in sorted(merged[k])]
        xs = [x for x, _ in sorted(merged[k])]
        if len(toks) < 3 or len(toks) % 3:
            continue
        trip = []
        okrow = True
        for i in range(0, len(toks), 3):
            num, ans, pts = toks[i:i + 3]
            if not (num.isdigit() and (ans in CIRC or ans == "X") and pts.isdigit()):
                okrow = False; break
            trip.append((xs[i], int(num), CIRC.get(ans), int(pts)))
        if okrow:
            out.append(trip)
    return out


def group_x(rows):
    """Cluster triple x positions into column groups (sorted)."""
    xs = sorted(x for r in rows for x, *_ in r)
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] < 40:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [sum(g) / len(g) for g in groups]


def build(job, rows):
    cents = group_x(rows)
    gi = lambda x: min(range(len(cents)), key=lambda i: abs(cents[i] - x))
    items, errs = {}, []
    area = job["area"]
    for r in rows:
        for x, n, a, p in r:
            g = gi(x)
            if area == "kor":
                sec = "kor_common" if g <= 1 and n <= 34 else ("kor_speech_writing" if g == 2 else "kor_lang_media" if g == 3 else None)
                if sec is None or (sec != "kor_common" and not 35 <= n <= 45):
                    errs.append(f"unexpected group {g} n={n}"); continue
            else:
                sec = job["section"]
            k = f"{sec}-{n}"
            if k in items and (items[k]["answer"], items[k]["points"]) != (a, p):
                errs.append(f"dup {k}")
            items[k] = dict(section=sec, number=n, answer=a, points=p)
            if a is None:
                items[k]["note"] = "정답 없음 (official key: no correct answer; item excluded from the dataset)"
    return items, errs, len(cents)


def check(job, items):
    errs = []
    exp = job["expect"]
    for sec, (lo, hi) in exp.items():
        miss = [n for n in range(lo, hi + 1) if f"{sec}-{n}" not in items]
        if miss:
            errs.append(f"{sec} missing {miss}")
    if job["area"] == "kor":
        for el in ("kor_speech_writing", "kor_lang_media"):
            tot = sum(v["points"] for v in items.values() if v["section"] in ("kor_common", el))
            if tot != 100:
                errs.append(f"{el} points {tot}")
    elif job["area"] == "eng":
        tot = sum(v["points"] for v in items.values())
        if tot != 100:
            errs.append(f"points {tot}")
    else:
        tot = sum(v["points"] for v in items.values())
        if tot != 50:
            errs.append(f"points {tot}")
    for k, v in items.items():
        if v["points"] not in (1, 2, 3):
            errs.append(f"{k} points {v['points']}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--png-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.png_dir, exist_ok=True)
    res = {}
    for j in json.load(open(a.jobs)):
        doc = fitz.open(j["answer_key"])
        pages = [p for p in doc if "짝수" not in p.get_text()]
        if j.get("answer_subject") and len(pages) > 1:
            sub = [p for p in pages if norm(j["answer_subject"]) in norm(p.get_text())]
            pages = sub or pages
        page = pages[0] if pages else doc[0]
        if not page.get_text().strip():
            png = os.path.join(a.png_dir, j["key"] + ".png")
            page.get_pixmap(dpi=200).save(png)
            res[j["key"]] = dict(source_file=j["answer_key"], items={}, errors=["no text layer"], png=png, page=page.number + 1)
            print(j["key"], "NO TEXT LAYER ->", png)
            continue
        if j.get("answer_subject"):
            hdr = norm(page.get_text()[:200])
            if norm(j["answer_subject"]) not in hdr:
                print("WARN subject header mismatch", j["key"], hdr[:80])
        rows = parse_page(page)
        items, errs, ng = build(j, rows)
        errs += check(j, items)
        res[j["key"]] = dict(source_file=j["answer_key"], page=page.number + 1, items=items, errors=errs, groups=ng, method="text_layer")
        print(j["key"], len(items), "OK" if not errs else errs)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
