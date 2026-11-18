"""Parse official KICE 정답표 PDFs (text layer) for 2017-2021 가형/나형 papers.

Table: 4 column groups of (문항 번호, 정답, 배점); rows read by y, tokens sorted by x, chunked into triples. The page is
chosen by its title '(가형)'/'(나형)' and must not be the 짝수형 page (수능 keys hold 가홀/가짝/나홀/나짝 pages).
Checks: items 1-30 present, mc for 1-21 / sa for 22-30 (circled vs number), points sum 100, and points equal to the
'[n점]' markers in the paper's text layer (render index points_text).
Usage: python -I parse_keys_kice_old.py --jobs /data/work3/jobs.json --render /data/render3 --out /data/work3/keys_kice.json
"""
import argparse, json, os, re
import fitz

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}


def parse_page(page):
    rows = {}
    for w in page.get_text("words"):
        t = w[4].strip()
        if re.fullmatch(r"\d{1,3}|[①-⑤]", t):
            rows.setdefault(round((w[1] + w[3]) / 2 / 6), []).append((w[0], t))
    out = {}
    for k in sorted(rows):
        toks = [t for _, t in sorted(rows[k])]
        if len(toks) % 3 or len(toks) < 3:
            continue
        for num, ans, pts in (toks[i:i + 3] for i in range(0, len(toks), 3)):
            if not num.isdigit() or not pts.isdigit() or not 1 <= int(num) <= 30 or int(pts) not in (2, 3, 4):
                break
            a = CIRC.get(ans, None)
            out[int(num)] = dict(number=int(num), answer=a if a else int(ans), kind="mc" if a else "sa", points=int(pts), raw=ans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for j in json.load(open(a.jobs)):
        if not j["exam"].startswith(("csat", "kice")):
            continue
        doc = fitz.open(j["answer_key"])
        ko = j["track_ko"]
        pages = [(i, p) for i, p in enumerate(doc) if f"({ko})" in re.sub(r"\s", "", p.get_text()) and "짝수" not in re.sub(r"\s", "", p.get_text())]
        errs = []
        if len(pages) != 1:
            errs.append(f"page select {[i for i, _ in pages]}")
        items = parse_page(pages[0][1]) if pages else {}
        title = re.sub(r"\s+", " ", pages[0][1].get_text())[:80] if pages else ""
        if sorted(items) != list(range(1, 31)):
            errs.append(f"items {sorted(items)}")
        for n, v in items.items():
            if (n <= 21) != (v["kind"] == "mc"):
                errs.append(f"{n} kind {v['kind']}")
            if v["kind"] == "sa" and not 0 <= v["answer"] <= 999:
                errs.append(f"{n} sa range")
        tot = sum(v["points"] for v in items.values())
        if tot != 100:
            errs.append(f"points sum {tot}")
        idx = os.path.join(a.render, "crops", j["key"], "index.json")
        if os.path.exists(idx):
            pt = {c["number"]: c["points_text"] for c in json.load(open(idx))["crops"]}
            mism = [n for n, v in items.items() if pt.get(n) != [v["points"]]]
            if mism:
                errs.append(f"points vs paper text layer mismatch {[(n, items[n]['points'], pt.get(n)) for n in mism]}")
        for v in items.values():
            v["source"] = j["answer_key"].replace("/data/", "") + f"#p{pages[0][0] + 1}" if pages else None
        res[j["key"]] = dict(source_file=j["answer_key"], page=pages[0][0] + 1 if pages else None, title=title,
                             items={str(k): v for k, v in sorted(items.items())}, errors=errs)
        print(j["key"], len(items), "OK" if not errs else errs, "|", title[:60])
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
