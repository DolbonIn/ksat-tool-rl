"""Deterministic text-fidelity check: compare the Hangul in the OCR transcription with the Hangul that is real
text in the official PDF's text layer inside the same crop rectangle (math is in private-use glyphs, but Korean
sentences are extractable). Missing/extra Hangul runs reveal dropped sentences, truncation or hallucinated text.

Output JSON {pid: {"hangul_sim": float, "hangul_missing": str, "hangul_extra": str}} (figure descriptions excluded).
Usage: python -I verify_text.py --jobs a.json,b.json --render /data/render --ocr /data/ocr --out /data/work/verify_text.json
"""
import argparse, difflib, json, os, re
from collections import Counter
import fitz

TAG = {"csat": "csat", "kice_06": "06", "kice_09": "09", "edu_03": "edu03", "edu_04": "edu04", "edu_07": "edu07", "edu_10": "edu10"}


PUA_DIGITS = {0xE034 + i: str((i + 1) % 10) for i in range(10)}


HEADER_RE = re.compile(r"(\(\s*(확률과\s*통계|미적분|기하)\s*\)|[홀짝]\s*수\s*형|단\s*답\s*형|5\s*지\s*선\s*다\s*형|수\s*학\s*영\s*역|제\s*2\s*교시|이 문제지에 관한 저작권은.*)")


def clean_ref(ref):
    """Text-layer text inside a crop minus page furniture (headers, page numbers, the problem label, [n점])."""
    ref = re.sub(r"\[\s*\d\s*점\s*\]", "", ref)
    ref = HEADER_RE.sub("", ref)
    ref = "\n".join(l for l in ref.split("\n") if not re.fullmatch(r"\s*\d{1,2}\s*", l)
                    and not re.fullmatch(r"\s*(영|역|수|학|영\s*역|수\s*학|수\s*학\s*영)\s*", l))
    ref = re.sub(r"(?m)^\s*\d{1,2}\s*\.\s*", "", ref)
    return ref


def hangul(s):
    return re.sub(r"[^가-힣]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True)
    ap.add_argument("--ocr", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default="", help="comma list of override JSONs (same as build_dataset) applied first")
    a = ap.parse_args()
    out = {}
    ov = {}
    for f in [x for x in a.overrides.split(",") if x]:
        for k, v in json.load(open(f)).items():
            ov.setdefault(k, {}).update(v)
    for jf in a.jobs.split(","):
        for j in json.load(open(jf)):
            idx = json.load(open(os.path.join(a.render, "crops", j["key"], "index.json")))
            pdfs = {os.path.basename(p["pdf"]): p["pdf"] for p in j.get("parts", [])} or {os.path.basename(j["pdf"]): j["pdf"]}
            docs = {}
            for c in idx["crops"]:
                pdfp = pdfs.get(c.get("pdf"), j["pdf"])
                doc = docs.setdefault(pdfp, fitz.open(pdfp))
                ref = doc[c["pdf_page"] - 1].get_text("text", clip=fitz.Rect(*c["rect"]))
                ref = clean_ref(ref)
                fn = os.path.join(a.ocr, j["key"], c["name"] + ".json")
                if not os.path.exists(fn):
                    continue
                f = dict(json.load(open(fn)).get("final") or {})
                pid0 = f"{j['school_year']}-{TAG[j['exam']]}-{c['section']}-{c['number']}"
                if pid0 in ov:
                    f.update(ov[pid0])
                    if "choices_list" in ov[pid0] and "choices" not in ov[pid0]:
                        f["choices"] = "\n".join(x or "" for x in ov[pid0]["choices_list"])
                ch = f.get("choices") or ""
                ocr = (f.get("stem") or "") + "\n" + ("" if ch.strip() == "없음" else ch)
                ocr = re.sub(r"\[\s*\d\s*점\s*\]", "", ocr)
                ocr = HEADER_RE.sub("", ocr)      # build_dataset strips page headers from stems too
                h1, h2 = hangul(ref), hangul(ocr)
                c1, c2 = Counter(h1), Counter(h2)
                miss = "".join((c1 - c2).elements()); extra = "".join((c2 - c1).elements())
                sm = difflib.SequenceMatcher(None, h1, h2, autojunk=False)
                # digits: HWP equation font puts digits in private-use glyphs U+E034..U+E03D (= 1..9, 0)
                refd = ref.translate(PUA_DIGITS)
                d1 = Counter(re.findall(r"\d", refd))
                d2 = Counter(re.findall(r"\d", ocr))
                fig = f.get("figure") or ""
                d3 = Counter(re.findall(r"\d", fig))
                dmiss = d1 - d2; dextra = d2 - d1
                pid = f"{j['school_year']}-{TAG[j['exam']]}-{c['section']}-{c['number']}"
                out[pid] = dict(hangul_sim=round(sm.ratio(), 4), hangul_ref_len=len(h1), hangul_missing=miss[:200], hangul_extra=extra[:200],
                                hangul_diff=len(miss) + len(extra),
                                digits_ref=sum(d1.values()), digits_missing="".join(sorted(dmiss.elements())),
                                digits_extra="".join(sorted(dextra.elements())),
                                digits_missing_unexplained="".join(sorted((dmiss - d3).elements())))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    bad = sorted(((-v["hangul_diff"], k, v["hangul_missing"][:60], v["hangul_extra"][:60]) for k, v in out.items()))
    print(f"{len(out)} items; hangul multiset diff 0: {sum(b[0] == 0 for b in bad)}; >=3 chars: {sum(b[0] <= -3 for b in bad)}")
    for b in bad[:int(os.environ.get('SHOW', 40))]:
        if b[0] == 0: break
        print(f"  {b[1]:26s} diff={-b[0]} missing={b[2]!r} extra={b[3]!r}")
    dd = sorted(((len(v["digits_extra"]) + len(v["digits_missing_unexplained"]), k) for k, v in out.items()), reverse=True)
    print(f"digits: exact multiset match {sum(1 for v in out.values() if not v['digits_extra'] and not v['digits_missing'])}/{len(out)}; "
          f"with extra or unexplained-missing digits: {sum(1 for n, _ in dd if n)}")
    for n, k in dd[:int(os.environ.get('SHOW', 40))]:
        if not n: break
        v = out[k]
        print(f"  {k:26s} extra={v['digits_extra']!r} missing_unexpl={v['digits_missing_unexplained']!r} (ref {v['digits_ref']})")


if __name__ == "__main__":
    main()
