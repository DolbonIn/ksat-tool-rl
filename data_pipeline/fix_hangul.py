"""Repair Korean-character substitutions in OCR output using the official PDF text layer as ground truth.

Only equal-length Hangul *substitutions* are applied (e.g. OCR '골' vs PDF '귤', particle '와' vs '과'), mapped back
to their positions in the transcription; insertions/deletions are never auto-applied (they are reported for review).
Math is untouched (it is not Hangul). Writes an overrides JSON {pid: {"stem": ..., "choices_list": [...]}} and a
log of applied substitutions {pid: {"hangul_autofix": ["골→귤", ...]}} usable as a build_dataset --verify-extra file.

Usage: python -I fix_hangul.py --jobs a.json,b.json --ocr /data/ocr --out-overrides ov.json --out-log log.json
"""
import argparse, difflib, json, os, re
import fitz

TAG = {"csat": "csat", "kice_06": "06", "kice_09": "09", "edu_03": "edu03", "edu_04": "edu04", "edu_07": "edu07", "edu_10": "edu10"}
SEP = "\n⁣⁣\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", default="/data/render"); ap.add_argument("--ocr", required=True)
    ap.add_argument("--out-overrides", required=True); ap.add_argument("--out-log", required=True)
    a = ap.parse_args()
    ov, log = {}, {}
    for jf in a.jobs.split(","):
        for j in json.load(open(jf)):
            idx = json.load(open(os.path.join(a.render, "crops", j["key"], "index.json")))
            pdfs = {os.path.basename(p["pdf"]): p["pdf"] for p in j.get("parts", [])} or {os.path.basename(j["pdf"]): j["pdf"]}
            docs = {}
            for c in idx["crops"]:
                fn = os.path.join(a.ocr, j["key"], c["name"] + ".json")
                if not os.path.exists(fn):
                    continue
                f = json.load(open(fn)).get("final") or {}
                pdfp = pdfs.get(c.get("pdf"), j["pdf"])
                doc = docs.setdefault(pdfp, fitz.open(pdfp))
                ref = re.sub(r"[^가-힣]", "", doc[c["pdf_page"] - 1].get_text("text", clip=fitz.Rect(*c["rect"])))
                stem = f.get("stem") or ""
                chs = f.get("choices_list")
                s = stem + (SEP + SEP.join(x or "" for x in chs) if chs else "")
                pos = [i for i, ch in enumerate(s) if "가" <= ch <= "힣"]
                hyp = "".join(s[i] for i in pos)
                if sorted(hyp) == sorted(ref):
                    continue
                sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
                chars = list(s); subs = []
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op == "replace" and i2 - i1 == j2 - j1 and sorted(ref[i1:i2]) != sorted(hyp[j1:j2]):
                        for k in range(i2 - i1):
                            if ref[i1 + k] != hyp[j1 + k]:
                                chars[pos[j1 + k]] = ref[i1 + k]
                        subs.append(f"{hyp[j1:j2]}→{ref[i1:i2]}")
                pid = f"{j['school_year']}-{TAG[j['exam']]}-{c['section']}-{c['number']}"
                if subs:
                    parts = "".join(chars).split(SEP)
                    o = {"stem": parts[0]}
                    if chs:
                        o["choices_list"] = parts[1:]
                        o["choices"] = "\n".join(f"{'①②③④⑤'[i]} {x}" for i, x in enumerate(parts[1:]))
                    ov[pid] = o
                    log[pid] = {"hangul_autofix": subs}
    json.dump(ov, open(a.out_overrides, "w"), ensure_ascii=False, indent=1)
    json.dump(log, open(a.out_log, "w"), ensure_ascii=False, indent=1)
    print(f"auto-fixed {len(ov)} items")
    for k, v in log.items():
        print(" ", k, v["hangul_autofix"])


if __name__ == "__main__":
    main()
