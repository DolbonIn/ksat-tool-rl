"""Select / repair OCR transcriptions using the official PDF text layer as an oracle for Korean text and digits.

For every crop we know (from the PDF text layer, inside the crop rectangle) the multiset of Hangul characters and
of digits (HWP equation font digits are private-use glyphs U+E034..E03D = 1..9,0). A transcription's score is
  hangul_missing + hangul_extra + 3 * (digits_extra + digits_missing_not_in_figure_description)
1. Selection: between pass A and the proofread pass B, keep the lower-score one (tie -> B).
2. Repair: items still above threshold get K extra pass-A samples (temperature 0.4) -> best by score -> pass B ->
   selection again. Everything is recorded in rec["select"].

Usage: python -I repair_ocr.py --jobs a.json,b.json --render /data/render --ocr /data/ocr [--repair --k 6]
"""
import argparse, asyncio, json, os, re, sys
from collections import Counter
import fitz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_vlm  # noqa: E402
from verify_text import clean_ref, PUA_DIGITS, HEADER_RE  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402


def score(f, ref):
    if not f or not f.get("stem"):
        return 10 ** 6, {}
    ch = f.get("choices") or ""
    ocr = (f.get("stem") or "") + "\n" + ("" if ch.strip() == "없음" else ch)
    ocr = re.sub(r"\[\s*\d\s*점\s*\]", "", ocr)
    ocr = HEADER_RE.sub("", ocr)          # page headers are stripped from stems at build time anyway
    h1, h2 = Counter(re.sub(r"[^가-힣]", "", ref)), Counter(re.sub(r"[^가-힣]", "", ocr))
    d1, d2 = Counter(re.findall(r"\d", ref.translate(PUA_DIGITS))), Counter(re.findall(r"\d", ocr))
    d3 = Counter(re.findall(r"\d", f.get("figure") or ""))
    hm, he = sum((h1 - h2).values()), sum((h2 - h1).values())
    de, dm = sum((d2 - d1).values()), sum(((d1 - d2) - d3).values())
    det = dict(hangul_missing=hm, hangul_extra=he, digits_extra=de, digits_missing=dm)
    return hm + he + 3 * (de + dm), det


def bad(det, s):
    return det.get("hangul_missing", 99) >= 3 or det.get("digits_extra", 99) >= 1 or det.get("digits_missing", 99) >= 2


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--ocr", required=True)
    ap.add_argument("--repair", action="store_true"); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--base-url", default="http://127.0.0.1:31801/v1")
    a = ap.parse_args()
    todo = []
    nsel = Counter()
    for jf in a.jobs.split(","):
        for j in json.load(open(jf)):
            idx = json.load(open(os.path.join(a.render, "crops", j["key"], "index.json")))
            pdfs = {os.path.basename(p["pdf"]): p["pdf"] for p in j.get("parts", [])} or {os.path.basename(j["pdf"]): j["pdf"]}
            docs = {}
            for c in idx["crops"]:
                fn = os.path.join(a.ocr, j["key"], c["name"] + ".json")
                rec = json.load(open(fn))
                pdfp = pdfs.get(c.get("pdf"), j["pdf"])
                doc = docs.setdefault(pdfp, fitz.open(pdfp))
                ref = clean_ref(doc[c["pdf_page"] - 1].get_text("text", clip=fitz.Rect(*c["rect"])))
                cands = {"a": rec.get("a_parsed")}
                if rec.get("b"):
                    tmp = ocr_vlm.finalize({"a_parsed": rec["a_parsed"], "b": rec["b"]})
                    if tmp.get("b_changed"):
                        cands["b"] = tmp["final"]
                for i, alt in enumerate(rec.get("alts", [])):
                    cands[f"alt{i}"] = alt.get("final") or alt.get("parsed")
                scored = {k: score(v, ref) for k, v in cands.items() if v}
                # tie-break: prefer proofread b, then a, then alts
                order = {"b": 0, "a": 1}
                best = min(scored, key=lambda k: (scored[k][0], order.get(k, 2)))
                rec["final"] = cands[best]
                rec["select"] = dict(chosen=best, scores={k: v[0] for k, v in scored.items()}, detail=scored[best][1])
                nsel[best if best in ("a", "b") else "alt"] += 1
                json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
                if bad(scored[best][1], scored[best][0]):
                    todo.append((j["key"], c, fn, ref))
    print("selection:", dict(nsel), "| still flagged:", len(todo))
    for key, c, fn, ref in todo:
        print("  ", key, c["name"], json.load(open(fn))["select"])
    if not a.repair or not todo:
        return
    client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=3 * 3600)
    sem = asyncio.Semaphore(64)

    async def fix(key, c, fn, ref):
        rec = json.load(open(fn))
        url = ocr_vlm.img_url(os.path.join(a.render, c["png"]))
        outs = await asyncio.gather(*[ocr_vlm.call(client, sem, [{"type": "image_url", "image_url": {"url": url}},
                                                                {"type": "text", "text": ocr_vlm.PROMPT_A}], False, 8192, temperature=0.4)
                                      for _ in range(a.k)])
        alts = [dict(raw=o, parsed=ocr_vlm.parse(o["content"])) for o in outs if o["finish"] == "stop"]
        alts.sort(key=lambda x: score(x["parsed"], ref)[0])
        if alts:
            best = alts[0]
            content = [{"type": "image_url", "image_url": {"url": url}},
                       {"type": "text", "text": ocr_vlm.PROMPT_B + ocr_vlm.fmt(best["parsed"])}]
            b = await ocr_vlm.call(client, sem, content, True, 32768, temperature=0.6)
            tmp = {"a_parsed": best["parsed"], "b": b}
            ocr_vlm.finalize(tmp)
            best["final"] = tmp["final"] if tmp.get("b_changed") and score(tmp["final"], ref)[0] <= score(best["parsed"], ref)[0] else best["parsed"]
            best["b_errors"] = tmp.get("b_errors")
            rec.setdefault("alts", []).append(best)
            s_new, d_new = score(best["final"], ref)
            if s_new < rec["select"]["scores"][rec["select"]["chosen"]]:
                rec["final"] = best["final"]
                rec["select"].update(chosen=f"alt{len(rec['alts']) - 1}", detail=d_new, repaired=True)
                rec["select"]["scores"][f"alt{len(rec['alts']) - 1}"] = s_new
            json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
        print("repaired" if rec["select"].get("repaired") else "unchanged", key, c["name"], rec["select"], flush=True)

    await asyncio.gather(*[fix(*t) for t in todo])


if __name__ == "__main__":
    asyncio.run(main())
