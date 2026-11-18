"""Deterministic fidelity check of transcriptions against the PDF text-layer draft, attempt selection and Hangul
substitution auto-fix.

For each segment and each transcription attempt:
  missing = chars of the text layer not found in the transcription (whole output incl. figure descriptions)
  extra   = chars in the transcription outside [그림]…[/그림] blocks that are not in the text layer
for Hangul, digits (PUA equation digits mapped) and Latin letters. score = hangul_m + hangul_e + 3*(digit_m + digit_e) + latin_m + latin_e.
The best-scoring attempt becomes the final; equal-length Hangul substitutions vs the text layer are then auto-fixed
(fix_hangul.py logic). Writes rec["select"], rec["final_fixed"], rec["verify"] into OCR_DIR/<key>/<sid>.json.

Usage: python -I verify_seg.py --render /data/render2 --ocr /data/ocr2 --keys k1,k2|all [--report out.json]
"""
import argparse, difflib, json, os, re, sys
from collections import Counter

FIG_RE = re.compile(r"\[그림\].*?\[/그림\]", re.S)
TAG_RE = re.compile(r"</?u>|\[/?[A-H]\]|</보기>|\[/?표\]|<br\s*/?>|\((?:[A-E])\)")
IGNORE_DRAFT = [re.compile(r"이제 듣기 문제가 끝났습니다\..*?답을 하시기 바랍니다\.", re.S),
                re.compile(r"(?m)^\s*(물리학|화학|생명\s*과학|지구\s*과학)\s*(Ⅰ|Ⅱ|I{1,2}|1|2)\s*$"),
                re.compile(r"(?m)^\s*(과학|사회)탐구\s*영역\s*$")]
MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.S)
HDR_RE = re.compile(r"(^|,\s*)(\([가-하A-Z]\)|[A-Zㄱ-ㅎ가-힣0-9() ]{1,12})\s*:\s")


def norm_box(s):
    return re.sub(r"<\s*보\s*기\s*>", "<보기>", s or "")


def seg_text(p, kind, for_extra=False):
    if not p:
        return ""
    if kind == "passage":
        return norm_box(p.get("passage") or "")
    stem = norm_box(p.get("stem") or "")
    ch = p.get("choices_list") or []
    if for_extra:
        if all(m in stem for m in "①②③④⑤"):
            ch = []            # choices are inline in the passage; the choice list only repeats them
        else:                  # table-form choices: drop repeated column headers ("A: ", "(가): ")
            ch = [HDR_RE.sub(lambda m: m.group(1), c or "") for c in ch]
    return stem + "\n" + "\n".join(c or "" for c in ch)


def draft_text(seg):
    d = norm_box(seg["draft"]).replace("¦", "")
    for r in IGNORE_DRAFT:
        d = r.sub("", d)
    if seg["kind"] == "question":
        d = re.sub(r"^\s*\d{1,2}\s*\.\s*", "", d, count=1)
    return TAG_RE.sub("", d)


DIGMAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹０１２３４５６７８９", "0123456789" * 3)
SUBJ_HDR = {"phys1": "물리학Ⅰ", "chem1": "화학Ⅰ", "bio1": "생명과학Ⅰ", "earth1": "지구과학Ⅰ", "phys2": "물리학Ⅱ", "chem2": "화학Ⅱ",
            "bio2": "생명과학Ⅱ", "earth2": "지구과학Ⅱ", "ethics_life": "생활과윤리", "ethics_thought": "윤리와사상",
            "geo_korea": "한국지리", "geo_world": "세계지리", "east_asian_hist": "동아시아사", "world_hist": "세계사",
            "economics": "경제", "politics_law": "정치와법", "society_culture": "사회문화"}


def header_only(missing, section, kind):
    """Missing chars fully explained by the page header (subject name / '영역') that leaked into the draft."""
    name = SUBJ_HDR.get(section, "")
    pool = (name + "과학탐구사회탐구영역") if kind == "h" else ("I" * 4 + "II")
    from collections import Counter as C
    return bool(missing) and not (C(missing) - C(pool * 2))


def counts(s, kind):
    s = s.translate(DIGMAP)
    if kind == "h":
        return Counter(re.findall(r"[가-힣]", s))
    if kind == "d":
        return Counter(re.findall(r"\d", s))
    return Counter(re.findall(r"[A-Za-z]", s))


def compare(seg, parsed):
    ref = draft_text(seg)
    out = TAG_RE.sub("", seg_text(parsed, seg["kind"]))
    # LaTeX commands are not text; drop \command names before counting Latin letters
    out_l = re.sub(r"\\[A-Za-z]+", "", out)
    ref_l = ref
    nonfig = FIG_RE.sub("", TAG_RE.sub("", seg_text(parsed, seg["kind"], for_extra=True)))
    # Latin letters inside math come from equation glyphs the text layer cannot decode ('□'): not counted as extra
    nonfig_l = re.sub(r"\\[A-Za-z]+", "", MATH_RE.sub("", nonfig)) if "□" in ref else re.sub(r"\\[A-Za-z]+", "", nonfig)
    det = {}
    for k, (r, o_all, o_nf) in {"h": (ref, out, nonfig), "d": (ref, out, nonfig), "l": (ref_l, out_l, nonfig_l)}.items():
        cr, ca, cn = counts(r, k), counts(o_all, k), counts(o_nf, k)
        det[k + "_missing"] = "".join(sorted((cr - ca).elements()))
        det[k + "_extra"] = "".join(sorted((cn - cr).elements()))
    sc = (len(det["h_missing"]) + len(det["h_extra"]) + 3 * (len(det["d_missing"]) + len(det["d_extra"]))
          + len(det["l_missing"]) + len(det["l_extra"]))
    return sc, det


def structural(seg, parsed, job_area):
    """Structure checks: number, choices count, marker leftovers."""
    errs = []
    if not parsed:
        return ["no parse"]
    if seg["kind"] == "question":
        try:
            if parsed.get("number") is not None and int(re.sub(r"\D", "", parsed.get("number") or "x")) != seg["number"]:
                errs.append("number mismatch")
        except ValueError:
            errs.append("number unreadable")
        if all(m in (parsed.get("stem") or "") for m in "①②③④⑤") and not any(parsed.get("choices_list") or []):
            parsed["choices_list"] = list("①②③④⑤")     # inline-marked choices (문장 삽입·무관한 문장 등)
        ch = parsed.get("choices_list") or []
        if len([c for c in ch if c and c.strip()]) != 5:
            errs.append(f"choices {len([c for c in ch if c and c.strip()])}")
        if not (parsed.get("stem") or "").strip():
            errs.append("empty stem")
    else:
        if not (parsed.get("passage") or "").strip():
            errs.append("empty passage")
    t = seg_text(parsed, seg["kind"])
    if "¦" in t:
        errs.append("junction marker left")
    if t.count("[그림]") != t.count("[/그림]"):
        errs.append("figure block unbalanced")
    if t.count("[표]") != t.count("[/표]"):
        errs.append("table block unbalanced")
    return errs


def autofix(seg, parsed):
    """Apply equal-length Hangul substitutions (text layer is ground truth) outside figure blocks."""
    SEP = "\n⁣⁣\n"
    kind = seg["kind"]
    if kind == "passage":
        fields = [parsed.get("passage") or ""]
    else:
        fields = [parsed.get("stem") or ""] + [c or "" for c in (parsed.get("choices_list") or [])]
    s = SEP.join(fields)
    # mask figure blocks
    masks = [(m.start(), m.end()) for m in FIG_RE.finditer(s)]
    infig = lambda i: any(a <= i < b for a, b in masks)
    pos = [i for i, ch in enumerate(s) if "가" <= ch <= "힣" and not infig(i)]
    hyp = "".join(s[i] for i in pos)
    ref = re.sub(r"[^가-힣]", "", draft_text(seg))
    if sorted(hyp) == sorted(ref):
        return parsed, []
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    chars = list(s); subs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1 and i2 - i1 <= 3 and sorted(ref[i1:i2]) != sorted(hyp[j1:j2]):
            for k in range(i2 - i1):
                if ref[i1 + k] != hyp[j1 + k]:
                    chars[pos[j1 + k]] = ref[i1 + k]
            subs.append(f"{hyp[j1:j2]}→{ref[i1:i2]}")
    if not subs:
        return parsed, []
    parts = "".join(chars).split(SEP)
    new = dict(parsed)
    if kind == "passage":
        new["passage"] = parts[0]
    else:
        new["stem"] = parts[0]
        if parsed.get("choices_list"):
            new["choices_list"] = parts[1:]
    return new, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True); ap.add_argument("--ocr", required=True)
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", default="all"); ap.add_argument("--report", default=None)
    a = ap.parse_args()
    jobs = {j["key"]: j for f in a.jobs.split(",") for j in json.load(open(f))}
    keys = list(jobs) if a.keys == "all" else [k for k in jobs if any(k == x or (x.endswith("*") and k.startswith(x[:-1])) for x in a.keys.split(","))]
    rep = {}
    tot = Counter()
    for key in keys:
        sf = os.path.join(a.render, key, "segments.json")
        if not os.path.exists(sf):
            continue
        for seg in json.load(open(sf))["segments"]:
            fn = os.path.join(a.ocr, key, seg["sid"] + ".json")
            if not os.path.exists(fn):
                tot["no_ocr"] += 1; continue
            rec = json.load(open(fn))
            best = None
            scores = []
            for i, att in enumerate(rec["attempts"]):
                p = att.get("parsed")
                if att.get("finish") != "stop" or not p:
                    scores.append(None); continue
                sc, det = compare(seg, p)
                se = structural(seg, p, jobs[key]["area"])
                tot_sc = sc + 50 * len(se)
                scores.append(tot_sc)
                if best is None or tot_sc < best[0] or (tot_sc == best[0] and att.get("tag") == "a" and False):
                    best = (tot_sc, i, det, se)
            if best is None:
                rec["select"] = dict(chosen=None, scores=scores); rec["verify"] = dict(ok=False, error="no usable attempt")
                json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1); tot["unusable"] += 1
                rep[f"{key}/{seg['sid']}"] = rec["verify"]; continue
            # an accepted figure-proofread attempt ("fig", text-layer score not worse) is preferred on ties
            figs = [j for j, att in enumerate(rec["attempts"]) if att.get("tag") == "fig" and att.get("status") == "changed"
                    and scores[j] is not None and scores[j] <= best[0]]
            if figs:
                j = figs[-1]
                sc_, det_ = compare(seg, rec["attempts"][j]["parsed"])
                best = (scores[j], j, det_, structural(seg, rec["attempts"][j]["parsed"], jobs[key]["area"]))
            # an old-Hangul / symbol re-read ("h") with fewer undecoded '□' and a text-layer score not worse is preferred
            hs = [j for j, att in enumerate(rec["attempts"]) if att.get("tag") == "h" and jobs[key]["area"] != "kor"
                  and scores[j] is not None and scores[j] <= best[0]
                  and seg_text(att["parsed"], seg["kind"]).count("□") < seg_text(rec["attempts"][best[1]]["parsed"], seg["kind"]).count("□")]
            if hs:
                j = hs[-1]
                sc_, det_ = compare(seg, rec["attempts"][j]["parsed"])
                best = (scores[j], j, det_, structural(seg, rec["attempts"][j]["parsed"], jobs[key]["area"]))
            _, i, det, se = best
            # agreement between the two best usable attempts from different models (for raster text the text layer lacks)
            agree = None
            others = [j for j, att in enumerate(rec["attempts"]) if j != i and att.get("finish") == "stop" and att.get("parsed")
                      and (att.get("model") != rec["attempts"][i].get("model") or att.get("temperature") != rec["attempts"][i].get("temperature"))]
            if others:
                h = lambda p: re.sub(r"[^가-힣A-Za-z0-9]", "", FIG_RE.sub("", seg_text(p, seg["kind"])))
                agree = max(difflib.SequenceMatcher(None, h(rec["attempts"][i]["parsed"]), h(rec["attempts"][j]["parsed"]), autojunk=False).ratio()
                            for j in others)
            final, subs = autofix(seg, rec["attempts"][i]["parsed"])
            sc2, det2 = compare(seg, final)
            se2 = structural(seg, final, jobs[key]["area"])
            hm_ok = len(det2["h_missing"]) <= 1 or header_only(det2["h_missing"], seg["section"], "h")
            lm_ok = len(det2["l_missing"]) <= 1 or header_only(det2["l_missing"], seg["section"], "l")
            ok = (hm_ok and len(det2["h_extra"]) <= 1 and not det2["d_missing"] and not det2["d_extra"]
                  and lm_ok and len(det2["l_extra"]) <= 1 and not se2)
            raster = seg["gfx"]["images"] > 0
            # raster text (text inside images, not in the text layer): accept extra text only if two attempts agree;
            # nothing from the text layer may be missing
            ok_cross = (not ok and raster and agree is not None and agree >= 0.95 and hm_ok and lm_ok
                        and not det2["d_missing"] and not se2)
            rec["select"] = dict(chosen=i, tag=rec["attempts"][i].get("tag"), scores=scores, cross_model_agreement=agree)
            rec["final_fixed"] = final
            rec["verify"] = dict(ok=ok or ok_cross, ok_textlayer=ok, ok_crossmodel=ok_cross, cross_model_agreement=agree,
                                 score=sc2, hangul_autofix=subs, structure=se2, raster=raster, **det2)
            json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
            tot["ok" if ok else "flag"] += 1
            rep[f"{key}/{seg['sid']}"] = rec["verify"]
    print(dict(tot))
    if a.report:
        json.dump(rep, open(a.report, "w"), ensure_ascii=False, indent=1)
    flagged = [(k, v) for k, v in rep.items() if not v.get("ok")]
    for k, v in flagged[:int(os.environ.get("SHOW", 60))]:
        print(" ", k, {x: v.get(x) for x in ("score", "cross_model_agreement", "structure", "h_missing", "h_extra", "d_missing", "d_extra", "l_missing", "l_extra") if v.get(x)})


if __name__ == "__main__":
    main()
