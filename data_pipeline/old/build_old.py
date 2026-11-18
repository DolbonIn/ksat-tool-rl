"""Assemble the 2017-2021 가형/나형 JSONL from DeepSeek OCR outputs + official answer keys (same schema as
data/ksat_math/train.jsonl, plus `track`; section = ga|na; id = <school_year>-<exam>-<ga|na>-<n>).

Answers always come from the official keys (keys_kice.json: 정답표 text layer; keys_edu.json: EBSi hsj text layer
cross-checked with both DeepSeek reads of the EBSi answer image). Points: official key (KICE) / '[n점]' markers in the
paper's text layer (교육청), must sum to 100 per paper. OCR gives only text, choices and the figure flag.
Text-layer checks (Hangul multiset + digits vs the PDF text layer inside the crop rect) and the equal-length Hangul
substitution autofix are those of data_pipeline/verify_text.py and fix_hangul.py. Solve-check results merge from
--solve (per-sample JSONL of solve_old.py) and manual/repair notes from --verify-extra JSON files.
Runs on the host (JSON only). Usage:
  python3 build_old.py --jobs work3/jobs.json --keys work3/keys_kice.json,work3/keys_edu.json --render render3 --ocr ocr3 \
     --out ksat_math_old/train_2017_2021_hard.jsonl --points 4 [--solve work3/solve_hard.jsonl] [--overrides ov.json]
"""
import argparse, collections, difflib, json, os, re, sys
sys.path.insert(0, "/home/b300-07/ksat-tool-rl/data_pipeline")
from build_dataset import latex_balanced, clean_stem, figure_text  # noqa: E402

D = "/raid/cache/b300-07/ksat-rl/data"
CIRC = "①②③④⑤"
PUA_DIGITS = {0xE034 + i: str((i + 1) % 10) for i in range(10)}
HEADER_RE = re.compile(r"(\(\s*(확률과\s*통계|미적분|기하)\s*\)|[홀짝]\s*수\s*형|단\s*답\s*형|5\s*지\s*선\s*다\s*형|수\s*학\s*영\s*역|제\s*2\s*교시|이 문제지에 관한 저작권은.*)")
SEP = "\n⁣⁣\n"
TRACK_KO = {"ga": "가형", "na": "나형"}


def norm_delims(s):
    """Match the existing dataset's math delimiters: \\( \\) -> $ $, \\[ \\] -> $$ $$; drop row-spacing args like \\\\[8pt]."""
    if not s:
        return s
    s = re.sub(r"\\\\\[\s*-?\d+(?:\.\d+)?\s*(?:pt|em|ex|mm|cm)\s*\]", r"\\\\", s)
    s = re.sub(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", lambda m: "$$" + m.group(1) + "$$", s, flags=re.S)
    s = re.sub(r"(?<!\\)\\\((.+?)(?<!\\)\\\)", lambda m: "$" + m.group(1).strip() + "$", s, flags=re.S)
    return s


def fix_boxed(s):
    """\\boxed{(가)} / \\boxed{\\text{(가)}} -> \\fbox{(가)} before build_dataset.clean_stem, whose regex eats one extra '}' when the
    blank is followed by '}' (e.g. \\dfrac{\\boxed{(나)}}{...})."""
    s = re.sub(r"\\boxed\{\s*\\text\{\s*(\(?[가-하]\)?)\s*\}\s*\}", r"\\fbox{\1}", s)
    return re.sub(r"\\boxed\{\s*(\(?[가-하]\)?)\s*\}", r"\\fbox{\1}", s)


def norm_fields(f):
    for k in ("stem", "figure", "choices"):
        if f.get(k):
            f[k] = fix_boxed(norm_delims(f[k]))
    if f.get("choices_list"):
        f["choices_list"] = [norm_delims(x) if x else x for x in f["choices_list"]]
    return f


def text_layer_check(f, ref):
    ch = f.get("choices") or ""
    ocr = (f.get("stem") or "") + "\n" + ("" if ch.strip() == "없음" else ch)
    ocr = HEADER_RE.sub("", re.sub(r"\[\s*\d\s*점\s*\]", "", ocr))
    h1, h2 = re.sub(r"[^가-힣]", "", ref), re.sub(r"[^가-힣]", "", ocr)
    c1, c2 = collections.Counter(h1), collections.Counter(h2)
    miss = "".join((c1 - c2).elements()); extra = "".join((c2 - c1).elements())
    d1 = collections.Counter(re.findall(r"\d", ref.translate(PUA_DIGITS))); d2 = collections.Counter(re.findall(r"\d", ocr))
    d3 = collections.Counter(re.findall(r"\d", f.get("figure") or ""))
    dm, de = d1 - d2, d2 - d1
    return dict(hangul_diff=len(miss) + len(extra), hangul_missing=miss[:200], hangul_extra=extra[:200], digits_ref=sum(d1.values()),
                digits_extra="".join(sorted(de.elements())), digits_missing_unexplained="".join(sorted((dm - d3).elements())))


def hangul_autofix(f, ref):
    """fix_hangul.py logic: apply only equal-length Hangul substitutions vs the text layer."""
    stem = f.get("stem") or ""; chs = f.get("choices_list")
    s = stem + (SEP + SEP.join(x or "" for x in chs) if chs else "")
    pos = [i for i, ch in enumerate(s) if "가" <= ch <= "힣"]
    hyp = "".join(s[i] for i in pos); refh = re.sub(r"[^가-힣]", "", ref)
    if sorted(hyp) == sorted(refh):
        return None, []
    sm = difflib.SequenceMatcher(None, refh, hyp, autojunk=False)
    chars = list(s); subs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1 and sorted(refh[i1:i2]) != sorted(hyp[j1:j2]):
            for k in range(i2 - i1):
                if refh[i1 + k] != hyp[j1 + k]:
                    chars[pos[j1 + k]] = refh[i1 + k]
            subs.append(f"{hyp[j1:j2]}→{refh[i1:i2]}")
    if not subs:
        return None, []
    parts = "".join(chars).split(SEP)
    o = {"stem": parts[0]}
    if chs:
        o["choices_list"] = parts[1:]
        o["choices"] = "\n".join(f"{CIRC[i]} {x}" for i, x in enumerate(parts[1:]))
    return o, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", required=True); ap.add_argument("--render", required=True)
    ap.add_argument("--ocr", required=True); ap.add_argument("--out", required=True); ap.add_argument("--points", default=None)
    ap.add_argument("--solve", default=""); ap.add_argument("--verify-extra", default=""); ap.add_argument("--overrides", default="")
    ap.add_argument("--exclude", default="", help="JSON {id: reason} of items to leave out (e.g. 전원 정답 처리)")
    ap.add_argument("--preambles", default="", help="preambles_final.json: shared stems prepended to the items of their group")
    a = ap.parse_args()
    keys = {}
    for f in a.keys.split(","):
        keys.update(json.load(open(f)))
    pts_filter = {int(x) for x in a.points.split(",")} if a.points else None
    extra, overrides = {}, {}
    for f in [x for x in a.verify_extra.split(",") if x]:
        for k, v in json.load(open(f)).items():
            extra.setdefault(k, {}).update(v)
    for f in [x for x in a.overrides.split(",") if x]:
        for k, v in json.load(open(f)).items():
            overrides.setdefault(k, {}).update(v)
    excl = json.load(open(a.exclude)) if a.exclude else {}
    pre = json.load(open(a.preambles)) if a.preambles else {}
    solve = collections.defaultdict(list)
    for f in [x for x in a.solve.split(",") if x]:
        for l in open(f):
            r = json.loads(l)
            if r.get("finish") != "error":
                solve[r["id"]].append(r)
    rows, report = [], []
    for j in json.load(open(a.jobs)):
        key = j["key"]; k = keys.get(key)
        idxf = os.path.join(a.render, "crops", key, "index.json")
        if not k or not os.path.exists(idxf):
            report.append(f"{key}: no key or no crops"); continue
        crops = {c["number"]: c for c in json.load(open(idxf))["crops"]}
        items = k["items"]
        tot = sum((items[str(n)]["points"] or 0) for n in range(1, 31))
        if tot != 100:
            report.append(f"{key}: points sum {tot} -> skipped"); continue
        for n in range(1, 31):
            it = items[str(n)]; c = crops.get(n)
            pid = f"{j['school_year']}-{j['exam']}-{j['track']}-{n}"
            if pts_filter and it["points"] not in pts_filter:
                continue
            if pid in excl:
                report.append(f"{pid}: excluded ({excl[pid]})"); continue
            if it.get("answer") is None or (it.get("status") not in (None, "ok") and not str(it.get("status")).startswith("resolved")):
                report.append(f"{pid}: key status {it.get('status')} -> skipped"); continue
            fn = os.path.join(a.ocr, key, f"{j['track']}-{n}.json")
            if not c or not os.path.exists(fn):
                report.append(f"{pid}: no OCR"); continue
            o = json.load(open(fn))
            f = dict(o.get("final") or {})
            v = {"answer_source": "official_key"}
            if j["exam"].startswith(("csat", "kice")):
                v["answer_key_file"] = it["source"]
                v["points_source"] = "official_key (= paper '[n점]' text layer)"
            else:
                v["answer_key_file"] = it["source"].replace("hsj:", "edu/" + key[:11] + "/").replace(" + img:", " + edu/" + key[:11] + "/")
                v["answer_key_crosscheck"] = it.get("status")
                v["points_source"] = "paper '[n점]' text layer" if c.get("points_text") else "paper '[n점]' marker read by OCR (no text layer), sum=100 checked"
                v["sitting"] = dict(held=j.get("held"), title=j.get("title"))
            ref = c.get("ref") or ""
            if ref:
                fix, subs = hangul_autofix(f, ref)
                if fix:
                    f.update(fix); v["hangul_autofix"] = subs
            if pid in overrides:
                f.update(overrides[pid]); v["manual_override"] = sorted(overrides[pid].keys())
                if "choices_list" in overrides[pid] and "choices" not in overrides[pid]:
                    f["choices"] = "\n".join(f"{CIRC[i]} {x}" for i, x in enumerate(overrides[pid]["choices_list"]))
            norm_fields(f)
            stem = clean_stem(f.get("stem") or "")
            cut = "[잘림]" in stem
            stem = stem.replace("[잘림]", "").rstrip()
            choices = f.get("choices_list")
            if choices:
                choices = [(x or "").strip() or None for x in choices]
            try:
                v["ocr_number_match"] = int(re.sub(r"\D", "", str(f.get("number")))) == n
            except ValueError:
                v["ocr_number_match"] = False
            try:
                v["ocr_points_match"] = int(re.sub(r"\D", "", str(f.get("points")))) == it["points"]
            except ValueError:
                v["ocr_points_match"] = False
            v["points_marker_in_text"] = f"[{it['points']}점]" in stem.replace(" ", "")
            if not v["points_marker_in_text"] and not re.search(r"\[\s*\d\s*점\s*\]", stem):
                stem = stem.rstrip() + f" [{it['points']}점]"; v["points_marker_added"] = True
            kind = it["kind"]
            if kind == "mc":
                v["n_choices"] = sum(1 for x in (choices or []) if x)
                v["kind_consistent"] = v["n_choices"] == 5
            else:
                v["kind_consistent"] = not choices
            v["latex_errors"] = latex_balanced(stem + "\n" + "\n".join(x or "" for x in (choices or [])))
            fa = (o.get("a") or {}).get("finish"); fb = (o.get("b") or {}).get("finish")
            v["ocr_finish"] = [fa, fb]
            v["truncated"] = fa != "stop" or not stem or cut
            if cut:
                v["crop_cut_reported_by_ocr"] = True
            v["proofread_changed"] = o.get("b_changed")
            sel = o.get("select") or {}
            v["ocr_selected"] = sel.get("chosen")
            if sel.get("repaired"):
                v["ocr_repaired"] = True
            if o.get("b_changed") and sel.get("chosen") == "b":
                v["proofread_errors"] = (o.get("b_errors") or "")[:1500]
            v["ocr_model"] = (o.get("a") or {}).get("model")
            # producing models (user instruction 2026-11-18: eco only from then on; earlier items used both models)
            chosen = sel.get("chosen") or "a"
            if chosen == "a":
                m_sel = (o.get("a") or {}).get("model")
            elif chosen == "b":
                m_sel = (o.get("b") or {}).get("model")
            else:
                alt = (o.get("alts") or [{}])[int(chosen[3:])] if chosen[3:].isdigit() and int(chosen[3:]) < len(o.get("alts") or []) else {}
                m_sel = ((alt.get("b") or {}).get("model") if alt.get("final") is not None and alt.get("final") != alt.get("parsed") else (alt.get("raw") or {}).get("model"))
            calls = {(o.get("a") or {}).get("model"), (o.get("b") or {}).get("model")}
            for alt in o.get("alts") or []:
                calls |= {(alt.get("raw") or {}).get("model"), (alt.get("b") or {}).get("model")}
            calls.discard(None)
            v["models"] = dict(ocr_pass_a=(o.get("a") or {}).get("model"), ocr_pass_b=(o.get("b") or {}).get("model"), ocr_selected=m_sel,
                               ocr_all_calls=sorted(calls))
            if ref:
                v["text_layer"] = text_layer_check(f, ref)
            else:
                v["text_layer"] = "n/a (paper has no text layer)"
            if c.get("top_ink"):
                v["crop_top_edge_ink"] = True
            fig = figure_text(f)
            pg = pre.get(key)
            if pg and pg["lo"] <= n <= pg["hi"]:      # shared stem '[lo～hi] …' printed above item lo: prepend it (checks above ran on the item part)
                stem = clean_stem(fix_boxed(norm_delims(pg["stem"]))) + "\n\n" + stem
                pf = norm_delims(pg.get("figure") or "")
                if pf and re.sub(r"[\s.]", "", pf) not in ("없음", ""):
                    fig = (pf + (" / 문항 그림: " + fig if fig else "")) if fig else pf
                v["shared_preamble"] = dict(group=f"{pg['lo']}-{pg['hi']}", source_crop=pg["png"], ocr_chosen=pg["ocr_chosen"],
                                            text_layer_scores=pg["text_layer_scores"], figure_manual=pg.get("figure_manual", False))
            has_fig = bool(fig) or "[그림]" in stem or str(f.get("has_figure", "")).lower().startswith("y")
            if str(f.get("has_figure", "")).lower().startswith("y") and not fig:
                v["figure_description_missing"] = True
            text = stem + (f"\n[그림] {fig} [/그림]" if fig else "")
            if kind == "mc" and choices:
                text += "\n" + "\n".join(f"{CIRC[i]} {x}" for i, x in enumerate(choices))
            ss = solve.get(pid)
            if ss:
                ss = sorted(ss, key=lambda r: r["sample"])
                nc = sum(r["correct"] for r in ss)
                v["solve_check_deepseek"] = dict(model="deepseek/deepseek-v4.1-flash (ultrafast/eco alternated)", n=len(ss), n_correct=nc,
                                                 accuracy=round(nc / len(ss), 3), preds=[r["pred"] for r in ss],
                                                 models=[r["model"] for r in ss], finish=[r["finish"] for r in ss], suspect=nc == 0)
            v.update(extra.get(pid, {}))
            sm = set((v.get("solve_check_deepseek") or {}).get("models") or [])
            v["models"]["solve"] = sorted(m for m in sm if m)
            v["ultrafast_used"] = any("ultrafast" in (m or "") for m in list(v["models"]["ocr_all_calls"]) + v["models"]["solve"])
            rows.append(dict(id=pid, exam=j["exam"], school_year=j["school_year"], section=j["track"], track=TRACK_KO[j["track"]],
                             number=n, kind=kind, points=it["points"], text=text, choices=choices if kind == "mc" else None,
                             answer=it["answer"], has_figure=bool(has_fig), source_file=j["pdf"].replace("/data/", D + "/"), verify=v))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("rows", len(rows), "->", a.out)
    for r in report:
        print("NOTE", r)


if __name__ == "__main__":
    main()
