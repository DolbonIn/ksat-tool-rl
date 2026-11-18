"""Assemble train.jsonl / test_2026.jsonl / problems_2026.py from OCR outputs + official answer keys.

Answers, points and kind (mc/sa) always come from the official 정답표 (answer_keys JSON). OCR is only used for
problem text, choices and the figure flag. Every record carries a `verify` dict with the checks run so far;
later stages (solve-check, KoreanSAT diff, manual review) merge into it via --verify-extra JSON files.

Usage:
  python3 build_dataset.py --jobs jobs_kice.json[,jobs_edu.json] --keys answer_keys_kice.json[,answer_keys_edu.json] \
      --ocr /raid/.../data/ocr --out /raid/.../data/ksat_math [--verify-extra a.json,b.json] [--overrides overrides.json]
"""
import argparse, json, os, pprint, re

EXAM_TAG = {"csat": "csat", "kice_06": "06", "kice_09": "09", "edu_03": "edu03", "edu_04": "edu04", "edu_07": "edu07", "edu_10": "edu10"}
SECTIONS = ["common", "prob_stat", "calculus", "geometry"]
CIRC = "①②③④⑤"
DATA_ROOT = "/raid/cache/b300-07/ksat-rl/data"


def latex_balanced(s):
    errs = []
    t = s.replace("\\$", "")
    # $$ blocks count as one delimiter pair each
    dd = t.count("$$")
    single = t.replace("$$", "").count("$")
    if dd % 2:
        errs.append("odd $$")
    if single % 2:
        errs.append("odd $")
    depth = 0
    for i, ch in enumerate(t):
        if ch == "{" and (i == 0 or t[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or t[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        errs.append(f"brace depth {depth}")
    b = re.findall(r"\\begin\{(\w+\*?)\}", t); e = re.findall(r"\\end\{(\w+\*?)\}", t)
    if sorted(b) != sorted(e):
        errs.append(f"begin/end {b} vs {e}")
    if len(re.findall(r"\\left\b|\\left[\(\[\.|\\{]", t)) != len(re.findall(r"\\right\b|\\right[\)\]\.|\\}]", t)):
        errs.append("left/right")
    if t.count("[그림]") != t.count("[/그림]"):
        errs.append("figure block")
    return errs


def html_table_to_md(m):
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), re.S)
    out = []
    for i, r in enumerate(rows):
        cells = [re.sub(r"\s+", " ", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
        out.append("| " + " | ".join(cells) + " |")
        if i == 0:
            out.append("|" + "---|" * len(cells))
    return "\n".join(out)


def clean_stem(stem):
    s = stem.strip()
    s = re.sub(r"<table[^>]*>.*?</table>", html_table_to_md, s, flags=re.S)
    s = re.sub(r"^\s*(\(\s*(확률과\s*통계|미적분|기하)\s*\)\s*)?([홀짝]수형\s*)?(수학\s*영역\s*)?", "", s)  # page header caught in crop
    s = re.sub(r"^\s*\d{1,2}\s*\.\s*", "", s)          # stray problem number
    # blank boxes like "(가)" are often transcribed as \boxed{...}; use \fbox so they never clash with \boxed{} answers
    s = re.sub(r"\\boxed\{\s*(?:\\text\{\s*)?(\(?[가-하]\)?)\s*\}?\s*\}", r"\\fbox{\1}", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def figure_text(f):
    fig = (f.get("figure") or "").strip()
    if not fig or re.sub(r"[\s.]", "", fig) in ("없음", "none", "None"):
        return ""
    return fig


def src_pdf(job, sec):
    for p in job.get("parts", []):
        if sec in (p.get("sections") or []):
            return p["pdf"]
    return job["pdf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", required=True)
    ap.add_argument("--ocr", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--verify-extra", default=""); ap.add_argument("--overrides", default="")
    ap.add_argument("--test-key", default="2026_csat")
    a = ap.parse_args()
    jobs = [j for f in a.jobs.split(",") for j in json.load(open(f))]
    keys = {}
    for f in a.keys.split(","):
        keys.update(json.load(open(f)))
    extra = {}
    for f in [x for x in a.verify_extra.split(",") if x]:
        for k, v in json.load(open(f)).items():
            extra.setdefault(k, {}).update(v)
    overrides = {}
    for f in [x for x in a.overrides.split(",") if x]:      # later files win (e.g. hangul autofix, then manual review)
        for k, v in json.load(open(f)).items():
            overrides.setdefault(k, {}).update(v)
    os.makedirs(a.out, exist_ok=True)
    train, test, problems = [], [], []
    report = []
    for j in jobs:
        key = j["key"]; yr = j["school_year"]; exam = j["exam"]
        ak = keys.get(key, {}).get("items", {})
        if not ak:
            report.append(f"{key}: no answer key -> skipped"); continue
        recs = []
        for sec in SECTIONS:
            for n in (range(1, 23) if sec == "common" else range(23, 31)):
                name = f"{sec}-{n}"
                pid = f"{yr}-{EXAM_TAG[exam]}-{sec}-{n}"
                fn = os.path.join(a.ocr, key, name + ".json")
                if not os.path.exists(fn):
                    report.append(f"{pid}: no OCR"); continue
                o = json.load(open(fn))
                f = dict(o.get("final") or {})
                if pid in overrides:
                    f.update(overrides[pid])
                k = ak[name]
                stem = clean_stem(f.get("stem") or "")
                choices = f.get("choices_list")
                if choices:
                    choices = [(c or "").strip() or None for c in choices]
                v = {}
                v["answer_source"] = "official_key"
                v["answer_key_file"] = k.get("source") or keys[key]["source_file"].split("/dl/")[-1]
                if k.get("hsj_check"):
                    v["answer_key_crosscheck"] = k["hsj_check"]
                try:
                    v["ocr_number_match"] = int(re.sub(r"\D", "", str(f.get("number")))) == n
                except ValueError:
                    v["ocr_number_match"] = False
                try:
                    v["ocr_points_match"] = int(re.sub(r"\D", "", str(f.get("points")))) == k["points"]
                except ValueError:
                    v["ocr_points_match"] = False
                v["points_marker_in_text"] = f"[{k['points']}점]" in stem.replace(" ", "")
                if not v["points_marker_in_text"] and not re.search(r"\[\s*\d\s*점\s*\]", stem):
                    stem = stem.rstrip() + f" [{k['points']}점]"      # marker is printed on the paper; OCR dropped it
                    v["points_marker_added"] = True
                if k["kind"] == "mc":
                    v["n_choices"] = sum(1 for c in (choices or []) if c)
                    v["kind_consistent"] = v["n_choices"] == 5
                else:
                    v["kind_consistent"] = choices is None
                v["latex_errors"] = latex_balanced(stem + "\n" + "\n".join(c or "" for c in (choices or [])))
                fa = (o.get("a") or {}).get("finish"); fb = (o.get("b") or {}).get("finish")
                v["ocr_finish"] = [fa, fb]
                v["truncated"] = fa != "stop" or not stem
                if fb is not None and fb != "stop":
                    v["proofread_incomplete"] = True      # proofread hit max tokens; pass-A transcription kept
                v["proofread_changed"] = o.get("b_changed")
                if o.get("select"):
                    v["ocr_selected"] = o["select"]["chosen"]          # a / b / altN (picked by text-layer agreement)
                    if o["select"].get("repaired") or str(o["select"]["chosen"]).startswith("alt"):
                        v["ocr_repaired"] = True
                if o.get("b_changed"):
                    v["proofread_errors"] = (o.get("b_errors") or "")[:1500]
                if pid in overrides:
                    v["manual_override"] = sorted(overrides[pid].keys())
                fig = figure_text(f)
                has_fig = bool(fig) or "[그림]" in stem or str(f.get("has_figure", "")).lower().startswith("y")
                if str(f.get("has_figure", "")).lower().startswith("y") and not fig:
                    v["figure_description_missing"] = True
                text = stem + (f"\n[그림] {fig} [/그림]" if fig else "")
                if k["kind"] == "mc" and choices:
                    text += "\n" + "\n".join(f"{CIRC[i]} {c}" for i, c in enumerate(choices))
                v.update(extra.get(pid, {}))
                rec = dict(id=pid, exam=exam, school_year=yr, section=sec, number=n, kind=k["kind"], points=k["points"],
                           text=text, choices=choices if k["kind"] == "mc" else None, answer=k["answer"],
                           has_figure=bool(has_fig), source_file=DATA_ROOT + "/dl/" + src_pdf(j, sec).split("/dl/")[-1], verify=v)
                recs.append(rec)
                if key == a.test_key:
                    problems.append(dict(id=f"{sec}-{n}", section=sec, kind=k["kind"], points=k["points"], text=text,
                                         choices=rec["choices"], answer=k["answer"]))
        # per-exam checks
        for s in ("prob_stat", "calculus", "geometry"):
            tot = sum(r["points"] for r in recs if r["section"] in ("common", s))
            if tot != 100:
                report.append(f"{key}: {s} points {tot}")
        if len(recs) != 46:
            report.append(f"{key}: {len(recs)} records")
        (test if key == a.test_key else train).extend(recs)
    for name, rows in (("train.jsonl", train), ("test_2026.jsonl", test)):
        with open(os.path.join(a.out, name), "w") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "problems_2026.py"), "w") as fo:
        fo.write('"""2026학년도 대학수학능력시험 수학 영역(홀수형) 46문항. 정답·배점은 평가원 공식 정답표.\n'
                 'Generated by data_pipeline/build_dataset.py — do not commit (data boundary)."""\n')
        fo.write("PROBLEMS = " + pprint.pformat(problems, width=120, sort_dicts=False) + "\n")
    print(f"train {len(train)}  test {len(test)}  problems_2026 {len(problems)}")
    for r in report:
        print("WARN", r)


if __name__ == "__main__":
    main()
