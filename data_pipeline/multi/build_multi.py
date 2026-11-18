"""Assemble {area}_train.jsonl / {area}_test_2026.jsonl (+ units_2026.json) from segments, verified transcriptions,
official answer keys and (English) official listening scripts.

Answers and points always come from the official 정답표 (answer-keys JSON). Transcriptions only supply text.
Every record carries `verify` (answer-key provenance, points-marker check, text-layer check, OCR selection, plus any
--verify-extra JSON merged by id: solve checks, manual review).

Usage (container, python -I):
  python -I build_multi.py --jobs /data/work2/jobs_kice.json --keys /data/work2/answer_keys_kice.json \
     --render /data/render2 --ocr /data/ocr2 --out /data/ksat_multi [--overrides ov.json] [--verify-extra a.json,b.json]
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
import fitz

CIRC = "①②③④⑤"
DATA_HOST = "/raid/cache/b300-07/ksat-rl/data"
SUBTAG = {"kor_common": "common", "kor_speech_writing": "speech_writing", "kor_lang_media": "lang_media"}
LISTEN_HDR = "[듣기 대본 — 실제 시험에서는 음성으로 들려주는 내용]"
FIG_RE = re.compile(r"\[그림\].*?\[/그림\]", re.S)


def host(p):
    return p.replace("/data/", DATA_HOST + "/", 1) if p.startswith("/data/") else p


def clean(s):
    if s is None:
        return None
    s = re.sub(r"<\s*보\s*기\s*>", "<보기>", s)
    s = re.sub(r"\n?</보기>\n?", "\n", s)
    s = re.sub(r"<br\s*/?>", " / ", s)
    s = s.replace("¦", " ")
    s = balance_blocks(s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def balance_blocks(s):
    """Balance [표]/[그림] blocks with a stack: figures inside table cells are fine; a table opened inside a figure and
    never closed is closed before the figure's [/그림]; stray closers are dropped; open blocks are closed at the end."""
    out, stack = [], []
    for t in re.split(r"(\[/?표\]|\[/?그림\])", s):
        if t in ("[표]", "[그림]"):
            stack.append(t[1:-1]); out.append(t)
        elif t in ("[/표]", "[/그림]"):
            name = t[2:-1]
            if name not in stack:
                continue
            while stack and stack[-1] != name:
                out.append(f"[/{stack.pop()}]\n")
            stack.pop(); out.append(t)
        else:
            out.append(t)
    while stack:
        out.append(f"\n[/{stack.pop()}]")
    return "".join(out)


def parse_script(pdf):
    """Official listening script PDF (KICE or 교육청): stream all lines, split at the Korean question lines "n. …"
    (1-15) and the shared "[16～17]" block; announcements (ANN) and page furniture are dropped."""
    doc = fitz.open(pdf)
    lines = []
    for page in doc:
        for l in page.get_text("text").split("\n"):
            if not l.strip():
                continue
            if ("저작권" in l or re.fullmatch(r"\s*-?\s*\d+\s*-?\s*", l) or re.search(r"학년도|전국연합학력평가|대학수학능력시험|듣기\s*(평가\s*)?대본|듣기 평가$|시행\s*:|^\s*○|실시\s*-", l)
                    or re.fullmatch(r"\s*(영어|영역|영어 영역|\d+월 고3)\s*", l)):
                continue
            lines.append(l.rstrip())
    blocks, cur, skipping_q, ann = {}, None, False, False
    for l in lines:
        st = l.strip()
        if re.match(r"^\[\s*16\s*[～~]\s*17\s*\]", st) or re.match(r"^16\s*번?\s*([～~]|과)\s*17\s*번\s*$", st):
            cur = "16_17"; blocks[cur] = []; skipping_q = False; ann = False; continue
        if re.match(r"^16번부터 17번까지는", st):
            continue
        m = re.match(r"^(\d{1,2})\.\s*[가-힣0-9A-Za-z]", st) or re.match(r"^(\d{1,2})\s*번\s*$", st)
        if m and 1 <= int(m.group(1)) <= 17 and re.search(r"[가-힣]", st):
            n = int(m.group(1))
            if cur == "16_17" and n in (16, 17):
                cur = None; continue
            cur = str(n); blocks[cur] = []; skipping_q = True; ann = False; continue
        if re.fullmatch(r"\(\d+\s*seconds?\)", st):
            continue
        if st.startswith("ANN"):
            ann = True; continue
        if cur is None or ann:
            continue
        if skipping_q:
            if re.match(r"^(M|W|[A-Z][A-Za-z]+)\s*:|^\[", st):
                skipping_q = False
            else:
                continue                      # wrapped question line / "[3점]"
        if re.search(r"[가-힣]", st) and not re.search(r"[A-Za-z]{2,}", st):
            continue                          # Korean broadcast remarks ("다시 한 번 듣겠습니다.")
        st = re.sub(r"\s*다시 한 번 듣겠습니다\.?\s*$", "", st)
        blocks[cur].append(st)
    res = {}
    for n, body in blocks.items():
        paras = []
        for l in body:
            if not paras or re.match(r"^(M|W|[A-Z][A-Za-z]+)\s*:|^\[", l):
                paras.append(l)
            else:
                paras[-1] = paras[-1] + " " + l
        if paras:
            res[n] = "\n".join(re.sub(r"\s+", " ", x).replace("\xad", "-").strip() for x in paras)
    return res


def load_extra(paths):
    extra = defaultdict(dict)
    for f in [x for x in (paths or "").split(",") if x]:
        for k, v in json.load(open(f)).items():
            extra[k].update(v)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", required=True)
    ap.add_argument("--render", required=True); ap.add_argument("--ocr", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default=""); ap.add_argument("--verify-extra", default="")
    ap.add_argument("--test-year", type=int, default=2026)
    a = ap.parse_args()
    jobs = [j for f in a.jobs.split(",") for j in json.load(open(f))]
    keys = {}
    for f in a.keys.split(","):
        keys.update(json.load(open(f)))
    extra = load_extra(a.verify_extra)
    overrides = load_extra(a.overrides)       # {item_id or "key/sid": {field: value}}
    os.makedirs(a.out, exist_ok=True)
    rows = defaultdict(list)                  # (area, split) -> records
    report, excluded = [], []
    for j in jobs:
        key, area, exam, yr = j["key"], j["area"], j["exam"], j["school_year"]
        ak = keys.get(key)
        if not ak or not ak.get("items") or ak.get("errors"):
            report.append(f"{key}: answer key unusable {ak.get('errors') if ak else 'missing'} -> skipped"); continue
        sf = os.path.join(a.render, key, "segments.json")
        if not os.path.exists(sf):
            report.append(f"{key}: no segments -> skipped"); continue
        segs = json.load(open(sf))["segments"]
        tx = {}
        for s in segs:
            fn = os.path.join(a.ocr, key, s["sid"] + ".json")
            if os.path.exists(fn):
                tx[s["sid"]] = json.load(open(fn))
        script = parse_script(j["script"]) if area == "eng" and j.get("script") else {}
        passages = {}
        segs = [s for s in segs if s["section"] not in (j.get("skip_sections") or [])]
        for s in segs:
            if s["kind"] != "passage":
                continue
            r = tx.get(s["sid"])
            ov = overrides.get(f"{key}/{s['sid']}", {})
            p = clean(ov.get("passage") or ((r or {}).get("final_fixed") or {}).get("passage"))
            sub = SUBTAG.get(s["section"])
            pid = f"{yr}-{exam}-{area}" + (f"-{sub}" if sub else (f"-{j['subject']}" if area in ("sci", "soc") else "")) + f"-p{s['range'][0]}_{s['range'][1]}"
            psel = (r or {}).get("select") or {}
            pmodel = (r["attempts"][psel["chosen"]].get("model") if r and psel.get("chosen") is not None else None)
            passages[(s["section"], s["range"][0], s["range"][1])] = dict(
                pid=pid, text=p, verify=(r or {}).get("verify"), model=pmodel, has_fig=bool(p and "[그림]" in p) or
                str(((r or {}).get("final_fixed") or {}).get("has_figure", "")).lower().startswith("y"))
        split = "test" if (exam == "csat" and yr == a.test_year) else "train"
        recs = []
        for s in segs:
            if s["kind"] != "question":
                continue
            n, sec = s["number"], s["section"]
            kk = ak["items"].get(f"{sec}-{n}")
            sub = SUBTAG.get(sec)
            iid = f"{yr}-{exam}-{area}-" + (f"{sub}-" if sub else (f"{j['subject']}-" if area in ("sci", "soc") else "")) + str(n)
            if kk is None:
                report.append(f"{iid}: no answer-key entry"); continue
            if kk.get("answer") is None:
                excluded.append(dict(id=iid, reason=kk.get("note", "no answer"))); continue
            r = tx.get(s["sid"])
            if not r or not r.get("final_fixed"):
                report.append(f"{iid}: no verified transcription"); continue
            f = dict(r["final_fixed"])
            ov = overrides.get(iid, {})
            f.update({k: v for k, v in ov.items() if k in ("stem", "choices_list", "has_figure")})
            stem = clean(f.get("stem") or "")
            stem = re.sub(r"^\s*\d{1,2}\s*\.\s+", "", stem)
            ch = f.get("choices_list") or []
            ch = [clean(c) if c is not None else None for c in ch]
            if all(m in stem for m in CIRC) and not any(ch):
                ch = list(CIRC)
            pas = None
            for (psec, lo, hi), pv in passages.items():
                if psec == sec and lo <= n <= hi:
                    pas = pv
            v = {}
            v["answer_source"] = "official_key"
            v["answer_key_file"] = host(ak["source_file"])
            v["answer_key_method"] = ak.get("method", "text_layer")
            if ak.get("reads"):
                v["answer_key_reads"] = ak["reads"]
            marker3 = 3 in s["points_text"]
            v["points_marker_match"] = (kk["points"] == 3) == marker3 or (area == "eng" and n <= 17 and kk["points"] == 3)
            v["n_choices"] = sum(1 for c in ch if c)
            vv = r.get("verify") or {}
            v["textlayer_check"] = {k: vv.get(k) for k in ("ok_textlayer", "ok_crossmodel", "cross_model_agreement", "h_missing", "h_extra",
                                                            "d_missing", "d_extra", "l_missing", "l_extra", "structure") if vv.get(k) not in (None, "", [])}
            v["textlayer_ok"] = bool(vv.get("ok"))
            if vv.get("hangul_autofix"):
                v["hangul_autofix"] = vv["hangul_autofix"]
            sel = r.get("select") or {}
            att = r["attempts"][sel["chosen"]] if sel.get("chosen") is not None else {}
            v["ocr_model"] = att.get("model"); v["ocr_attempt"] = att.get("tag")
            if pas is not None:
                v["passage_textlayer_ok"] = bool((pas["verify"] or {}).get("ok"))
                v["passage_ocr_model"] = pas.get("model")
            if ov:
                v["manual_override"] = sorted(ov)
            has_fig = bool(FIG_RE.search(stem)) or str(f.get("has_figure", "")).lower().startswith("y") or any("[그림]" in (c or "") for c in ch)
            passage_text = pas["text"] if pas else None
            pid = pas["pid"] if pas else None
            if area == "eng" and n <= 17:
                sk = "16_17" if n in (16, 17) else str(n)
                sc = script.get(sk)
                if sc:
                    passage_text = LISTEN_HDR + "\n" + sc
                    pid = f"{yr}-{exam}-eng-p16_17" if n in (16, 17) else None
                    v["listening_script"] = "official KICE 듣기평가 대본 (text layer)"
                else:
                    v["listening_script"] = "MISSING"
            if pas and pas["has_fig"]:
                has_fig = True
            text = (passage_text + "\n\n" if passage_text else "") + stem + "\n" + "\n".join(f"{CIRC[i]} {c}" for i, c in enumerate(ch) if i < 5)
            rec = dict(id=iid, area=area, subject=sec if area != "eng" else "eng", exam=exam, school_year=yr, number=n, points=kk["points"],
                       passage_id=pid, passage=passage_text, question=stem, choices=ch, answer=kk["answer"], has_figure=bool(has_fig),
                       source_file=host(j["pdf"]), verify=v, text=text)
            rec["verify"].update(extra.get(iid, {}))
            recs.append(rec)
        # sitting checks
        exp = j["expect"]
        for secx, (lo, hi) in exp.items():
            if secx in (j.get("skip_sections") or []):
                continue
            got = sorted(r["number"] for r in recs if r["subject"] == (secx if area != "eng" else "eng"))
            exc = [x for x in excluded if x["id"].startswith(f"{yr}-{exam}-{area}-") and (area != "eng")]
            if got != list(range(lo, hi + 1)):
                missing = sorted(set(range(lo, hi + 1)) - set(got))
                report.append(f"{key}/{secx}: {len(got)} items (missing {missing})")
        if area == "kor":
            for el in [x for x in ("kor_speech_writing", "kor_lang_media") if x in j["expect"]]:
                tot = sum(v["points"] for v in ak["items"].values() if v["section"] in ("kor_common", el))
                if tot != 100:
                    report.append(f"{key}: {el} key points {tot}")
        rows[(area, split)].extend(recs)
    for (area, split), rs in sorted(rows.items()):
        fn = os.path.join(a.out, f"{area}_train.jsonl" if split == "train" else f"{area}_test_{a.test_year}.jsonl")
        with open(fn, "w") as fo:
            for r in rs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{os.path.basename(fn)}: {len(rs)} items, figures {sum(r['has_figure'] for r in rs)}, "
              f"textlayer_ok {sum(r['verify']['textlayer_ok'] for r in rs)}")
    json.dump(dict(excluded=excluded, report=report), open(os.path.join(a.out, "build_report.json"), "w"), ensure_ascii=False, indent=1)
    for x in excluded:
        print("EXCLUDED", x)
    for r in report:
        print("WARN", r)


if __name__ == "__main__":
    main()
