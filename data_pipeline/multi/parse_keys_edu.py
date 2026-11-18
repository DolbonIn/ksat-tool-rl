"""Official answer keys for 교육청 학평 (EBSi): answers from the 정답 및 해설 PDF text layer (answer table "1 ④ 2 ① …";
국어 elective table after "<선택과목> 정답"), cross-checked against the EBSi 정답 PNG read by DeepSeek (both models when
available). Points from the "[3점]" markers in the official paper's text layer (segments.json points_text), with the
sitting sum check (국어/영어 100, 탐구 50).

Acceptance: hsj text-layer answers present and equal to at least one PNG read -> OK; no hsj table -> two PNG reads by
different models must agree. Anything else -> errors (sitting skipped).
Usage: python -I parse_keys_edu.py --jobs /data/work2/jobs_edu.json --render /data/render2 --out /data/work2/answer_keys_edu.json
"""
import argparse, asyncio, base64, json, os, re, sys
import fitz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw  # noqa: E402

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
ELNAME = {"kor_speech_writing": r"화법과\s*작문", "kor_lang_media": r"언어와\s*매체"}


def runs(toks, lo, hi):
    out = []
    L = hi - lo + 1
    for i in range(len(toks) - 2 * L + 1):
        if all(toks[i + 2 * k] == str(lo + k) and toks[i + 2 * k + 1] in CIRC for k in range(L)):
            out.append((i, [CIRC[toks[i + 2 * k + 1]] for k in range(L)]))
    return out


def hsj_answers(job):
    t = "\n".join(p.get_text() for p in fitz.open(job["hsj"]))
    toks = re.findall(r"[①-⑤]|\d{1,2}|화법과\s*작문|언어와\s*매체", t)
    toks = [re.sub(r"\s+", " ", x) for x in toks]
    res = {}
    if job["area"] == "kor":
        r = runs(toks, 1, 34)
        if r:
            res.update({f"kor_common-{n}": a for n, a in zip(range(1, 35), r[0][1])})
        el = [s for s in job["expect"] if s != "kor_common"][0]
        r = runs(toks, 35, 45)
        name = re.compile(ELNAME[el])
        good = [x for x in r if any(name.fullmatch(toks[j]) for j in range(max(0, x[0] - 4), x[0]))]
        pick = good[0] if good else (r[0] if len(r) == 1 else None)
        if pick:
            res.update({f"{el}-{n}": a for n, a in zip(range(35, 46), pick[1])})
    else:
        sec = job["section"]; hi = 45 if job["area"] == "eng" else 20
        r = runs(toks, 1, hi)
        if r:
            res.update({f"{sec}-{n}": a for n, a in zip(range(1, hi + 1), r[0][1])})
    return res


PROMPT = ('이 이미지는 시험 정답표입니다. 표에 있는 모든 문항 번호와 정답(원문자)을 읽어 JSON 객체 하나로만 출력하세요. '
          '키는 문항 번호(문자열), 값은 정답 원문자입니다. 예: {"1": "②", "2": "⑤"}. 선택과목 열이 따로 있으면 그 번호(35~45)도 포함합니다. 다른 설명은 쓰지 마세요.')


async def read_png(path, model):
    url = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
    r = await gw.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}],
                      max_tokens=3000, temperature=0.0, thinking=False, tag=f"edukey:{os.path.basename(path)}", prefer=model, strict=True, retries=12)
    m = re.search(r"\{.*\}", r["content"] or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return {str(k): CIRC.get(str(v).strip()) for k, v in d.items()}, r["model"]
    except Exception:
        return {}, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--usage-log", default="/data/work2/usage.jsonl")
    ap.add_argument("--manual", default=None, help="JSON {key: {sec-n: answer | '모두 정답'}} from a manual read of the 정답 PNG")
    a = ap.parse_args()
    manual = json.load(open(a.manual)) if a.manual and os.path.exists(a.manual) else {}
    gw.init(int(os.environ.get("CONC", "10")), a.usage_log)
    jobs = json.load(open(a.jobs))
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    need = [(j, m) for j in jobs for m in gw.MODELS if m not in (prev.get(j["key"], {}).get("png_reads") or {})]
    reads = await asyncio.gather(*[read_png(j["answer_png"], m) for j, m in need])
    pr = {}
    for (j, m), (d, mod) in zip(need, reads):
        if d and mod:
            pr.setdefault(j["key"], {})[m] = d
    res = {}
    for j in jobs:
        key = j["key"]
        png = dict((prev.get(key, {}).get("png_reads") or {})); png.update(pr.get(key, {}))
        hs = hsj_answers(j)
        sf = os.path.join(a.render, key, "segments.json")
        pts = {}
        if os.path.exists(sf):
            for s in json.load(open(sf))["segments"]:
                if s["kind"] == "question":
                    pts[f"{s['section']}-{s['number']}"] = 3 if 3 in s["points_text"] else 2
        items, errs, notes = {}, [], []
        for sec, (lo, hi) in j["expect"].items():
            for n in range(lo, hi + 1):
                e = f"{sec}-{n}"
                pv = [d.get(str(n)) for d in png.values()]
                h = hs.get(e)
                mv = manual.get(key, {}).get(e, "absent")
                if mv == "모두 정답":
                    items[e] = dict(section=sec, number=n, answer=None, points=pts.get(e), source="manual_png_read",
                                    note="모두 정답 (official key: every choice accepted; item excluded from the dataset)")
                    continue
                if h is not None:
                    if not any(x == h for x in pv) and mv != h:
                        errs.append(f"{e}: hsj {h} vs png reads {pv}")
                    src = "hsj_text_layer" + ("+png" if any(x == h for x in pv) else "+manual_png_read" if mv == h else "")
                    ans = h
                else:
                    if len(pv) >= 2 and len(set(pv)) == 1 and pv[0]:
                        ans, src = pv[0], "png_read_x2"
                    else:
                        errs.append(f"{e}: no hsj answer, png reads {pv}"); ans, src = None, None
                if e not in pts:
                    errs.append(f"{e}: no points (segment missing)")
                items[e] = dict(section=sec, number=n, answer=ans, points=pts.get(e), source=src)
        secs = list(j["expect"])
        tot = sum(v["points"] or 0 for v in items.values())
        exp = 100 if j["area"] in ("kor", "eng") else 50
        if tot != exp:
            errs.append(f"points sum {tot} != {exp}")
        # 국어 언매 PDF: elective 35-45 sum must be 100 - common
        res[key] = dict(source_file=j["hsj"], answer_png=j["answer_png"], items=items, errors=errs, method="edu_hsj+png",
                        png_reads=png, notes=notes)
        print(key, "OK" if not errs else errs[:4])
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
