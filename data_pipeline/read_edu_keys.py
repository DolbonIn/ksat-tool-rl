"""Build official answer keys for 교육청 학평 sittings (EBSi).

Sources (both official EBSi files):
  * 정답표 PNG, one per 선택과목 (공통 1-22 + that elective's 23-30). PNG has no text layer, so the table is
    read by the local VLM (greedy, thinking off). Common answers are therefore read 3 times (once per PNG).
  * 정답 및 해설 PDF text layer: the circled (객관식) answers are real text; their order is cross-checked.
Points come from the "[n점]" markers in the official paper's text layer (render_crops index points_text).

Output: answer_keys_edu.json in the same format as parse_answer_keys.py, with per-item provenance.
Usage: python -I read_edu_keys.py --jobs /data/work/jobs_edu.json --render /data/render --out /data/work/answer_keys_edu.json
"""
import argparse, asyncio, base64, json, re
import fitz
from openai import AsyncOpenAI

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
PROMPT = ("이 이미지는 수학 영역 정답표입니다. 표에 있는 모든 문항 번호와 정답을 빠짐없이 읽어 JSON 객체 하나로만 출력하세요. "
          "키는 문항 번호(문자열), 값은 정답을 이미지에 적힌 그대로(객관식은 ①②③④⑤ 원문자, 단답형은 숫자) 적습니다. "
          '예: {"1": "②", "2": "①", "16": "10"}. 다른 설명은 쓰지 마세요.')


async def read_png(client, path):
    url = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
    r = await client.chat.completions.create(
        model="vlm", temperature=0.0, max_tokens=2048,
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1})
    txt = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def to_ans(v):
    v = str(v).strip()
    if v in CIRC:
        return CIRC[v], "mc"
    m = re.fullmatch(r"\d{1,3}", v)
    return (int(v), "sa") if m else (None, None)


def hsj_circled(paths):
    """Ordered circled answers in the 해설 PDFs' answer tables (text before each first '[출제의도]')."""
    seqs = []
    for p in paths:
        doc = fitz.open(p)
        txt = "\n".join(pg.get_text() for pg in doc)
        seqs.append(re.findall(r"[①-⑤]", txt.split("[출제의도]")[0]))
    return seqs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:31801/v1")
    a = ap.parse_args()
    client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=600)
    jobs = json.load(open(a.jobs))
    reads = await asyncio.gather(*[read_png(client, j["answer_pngs"][s]) for j in jobs for s in ("prob_stat", "calculus", "geometry")])
    res = {}
    k = 0
    for j in jobs:
        idx = json.load(open(f"{a.render}/crops/{j['key']}/index.json"))
        pts = {c["name"]: (c["points_text"][0] if len(c["points_text"]) == 1 else None) for c in idx["crops"]}
        items, errs, notes = {}, [], []
        common_reads = []
        for s in ("prob_stat", "calculus", "geometry"):
            rd = reads[k]; k += 1
            common_reads.append({n: rd.get(str(n)) for n in range(1, 23)})
            for n in range(23, 31):
                ans, kind = to_ans(rd.get(str(n)))
                items[f"{s}-{n}"] = dict(section=s, number=n, answer=ans, kind=kind, points=pts.get(f"{s}-{n}"), raw=rd.get(str(n)),
                                         source=f"png:{j['answer_pngs'][s].split('/')[-1]}")
        for n in range(1, 23):
            vals = [cr[n] for cr in common_reads]
            if len(set(map(str, vals))) != 1:
                errs.append(f"common-{n} disagree across PNG reads {vals}")
            ans, kind = to_ans(vals[0])
            items[f"common-{n}"] = dict(section="common", number=n, answer=ans, kind=kind, points=pts.get(f"common-{n}"), raw=vals[0],
                                        source="png x3 (prob/calc/geom answer images)")
        # kind sanity: mc for 1-15 and 23-28 in every 2022+ paper
        for e, v in items.items():
            exp_mc = v["number"] <= 15 or 23 <= v["number"] <= 28
            if v["answer"] is None:
                errs.append(f"{e} unreadable {v['raw']}")
            elif (v["kind"] == "mc") != exp_mc:
                errs.append(f"{e} kind {v['kind']} unexpected")
            if v["points"] is None:
                errs.append(f"{e} points missing")
        # cross-check circled answers against the 해설 PDF text layer (common 1-15 come first)
        seqs = hsj_circled(j["hsj"])
        mc_common = [items[f"common-{n}"]["answer"] for n in range(1, 16)]
        for p, sq in zip(j["hsj"], seqs):
            got = [CIRC[c] for c in sq[:15]]
            if len(got) == 15:
                if got != mc_common:
                    errs.append(f"hsj {p.split('/')[-1]} common mc mismatch {got} vs {mc_common}")
                else:
                    notes.append(f"hsj {p.split('/')[-1]}: common 1-15 circled answers match")
            else:
                notes.append(f"hsj {p.split('/')[-1]}: only {len(got)} circled answers in text layer")
        for s in ("prob_stat", "calculus", "geometry"):
            tot = sum((items[e]["points"] or 0) for e in items if e.startswith("common") or e.startswith(s))
            if tot != 100:
                errs.append(f"{s} points sum {tot}")
        res[j["key"]] = dict(source_file=";".join(j["answer_pngs"].values()), items=items, errors=errs, notes=notes)
        print(j["key"], "OK" if not errs else errs, "|", "; ".join(notes)[:200])
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
