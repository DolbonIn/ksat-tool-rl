"""Repair DeepSeek solve-check suspects (0/n correct vs the OFFICIAL answer) of the 2017-2021 set.

For each suspect: candidate re-transcriptions are produced
  (v) a focused proofread of the current item text vs the image (thinking on),
  (t) the text-layer-assisted transcription (image + PDF Korean/digit text; papers with a text layer),
  (r) a fresh transcription with thinking on (T=0.3);
candidates that differ materially from the current text are re-solved (n=4, thinking, 65536 tokens). A candidate is
adopted only if it gets >=1 correct sample AND does not score worse on the text-layer check. Items that stay at 0
correct get 4 extra solve samples of the current text (pass@8). Never changes official answers.
Outputs: --ov overrides JSON {id: {stem, figure, choices_list, choices}}, --extra verify JSON {id: {...}}, --log JSONL.
Usage: python -I repair_suspects.py --items /data/ksat_math_old/X.jsonl --solve /data/work3/solve_hard.jsonl \
          --jobs /data/work3/jobs.json --render /data/render3 --ocr /data/ocr3 --ov /data/work3/ov_repair.json \
          --extra /data/work3/verify_repair.json --log /data/work3/repair_log.jsonl
"""
import argparse, asyncio, json, os, re, sys
sys.path.insert(0, "/pipe"); sys.path.insert(0, "/pipe/multi"); sys.path.insert(0, "/code")
import gw  # noqa: E402
# model policy (user instruction 2026-11-18): eco only from now on; override with GW_MODELS=a,b
gw.MODELS = os.environ.get("GW_MODELS", "deepseek/deepseek-v4.1-flash-eco").split(",")
import ocr_vlm  # noqa: E402
from repair_ocr import score  # noqa: E402
from solve_check import INSTR_MC, INSTR_SA, grade  # noqa: E402
from ocr_old import pass_tl, img, NOTE_A  # noqa: E402
from build_dataset import clean_stem, figure_text  # noqa: E402

CIRC = "①②③④⑤"
P_VERIFY = r"""아래는 이 수학 문항 이미지를 텍스트(한국어 + LaTeX)로 옮긴 결과입니다. 이 전사로 문제를 풀면 공식 정답이 나오지 않아, 전사 오류가 의심됩니다.
이미지와 한 기호씩 대조해 문제의 의미를 바꾸는 오류를 모두 찾으세요: 숫자, 부호, 지수·첨자, 분수의 분자·분모, 근호, 절댓값, 부등호 방향과 등호 포함 여부,
괄호, 함수·변수 이름, 로그의 밑, 극한·적분·합의 범위, 조건 (가)(나)(다)와 <보기> 항목의 누락·변형, 선택지 값과 순서, 그림 설명의 누락·오류
(그림에서 읽히는 좌표·길이·각·교점·점의 위치, 그림에 없는 정보를 지어낸 것 포함). 표기 방식 차이는 오류가 아닙니다. 풀이는 하지 마세요.

출력 형식:
<errors>
찾은 오류를 한 줄에 하나씩(없으면 '없음')
</errors>
오류가 있으면 이어서 수정된 전체 전사를 <number>, <points>, <stem>, <has_figure>, <figure>, <choices> 형식으로 출력하세요.

[전사 결과]
"""


def fmt_item(f):
    return ocr_vlm.fmt(dict(number=f.get("number"), points=f.get("points"), stem=f.get("stem"), has_figure=f.get("has_figure") or "no",
                            figure=f.get("figure") or "없음", choices=f.get("choices") if f.get("choices_list") else None))


def item_text(f, kind, pts):
    stem = clean_stem(f.get("stem") or "").replace("[잘림]", "").rstrip()
    if not re.search(r"\[\s*\d\s*점\s*\]", stem):
        stem += f" [{pts}점]"
    fig = figure_text(f)
    t = stem + (f"\n[그림] {fig} [/그림]" if fig else "")
    ch = f.get("choices_list")
    if kind == "mc" and ch:
        t += "\n" + "\n".join(f"{CIRC[i]} {(c or '').strip()}" for i, c in enumerate(ch))
    return t


def norm(t):
    return re.sub(r"\s+|\\[,;!]|\\dfrac|\\frac|\\left|\\right|\$", "", t)


async def solve(it, text, n, tag, offset=0):
    instr = INSTR_MC if it["kind"] == "mc" else INSTR_SA
    async def one(k):
        r = await gw.chat([{"role": "user", "content": f"{instr}\n\n{text}"}], max_tokens=65536, temperature=0.6, thinking=True,
                          tag=f"repair-solve:{it['id']}:{tag}:{k}", prefer=gw.MODELS[k % len(gw.MODELS)], retries=10)
        pred, ok = grade(it, r["content"]) if r["finish"] != "error" else (None, False)
        return dict(sample=k, model=r["model"], pred=pred, correct=bool(ok), finish=r["finish"], tokens=(r.get("usage") or {}).get("completion"))
    return await asyncio.gather(*[one(k) for k in range(offset, offset + n)])


async def repair(a, it, job, crop, rec):
    path = os.path.join(a.render, crop["png"]); ref = crop.get("ref") or ""
    cur = dict(rec["final"])
    cur_text = item_text(cur, it["kind"], it["points"])
    tag = f"repair:{it['id']}"
    cands = {}
    rv = await gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": P_VERIFY + fmt_item(cur)}]}],
                       max_tokens=32768, temperature=0.3, thinking=True, tag=tag + ":v")
    fin = ocr_vlm.finalize({"a_parsed": cur, "b": rv})
    v_errors = fin.get("b_errors")
    if fin.get("b_changed"):
        cands["v"] = fin["final"]
    if ref:
        o = await pass_tl(path, ref, tag + ":t")
        if o["finish"] == "stop":
            cands["t"] = ocr_vlm.parse(o["content"])
    rr = await gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": ocr_vlm.PROMPT_A + NOTE_A}]}],
                       max_tokens=32768, temperature=0.3, thinking=True, tag=tag + ":r")
    if rr["finish"] == "stop":
        cands["r"] = ocr_vlm.parse(rr["content"])
    cur_sc = score(cur, ref)[0] if ref else None
    tried = {}
    for k, c in cands.items():
        if not c or not c.get("stem"):
            continue
        if it["kind"] == "mc" and (not c.get("choices_list") or sum(1 for x in c["choices_list"] if x) != 5):
            continue
        t = item_text(c, it["kind"], it["points"])
        if norm(t) == norm(cur_text) or any(norm(t) == norm(x["text"]) for x in tried.values()):
            continue
        sc = score(c, ref)[0] if ref else None
        res = await solve(it, t, 4, k)
        tried[k] = dict(text=t, cand=c, text_layer_score=sc, n_correct=sum(r["correct"] for r in res), samples=res)
    ok = [k for k, x in tried.items() if x["n_correct"] > 0 and (cur_sc is None or x["text_layer_score"] is None or x["text_layer_score"] <= cur_sc)]
    adopted = max(ok, key=lambda k: tried[k]["n_correct"]) if ok else None
    out = dict(id=it["id"], v_errors=(v_errors or "")[:1500], candidates={k: dict(n_correct=x["n_correct"], text_layer_score=x["text_layer_score"],
               preds=[r["pred"] for r in x["samples"]]) for k, x in tried.items()}, current_text_layer_score=cur_sc, adopted=adopted)
    if adopted:
        c = tried[adopted]["cand"]
        out["override"] = dict(stem=c.get("stem"), figure=c.get("figure") or "없음", has_figure=c.get("has_figure"),
                               choices_list=c.get("choices_list"), choices=c.get("choices"))
        out["solve_after"] = tried[adopted]["samples"]
    else:
        out["extra_samples"] = await solve(it, cur_text, 4, "extra", offset=4)
    return out


async def main():
    ap = argparse.ArgumentParser()
    for x in ("items", "solve", "jobs", "render", "ocr", "ov", "extra", "log"):
        ap.add_argument("--" + x, required=True)
    ap.add_argument("--ids", default=None); ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--usage-log", default="/data/work3/usage.jsonl")
    a = ap.parse_args()
    gw.init(a.concurrency, a.usage_log)
    items = {json.loads(l)["id"]: json.loads(l) for l in open(a.items)}
    by = {}
    for f in a.solve.split(","):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish") != "error":
                by.setdefault(r["id"], []).append(r)
    sus = [i for i in items if i in by and not any(r["correct"] for r in by[i])]
    if a.ids:
        sus = [i for i in sus if i in a.ids.split(",")]
    done = {}
    if os.path.exists(a.log):
        for l in open(a.log):
            r = json.loads(l); done[r["id"]] = r
    jobs = {j["key"]: j for j in json.load(open(a.jobs))}
    todo = []
    for pid in sus:
        if pid in done:
            continue
        sy, ex, tr, n = pid.split("-")
        key = f"{sy}_{ex}_{tr}"
        crop = [c for c in json.load(open(os.path.join(a.render, "crops", key, "index.json")))["crops"] if c["number"] == int(n)][0]
        rec = json.load(open(os.path.join(a.ocr, key, f"{tr}-{n}.json")))
        todo.append((items[pid], jobs[key], crop, rec))
    print("suspects", len(sus), "todo", len(todo), flush=True)
    flog = open(a.log, "a")

    async def run(t):
        r = await repair(a, *t)
        flog.write(json.dumps(r, ensure_ascii=False) + "\n"); flog.flush()
        print(r["id"], "adopted", r["adopted"], {k: v["n_correct"] for k, v in r["candidates"].items()},
              "extra", [s["pred"] for s in r.get("extra_samples", [])], flush=True)
        done[r["id"]] = r
    await asyncio.gather(*[run(t) for t in todo])
    ov, extra = {}, {}
    for pid, r in done.items():
        before = by.get(pid, [])
        e = dict(suspect_repair=dict(candidates=r["candidates"], adopted=r["adopted"], proofread_errors=r["v_errors"][:600],
                                     current_text_layer_score=r["current_text_layer_score"]))
        nb = len(before)
        if r["adopted"]:
            ov[pid] = r["override"]
            s = r["solve_after"]; nc = sum(x["correct"] for x in s)
            e["solve_check_deepseek_before_repair"] = dict(n=nb, n_correct=0, preds=[x["pred"] for x in before])
            e["solve_check_deepseek"] = dict(model="deepseek/deepseek-v4.1-flash (ultrafast/eco alternated)", n=len(s), n_correct=nc,
                                             accuracy=round(nc / len(s), 3), preds=[x["pred"] for x in s], models=[x["model"] for x in s],
                                             finish=[x["finish"] for x in s], suspect=nc == 0, after_repair=True)
        else:
            s = sorted(before, key=lambda x: x["sample"]) + r["extra_samples"]; nc = sum(x["correct"] for x in s)
            e["solve_check_deepseek_pass8"] = dict(n=len(s), n_correct=nc, preds=[x["pred"] for x in s])
            e["suspect_unresolved"] = nc == 0
        extra[pid] = e
    json.dump(ov, open(a.ov, "w"), ensure_ascii=False, indent=1)
    json.dump(extra, open(a.extra, "w"), ensure_ascii=False, indent=1)
    print("adopted", len(ov), "of", len(done), "| still 0 after pass@8:", sum(1 for e in extra.values() if e.get("suspect_unresolved")))


if __name__ == "__main__":
    asyncio.run(main())
