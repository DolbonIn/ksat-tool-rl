"""DeepSeek V4.1 Flash OCR for 2017-2021 가형/나형 crops (opengateway, models alternated by multi/gw.py, <=32 in flight).

Per item: pass A (transcribe; image only, thinking off, temperature 0; PROMPT_A of data_pipeline/ocr_vlm.py) ->
pass B (proofread vs image, thinking on) -> selection between A and B by agreement with the official PDF text layer
(Hangul multiset + digits, score of data_pipeline/repair_ocr.py) -> if still flagged: K extra pass-A samples
(temperature 0.4) -> best by score -> pass B -> selection again. Everything is kept in the record.
Output: OCR/<key>/<name>.json  {a, a_parsed, b, b_changed, b_errors, final, select, alts}
Usage: python -I ocr_old.py --jobs /data/work3/jobs.json --render /data/render3 --out /data/ocr3 --points 4 [--keys k1,k2] [--names k/n,...]
"""
import argparse, asyncio, json, os, re, sys, time
sys.path.insert(0, "/pipe"); sys.path.insert(0, "/pipe/multi")
import gw  # noqa: E402
# model policy (user instruction 2026-11-18): eco only from now on; override with GW_MODELS=a,b
gw.MODELS = os.environ.get("GW_MODELS", "deepseek/deepseek-v4.1-flash-eco").split(",")
import ocr_vlm  # noqa: E402
from repair_ocr import score, bad  # noqa: E402

NOTE_A = ("\n\n추가 규칙: 분수·근호·적분처럼 위아래로 큰 수식이 첫 줄에 있어도 이미지에 보이는 그대로 옮기고, "
          "이미지 가장자리에서 잘려 보이는 부분이 있으면 추측해 채우지 말고 <stem> 맨 끝에 '[잘림]'이라고 적으세요.")


TL_HEAD = ("아래 [텍스트 레이어]는 이 문항 영역에서 PDF 텍스트 레이어를 추출한 것입니다. 한국어 문장, <보기>·조건 상자의 글, 일부 숫자는 "
           "정확하지만 수식 기호는 '□'로 빠져 있고 줄 순서가 섞여 있을 수 있습니다. 한국어 문장은 [텍스트 레이어]의 글자를 그대로 따르되 "
           "하나도 빠뜨리지 말고(특히 <보기>의 ㄱ, ㄴ, ㄷ 항목과 (가), (나) 조건), 수식·배치·그림은 이미지에서 읽으세요.\n\n")
PUA = {0xE034 + i: str((i + 1) % 10) for i in range(10)}


def tl_text(ref):
    t = ref.translate(PUA)
    t = re.sub(r"[\ue000-\uf8ff]+", "□", t)
    return re.sub(r"[ \t]+", " ", t).strip()


async def pass_tl(path, ref, tag):
    prompt = ocr_vlm.PROMPT_A + NOTE_A + "\n\n" + TL_HEAD + "[텍스트 레이어]\n" + tl_text(ref) + "\n[/텍스트 레이어]"
    r = await gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": prompt}]}],
                      max_tokens=8192, temperature=0.0, thinking=False, tag=tag)
    return slim(r)


async def do_tl(args, key, crop):
    """Text-layer-assisted transcription as an extra candidate ('tl'), then proofread + re-selection."""
    fn = os.path.join(args.out, key, crop["name"] + ".json")
    rec = json.load(open(fn))
    if any(x.get("tag") == "tl" for x in rec.get("alts", [])) and not args.force:
        return
    path = os.path.join(args.render, crop["png"]); ref = crop["ref"]; tag = f"ocr:{key}/{crop['name']}:tl"
    o = await pass_tl(path, ref, tag)
    if o["finish"] != "stop":
        print("tl failed", key, crop["name"], o["error"], flush=True); return
    alt = dict(tag="tl", raw=o, parsed=ocr_vlm.parse(o["content"]))
    b = await pass_b(path, alt["parsed"], tag + "b")
    tmp = ocr_vlm.finalize({"a_parsed": alt["parsed"], "b": b})
    alt["b"] = b; alt["b_errors"] = tmp.get("b_errors")
    alt["final"] = tmp["final"] if tmp.get("b_changed") and score(tmp["final"], ref)[0] <= score(alt["parsed"], ref)[0] else alt["parsed"]
    rec.setdefault("alts", []).append(alt)
    rec["flagged"] = bool(select(rec, ref))
    json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
    print(f"TL {key}/{crop['name']}: sel={rec['select']['chosen']} sc={rec['select']['scores']} flag={rec['flagged']}", flush=True)


def img(path):
    return {"type": "image_url", "image_url": {"url": ocr_vlm.img_url(path)}}


def slim(r):
    return dict(content=r["content"], finish=r["finish"], model=r["model"], usage=r["usage"], error=r["error"],
                reasoning_len=len(r.get("reasoning") or ""))


async def pass_a(path, tag, temperature=0.0):
    r = await gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": ocr_vlm.PROMPT_A + NOTE_A}]}],
                      max_tokens=8192, temperature=temperature, thinking=False, tag=tag)
    return slim(r)


async def pass_b(path, parsed, tag):
    r = await gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": ocr_vlm.PROMPT_B + ocr_vlm.fmt(parsed)}]}],
                      max_tokens=32768, temperature=0.3, thinking=True, tag=tag)
    return slim(r)


def select(rec, ref):
    cands = {"a": rec.get("a_parsed")}
    if rec.get("b"):
        tmp = ocr_vlm.finalize({"a_parsed": rec["a_parsed"], "b": rec["b"]})
        rec["b_changed"], rec["b_errors"] = tmp.get("b_changed"), tmp.get("b_errors")
        if tmp.get("b_changed"):
            cands["b"] = tmp["final"]
    for i, alt in enumerate(rec.get("alts", [])):
        cands[f"alt{i}"] = alt.get("final") or alt.get("parsed")
    scored = {k: score(v, ref) for k, v in cands.items() if v}
    if rec.get("number", 99) <= 21:            # 1-21 are 5지선다형: a candidate without exactly 5 choices is not acceptable
        for k, v in cands.items():
            if k in scored and sum(1 for x in (v.get("choices_list") or []) if x) != 5:
                scored[k] = (scored[k][0] + 100, dict(scored[k][1], bad_choices=True))
    order = {"b": 0, "a": 1}
    if not ref:      # no text layer (2021_edu_04): no oracle -> proofread B if it changed something, else A (choice-count rule kept)
        best = min((k for k in scored if k in ("a", "b")), key=lambda k: (scored[k][0] >= 100, order[k]))
        rec["final"] = cands[best]
        rec["select"] = dict(chosen=best, scores={}, detail={}, repaired=False, rule="no text layer: b if changed else a")
        return False
    best = min(scored, key=lambda k: (scored[k][0], order.get(k, 2)))
    rec["final"] = cands[best]
    rec["select"] = dict(chosen=best, scores={k: v[0] for k, v in scored.items()}, detail=scored[best][1],
                         repaired=best.startswith("alt"))
    return bad(scored[best][1], scored[best][0])


async def do_item(args, key, crop):
    od = os.path.join(args.out, key); os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, crop["name"] + ".json")
    rec = json.load(open(fn)) if os.path.exists(fn) else {"name": crop["name"], "number": crop["number"], "png": crop["png"]}
    path = os.path.join(args.render, crop["png"]); ref = crop["ref"]; tag = f"ocr:{key}/{crop['name']}"
    t0 = time.time()
    if "a" not in rec or rec["a"]["finish"] != "stop" or args.force:
        rec["a"] = await pass_a(path, tag + ":a")
        rec["a_parsed"] = ocr_vlm.parse(rec["a"]["content"])
        rec.pop("b", None); rec.pop("alts", None)
        json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
    if rec["a"]["finish"] != "stop":
        print("A failed", key, crop["name"], rec["a"]["error"], flush=True); return
    if "b" not in rec or rec["b"]["finish"] == "error":
        rec["b"] = await pass_b(path, rec["a_parsed"], tag + ":b")
    flagged = select(rec, ref)
    if flagged and args.k and not rec.get("alts"):
        outs = await asyncio.gather(*[pass_a(path, tag + f":alt{i}", temperature=0.4) for i in range(args.k)])
        alts = [dict(raw=o, parsed=ocr_vlm.parse(o["content"])) for o in outs if o["finish"] == "stop"]
        alts.sort(key=lambda x: score(x["parsed"], ref)[0])
        if alts:
            best = alts[0]
            b = await pass_b(path, best["parsed"], tag + ":altb")
            tmp = ocr_vlm.finalize({"a_parsed": best["parsed"], "b": b})
            best["b"] = b; best["b_errors"] = tmp.get("b_errors")
            best["final"] = tmp["final"] if tmp.get("b_changed") and score(tmp["final"], ref)[0] <= score(best["parsed"], ref)[0] else best["parsed"]
            rec["alts"] = alts[:3]
        flagged = select(rec, ref)
    rec["flagged"] = bool(flagged)
    json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
    print(f"{key}/{crop['name']}: {time.time()-t0:.0f}s sel={rec['select']['chosen']} sc={rec['select']['scores']} flag={flagged}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default=None); ap.add_argument("--names", default=None, help="key/name,...")
    ap.add_argument("--points", default="4", help="comma list of point values to process (from the paper text layer)")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--force", action="store_true")
    ap.add_argument("--tl", action="store_true", help="text-layer-assisted pass for flagged items (or --names)")
    ap.add_argument("--concurrency", type=int, default=30); ap.add_argument("--usage-log", default="/data/work3/usage.jsonl")
    args = ap.parse_args()
    gw.init(args.concurrency, args.usage_log)
    pts = {int(x) for x in args.points.split(",")} if args.points else None
    names = set(args.names.split(",")) if args.names else None
    tasks = []
    for j in json.load(open(args.jobs)):
        if args.keys and j["key"] not in args.keys.split(","):
            continue
        idx = os.path.join(args.render, "crops", j["key"], "index.json")
        if not os.path.exists(idx):
            continue
        for c in json.load(open(idx))["crops"]:
            if names and f"{j['key']}/{c['name']}" not in names:
                continue
            p = c.get("points_override") or (c["points_text"][0] if len(c["points_text"]) == 1 else None)
            if not names and pts and p not in pts:
                continue
            if args.tl:
                fn = os.path.join(args.out, j["key"], c["name"] + ".json")
                if not os.path.exists(fn) or not c.get("ref"):
                    continue
                if names or json.load(open(fn)).get("flagged"):
                    tasks.append(do_tl(args, j["key"], c))
                continue
            tasks.append(do_item(args, j["key"], c))
    print("items:", len(tasks), flush=True)
    t0 = time.time()
    await asyncio.gather(*tasks)
    print(f"done {len(tasks)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
