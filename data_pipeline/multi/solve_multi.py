"""DeepSeek solve check for MC items: n samples per item, final answer in \\boxed{}, compared with the OFFICIAL answer.
Items with 0/n correct are transcription suspects (the official answer is never changed). Resumable.

Usage: python -I solve_multi.py --in /data/ksat_multi/kor_test_2026.jsonl --out /data/work2/solve/kor_test.jsonl --n 4
"""
import argparse, asyncio, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw  # noqa: E402

AREA = {"kor": "국어", "eng": "영어", "sci": "과학탐구", "soc": "사회탐구"}
CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}


def last_boxed(s):
    i = (s or "").rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1:k].strip()
    return None


def pred_of(content):
    b = last_boxed(content or "")
    if b is None:
        return None
    for k, v in CIRCLED.items():
        b = b.replace(k, v)
    m = re.search(r"[1-5]", b)
    return int(m.group()) if m else None


def prompt(it):
    head = f"다음은 대학수학능력시험(또는 모의평가) {AREA[it['area']]} 영역 {it['number']}번 문제입니다."
    if it.get("has_figure"):
        head += " 그림·그래프·표는 [그림]…[/그림], [표]…[/표] 형식의 텍스트로 옮겨져 있습니다."
    return (f"{head} 제시된 지문/자료를 바탕으로 문제를 풀고, 마지막 줄에 정답 선택지 번호(1~5 중 하나)만 \\boxed{{}} 안에 넣으세요. "
            f"예: \\boxed{{3}}\n\n{it['text']}")


async def one(it, k, args):
    r = await gw.chat([{"role": "user", "content": prompt(it)}], max_tokens=args.max_tokens, temperature=0.6, thinking=True,
                      tag=f"solve:{it['id']}", prefer=gw.MODELS[k % len(gw.MODELS)] if args.alternate else args.prefer)
    p = pred_of(r["content"])
    return dict(id=it["id"], sample=k, model=r["model"], pred=p, correct=p == it["answer"], finish=r["finish"],
                usage=r["usage"], error=r["error"], content=(r["content"] or "")[-1500:])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16000); ap.add_argument("--ids", default=None)
    ap.add_argument("--prefer", default=None); ap.add_argument("--alternate", action="store_true")
    ap.add_argument("--redo-length", action="store_true", help="re-sample items whose earlier sample hit max_tokens")
    ap.add_argument("--redo-ids", default=None, help="file/comma list of ids whose existing samples are discarded (text changed)")
    ap.add_argument("--usage-log", default="/data/work2/usage.jsonl")
    a = ap.parse_args()
    gw.init(a.concurrency, a.usage_log)
    items = [json.loads(l) for f in a.inp.split(",") for l in open(f)]
    if a.ids:
        keep = set(open(a.ids).read().split()) if os.path.exists(a.ids) else set(a.ids.split(","))
        items = [x for x in items if x["id"] in keep]
    done = {}
    redo = set()
    if a.redo_ids:
        redo = set(open(a.redo_ids).read().replace(",", " ").split()) if os.path.exists(a.redo_ids) else set(a.redo_ids.split(","))
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r.get("finish") == "error" or r["id"] in redo:
                continue
            if r.get("finish") == "length" and a.redo_length:
                continue
            if r.get("pred") is not None or r.get("finish") == "length":
                done[(r["id"], r["sample"])] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fo = open(a.out, "a")
    t0 = time.time()
    cnt = [0]

    async def run(it, k):
        r = await one(it, k, a)
        fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
        cnt[0] += 1
        if cnt[0] % 200 == 0:
            print(f"{cnt[0]} done {time.time()-t0:.0f}s", flush=True)
        return r

    res = list(done.values()) + await asyncio.gather(*[run(it, k) for it in items for k in range(a.n) if (it["id"], k) not in done])
    fo.close()
    by = {}
    for r in res:
        by.setdefault(r["id"], []).append(r)
    sus = sorted(i for i, rs in by.items() if not any(r["correct"] for r in rs))
    acc = sum(r["correct"] for r in res) / max(1, len(res))
    print(f"{len(items)} items x {a.n}: acc {acc:.3f}, {len(sus)} with 0/{a.n}, errors {sum(r['finish'] == 'error' for r in res)}, "
          f"no-box {sum(r['pred'] is None for r in res)}, {time.time()-t0:.0f}s")
    print("suspects:", ",".join(sus))


if __name__ == "__main__":
    asyncio.run(main())
