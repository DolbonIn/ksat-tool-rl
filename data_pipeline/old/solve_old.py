"""DeepSeek solve check (n samples per item) for the 2017-2021 set: compare the last \\boxed{} with the OFFICIAL answer.
Reuses prompts + grading of data_pipeline/solve_check.py; transport = multi/gw.py (models alternated per sample:
even samples prefer eco, odd prefer ultrafast; retry/backoff on 429/5xx). max_tokens 65536, thinking on, T=0.6.
Items with 0/n correct are transcription suspects (the official answer is never changed). Resumable.
Usage: python -I solve_old.py --in /data/ksat_math_old/train_2017_2021_hard.jsonl --out /data/work3/solve_hard.jsonl --n 4
"""
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, "/pipe"); sys.path.insert(0, "/pipe/multi")
import gw  # noqa: E402
# model policy (user instruction 2026-11-18): eco only from now on; override with GW_MODELS=a,b
gw.MODELS = os.environ.get("GW_MODELS", "deepseek/deepseek-v4.1-flash-eco").split(",")
from solve_check import INSTR_MC, INSTR_SA, grade  # noqa: E402


async def one(it, k, args, fout):
    instr = INSTR_MC if it["kind"] == "mc" else INSTR_SA
    r = await gw.chat([{"role": "user", "content": f"{instr}\n\n{it['text']}"}], max_tokens=args.max_tokens, temperature=0.6,
                      thinking=True, tag=f"solve:{it['id']}:{k}", prefer=gw.MODELS[k % len(gw.MODELS)], retries=10)
    pred, ok = grade(it, r["content"]) if r["finish"] != "error" else (None, False)
    rec = dict(id=it["id"], sample=k, tag=args.tag, model=r["model"], pred=pred, correct=bool(ok), finish=r["finish"],
               tokens=(r.get("usage") or {}).get("completion"), prompt_tokens=(r.get("usage") or {}).get("prompt"),
               reasoning_len=len(r.get("reasoning") or ""), error=r.get("error"), content=(r["content"] or "")[-2500:])
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--ids", default=None); ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--tag", default="main"); ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--usage-log", default="/data/work3/usage.jsonl")
    a = ap.parse_args()
    gw.init(a.concurrency, a.usage_log)
    items = [json.loads(l) for l in open(a.inp)]
    if a.ids:
        keep = set(open(a.ids).read().split()) if os.path.exists(a.ids) else set(a.ids.split(","))
        items = [x for x in items if x["id"] in keep]
    done = {}
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r.get("finish") != "error":
                done[(r["id"], r["sample"])] = r
    with open(a.out, "w") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fout = open(a.out, "a")
    t0 = time.time()
    todo = [(it, k) for it in items for k in range(a.sample_offset, a.sample_offset + a.n) if (it["id"], k) not in done]
    print("items", len(items), "samples to run", len(todo), flush=True)
    res = await asyncio.gather(*[one(it, k, a, fout) for it, k in todo])
    fout.close()
    allr = list(done.values()) + res
    by = {}
    for r in allr:
        by.setdefault(r["id"], []).append(r)
    sus = sorted(i for i, rs in by.items() if not any(r["correct"] for r in rs))
    print(f"{len(items)} items: {len(sus)} with 0 correct; errors {sum(r['finish'] == 'error' for r in res)}; "
          f"length {sum(r['finish'] == 'length' for r in res)}; {time.time() - t0:.0f}s")
    print("suspects:", ",".join(sus))


if __name__ == "__main__":
    asyncio.run(main())
