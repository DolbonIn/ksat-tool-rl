"""Evaluate a model on 2026 CSAT math (or any KSAT JSONL) against an SGLang OpenAI-compatible server.

Identical protocol to experiments/csat2026_baseline/code/solve.py + report.py so numbers are comparable
to the recorded baseline: same instructions (with the exam label), temperature 0.6 / top_p 0.95 /
top_k 20, thinking on, max_tokens 65536, n samples per item, grade = last \\boxed{} in content.
Writes per-generation JSONL and prints the expected / per-sample / majority scores per elective track.
"""
import argparse, asyncio, collections, json, os, sys, time
from openai import AsyncOpenAI
sys.path.insert(0, "/rl")
from ksat_rl.reward import grade
from ksat_rl.make_parquet import instruction

BASELINE_2026 = ("다음은 2026학년도 대학수학능력시험 수학 영역")


async def run_one(client, sem, args, p, k):
    instr = instruction(p)
    async with sem:
        t0 = time.time()
        try:
            r = await client.chat.completions.create(
                model=args.model_name, messages=[{"role": "user", "content": f"{instr}\n\n{p['text']}"}],
                temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                extra_body={"top_k": args.top_k, "chat_template_kwargs": {"enable_thinking": True}})
            msg = r.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            toks, finish = r.usage.completion_tokens, r.choices[0].finish_reason
        except Exception as e:  # infra errors are recorded separately, not as wrong answers
            content, reasoning, toks, finish = "", "", 0, f"error: {e}"
        pred, ok = grade(p["kind"], int(p["answer"]), p.get("choices"), content)
        return dict(id=p["id"], sample=k, pred=pred, correct=ok, tokens=toks, finish=finish,
                    secs=round(time.time() - t0, 1), content=content, reasoning=reasoning)


def report(problems, res, out=sys.stdout):
    by = collections.defaultdict(list)
    for r in res:
        by[r["id"]].append(r)
    n = max(r["sample"] for r in res) + 1
    maj = {}
    lines = [f"{'문항':<15}{'배점':>4}{'정답':>6}{'정답률':>8}{'다수결':>10}{'평균토큰':>10}"]
    for p in problems:
        rs = by[p["id"]]
        acc = sum(r["correct"] for r in rs) / len(rs)
        votes = collections.Counter(r["pred"] for r in rs if r["pred"] is not None)
        top = votes.most_common(1)[0][0] if votes else None
        maj[p["id"]] = next((r["correct"] for r in rs if r["pred"] == top), False)
        lines.append(f"{p['id']:<15}{p['points']:>4}{p['answer']:>6}{acc:>8.0%}{str(top)[:8]:>10}{sum(r['tokens'] for r in rs)/len(rs):>10.0f}")
    common = [p for p in problems if p["section"] == "common"]
    summary = {}
    lines.append("")
    for sec, name in [("prob_stat", "확률과 통계"), ("calculus", "미적분"), ("geometry", "기하")]:
        ps = common + [p for p in problems if p["section"] == sec]
        if len(ps) == len(common):
            continue
        exp = sum(p["points"] * sum(r["correct"] for r in by[p["id"]]) / len(by[p["id"]]) for p in ps)
        per = [sum(p["points"] for p in ps if next(r for r in by[p["id"]] if r["sample"] == k)["correct"]) for k in range(n)]
        mj = sum(p["points"] for p in ps if maj[p["id"]])
        summary[sec] = dict(expected=round(exp, 2), per_sample=per, majority=mj)
        lines.append(f"[{name}] 기대점수 {exp:.1f}/100 | 회차별 {per} (min {min(per)}, max {max(per)}) | 다수결({n}회) {mj}/100")
    trunc = sum(r["finish"] == "length" for r in res)
    err = sum(str(r["finish"]).startswith("error") for r in res)
    acc_all = sum(r["correct"] for r in res) / len(res)
    lines.append(f"\n잘림(length) {trunc}건, 오류 {err}건, 총 {len(res)}건, 전체 정답률 {acc_all:.4f}, 평균 토큰 {sum(r['tokens'] for r in res)/len(res):.0f}")
    print("\n".join(lines), file=out)
    summary.update(trunc=trunc, errors=err, acc=acc_all, n=n)
    return summary


def _ok(r):
    f = str(r.get("finish"))
    return not (f.startswith("error") or f == "abort")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="/data/ksat_math/test_2026.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model-name", default="nanbeige")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--no-report", action="store_true", help="skip CSAT score report (e.g. train-set pass-rate runs)")
    ap.add_argument("--reverse", action="store_true", help="process problems from the end of the file (to split work with another run)")
    ap.add_argument("--resume", action="store_true", help="skip (id, sample) pairs already completed in <out>.partial.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    problems = [json.loads(l) for l in open(args.problems) if l.strip()]
    client = AsyncOpenAI(base_url=args.url, api_key="x", timeout=6 * 3600)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    # Every finished generation is appended to <out>.partial.jsonl immediately, so a run can be stopped,
    # split across servers, or resumed without losing completed work.
    partial = args.out + ".partial.jsonl"
    done = {}
    if args.resume and os.path.exists(partial):
        for l in open(partial):
            r = json.loads(l)
            if _ok(r):
                done[(r["id"], r["sample"])] = r
    order = problems[::-1] if args.reverse else problems
    todo = [(p, k) for p in order for k in range(args.n) if (p["id"], k) not in done]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)
    fpart = open(partial, "a")

    async def run_and_log(p, k):
        r = await run_one(client, sem, args, p, k)
        fpart.write(json.dumps(r, ensure_ascii=False) + "\n"); fpart.flush()
        return r

    new = await asyncio.gather(*[run_and_log(p, k) for p, k in todo])
    fpart.close()
    res = list(done.values()) + list(new)
    with open(args.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not args.no_report:
        with open(args.out.replace(".jsonl", "") + ".report.txt", "w") as f:
            s = report(problems, res, f)
        report(problems, res)
        json.dump(s, open(args.out.replace(".jsonl", "") + ".summary.json", "w"), ensure_ascii=False, indent=1)
    print(f"wall {time.time()-t0:.0f}s, {len(res)} generations")


if __name__ == "__main__":
    asyncio.run(main())
