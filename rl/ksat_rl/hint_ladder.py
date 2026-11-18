"""Hint ladders + verified reference solutions for hard items (input for hint-guided STaR, strategy 3).

Per hard item (policy pass@8 <= 2/8, or killer-position real items <= 4/8; never test/val):
  1. SOLVE   deepseek-v4.1-flash-eco solves it N times (65k budget); keep solutions whose final \\boxed{}
             matches the official / consensus answer (graded with ksat_rl.reward.grade).
  2. LADDER  from the shortest correct solution, eco writes three hints of increasing strength:
             h1 = the decisive idea (1-2 sentences), h2 = solution direction + first steps,
             h3 = near-complete outline stopping before the final computation. No final answer.
  3. LEAK    reject a ladder that states the final answer (sa: the integer as a standalone token in h1/h2,
             or anywhere in h3 together with "답"; mc: "정답"/"답은" with a choice marker); one retry.
Later (GPU phase): the policy is rolled out WITH a hint; its own correct solutions become training data
for the hint-free prompt (self-generated rationalisation, keeps Nanbeige's thinking style).
Output JSONL rows: id, kind, answer (grader only), hints, ref_solution, ref_reasoning, eco_n_correct, eco_n.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ksat_rl.make_parquet import instruction  # noqa: E402
from ksat_rl.reward import grade  # noqa: E402
from ksat_rl.synth_variants import parse_json  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash-eco"

LADDER_PROMPT = """다음은 대학수학능력시험 수학 문항과, 정답이 검증된 풀이입니다.
이 문항을 혼자서는 풀지 못한 학생에게 줄 **단계별 힌트 3개**를 만드세요.

[문항]
{text}

[검증된 풀이]
{solution}

규칙:
- h1: 결정적인 아이디어나 관찰을 1~2문장으로. 계산 결과는 쓰지 않습니다.
- h2: 풀이의 방향과 처음 몇 단계(식 세우기, 경우 나누기 기준 등)를 3~5문장으로.
- h3: 마지막 계산 직전까지의 풀이 개요. 중간 결과는 써도 되지만 **최종 답(정답 값·정답 번호)은 절대 쓰지 마세요.**
- 힌트는 학생에게 말하듯 한국어로, LaTeX 수식은 그대로 사용하세요.

최종 출력은 아래 JSON 하나만 쓰세요(코드블록 없이):
{{"h1": "...", "h2": "...", "h3": "..."}}"""


class Client:
    def __init__(self, concurrency):
        self.c = AsyncOpenAI(base_url=os.environ["OPENGATEWAY_BASE_URL"], api_key=os.environ["OPENGATEWAY_API_KEY"], timeout=3600)
        self.sem = asyncio.Semaphore(concurrency)
        self.usage = {"prompt": 0, "completion": 0, "calls": 0, "errors": 0}

    async def chat(self, prompt, max_tokens=65536, temperature=0.7):
        for attempt in range(6):
            async with self.sem:
                try:
                    r = await self.c.chat.completions.create(
                        model=MODEL, messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens, temperature=temperature)
                    u = r.usage
                    self.usage["prompt"] += u.prompt_tokens; self.usage["completion"] += u.completion_tokens; self.usage["calls"] += 1
                    m = r.choices[0].message
                    reasoning = getattr(m, "reasoning_content", None) or (getattr(m, "model_extra", None) or {}).get("reasoning_content") or ""
                    return m.content or "", reasoning, r.choices[0].finish_reason
                except Exception as e:
                    self.usage["errors"] += 1
                    if "budget" in str(e).lower():
                        raise
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        return None, "", "error"


def leaks(hints, p):
    ans = int(p["answer"])
    if p["kind"] == "sa":
        tok = re.compile(rf"(?<![\d.]){ans}(?![\d.])")
        if ans >= 10 and (tok.search(hints.get("h1", "")) or tok.search(hints.get("h2", ""))):
            return True
        if tok.search(hints.get("h3", "")) and re.search(r"답|구하는 값|따라서", hints.get("h3", "")[-200:]):
            return True
        return False
    marker = "①②③④⑤"[ans - 1]
    return bool(re.search(rf"(정답|답은|답:)\s*\S{{0,6}}({marker}|{ans}번)", " ".join(hints.values())))


async def process(cli, p, args):
    instr = instruction(p)
    prompt = f"{instr}\n\n{p['text']}"
    sols = await asyncio.gather(*[cli.chat(prompt) for _ in range(args.n_solve)])
    good = []
    for content, reasoning, finish in sols:
        if content is None:
            continue
        pred, ok = grade(p["kind"], int(p["answer"]), p.get("choices"), content)
        if ok:
            good.append((len(reasoning) + len(content), content, reasoning))
    rec = {"id": p["id"], "kind": p["kind"], "answer": int(p["answer"]), "section": p["section"], "number": p.get("number"),
           "exam": p["exam"], "eco_n": args.n_solve, "eco_n_correct": len(good), "model": MODEL}
    if not good:
        rec["status"] = "eco_unsolved"
        return rec
    good.sort()
    _, content, reasoning = good[0]
    rec["ref_solution"], rec["ref_reasoning"] = content, reasoning
    for attempt in range(2):
        out, _, _ = await cli.chat(LADDER_PROMPT.format(text=p["text"], solution=content), max_tokens=32768, temperature=0.5)
        hints = parse_json(out or "")
        if hints and all(isinstance(hints.get(k), str) and hints[k].strip() for k in ("h1", "h2", "h3")):
            if not leaks(hints, p):
                rec["hints"] = {k: hints[k].strip() for k in ("h1", "h2", "h3")}
                rec["status"] = "ok"
                return rec
            rec["leak_retry"] = attempt + 1
    rec["status"] = "ladder_fail"
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="JSONL with all candidate problems (real + synth)")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--test", required=True, help="test JSONL; any overlap aborts")
    ap.add_argument("--n-solve", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep = [l.strip() for l in open(a.ids) if l.strip()]
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(a.pool) if l.strip()}
    test_ids = {json.loads(l)["id"] for l in open(a.test) if l.strip()}
    assert not set(keep) & test_ids, "test item in hint seeds"
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [pool[i] for i in keep if i in pool and i not in done]
    print(f"resume: {len(done)} done, {len(todo)} to run", flush=True)
    cli = Client(a.concurrency)
    t0 = time.time()
    stats = {}
    with open(a.out, "a") as fo:
        for fut in asyncio.as_completed([process(cli, p, a) for p in todo]):
            rec = await fut
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
    print(json.dumps({"stats": stats, "secs": round(time.time() - t0), "usage": cli.usage}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
