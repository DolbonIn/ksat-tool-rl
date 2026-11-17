"""Solve 2026 CSAT math in parallel against a local SGLang OpenAI-compatible server."""
import argparse, asyncio, json, re, time, collections
from openai import AsyncOpenAI
from problems import PROBLEMS

SYS = None
INSTR_MC = "다음은 2026학년도 대학수학능력시험 수학 영역 객관식 문제입니다. 단계적으로 풀이한 뒤, 마지막에 정답 선택지 번호(1~5 중 하나)만 \\boxed{} 안에 넣으세요. 예: \\boxed{3}"
INSTR_SA = "다음은 2026학년도 대학수학능력시험 수학 영역 단답형 문제입니다. 단계적으로 풀이한 뒤, 마지막에 최종 답(0 이상 999 이하의 정수)만 \\boxed{} 안에 넣으세요."
CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}

def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    d, k = 0, j
    while k < len(s):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[j+1:k].strip()
        k += 1
    return None

def norm(x):
    x = x.replace("$", "").replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", x)

def grade(p, content):
    b = last_boxed(content or "")
    if b is None: return None, False
    b2 = b
    for k, v in CIRCLED.items(): b2 = b2.replace(k, v)
    b2 = re.sub(r"^\(?([1-5])\)?\.?$", r"\1", norm(b2))
    if p["kind"] == "mc":
        if b2 == str(p["answer"]): return b, True
        # also accept the value of the correct choice
        if norm(b) == norm(p["choices"][p["answer"]-1]): return b, True
        return b, False
    m = re.fullmatch(r"-?\d+", b2.replace(",", ""))
    return b, bool(m and int(m.group()) == p["answer"])

async def run_one(client, sem, args, p, k):
    instr = INSTR_MC if p["kind"] == "mc" else INSTR_SA
    async with sem:
        t0 = time.time()
        try:
            r = await client.chat.completions.create(
                model="nanbeige", messages=[{"role": "user", "content": f"{instr}\n\n{p['text']}"}],
                temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})
            msg = r.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            toks = r.usage.completion_tokens; finish = r.choices[0].finish_reason
        except Exception as e:
            content, reasoning, toks, finish = "", "", 0, f"error: {e}"
        pred, ok = grade(p, content)
        print(f"[{p['id']:>14} #{k}] {'O' if ok else 'X'} pred={pred!s:<12} ans={p['answer']} toks={toks} {time.time()-t0:.0f}s", flush=True)
        return dict(id=p["id"], sample=k, pred=pred, correct=ok, tokens=toks, finish=finish, content=content, reasoning=reasoning)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=65536); ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()
    client = AsyncOpenAI(base_url="http://127.0.0.1:30000/v1", api_key="x", timeout=6*3600)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    res = await asyncio.gather(*[run_one(client, sem, args, p, k) for p in PROBLEMS for k in range(args.n)])
    with open(args.out, "w") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal wall time: {time.time()-t0:.0f}s, {len(res)} generations")

if __name__ == "__main__":
    asyncio.run(main())
