"""Parallel MC solver for CSAT 국어/영어/과탐 units (answer = choice number 1-5)."""
import argparse, asyncio, json, re, time
from openai import AsyncOpenAI
from solve import last_boxed, CIRCLED

SUBJ = {"korean": "국어", "english": "영어", "science": "과학탐구"}
def prompt(u):
    head = f"다음은 2026학년도 대학수학능력시험 {SUBJ[u['exam']]} 영역 {u['no']}번 문제입니다."
    if u.get("note"): head += " " + u["note"]
    return (f"{head} 제시된 지문/자료를 바탕으로 {u['no']}번 문제를 풀고, 마지막에 정답 선택지 번호(1~5 중 하나)만 "
            f"\\boxed{{}} 안에 넣으세요. 예: \\boxed{{3}}\n\n{u['text']}")

SYS_V2 = ("당신은 한국 수능 문제를 푸는 수험생입니다. 풀이를 마친 뒤 응답의 맨 마지막 줄에는 반드시 "
          "\\boxed{N} 형식으로 정답 번호를 쓰세요. N은 1, 2, 3, 4, 5 중 하나의 아라비아 숫자이며, "
          "①②③④⑤ 같은 원문자나 '정답: ④' 같은 다른 표기는 쓰지 마세요.")
REMIND_V2 = ("\n\n[답안 형식] 풀이 후 마지막 줄에 \\boxed{N} (N은 1~5의 아라비아 숫자) 형식으로만 정답을 쓰세요. "
             "예: \\boxed{4}")
DEMO_Q = ("다음은 예시 문제입니다. 문제를 풀고, 마지막에 정답 선택지 번호(1~5 중 하나)만 \\boxed{} 안에 넣으세요.\n\n"
          "1. 다음 중 짝수인 것은?\n① 1  ② 3  ③ 4  ④ 5  ⑤ 7")
DEMO_A = "선택지 중 2로 나누어떨어지는 수는 4뿐이므로 ③이 정답이다.\n\n\\boxed{3}"
def messages(u, v2):
    if v2 == "v3":
        return [{"role": "user", "content": DEMO_Q}, {"role": "assistant", "content": DEMO_A},
                {"role": "user", "content": prompt(u)}]
    if not v2: return [{"role": "user", "content": prompt(u)}]
    return [{"role": "system", "content": SYS_V2}, {"role": "user", "content": prompt(u) + REMIND_V2}]

def parse(content):
    b = last_boxed(content or "")
    if b is None: return None
    for k, v in CIRCLED.items(): b = b.replace(k, v)
    m = re.search(r"[1-5]", b)
    return int(m.group()) if m else None

async def run_one(client, sem, args, u, k):
    async with sem:
        t0 = time.time()
        try:
            r = await client.chat.completions.create(
                model=args.model, messages=messages(u, args.format_v2),
                temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens,
                extra_body={"top_k": args.top_k, "chat_template_kwargs": {"enable_thinking": True}})
            m = r.choices[0].message
            content, reasoning = m.content or "", getattr(m, "reasoning_content", None) or ""
            toks, finish = r.usage.completion_tokens, r.choices[0].finish_reason
        except Exception as e:
            content, reasoning, toks, finish = "", "", 0, f"error: {e}"
        pred = parse(content); ok = pred == u["answer"]
        print(f"[{u['id']:>24} #{k}] {'O' if ok else 'X'} pred={pred} ans={u['answer']} toks={toks} {time.time()-t0:.0f}s", flush=True)
        return dict(id=u["id"], sample=k, pred=pred, correct=ok, tokens=toks, finish=finish, content=content, reasoning=reasoning)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units"); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--port", type=int, default=30000); ap.add_argument("--model", default="nanbeige")
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--format-v2", action="store_const", const="v2", default=None); ap.add_argument("--format-v3", dest="format_v2", action="store_const", const="v3"); ap.add_argument("--top-k", type=int, default=20); ap.add_argument("--max-tokens", type=int, default=65536)
    args = ap.parse_args()
    units = json.load(open(args.units))
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="x", timeout=6*3600)
    sem = asyncio.Semaphore(1024); t0 = time.time()
    res = await asyncio.gather(*[run_one(client, sem, args, u, k) for u in units for k in range(args.n)])
    with open(args.out, "w") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal wall time: {time.time()-t0:.0f}s, {len(res)} generations")

if __name__ == "__main__":
    asyncio.run(main())
