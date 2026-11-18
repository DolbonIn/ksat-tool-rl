"""Decode throughput of an SGLang server: C concurrent requests, fixed output length (ignore_eos)."""
import asyncio, json, sys, time, aiohttp
URL, C, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PROMPT = "다음은 대학수학능력시험 수학 영역 단답형 문제입니다. 함수 f(x)=x^3-3x^2+2 에 대하여 f(x)=0 의 실근의 합을 구하시오."
async def one(s):
    body = {"text": f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            "sampling_params": {"temperature": 0.8, "max_new_tokens": L, "ignore_eos": True}}
    t = time.time()
    async with s.post(URL + "/generate", json=body) as r:
        d = await r.json()
    return d["meta_info"]["completion_tokens"], time.time() - t
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as s:
        await one(s)  # warmup
        t0 = time.time(); res = await asyncio.gather(*[one(s) for _ in range(C)]); dt = time.time() - t0
    toks = sum(r[0] for r in res)
    print(json.dumps({"concurrency": C, "len": L, "wall_s": round(dt, 1), "total_tok_s": round(toks / dt), "per_seq_tok_s": round(L / max(r[1] for r in res), 1)}))
asyncio.run(main())
