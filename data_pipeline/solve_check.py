"""Solve-check transcribed problems: sample n solutions, compare the last \\boxed{} with the OFFICIAL answer.

Items where no sample matches are suspects for transcription errors (the official answer is never changed).

Providers:
  --provider deepseek : opengateway (OpenAI-compatible), model deepseek-v4.1-flash-ultrafast (fallback -eco).
                        Needs OPENGATEWAY_API_KEY / OPENGATEWAY_BASE_URL in the environment (docker --env-file .env).
                        The key is never printed.
  --provider local    : the local transcription VLM server (SGLang, served name "vlm"), text-only, thinking on.
                        This is a non-DeepSeek sanity check, recorded separately.

Usage: python -I solve_check.py --in items.jsonl --out results.jsonl --provider deepseek --n 4
       python -I solve_check.py --ping --provider deepseek     # one tiny request, prints status only
"""
import argparse, asyncio, json, os, re, sys, time
from openai import AsyncOpenAI

INSTR_MC = "다음은 대학수학능력시험 수학 영역 객관식 문제입니다. 단계적으로 풀이한 뒤, 마지막에 정답 선택지 번호(1~5 중 하나)만 \\boxed{} 안에 넣으세요. 예: \\boxed{3}"
INSTR_SA = "다음은 대학수학능력시험 수학 영역 단답형 문제입니다. 단계적으로 풀이한 뒤, 마지막에 최종 답(0 이상 999 이하의 정수)만 \\boxed{} 안에 넣으세요."
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


def norm(x):
    x = x.replace("$", "").replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "").replace("\\text{", "{")
    return re.sub(r"\s+", "", x)


def grade(item, content):
    b = last_boxed(content)
    if b is None:
        return None, False
    b2 = b
    for k, v in CIRCLED.items():
        b2 = b2.replace(k, v)
    b2 = re.sub(r"^\(?([1-5])\)?\.?$", r"\1", norm(b2))
    if item["kind"] == "mc":
        if b2 == str(item["answer"]):
            return b, True
        ch = item.get("choices") or []
        if ch and norm(b) == norm(ch[item["answer"] - 1]):
            return b, True
        return b, False
    m = re.fullmatch(r"-?\d+", b2.replace(",", ""))
    return b, bool(m and int(m.group()) == item["answer"])


def make_client(provider):
    if provider == "deepseek":
        base = os.environ.get("OPENGATEWAY_BASE_URL", "https://apis.opengateway.ai/v1")
        key = os.environ.get("OPENGATEWAY_API_KEY")
        if not key:
            sys.exit("OPENGATEWAY_API_KEY not set")
        # the gateway requires canonical owner-prefixed ids (bare names only resolve if implicit resolution is enabled)
        return AsyncOpenAI(base_url=base, api_key=key, timeout=3600), ["deepseek/deepseek-v4.1-flash-ultrafast", "deepseek/deepseek-v4.1-flash-eco"]
    return AsyncOpenAI(base_url=os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:31801/v1"), api_key="x", timeout=6 * 3600), ["vlm"]


async def one(client, models, sem, item, k, args):
    instr = INSTR_MC if item["kind"] == "mc" else INSTR_SA
    msg = [{"role": "user", "content": f"{instr}\n\n{item['text']}"}]
    async with sem:
        last_err = None
        rot = k % len(models)                      # spread samples across models (deepseek: ultrafast / eco)
        for model in models[rot:] + models[:rot]:
            for attempt in range(3):
                try:
                    kw = dict(model=model, messages=msg, max_tokens=args.max_tokens, temperature=0.6, top_p=0.95)
                    if args.provider == "local":
                        kw["extra_body"] = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
                    r = await client.chat.completions.create(**kw)
                    content = r.choices[0].message.content or ""
                    pred, ok = grade(item, content)
                    return dict(id=item["id"], sample=k, model=model, pred=pred, correct=ok,
                                finish=r.choices[0].finish_reason, tokens=getattr(r.usage, "completion_tokens", None), content=content[-3000:])
                except Exception as e:
                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
                    if "budget" in last_err or "401" in last_err or "403" in last_err or "non_canonical" in last_err:
                        break
                    await asyncio.sleep(5 * 2 ** attempt)   # backoff on 429 / 5xx / timeouts
        return dict(id=item["id"], sample=k, model=None, pred=None, correct=False, finish="error", error=last_err)


async def ping(args):
    client, models = make_client(args.provider)
    for m in models:
        try:
            r = await client.chat.completions.create(model=m, messages=[{"role": "user", "content": "1+1=? 숫자만"}], max_tokens=8)
            print(m, "OK", repr((r.choices[0].message.content or "")[:20]))
        except Exception as e:
            print(m, "FAIL", type(e).__name__, str(e)[:160])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp"); ap.add_argument("--out")
    ap.add_argument("--provider", choices=["deepseek", "local"], default="deepseek")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--concurrency", type=int, default=64)  # keep <= 64 in flight for the gateway
    ap.add_argument("--max-tokens", type=int, default=32768); ap.add_argument("--ids", default=None)
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()
    if args.ping:
        return await ping(args)
    client, models = make_client(args.provider)
    items = [json.loads(l) for l in open(args.inp)]
    if args.ids:
        keep = set(args.ids.split(","))
        items = [x for x in items if x["id"] in keep]
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    done = {}
    if os.path.exists(args.out):                       # resume: keep finished (non-error) samples
        for l in open(args.out):
            r = json.loads(l)
            if r.get("finish") != "error":
                done[(r["id"], r["sample"])] = r
    with open(args.out, "w") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fout = open(args.out, "a")

    async def run(it, k):
        r = await one(client, models, sem, it, k, args)
        fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush()
        return r

    res = list(done.values()) + await asyncio.gather(*[run(it, k) for it in items for k in range(args.n) if (it["id"], k) not in done])
    fout.close()
    by = {}
    for r in res:
        by.setdefault(r["id"], []).append(r)
    sus = [i for i, rs in by.items() if not any(r["correct"] for r in rs)]
    errs = sum(r["finish"] == "error" for r in res)
    print(f"{len(items)} items x {args.n}: {len(sus)} with no correct sample, {errs} errors, {time.time()-t0:.0f}s")
    print("suspects:", ",".join(sus))


if __name__ == "__main__":
    asyncio.run(main())
