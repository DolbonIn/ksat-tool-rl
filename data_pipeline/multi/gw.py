"""Shared opengateway client: DeepSeek V4.1 Flash (ultrafast / eco), alternating models, global concurrency cap,
retry with backoff on 429 / 5xx / timeouts, token-usage logging. The API key is read from the environment and never printed.
"""
import asyncio, json, os, random, time
from openai import AsyncOpenAI

# Per user instruction (mid-run): eco only from now on. Earlier outputs from ultrafast are kept and their model is recorded.
MODELS = ["deepseek/deepseek-v4.1-flash-eco"]
ULTRAFAST = "deepseek/deepseek-v4.1-flash-ultrafast"   # no longer used
_client = None
_sem = None
_usage_path = None
_rr = 0
_cool = {}          # model -> time until which we avoid it (after a concurrency 429)


def init(concurrency=40, usage_log=None):
    global _client, _sem, _usage_path
    key = os.environ.get("OPENGATEWAY_API_KEY")
    if not key:
        raise SystemExit("OPENGATEWAY_API_KEY not set")
    _client = AsyncOpenAI(base_url=os.environ.get("OPENGATEWAY_BASE_URL", "https://apis.opengateway.ai/v1"), api_key=key,
                          timeout=900, max_retries=0)
    _sem = asyncio.Semaphore(concurrency)
    _usage_path = usage_log


def _log(rec):
    if _usage_path:
        with open(_usage_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _pick(prefer=None):
    global _rr
    now = time.time()
    order = MODELS[:] if prefer is None else ([prefer] + [m for m in MODELS if m != prefer])
    if prefer is None:
        _rr += 1
        if _rr % 2:
            order = order[::-1]
    ok = [m for m in order if _cool.get(m, 0) <= now]
    return ok[0] if ok else min(order, key=lambda m: _cool.get(m, 0))


async def chat(messages, max_tokens=8192, temperature=0.0, thinking=False, tag="", prefer=None, retries=8, strict=False):
    """Returns dict(content, reasoning, finish, model, usage, error)."""
    last = None
    for attempt in range(retries):
        if prefer not in MODELS:
            prefer = None
        model = prefer if (strict and prefer) else _pick(prefer)
        wait = _cool.get(model, 0) - time.time()
        if wait > 0:
            await asyncio.sleep(min(wait, 30))
        async with _sem:
            t0 = time.time()
            try:
                kw = dict(model=model, messages=messages, max_tokens=max_tokens)
                if temperature is not None:
                    kw["temperature"] = temperature
                if not thinking:
                    kw["extra_body"] = {"thinking": {"type": "disabled"}}
                r = await _client.chat.completions.create(**kw)
                m = r.choices[0].message
                u = r.usage.model_dump() if r.usage else {}
                _log(dict(t=time.time(), tag=tag, model=model, prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens"),
                          sec=round(time.time() - t0, 1)))
                return dict(content=m.content or "", reasoning=getattr(m, "reasoning_content", None) or "",
                            finish=r.choices[0].finish_reason, model=model,
                            usage=dict(prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens")), error=None)
            except Exception as e:
                s = f"{type(e).__name__}: {str(e)[:300]}"
                last = s
                code = getattr(e, "status_code", None)
                if "concurrency" in s or code == 429:
                    _cool[model] = time.time() + 5 + random.random() * 10
                    await asyncio.sleep(1 + random.random() * 3)
                    continue
                if code in (400, 401, 403, 404) and "timeout" not in s.lower():
                    _log(dict(t=time.time(), tag=tag, model=model, error=s[:200]))
                    return dict(content="", reasoning="", finish="error", model=model, usage={}, error=s)
                await asyncio.sleep(min(60, 3 * 2 ** attempt) + random.random() * 3)
    _log(dict(t=time.time(), tag=tag, error=(last or "")[:200]))
    return dict(content="", reasoning="", finish="error", model=None, usage={}, error=last)
