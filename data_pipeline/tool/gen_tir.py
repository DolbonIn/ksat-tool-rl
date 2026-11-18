"""Generate tool-integrated reasoning (TIR) trajectories for KSAT math with DeepSeek V4.1 Flash eco.

Flow per problem (pool.jsonl from build_pool.py; train items only):
  system + user prompt -> eco (OpenAI tools=[python]) -> tool_calls -> REAL execution in the sandbox service
  (ksat_rl.tools.python_sandbox, isolated container) -> tool message with the actual observation -> ... ->
  final answer in \\boxed{}.  <= 6 tool calls (after that tool_choice="none"), <= 8 assistant turns.
Graded with ksat_rl.reward.grade against the official answer (synthetic: DeepSeek-consensus label).

Kept (status "verified"): correct, >= 1 tool call, >= 1 call whose output re-appears in the following
reasoning (trace_utils.output_used), no chat/tool markup inside text or code, no truncated turn.
Every attempt (verified or rejected, with reasons) is appended to --raw; finalize_tir.py writes the
deliverables.  Re-running resumes: ids already in --raw are skipped unless their status is retryable.

Run inside a container attached to the internet (gateway) AND ksat-tool-sbx-net (sandbox); see run_gen.sh.
The API key is read from the environment and never printed.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

from openai import AsyncOpenAI

sys.path.insert(0, os.environ.get("KSAT_RL_DIR", "/rl"))
from ksat_rl.reward import grade  # noqa: E402
from ksat_rl.tools.python_sandbox import arun_python, format_observation  # noqa: E402
from ksat_rl.tools.schema import MAX_TOOL_CALLS, TOOL_SYSTEM_PROMPT, TOOLS, tool_user_prompt  # noqa: E402
from ksat_rl.tools.trace_utils import forbidden_markup, meta_tool_commentary, output_used  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash-eco"   # user instruction: eco only, never ultrafast
MAX_TURNS = 8
GEN_PROMPT_VERSION = "gen_v2"
# The teacher gets exactly the policy-facing system prompt (TOOL_SYSTEM_PROMPT), so stored traces never refer to
# instructions the policy does not see (gen_v1 appended teacher-only guidance; its reasoning echoed it).
GEN_GUIDANCE = ""

RETRYABLE = {"api_error", "sandbox_error"}


class Budget(Exception):
    pass


class Gen:
    def __init__(self, a):
        self.a = a
        self.client = AsyncOpenAI(api_key=os.environ["OPENGATEWAY_API_KEY"],
                                  base_url=os.environ.get("OPENGATEWAY_BASE_URL", "https://apis.opengateway.ai/v1"),
                                  timeout=a.request_timeout, max_retries=0)
        self.api_sem = asyncio.Semaphore(a.concurrency)      # <= 32 concurrent gateway requests
        self.lock = asyncio.Lock()
        self.stop = False
        self.n_verified = 0
        self.n_done = 0
        self.tokens = {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0, "requests": 0}
        self.t0 = time.time()

    async def chat(self, messages, tool_choice):
        last = None
        for attempt in range(self.a.api_retries):
            if self.stop:
                raise Budget("stopped")
            try:
                async with self.api_sem:
                    r = await self.client.chat.completions.create(
                        model=MODEL, messages=messages, tools=TOOLS, tool_choice=tool_choice,
                        max_tokens=self.a.max_tokens, temperature=self.a.temperature)
                if not r.choices:
                    raise RuntimeError("empty choices")
                return r
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                last = msg
                if "budget_exceeded" in msg or "insufficient" in msg.lower():
                    self.stop = True
                    raise Budget(msg)
                if "non_canonical_model_id" in msg or "invalid_request" in msg.lower() and "tool" not in msg.lower():
                    raise
                await asyncio.sleep(min(90, 2 * 2 ** attempt) * (0.5 + random.random()))
        raise RuntimeError(f"api failed after retries: {last}")

    def add_usage(self, u, rec):
        if u is None:
            return
        d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
        p, c = d.get("prompt_tokens") or 0, d.get("completion_tokens") or 0
        rt = ((d.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
        ca = ((d.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        for k, v in (("prompt", p), ("completion", c), ("reasoning", rt), ("cached", ca)):
            rec[k] = rec.get(k, 0) + v
            self.tokens[k] += v
        rec["requests"] = rec.get("requests", 0) + 1
        self.tokens["requests"] += 1

    async def solve(self, p):
        user = tool_user_prompt(p)
        api_msgs = [{"role": "system", "content": TOOL_SYSTEM_PROMPT + GEN_GUIDANCE}, {"role": "user", "content": user}]
        msgs = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, {"role": "user", "content": user}]
        runs, usage, reasons = [], {}, []
        n_calls, final, status = 0, None, None
        t0 = time.time()
        try:
            for turn in range(MAX_TURNS):
                force_final = n_calls >= MAX_TOOL_CALLS
                r = await self.chat(api_msgs, "none" if force_final else "auto")
                self.add_usage(r.usage, usage)
                ch = r.choices[0]
                m = ch.message.model_dump()
                reasoning = (m.get("reasoning_content") or "").strip()
                content = (m.get("content") or "").strip()
                tcs = m.get("tool_calls") or []
                if ch.finish_reason == "length":
                    reasons.append("turn_truncated_length")
                    status = "rejected"
                    break
                if tcs and force_final:
                    reasons.append("tool_call_after_budget")
                    status = "rejected"
                    break
                if tcs and n_calls + len(tcs) > MAX_TOOL_CALLS:
                    reasons.append("exceeds_tool_budget")
                    status = "rejected"
                    break
                if tcs:
                    calls = []
                    for tc in tcs:
                        calls.append({"id": tc["id"], "type": "function",
                                      "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}})
                    a_msg = {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}
                    msgs.append(a_msg)
                    api_msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls,
                                     "reasoning_content": m.get("reasoning_content") or ""})
                    for c in calls:
                        n_calls += 1
                        name = c["function"]["name"]
                        try:
                            args = json.loads(c["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            args = None
                        if name != "python" or not isinstance(args, dict) or not isinstance(args.get("code"), str):
                            res = {"stdout": "", "stderr": "Error: invalid tool call (expected python with a string "
                                   "argument 'code')", "status": "error", "truncated": False}
                            code = None
                        else:
                            code = args["code"]
                            res = await arun_python(code)
                            if res.get("status") == "sandbox_error":
                                status = "sandbox_error"
                                reasons.append("sandbox_error")
                                raise RuntimeError("sandbox_error")
                        obs = format_observation(res)
                        msgs.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": obs})
                        api_msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
                        runs.append({"call_id": c["id"], "turn": turn, "code": code, "status": res.get("status"),
                                     "stdout": res.get("stdout"), "stderr": res.get("stderr"),
                                     "truncated": res.get("truncated"), "wall_ms": res.get("wall_ms"),
                                     "observation": obs})
                    continue
                final = {"role": "assistant", "content": content, "reasoning_content": reasoning}
                msgs.append(final)
                break
            else:
                reasons.append("max_turns_without_final")
                status = "rejected"
        except Budget as e:
            status, reasons = "api_error", [f"budget_or_stop: {str(e)[:200]}"]
        except Exception as e:  # noqa: BLE001
            if status != "sandbox_error":
                status, reasons = "api_error", [f"{type(e).__name__}: {str(e)[:200]}"]

        pred, ok = None, False
        used_any = False
        if status is None and final is not None:
            pred, ok = grade(p["kind"], int(p["answer"]), p.get("choices"), final["content"])
            # per call: does its output re-appear in the assistant text that follows the tool response?
            idx = {r_["call_id"]: r_ for r_ in runs}
            for i, mm in enumerate(msgs):
                if mm["role"] != "tool":
                    continue
                j = i + 1
                while j < len(msgs) and msgs[j]["role"] == "tool":
                    j += 1
                follow = []
                if j < len(msgs):
                    nxt = msgs[j]
                    follow = [nxt.get("reasoning_content", ""), nxt.get("content", "")]
                    follow += [tc["function"]["arguments"] for tc in nxt.get("tool_calls") or []]
                run = idx[mm["tool_call_id"]]
                hits = output_used(run["stdout"] or "", "\n".join(follow)) if run["status"] == "ok" else []
                run["used_tokens"] = hits[:10]
                run["used"] = bool(hits)
                used_any |= bool(hits)
            bad = []
            for mm in msgs[2:]:
                for fld in ("content", "reasoning_content"):
                    bad += forbidden_markup(mm.get(fld) or "")
                for tc in mm.get("tool_calls") or []:
                    bad += forbidden_markup(tc["function"]["arguments"])
            if pred is None:
                reasons.append("no_boxed")
            elif not ok:
                reasons.append("wrong_answer")
            if n_calls == 0:
                reasons.append("no_tool_call")
            elif not used_any:
                reasons.append("tool_output_unused")
            if bad:
                reasons.append("forbidden_markup:" + ",".join(sorted(set(bad))))
            if not any(r_["status"] == "ok" for r_ in runs) and n_calls:
                reasons.append("no_successful_tool_run")
            meta = meta_tool_commentary(msgs)
            if meta and self.a.reject_meta:
                reasons.append("meta_tool_commentary")
            status = "verified" if not reasons else "rejected"
        elif status is None:
            status = "rejected"

        rec = {"id": p["id"], "source": p["source"], "section": p["section"], "exam": p["exam"],
               "school_year": p["school_year"], "number": p.get("number"), "kind": p["kind"], "points": p.get("points"),
               "family_id": p.get("family_id"), "passrate": p.get("passrate"), "affinity": p.get("affinity"),
               "label_source": p.get("label_source") or "official_key",
               "status": status, "reasons": reasons, "n_calls": n_calls,
               "answer_check": {"pred": pred, "correct": bool(ok), "answer": int(p["answer"]), "kind": p["kind"]},
               "tool_output_used": used_any, "meta_tool_commentary": meta_tool_commentary(msgs)[:5], "messages": msgs, "tool_runs": runs, "usage": usage,
               "model": MODEL, "gen_prompt_version": GEN_PROMPT_VERSION, "temperature": self.a.temperature,
               "secs": round(time.time() - t0, 1)}
        return rec

    async def worker(self, queue, out):
        while not queue.empty() and not self.stop:
            if self.n_verified >= self.a.target:
                return
            p = queue.get_nowait()
            rec = await self.solve(p)
            async with self.lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                self.n_done += 1
                self.n_verified += rec["status"] == "verified"
                if self.n_done % 10 == 0 or rec["status"] in RETRYABLE:
                    el = time.time() - self.t0
                    print(json.dumps({"done": self.n_done, "verified": self.n_verified, "last": rec["id"],
                                      "last_status": rec["status"], "reasons": rec["reasons"][:3],
                                      "tokens": self.tokens, "elapsed_min": round(el / 60, 1)}), flush=True)
                tot = self.tokens["prompt"] + self.tokens["completion"]
                if self.a.max_total_tokens and tot > self.a.max_total_tokens:
                    print(json.dumps({"event": "token_budget_reached", "tokens": self.tokens}), flush=True)
                    self.stop = True


async def amain(a):
    pool = [json.loads(l) for l in open(a.pool)]
    pool.sort(key=lambda p: p["order"])
    done = {}
    if os.path.exists(a.raw):
        for l in open(a.raw):
            r = json.loads(l)
            done[r["id"]] = r["status"]
    todo = [p for p in pool if done.get(p["id"]) in (None, *RETRYABLE)]
    todo = todo[a.offset:a.offset + a.limit] if a.limit else todo[a.offset:]
    g = Gen(a)
    g.n_verified = sum(1 for s in done.values() if s == "verified")
    print(json.dumps({"event": "start", "model": MODEL, "pool": len(pool), "already": len(done),
                      "already_verified": g.n_verified, "todo": len(todo), "target": a.target,
                      "concurrency": a.concurrency}), flush=True)
    queue = asyncio.Queue()
    for p in todo:
        queue.put_nowait(p)
    os.makedirs(os.path.dirname(a.raw), exist_ok=True)
    with open(a.raw, "a") as out:
        await asyncio.gather(*[g.worker(queue, out) for _ in range(a.trajectories)])
    print(json.dumps({"event": "end", "done": g.n_done, "verified_total": g.n_verified, "tokens": g.tokens,
                      "elapsed_min": round((time.time() - g.t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/data/tool/pool.jsonl")
    ap.add_argument("--raw", default="/data/tool/gen/tir_raw.jsonl")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target", type=int, default=600, help="stop launching new problems at this many verified")
    ap.add_argument("--concurrency", type=int, default=32, help="max concurrent gateway requests (<= 32)")
    ap.add_argument("--trajectories", type=int, default=40, help="trajectories in flight")
    ap.add_argument("--max-tokens", type=int, default=16384, help="per request")
    ap.add_argument("--max-total-tokens", type=int, default=0, help="stop when prompt+completion exceeds this")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--api-retries", type=int, default=7)
    ap.add_argument("--reject-meta", action="store_true", help="reject trajectories whose reasoning calls the tool "
                    "unnecessary / prompt-driven (trace_utils.META_TOOL_COMMENTARY)")
    a = ap.parse_args()
    assert a.concurrency <= 32
    asyncio.run(amain(a))
