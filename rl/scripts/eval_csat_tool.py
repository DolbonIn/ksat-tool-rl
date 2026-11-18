"""Tool-mode evaluation on 2026 CSAT math (or any KSAT JSONL): multi-turn with the python sandbox.

Same protocol as eval_csat.py (instruction per exam, T=0.6 / top_p 0.95 / top_k 20, thinking on, 65,536-token
budget, n samples per item, grade = last \\boxed{}, same score report per elective track) but each episode runs
through veRL's real tool agent loop (KsatToolAgentLoop + nanbeige parser + Nanbeige Continuous Token builder +
KsatPythonTool), token-in/token-out against an SGLang server's native /generate endpoint: the policy sees exactly
the environment it is trained in (rl/scripts/run_math_tool_v1.sh), including <= 6 tool calls, <= 8 assistant turns,
1000-char tool outputs, defanged markup, and the 65,536-token budget counting tool-response tokens.
Grading uses the final assistant turn only (ksat_rl/tools/reward_tool.py).  Sandbox infra faults are recorded
(finish="infra_fault") and excluded from the score by --resume reruns, like API errors in eval_csat.py.

  python /rl/scripts/eval_csat_tool.py --url http://<sglang>:30600 --out /out/tool_base.jsonl [--tool-demos 0]
Run via rl/scripts/docker_eval_tool.sh (needs the veRL patch and the sandbox network).
"""
import argparse
import asyncio
import collections
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, "/rl")
from transformers import AutoTokenizer  # noqa: E402

from ksat_rl.tools.agent_harness import SGLangGenerateClient, build_agent_loop  # noqa: E402
from ksat_rl.tools.reward_tool import score_response  # noqa: E402
from ksat_rl.tools.schema import build_messages  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_csat", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_csat.py"))
eval_csat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_csat)


async def run_one(args, tok, client, tools, demos, sem, p, k):
    async with sem:
        t0 = time.time()
        msgs = build_messages(p, demos, args.tool_demos)
        try:
            loop = build_agent_loop(tok, client, prompt_length=args.prompt_length, response_length=args.max_tokens,
                                    tools=tools, max_user_turns=args.max_user_turns,
                                    max_assistant_turns=args.max_assistant_turns)
            out = await loop.run({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                                  "repetition_penalty": 1.0, "logprobs": False},
                                 raw_prompt=msgs, extra_info={}, tools_kwargs={})
        except Exception as e:  # infra errors are recorded separately, not as wrong answers
            return dict(id=p["id"], sample=k, pred=None, correct=False, tokens=0, finish=f"error: {e}"[:300],
                        secs=round(time.time() - t0, 1), content="", n_calls=0)
        text = tok.decode(out.response_ids, skip_special_tokens=True)
        r = score_response(text, p["kind"], int(p["answer"]), p.get("choices"))
        ef = out.extra_fields
        if ef.get("ksat_infra_fault"):
            finish = "infra_fault"
        elif len(out.response_mask) >= args.max_tokens:
            finish = "length"
        elif r["unfinished"]:
            finish = "tool_budget"
        else:
            finish = "stop"
        return dict(id=p["id"], sample=k, pred=r["pred"], correct=r["ok"], tokens=len(out.response_ids),
                    gen_tokens=int(sum(out.response_mask)), finish=finish, secs=round(time.time() - t0, 1),
                    n_calls=int(ef.get("ksat_tool_calls", 0)), num_turns=out.num_turns,
                    tool_status={kk[len("ksat_tool_"):]: int(v) for kk, v in ef.items() if kk.startswith("ksat_tool_") and v},
                    content=r["content"], response=text)


def _ok(r):
    f = str(r.get("finish"))
    return not (f.startswith("error") or f in ("abort", "infra_fault"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="/data/ksat_math/test_2026.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:30600", help="SGLang server base URL (native /generate)")
    ap.add_argument("--tokenizer", default="/models/Nanbeige4.2-3B")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=65536, help="episode budget incl. tool-response tokens")
    ap.add_argument("--prompt-length", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--tool-demos", type=int, default=0, help="few-shot demos in the system prompt (0 = zero-shot)")
    ap.add_argument("--demos", default="/data/tool/fewshot_demos.json")
    ap.add_argument("--max-user-turns", type=int, default=6)
    ap.add_argument("--max-assistant-turns", type=int, default=8)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip (id, sample) pairs already completed in <out>.partial.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    problems = [json.loads(l) for l in open(args.problems) if l.strip()]
    demos = json.load(open(args.demos))["demos"] if args.tool_demos > 0 else []
    if demos:
        assert not {d["id"] for d in demos} & {p["id"] for p in problems}, "a demo item is in the evaluation set"
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    from verl.tools.tool_registry import load_all_tools
    tools = load_all_tools(os.environ.get("KSAT_TOOL_CONFIG", "/rl/configs/tool/python_tool.yaml"), None)
    client = SGLangGenerateClient(args.url, args.prompt_length, args.max_tokens, args.max_model_len)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    partial = args.out + ".partial.jsonl"
    done = {}
    if args.resume and os.path.exists(partial):
        for l in open(partial):
            r = json.loads(l)
            if _ok(r):
                done[(r["id"], r["sample"])] = r
    todo = [(p, k) for p in problems for k in range(args.n) if (p["id"], k) not in done]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)
    fpart = open(partial, "a")

    async def run_and_log(p, k):
        r = await run_one(args, tok, client, tools, demos, sem, p, k)
        fpart.write(json.dumps(r, ensure_ascii=False) + "\n")
        fpart.flush()
        return r

    new = await asyncio.gather(*[run_and_log(p, k) for p, k in todo])
    fpart.close()
    await client.close()
    for t in tools:
        if hasattr(t, "aclose"):
            await t.aclose()
    res = list(done.values()) + list(new)
    with open(args.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    calls = collections.Counter(r.get("n_calls", 0) for r in res)
    status = collections.Counter()
    for r in res:
        status.update(r.get("tool_status") or {})
    tool_summary = {"calls_per_episode": dict(sorted(calls.items())),
                    "mean_calls": round(sum(r.get("n_calls", 0) for r in res) / max(1, len(res)), 3),
                    "episodes_with_tool": sum(1 for r in res if r.get("n_calls")),
                    "tool_status": dict(status), "finish": dict(collections.Counter(r["finish"][:20] for r in res)),
                    "tool_demos": args.tool_demos}
    print(json.dumps(tool_summary, ensure_ascii=False))
    if not args.no_report:
        base = args.out.replace(".jsonl", "")
        with open(base + ".report.txt", "w") as f:
            s = eval_csat.report(problems, res, f)
            print(json.dumps(tool_summary, ensure_ascii=False), file=f)
        eval_csat.report(problems, res)
        s["tool"] = tool_summary
        json.dump(s, open(base + ".summary.json", "w"), ensure_ascii=False, indent=1)
    print(f"wall {time.time() - t0:.0f}s, {len(res)} episodes")


if __name__ == "__main__":
    asyncio.run(main())
