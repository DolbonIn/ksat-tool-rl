"""Pick 2-3 short, clean, diverse TIR demos for the ICRL 2->1->0 curriculum (ksat_tool_rl_v2 §4-B).

Only REAL train items with official answers (source real / real_old; 2026 수능, 2026 9월 모평 and their
families were already excluded from the pool).  Clean = verified, every tool run status ok and its output
used, no truncated output, 1-2 calls, short rendered assistant text.  Diverse = different sections / call
counts.  Writes data/tool/fewshot_demos.json with the demos, plus the ready system prompts for k=2,1,0
(ksat_rl.tools.schema.system_prompt_with_demos) and their Nanbeige token counts.
Run (CPU only):  data_pipeline/tool/run_format.sh fewshot
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("KSAT_RL_DIR", "/rl"))
from transformers import AutoTokenizer  # noqa: E402

from ksat_rl.tools.schema import system_prompt_with_demos  # noqa: E402
from ksat_rl.tools.trace_utils import meta_tool_commentary  # noqa: E402


HEDGED = re.compile(r"straightforward|maybe\W{0,3}(?:\w+\W+){0,3}?(?:verify|python|tool|check)|"
                    r"(?:verify|python|tool|check)\W{0,3}(?:\w+\W+){0,3}?maybe", re.I)


def skill(code: str) -> str:
    if re.search(r"itertools|product\(|permutations|combinations|for .+ in range\(", code):
        return "enumeration"
    if re.search(r"sympy|sp\.|solve\(|integrate\(|diff\(|factor\(", code):
        return "symbolic"
    return "arithmetic"


def genuine(tr) -> bool:
    """Tool did real work (enumeration / symbolic) and the reasoning does not call the tool unnecessary."""
    codes = [r["code"] or "" for r in tr["tool_runs"]]
    if all(skill(c) == "arithmetic" for c in codes):
        return False
    if meta_tool_commentary(tr["messages"]):
        return False
    # demos only: also skip hedged "verify maybe" calls on problems the teacher calls straightforward
    think = " ".join(m.get("reasoning_content", "") for m in tr["messages"] if m["role"] == "assistant")
    return not HEDGED.search(think)


def to_demo(tr, nb):
    steps, msgs = [], tr["messages"]
    outputs = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
    for m in msgs:
        if m["role"] != "assistant":
            continue
        st = {"think": m.get("reasoning_content", ""), "content": m.get("content", "")}
        if m.get("tool_calls"):
            st["tool_calls"] = [{"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"]),
                                 "output": outputs[tc["id"]]} for tc in m["tool_calls"]]
        steps.append(st)
    return {"id": tr["id"], "source": tr["source"], "section": tr["section"], "exam": tr["exam"],
            "school_year": tr["school_year"], "kind": tr["kind"], "label_source": tr["label_source"],
            "n_calls": tr["n_calls"], "answer_check": tr["answer_check"],
            "skills": sorted({skill(r["code"] or "") for r in tr["tool_runs"]}),
            "n_assistant_tokens": nb["n_assistant_tokens"], "n_tokens": nb["n_tokens"],
            "user": msgs[1]["content"], "steps": steps, "messages": msgs[1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/data/tool/tir_traces.jsonl")
    ap.add_argument("--nanbeige", default="/data/tool/tir_nanbeige.jsonl")
    ap.add_argument("--model", default="/models/Nanbeige4.2-3B")
    ap.add_argument("--out", default="/data/tool/fewshot_demos.json")
    ap.add_argument("--max-asst-tokens", type=int, default=2500)
    a = ap.parse_args()
    nb = {}
    for l in open(a.nanbeige):
        r = json.loads(l)
        nb[r["id"]] = {"n_assistant_tokens": r["n_assistant_tokens"], "n_tokens": r["n_tokens"]}
    cands = []
    for l in open(a.traces):
        tr = json.loads(l)
        if tr["source"] not in ("real", "real_old") or tr["label_source"] != "official_key" or tr["id"] not in nb:
            continue
        runs = tr["tool_runs"]
        if not (1 <= tr["n_calls"] <= 2) or not all(r["status"] == "ok" and r.get("used") and not r["truncated"] for r in runs):
            continue
        if any(len(r["observation"]) > 400 for r in runs) or len(tr["messages"][1]["content"]) > 700:
            continue
        if nb[tr["id"]]["n_assistant_tokens"] > a.max_asst_tokens or not genuine(tr):
            continue
        cands.append(tr)
    cands.sort(key=lambda t: nb[t["id"]]["n_assistant_tokens"])
    # diverse pick (shortest first within each rule): a 2-call demo, the other answer kind (mc/sa), then a new skill
    picked = []

    def sk(t):
        return {skill(r["code"] or "") for r in t["tool_runs"]}

    def fresh(t):
        return all(t["id"] != p["id"] and t["section"] != p["section"] for p in picked)

    def new_skill(t):
        return not (sk(t) <= set().union(*[sk(p) for p in picked])) if picked else True

    rules = [lambda t: t["n_calls"] == 2,
             lambda t: t["kind"] not in {p["kind"] for p in picked},
             lambda t: new_skill(t) and t["kind"] not in {p["kind"] for p in picked},
             new_skill,
             lambda t: True]
    for rule in rules:
        if len(picked) >= 3:
            break
        for t in cands:
            if fresh(t) and rule(t):
                picked.append(t)
                break
    demos = [to_demo(t, nb[t["id"]]) for t in picked]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    sys_prompts = {f"k{k}": system_prompt_with_demos(demos, k) for k in (2, 1, 0)}
    out = {"note": "ICRL curriculum demos (train items only, official answers). Insert with "
                   "ksat_rl.tools.schema.build_messages(p, demos, k_demos=k) for k = 2 -> 1 -> 0.",
           "n_candidates": len(cands), "demos": demos, "system_prompts": sys_prompts,
           "system_prompt_tokens": {k: len(tok(v, add_special_tokens=False)["input_ids"]) for k, v in sys_prompts.items()}}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"n_candidates": len(cands), "picked": [(d["id"], d["section"], d["kind"], d["n_calls"],
                                                              d["n_assistant_tokens"]) for d in demos],
                      "system_prompt_tokens": out["system_prompt_tokens"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
