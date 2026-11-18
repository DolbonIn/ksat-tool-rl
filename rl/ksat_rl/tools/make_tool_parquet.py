"""Build tool-RL parquet variants (ICRL demo curriculum 2 -> 1 -> 0) from the no-tool RL parquet files.

    python -m ksat_rl.tools.make_tool_parquet --inp /data/ksat_math/rl/rl_train.parquet --split train \\
        --tool-demos 2,1,0 --out /data/ksat_math/rl_tool/rl_train_tool_k{k}.parquet

Each row keeps the no-tool row's user prompt, reward_model and extra_info, and gets
  prompt      = [system: TOOL_SYSTEM_PROMPT (+ k few-shot demos from data/tool/fewshot_demos.json), user: same]
  agent_name  = ksat_tool_agent         (rl/configs/tool/agent_loop.yaml)
  data_source = ksat_math_tool          extra_info.tool_demos = k
The chat template adds the tool schema (tools=) at rollout time; demos use the exact XML call format.
All k-variants keep the same rows in the same order, so a run can switch phase (k2 -> k1 -> k0) by resuming
from its checkpoint with the next TRAIN_FILE (dataloader state stays valid).  Phase changes should follow the
tool-call validity / accuracy / completion metrics, not a fixed step (ksat_tool_rl_v2 §4-C).
Guards: for split=train no 2026 수능 id, no 2026학년도 9월 모평 id and no synthetic item of either family may
appear (hard assert); demo items and their synthetic variants are dropped from every split (a demo must
never be its own problem).  Run inside ksat-rl/verl-sgl (pandas + Nanbeige tokenizer for length stats).
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ksat_rl.tools.schema import TOOLS, system_prompt_with_demos  # noqa: E402

DATA = os.environ.get("KSAT_DATA", "/data")


def family(pid: str) -> str:
    """synth-<family>-<k> -> <family>; real ids are their own family."""
    return pid[len("synth-"):].rsplit("-", 1)[0] if pid.startswith("synth-") else pid


def held_out_ids() -> set:
    held = {json.loads(l)["id"] for l in open(f"{DATA}/ksat_math/test_2026.jsonl") if l.strip()}
    for f in (f"{DATA}/ksat_math/train.jsonl", f"{DATA}/ksat_math_old/train_2017_2021_hard.jsonl"):
        for l in open(f):
            p = json.loads(l)
            if p["exam"] == "kice_09" and int(p["school_year"]) == 2026:
                held.add(p["id"])
    return held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="no-tool RL parquet (make_parquet.to_row rows)")
    ap.add_argument("--out", required=True, help="output path; '{k}' is replaced by the demo count")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--tool-demos", default="0", help="comma list of demo counts, e.g. 2,1,0")
    ap.add_argument("--demos", default=f"{DATA}/tool/fewshot_demos.json")
    ap.add_argument("--tokenizer", default="/models/Nanbeige4.2-3B-train-fused", help="for prompt length stats ('' to skip)")
    a = ap.parse_args()
    ks = [int(x) for x in a.tool_demos.split(",")]
    demos = json.load(open(a.demos))["demos"]
    demo_fams = {family(d["id"]) for d in demos}
    df = pd.read_parquet(a.inp)
    ids = [r["id"] for r in df["extra_info"]]

    held = held_out_ids()
    if a.split == "train":
        leak = [i for i in ids if i in held or family(i) in held]
        assert not leak, f"held-out items in train input: {leak[:5]}"
        assert not any(family(d["id"]) in held for d in demos), "a few-shot demo is a held-out item"
    keep = [family(i) not in demo_fams for i in ids]
    dropped = [i for i, k in zip(ids, keep) if not k]
    df = df[keep].reset_index(drop=True)

    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    stats = {"input_rows": len(ids), "rows": len(df), "dropped_demo_items": dropped, "split": a.split}
    for k in ks:
        sysmsg = system_prompt_with_demos(demos, k)
        rows = []
        for _, r in df.iterrows():
            user = [m for m in list(r["prompt"]) if m["role"] == "user"]
            assert len(user) == 1
            ei = dict(r["extra_info"])
            ei["tool_demos"] = k
            rows.append({"data_source": "ksat_math_tool",
                         "prompt": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user[0]["content"]}],
                         "ability": r["ability"], "reward_model": dict(r["reward_model"]), "extra_info": ei,
                         "agent_name": "ksat_tool_agent"})
        out = a.out.replace("{k}", str(k))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
        st = {"out": out}
        if tok is not None:
            lens = sorted(len(tok(tok.apply_chat_template(x["prompt"], tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                                          preserve_thinking=True), add_special_tokens=False)["input_ids"])
                          for x in rows)
            st.update(prompt_tokens_max=lens[-1], prompt_tokens_p50=lens[len(lens) // 2])
        stats[f"k{k}"] = st
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


if __name__ == "__main__":
    main()
