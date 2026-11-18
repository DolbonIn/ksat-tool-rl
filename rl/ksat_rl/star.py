"""Hint-guided self-generated rationalisation (STaR-style) for hard items — strategy 3.

  make-prompts : hints.jsonl (+ problem pool) -> problems JSONL whose text carries a cumulative hint
                 block (level 1 = h1, 2 = h1+h2, 3 = h1+h2+h3). id = "<orig id>#h<level>".
                 These are rolled out with the POLICY (eval_csat.py, GPU phase).
  build-sft    : policy samples on hinted prompts -> SFT parquet for the ORIGINAL, hint-free prompt.
                 Keeps only correct samples (reward.grade), drops responses that refer to the hint
                 (the hint is absent at training time), prefers the weakest hint level, then the shortest
                 response, up to --per-item samples per problem. The user turn is byte-identical to the RL
                 prompt (make_parquet.instruction + text); the assistant turn keeps the policy's own
                 thinking as reasoning_content, so Nanbeige's template renders <think>…</think> + answer.
Test/val exclusion is enforced upstream (hint seeds never contain them) and re-checked here.
"""
import argparse
import collections
import json
import re
import sys

from ksat_rl.make_parquet import instruction
from ksat_rl.reward import grade

HINT_REF = re.compile(r"힌트|hint|Hint|HINT|단서에 따라|제시된 (방향|풀이)|주어진 (힌트|방향)")


def hint_block(h, level):
    parts = [h["h1"], h.get("h2", ""), h.get("h3", "")][:level]
    return "\n\n[힌트]\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(parts) if t)


def make_prompts(a):
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(a.pool) if l.strip()}
    n = 0
    with open(a.out, "w") as fo:
        for l in open(a.hints):
            r = json.loads(l)
            if r.get("status") != "ok" or r["id"] not in pool:
                continue
            p = pool[r["id"]]
            for level in a.levels:
                q = dict(p)
                q["id"] = f"{p['id']}#h{level}"
                q["text"] = p["text"] + hint_block(r["hints"], level)
                q["hint_level"] = level
                q["orig_id"] = p["id"]
                fo.write(json.dumps(q, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} hinted prompts -> {a.out}")


def build_sft(a):
    import pandas as pd
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(a.pool) if l.strip()}
    test_ids = {json.loads(l)["id"] for l in open(a.test) if l.strip()}
    by = collections.defaultdict(list)
    stats = collections.Counter()
    for f in a.samples:
        for l in open(f):
            r = json.loads(l)
            orig, level = r["id"].rsplit("#h", 1)
            p = pool.get(orig)
            if p is None or orig in test_ids or str(p.get("family_id", "")) in test_ids:
                stats["skipped_unknown_or_test"] += 1; continue
            stats["samples"] += 1
            _, ok = grade(p["kind"], int(p["answer"]), p.get("choices"), r.get("content") or "")
            if not ok:
                continue
            stats["correct"] += 1
            if HINT_REF.search((r.get("reasoning") or "") + (r.get("content") or "")):
                stats["dropped_mentions_hint"] += 1; continue
            if r.get("finish") == "length":
                continue
            by[orig].append((int(level), int(r.get("tokens") or 0), r))
    rows = []
    for orig, cands in by.items():
        cands.sort(key=lambda x: (x[0], x[1]))
        p = pool[orig]
        for level, _, r in cands[: a.per_item]:
            rows.append({
                "messages": [
                    {"role": "user", "content": f"{instruction(p)}\n\n{p['text']}"},
                    {"role": "assistant", "content": (r.get("content") or "").strip(), "reasoning_content": (r.get("reasoning") or "").strip()},
                ],
                "enable_thinking": True,
                "orig_id": orig, "hint_level": level, "tokens": int(r.get("tokens") or 0),
            })
    stats["sft_rows"] = len(rows); stats["items_covered"] = len(by)
    stats.update({f"level{lv}_rows": sum(1 for x in rows if x["hint_level"] == lv) for lv in (1, 2, 3)})
    pd.DataFrame(rows).to_parquet(a.out)
    print(json.dumps(dict(stats), ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-prompts")
    m.add_argument("--hints", required=True); m.add_argument("--pool", required=True); m.add_argument("--out", required=True)
    m.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    b = sub.add_parser("build-sft")
    b.add_argument("--samples", nargs="+", required=True); b.add_argument("--pool", required=True)
    b.add_argument("--test", required=True); b.add_argument("--out", required=True)
    b.add_argument("--per-item", type=int, default=2)
    a = ap.parse_args()
    make_prompts(a) if a.cmd == "make-prompts" else build_sft(a)
