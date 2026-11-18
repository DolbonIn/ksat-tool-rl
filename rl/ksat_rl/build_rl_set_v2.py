"""Phase-2 RL set, selected by the CURRENT policy (math_v1 step 40) pass rates.

Sources (each with its own policy pass-rate sample files):
  --items  JSONL problem files (real + synthetic + terms + 0/8 re-measure set); answers official or consensus
  --rates  eval_csat.py sample files with policy rollouts on those items (any n; rate = mean correct)
  --hinted hinted prompts (ksat_rl.star make-prompts) + --hint-rates: level-1 hinted versions are added only
           for items the policy still never solves without a hint (rate == 0 with >= --zero-min-n samples)
Selection: min_pass <= rate <= max_pass. Items with rate <= --dup-below are duplicated (2 copies) so the
hardest learnable items get 16 rollouts per epoch instead of 8.
Exclusions: val exam (2026 kice_09) + everything seeded from it, every 2026 수능 test id / family.
"""
import argparse, collections, json

import pandas as pd

from ksat_rl.make_parquet import to_row

ap = argparse.ArgumentParser()
ap.add_argument("--items", nargs="+", required=True)
ap.add_argument("--rates", nargs="+", required=True)
ap.add_argument("--hinted", default=None)
ap.add_argument("--hint-rates", nargs="*", default=[])
ap.add_argument("--hint-level", type=int, default=1)
ap.add_argument("--test", required=True)
ap.add_argument("--val", required=True, help="existing rl_val.parquet to copy (held-out 2026 9월 모평)")
ap.add_argument("--min-pass", type=float, default=0.125)
ap.add_argument("--max-pass", type=float, default=0.875)
ap.add_argument("--dup-below", type=float, default=0.25)
ap.add_argument("--zero-min-n", type=int, default=16)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()


def rates_of(files):
    by = collections.defaultdict(list)
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if str(r.get("finish", "")).startswith("error") or r.get("finish") == "abort":
                continue
            by[r["id"]].append(bool(r["correct"]))
    return by


test_ids = {json.loads(l)["id"] for l in open(a.test) if l.strip()}
is_val_id = lambda i: str(i).startswith("2026-09-")
items = {}
for f in a.items:
    for l in open(f):
        if l.strip():
            p = json.loads(l); items[p["id"]] = p
bad = lambda p: (p["id"] in test_ids or str(p.get("family_id", "")) in test_ids or is_val_id(p["id"])
                 or is_val_id(p.get("family_id", "")) or (p.get("exam") == "kice_09" and str(p.get("school_year")) == "2026"))
R = rates_of(a.rates)
rows, stats = [], collections.Counter()
chosen = []
for i, p in items.items():
    if bad(p):
        stats["excluded_val_test"] += 1; continue
    if i not in R:
        stats["no_rate"] += 1; continue
    rate = sum(R[i]) / len(R[i])
    if a.min_pass <= rate <= a.max_pass or (len(R[i]) >= 16 and 0 < rate < 1):
        copies = 2 if rate <= a.dup_below else 1
        chosen.append((p, rate, copies))
        stats["dup" if copies == 2 else "single"] += 1
src = lambda p: ("synth_" + p["id"].split("-")[0].split("_", 1)[1]) if p["id"].startswith("synth_") else ("synth" if p["exam"] == "synth" else ("term" if p["exam"] == "term" else "real"))
hint_added = 0
if a.hinted:
    HR = rates_of(a.hint_rates)
    unsolved = {i for i, v in R.items() if len(v) >= a.zero_min_n and not any(v) and i in items and not bad(items[i])}
    for l in open(a.hinted):
        q = json.loads(l)
        if q.get("hint_level") != a.hint_level or q["orig_id"] not in unsolved or bad(q):
            continue
        hr = HR.get(q["id"])
        if hr and 0 < sum(hr) / len(hr) < 1:
            chosen.append((q, sum(hr) / len(hr), 1)); hint_added += 1
for p, rate, copies in chosen:
    for c in range(copies):
        row = to_row(p, "train"); row["extra_info"]["copy"] = c; row["extra_info"]["pass_rate"] = rate
        rows.append(row)
df = pd.DataFrame(rows).sample(frac=1.0, random_state=1)
df.to_parquet(f"{a.out_dir}/rl_train.parquet")
pd.read_parquet(a.val).to_parquet(f"{a.out_dir}/rl_val.parquet")
summary = {"rows": len(rows), "unique_items": len(chosen), "hint_items": hint_added, **stats,
           "by_source": dict(collections.Counter(src(p) for p, _, _ in chosen)),
           "rate_hist": dict(sorted(collections.Counter(round(r * 8) for _, r, _ in chosen).items()))}
json.dump(summary, open(f"{a.out_dir}/rl_set_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
