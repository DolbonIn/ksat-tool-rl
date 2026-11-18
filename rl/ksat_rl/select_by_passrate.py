"""Pick RL training problems from a base-model sampling run (eval_csat.py output on the train set).

GRPO only learns from groups whose rewards differ, so problems the base model always solves (8/8) or
never solves (0/8) give zero advantage. Default keeps 0 < pass < 1. Never-solved items are listed
separately: they are either beyond the model or transcription/answer-key errors and are checked
(e.g. with DeepSeek) before being added.
"""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--problems", required=True)
ap.add_argument("--out-ids", required=True)
ap.add_argument("--max-pass", type=float, default=0.875)
ap.add_argument("--min-pass", type=float, default=0.125)
a = ap.parse_args()
by = collections.defaultdict(list)
for l in open(a.samples):
    r = json.loads(l); by[r["id"]].append(r)
probs = {json.loads(l)["id"]: json.loads(l) for l in open(a.problems) if l.strip()}
rate = {i: sum(r["correct"] for r in rs) / len(rs) for i, rs in by.items()}
trunc = {i: sum(r["finish"] == "length" for r in rs) / len(rs) for i, rs in by.items()}
keep = sorted(i for i, p in rate.items() if a.min_pass <= p <= a.max_pass)
zero = sorted(i for i, p in rate.items() if p == 0)
hist = collections.Counter(round(p * 8) for p in rate.values())
print("pass@8 histogram (k correct of 8):", dict(sorted(hist.items())))
print(f"keep {len(keep)} / {len(rate)}  | never solved {len(zero)} | always solved {sum(p == 1 for p in rate.values())}")
by_sec = collections.Counter(probs[i]["section"] for i in keep); print("kept by section", dict(by_sec))
open(a.out_ids, "w").write("\n".join(keep) + "\n")
open(a.out_ids + ".zero", "w").write("\n".join(zero) + "\n")
json.dump({"rate": rate, "trunc": trunc}, open(a.out_ids + ".stats.json", "w"), indent=0)
