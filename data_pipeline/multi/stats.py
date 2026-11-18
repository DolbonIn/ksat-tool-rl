"""Summary statistics of the final dataset + DeepSeek token usage.
Usage: python3 stats.py <ksat_multi_dir> <usage.jsonl>
"""
import glob, json, os, sys
from collections import Counter, defaultdict

D, U = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
    rs = [json.loads(l) for l in open(f)]
    name = os.path.basename(f)
    by_exam = Counter((r["school_year"], r["exam"]) for r in rs)
    by_subj = Counter(r["subject"] for r in rs)
    fig = sum(r["has_figure"] for r in rs)
    tl = sum(r["verify"].get("textlayer_ok", False) for r in rs)
    ds = [r["verify"].get("deepseek_solve") for r in rs]
    solved = [d for d in ds if d]
    acc = sum(d["n_correct"] for d in solved) / max(1, sum(d["n"] for d in solved))
    sus = [r["id"] for r, d in zip(rs, ds) if d and d["suspect"]]
    rev = Counter((r["verify"].get("manual_review") or {}).get("status", "-") for r in rs)
    print(f"== {name}: {len(rs)} items, has_figure {fig}, textlayer_ok {tl}, solved {len(solved)} (sample acc {acc:.3f}), 0/4 {len(sus)}")
    print("   subjects:", dict(sorted(by_subj.items())))
    print("   sittings:", len(by_exam), dict(Counter(e for _, e in by_exam)))
    if "test" in name:
        print("   visual review status:", dict(rev))
tot = defaultdict(lambda: [0, 0, 0])
for l in open(U):
    r = json.loads(l)
    if "error" in r:
        continue
    k = r["tag"].split(":")[0]
    tot[k][0] += 1; tot[k][1] += r.get("prompt") or 0; tot[k][2] += r.get("completion") or 0
print("DeepSeek usage by stage (calls, prompt tokens, completion tokens):")
for k, v in sorted(tot.items()):
    print(f"   {k:8s} {v[0]:7d} {v[1]:12,d} {v[2]:12,d}")
print(f"   total    {sum(v[0] for v in tot.values()):7d} {sum(v[1] for v in tot.values()):12,d} {sum(v[2] for v in tot.values()):12,d}")
