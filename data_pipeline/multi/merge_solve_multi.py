"""Turn solve_multi.py outputs into per-id verify records (build_multi.py --verify-extra).
Usage: python3 merge_solve_multi.py out.json a.jsonl [b.jsonl ...]   (later files override samples of earlier ones by (id, sample))
Record: {"deepseek_solve": {"n", "n_correct", "acc", "preds", "models", "suspect"}}; suspect = 0/n match the official answer.
"""
import json, sys
from collections import defaultdict, Counter

out = sys.argv[1]
by = defaultdict(dict)
for f in sys.argv[2:]:
    for l in open(f):
        r = json.loads(l)
        if r.get("finish") == "error":
            continue
        by[r["id"]][r["sample"]] = r
res = {}
for pid, d in by.items():
    rs = [d[k] for k in sorted(d)]
    nc = sum(bool(r["correct"]) for r in rs)
    res[pid] = {"deepseek_solve": dict(n=len(rs), n_correct=nc, acc=round(nc / len(rs), 3), preds=[r["pred"] for r in rs],
                                       models=dict(Counter((r.get("model") or "").split("/")[-1] for r in rs)), suspect=nc == 0)}
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
n = len(res); s = sum(v["deepseek_solve"]["suspect"] for v in res.values())
acc = sum(v["deepseek_solve"]["n_correct"] for v in res.values()) / max(1, sum(v["deepseek_solve"]["n"] for v in res.values()))
print(f"{n} items, sample acc {acc:.3f}, suspects (0/n) {s}")
