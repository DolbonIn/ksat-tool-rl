"""Turn solve_check.py outputs into per-id verify records for build_dataset.py --verify-extra.

Usage: python3 merge_solve.py --out verify_solve.json deepseek=a.jsonl,b.jsonl local=c.jsonl,d.jsonl
Each record: {"solve_check_<name>": {"model", "n", "n_correct", "preds", "suspect"}} where suspect = no sample
matched the official answer (=> candidate transcription error; answers are never changed).
"""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("groups", nargs="+")
a = ap.parse_args()
out = collections.defaultdict(dict)
for g in a.groups:
    name, files = g.split("=", 1)
    by = collections.defaultdict(list)
    for f in files.split(","):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish") != "error":
                by[r["id"]].append(r)
    for pid, rs in by.items():
        rs.sort(key=lambda r: r["sample"])
        nc = sum(r["correct"] for r in rs)
        out[pid][f"solve_check_{name}"] = dict(model=rs[0].get("model"), n=len(rs), n_correct=nc,
                                               preds=[r["pred"] for r in rs], suspect=nc == 0)
    sus = sorted(p for p, rs in by.items() if not any(r["correct"] for r in rs))
    print(f"{name}: {len(by)} items, {sum(len(v) for v in by.values())} samples, "
          f"acc {sum(r['correct'] for v in by.values() for r in v) / max(1, sum(len(v) for v in by.values())):.3f}, suspects {len(sus)}")
    print("  ", ",".join(sus))
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
