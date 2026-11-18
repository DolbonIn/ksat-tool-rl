"""Merge a forward run (A, full output file) and a reverse run (B, .partial.jsonl) of eval_csat.py.

For each problem: use A's samples if A finished all n without error/abort, else B's if B has all n, else the
problem goes to the residual list. Output: merged samples JSONL (exactly n per problem, single source each)
and residual problems JSONL.
"""
import collections, json, sys
probs_f, a_f, b_f, n, out_samples, out_resid = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
ok = lambda r: not (str(r["finish"]).startswith("error") or r["finish"] == "abort")
def load(f):
    d = collections.defaultdict(dict)
    try:
        for l in open(f):
            r = json.loads(l)
            if ok(r): d[r["id"]][r["sample"]] = r
    except FileNotFoundError:
        pass
    return d
A, B = load(a_f), load(b_f)
probs = [json.loads(l) for l in open(probs_f) if l.strip()]
na = nb = 0; resid = []
with open(out_samples, "w") as fo:
    for p in probs:
        src = A if len(A[p["id"]]) >= n else B if len(B[p["id"]]) >= n else None
        if src is None:
            resid.append(p); continue
        na += src is A; nb += src is B
        for k in range(n): fo.write(json.dumps(src[p["id"]][k], ensure_ascii=False) + "\n")
with open(out_resid, "w") as fr:
    for p in resid: fr.write(json.dumps(p, ensure_ascii=False) + "\n")
print(json.dumps({"problems": len(probs), "from_forward": na, "from_reverse": nb, "residual": len(resid)}))
