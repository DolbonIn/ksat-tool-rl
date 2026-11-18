"""Pool nb_logprobs --ref outputs (*.kl.tsv) into one quality row per variant.

  python summarize_kl.py DIR [DIR ...]
Columns: scored tokens, PPL(ref) -> PPL(test), mean / p99 / max KL(ref||test), mean |dlogp| of the target
token, top-1 agreement with the reference, and the fraction of tokens with |dlogp| > 0.5.
"""
import glob, json, os, sys
import numpy as np

rows = []
for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{d}/*.kl.tsv"))
    if not fs:
        continue
    def _load(f):  # tolerate a partially written last line while a run is still going
        rows_ = [l.split() for l in open(f)]
        return np.array([[float(x) for x in r] for r in rows_ if len(r) == 5], dtype=np.float64).reshape(-1, 5)
    a = np.concatenate([_load(f) for f in fs])
    kl, lref, ltest, agree = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    d_lp = np.abs(lref - ltest)
    rows.append(dict(name=os.path.basename(d.rstrip("/")), n=len(a),
                     ppl_ref=float(np.exp(-lref.mean())), ppl=float(np.exp(-ltest.mean())),
                     kl_mean=float(kl.mean()), kl_p99=float(np.percentile(kl, 99)), kl_max=float(kl.max()),
                     dlogp_mean=float(d_lp.mean()), top1=float(agree.mean()), frac_dlp_gt_0p5=float((d_lp > 0.5).mean())))
print(f"{'variant':<26}{'tokens':>8}{'PPL ref':>9}{'PPL':>9}{'KL mean':>10}{'KL p99':>9}{'KL max':>8}{'|dlogp|':>9}{'top1':>8}{'>0.5':>8}")
for r in rows:
    print(f"{r['name']:<26}{r['n']:>8}{r['ppl_ref']:>9.4f}{r['ppl']:>9.4f}{r['kl_mean']:>10.5f}{r['kl_p99']:>9.4f}{r['kl_max']:>8.3f}"
          f"{r['dlogp_mean']:>9.4f}{r['top1']:>8.4f}{r['frac_dlp_gt_0p5']:>8.4f}")
if os.environ.get("JSON_OUT"):
    json.dump(rows, open(os.environ["JSON_OUT"], "w"), indent=1)
