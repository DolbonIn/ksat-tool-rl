"""Compare two per-token log-prob dumps (nb_logprobs / hf_logprobs format).

  python compare_lp.py REF_DIR TEST_DIR [--names a,b,c] [--vocab 166144]
Prints per-sequence and pooled: mean/max |dlogp| of the target token, top-1 agreement, and
KL(ref || test) when both dirs hold *.full.f32 dumps.
"""
import argparse, glob, json, os
import numpy as np


def read_lp(path):
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--names", default=None)
    ap.add_argument("--vocab", type=int, default=166144)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    names = a.names.split(",") if a.names else sorted(os.path.basename(p)[:-7] for p in glob.glob(f"{a.ref}/*.lp.tsv"))
    rows, pool = [], dict(d=[], agree=[], kl=[])
    for n in names:
        rp, rt, rl, ra = read_lp(f"{a.ref}/{n}.lp.tsv")
        tp, tt, tl, ta = read_lp(f"{a.test}/{n}.lp.tsv")
        assert (rp == tp).all() and (rt == tt).all(), n
        d = np.abs(rl - tl)
        agree = (ra == ta)
        kl = None
        rf, tf = f"{a.ref}/{n}.full.f32", f"{a.test}/{n}.full.f32"
        if os.path.exists(rf) and os.path.exists(tf):
            R = np.fromfile(rf, dtype=np.float32).reshape(-1, a.vocab)
            T = np.fromfile(tf, dtype=np.float32).reshape(-1, a.vocab)
            kl = np.empty(len(R))
            for i in range(0, len(R), 256):
                r = R[i:i + 256].astype(np.float64)
                t = T[i:i + 256].astype(np.float64)
                kl[i:i + 256] = (np.exp(r) * (r - t)).sum(-1)
            pool["kl"].append(kl)
        pool["d"].append(d)
        pool["agree"].append(agree)
        rows.append(dict(name=n, n=len(d), mean_abs_dlogp=d.mean(), max_abs_dlogp=d.max(), top1_agree=agree.mean(),
                         kl_mean=(kl.mean() if kl is not None else None), kl_max=(kl.max() if kl is not None else None)))
    d = np.concatenate(pool["d"]); ag = np.concatenate(pool["agree"])
    kl = np.concatenate(pool["kl"]) if pool["kl"] else None
    rows.append(dict(name="ALL", n=len(d), mean_abs_dlogp=d.mean(), max_abs_dlogp=d.max(), top1_agree=ag.mean(),
                     kl_mean=(kl.mean() if kl is not None else None), kl_max=(kl.max() if kl is not None else None),
                     p99_abs_dlogp=float(np.percentile(d, 99))))
    print(f"{'seq':<16}{'n':>6}{'mean|dlp|':>11}{'max|dlp|':>10}{'top1':>8}{'KL mean':>11}{'KL max':>10}")
    for r in rows:
        klm = f"{r['kl_mean']:.2e}" if r["kl_mean"] is not None else "-"
        klx = f"{r['kl_max']:.2e}" if r["kl_max"] is not None else "-"
        print(f"{r['name']:<16}{r['n']:>6}{r['mean_abs_dlogp']:>11.5f}{r['max_abs_dlogp']:>10.4f}{r['top1_agree']:>8.4f}{klm:>11}{klx:>10}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
