"""Build the problem pool for TIR trajectory generation (train items only).

Sources (under /raid/cache/b300-07/ksat-rl/data):
  real  ksat_math/train.jsonl (4-point only), ksat_math_old/train_2017_2021_hard.jsonl     official answers
  synth synth/synth_pr1.jsonl, synth/synth_pr2.jsonl                                        consensus answers
Exclusions (held-out test/val):
  every id in ksat_math/test_2026.jsonl (2026 수능), every exam=="kice_09" & school_year==2026 item (2026 9월 모평),
  and every synthetic item whose family_id is one of those ids.
Also dropped: items with has_figure (the policy only sees text) and items without an integer answer.

Selection favours items where computation / enumeration helps (keyword "tool affinity"), items the policy
does not already solve 8/8 (runs/eval/passrate_*.jsonl), and spreads attempts over (source, section).
Writes data/tool/pool.jsonl (+ pool_summary.json); generation reads `order` to attempt in rank order.
"""
import argparse
import collections
import json
import os
import random
import re

ROOT = "/raid/cache/b300-07/ksat-rl"
D = f"{ROOT}/data"
EV = f"{ROOT}/runs/eval"

AFFINITY = [  # (regex, weight): computation, enumeration or exact-arithmetic signals
    (r"경우의 수", 3), (r"순서쌍", 3), (r"개수", 2), (r"a_\{?n ?\+ ?[12]\}?|a_\{n\+1\}", 3), (r"나머지", 2),
    (r"배수|약수", 2), (r"확률", 2), (r"\\sum|합", 1), (r"수열", 1), (r"\\int|정적분", 1), (r"넓이", 1),
    (r"최댓값|최솟값", 1), (r"\\lim", 1), (r"\\mathrm\{C\}|\\binom|이항", 1), (r"자연수", 1), (r"정수", 1),
    (r"삼차함수|사차함수|다항함수", 1), (r"벡터|좌표", 1), (r"\\log", 1), (r"방정식", 1),
]
SECTION_BONUS = {"prob_stat": 2, "common": 0, "calculus": 0, "geometry": 0, "ga": 0, "na": 0}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def passrates(files):
    acc = collections.defaultdict(list)
    for f in files:
        if os.path.exists(f):
            for r in load(f):
                acc[r["id"]].append(bool(r["correct"]))
    return {k: (sum(v), len(v)) for k, v in acc.items()}


def affinity(text, section):
    return sum(w for pat, w in AFFINITY if re.search(pat, text)) + SECTION_BONUS.get(section, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/tool/pool.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    test = load(f"{D}/ksat_math/test_2026.jsonl")
    train = load(f"{D}/ksat_math/train.jsonl")
    old = load(f"{D}/ksat_math_old/train_2017_2021_hard.jsonl")
    synth = load(f"{D}/synth/synth_pr1.jsonl") + load(f"{D}/synth/synth_pr2.jsonl")
    excluded = {p["id"] for p in test}
    excluded |= {p["id"] for p in train + old if p["exam"] == "kice_09" and int(p["school_year"]) == 2026}
    pr = passrates([f"{EV}/passrate_train.jsonl", f"{EV}/passrate_old_hard.jsonl",
                    f"{EV}/passrate_synth_pr1_merged.jsonl", f"{EV}/passrate_synth_pr1c.jsonl",
                    f"{EV}/passrate_synth_pr2.jsonl"])

    stats = collections.Counter()
    pool, seen = [], set()
    for src, rows in (("real", train), ("real_old", old), ("synth", synth)):
        for p in rows:
            stats[f"{src}_total"] += 1
            if p["id"] in excluded or p.get("family_id") in excluded:
                stats[f"{src}_excluded_test_val"] += 1
                continue
            if src == "real" and int(p.get("points") or 0) != 4:
                stats[f"{src}_not_4pt"] += 1
                continue
            if p.get("has_figure") in (True, "True"):
                stats[f"{src}_figure"] += 1
                continue
            if p["id"] in seen or p.get("answer") is None:
                continue
            seen.add(p["id"])
            k, n = pr.get(p["id"], (None, None))
            rate = None if n is None else k / n
            aff = affinity(p["text"], p["section"])
            score = aff + (2.0 * (1 - rate) if rate is not None else 1.0) + rnd.random() * 0.5
            pool.append({**p, "source": src, "passrate": rate, "pass_k": k, "pass_n": n,
                         "affinity": aff, "score": round(score, 3)})
    assert not any(p["id"] in excluded or p.get("family_id") in excluded for p in pool)

    # rank: round-robin over (source, section) strata, each stratum sorted by score (desc)
    strata = collections.defaultdict(list)
    for p in pool:
        strata[(p["source"], p["section"])].append(p)
    for v in strata.values():
        v.sort(key=lambda p: -p["score"])
    weights = {"real": 1.0, "real_old": 0.8, "synth": 1.0}   # attempts per round, per stratum
    order, credit = [], collections.Counter()
    keys = sorted(strata)
    while any(strata[k] for k in keys):
        for k in keys:
            credit[k] += weights[k[0]]
            while credit[k] >= 1 and strata[k]:
                order.append(strata[k].pop(0))
                credit[k] -= 1
    for i, p in enumerate(order):
        p["order"] = i
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for p in order:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    summ = {"n_pool": len(order), "filters": dict(stats), "excluded_ids": len(excluded),
            "by_source_section": dict(collections.Counter(f"{p['source']}/{p['section']}" for p in order)),
            "first_1000_by_source_section": dict(collections.Counter(f"{p['source']}/{p['section']}" for p in order[:1000]))}
    json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
