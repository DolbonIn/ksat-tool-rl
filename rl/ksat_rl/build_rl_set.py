"""Assemble the RL train / val parquet files.

train = real train items with base pass@8 in [min,max]  (official answers)
      + DeepSeek-verified synthetic variants with base pass@8 in [min,max]  (consensus answers)
val   = one held-out real exam (default 2026학년도 9월 모의평가, all 46 items) for in-training monitoring;
        it and every synthetic variant seeded from it are removed from train.
The 2026 수능 test set is never used here except as a hard exclusion check.
"""
import argparse, collections, json

import pandas as pd

from ksat_rl.make_parquet import to_row


def load_rates(paths):
    by = collections.defaultdict(list)
    for p in paths:
        for l in open(p):
            r = json.loads(l); by[r["id"]].append(r["correct"])
    return {i: sum(v) / len(v) for i, v in by.items()}


ap = argparse.ArgumentParser()
ap.add_argument("--train", nargs="+", required=True, help="real-problem JSONL files (official answers)")
ap.add_argument("--synth", nargs="*", default=[])
ap.add_argument("--rates", nargs="+", required=True, help="eval_csat.py sample files (train + synth pass-rate runs)")
ap.add_argument("--test", required=True)
ap.add_argument("--val-exam", default="kice_09:2026")
ap.add_argument("--min-pass", type=float, default=0.125)
ap.add_argument("--max-pass", type=float, default=0.875)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()

rate = load_rates(a.rates)
test_ids = {json.loads(l)["id"] for l in open(a.test) if l.strip()}
val_exam, val_year = a.val_exam.split(":")
real = [json.loads(l) for f in a.train for l in open(f) if l.strip()]
synth = [json.loads(l) for s in a.synth for l in open(s) if l.strip()]
assert not test_ids & {p["id"] for p in real}, "test leak in train"
assert not any(p.get("family_id") in test_ids for p in synth), "synthetic variant seeded from a test item"

is_val = lambda p: p["exam"] == val_exam and int(p["school_year"]) == int(val_year)
val = [p for p in real if is_val(p)]
val_ids = {p["id"] for p in val}
learn = lambda p: p["id"] in rate and a.min_pass <= rate[p["id"]] <= a.max_pass
train_real = [p for p in real if not is_val(p) and learn(p)]
train_syn = [p for p in synth if p.get("family_id") not in val_ids and learn(p)]
missing = sum(p["id"] not in rate for p in real if not is_val(p)) + sum(p["id"] not in rate for p in synth)

rows = [to_row(p, "train") for p in train_real + train_syn]
pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_parquet(f"{a.out_dir}/rl_train.parquet")
pd.DataFrame([to_row(p, "val") for p in val]).to_parquet(f"{a.out_dir}/rl_val.parquet")
test = [json.loads(l) for l in open(a.test) if l.strip()]
pd.DataFrame([to_row(p, "test") for p in test]).to_parquet(f"{a.out_dir}/rl_test2026.parquet")
summary = {
    "train_real": len(train_real), "train_synth": len(train_syn), "val": len(val), "test": len(test),
    "no_rate": missing,
    "train_real_by_section": dict(collections.Counter(p["section"] for p in train_real)),
    "train_real_by_exam": dict(collections.Counter(p["exam"] for p in train_real)),
    "train_real_by_era": dict(collections.Counter("2017-2021" if int(p["school_year"]) <= 2021 else "2022-2026" for p in train_real)),
    "train_synth_by_section": dict(collections.Counter(p["section"] for p in train_syn)),
    "real_rate_hist": dict(sorted(collections.Counter(round(rate[p["id"]] * 8) for p in real if p["id"] in rate).items())),
    "synth_rate_hist": dict(sorted(collections.Counter(round(rate[p["id"]] * 8) for p in synth if p["id"] in rate).items())),
}
json.dump(summary, open(f"{a.out_dir}/rl_set_summary.json", "w"), ensure_ascii=False, indent=1)
open(f"{a.out_dir}/rl_train_ids.txt", "w").write("\n".join(p["id"] for p in train_real + train_syn) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=1))
