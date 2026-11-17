import json, sys, collections
from problems import PROBLEMS
res = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")]
by = collections.defaultdict(list)
for r in res: by[r["id"]].append(r)
n = max(r["sample"] for r in res) + 1
maj = {}
print(f"{'문항':<15}{'배점':>4}{'정답':>6}{'정답률':>8}{'다수결':>10}{'평균토큰':>10}")
for p in PROBLEMS:
    rs = by[p["id"]]
    acc = sum(r["correct"] for r in rs) / len(rs)
    votes = collections.Counter(r["pred"] for r in rs if r["pred"] is not None)
    top = votes.most_common(1)[0][0] if votes else None
    maj[p["id"]] = next((r["correct"] for r in rs if r["pred"] == top), False)
    print(f"{p['id']:<15}{p['points']:>4}{p['answer']:>6}{acc:>8.0%}{str(top)[:8]:>10}{sum(r['tokens'] for r in rs)/len(rs):>10.0f}")
common = [p for p in PROBLEMS if p["section"] == "common"]
print()
for sec, name in [("prob_stat", "확률과 통계"), ("calculus", "미적분"), ("geometry", "기하")]:
    ps = common + [p for p in PROBLEMS if p["section"] == sec]
    exp = sum(p["points"] * sum(r["correct"] for r in by[p["id"]]) / len(by[p["id"]]) for p in ps)
    per = [sum(p["points"] for p in ps if next(r for r in by[p["id"]] if r["sample"] == k)["correct"]) for k in range(n)]
    mj = sum(p["points"] for p in ps if maj[p["id"]])
    print(f"[{name}] 기대점수 {exp:.1f}/100 | 회차별 {per} (min {min(per)}, max {max(per)}) | 다수결({n}회) {mj}/100")
trunc = sum(r["finish"] == "length" for r in res); err = sum(str(r["finish"]).startswith("error") for r in res)
print(f"\n잘림(length) {trunc}건, 오류 {err}건, 총 {len(res)}건")
