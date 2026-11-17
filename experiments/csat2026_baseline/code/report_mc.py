import json, sys, collections
units = {u["id"]: u for f in sys.argv[2:] for u in json.load(open(f))}
res = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in res: by[r["id"]].append(r)
n = max(r["sample"] for r in res) + 1
def maj_ok(rs, ans):
    v = collections.Counter(r["pred"] for r in rs if r["pred"] is not None)
    return bool(v) and v.most_common(1)[0][0] == ans
def score(ids):
    exp = sum(units[i]["points"] * sum(r["correct"] for r in by[i]) / len(by[i]) for i in ids)
    per = [sum(units[i]["points"] for i in ids if next(r for r in by[i] if r["sample"] == k)["correct"]) for k in range(n)]
    mj = sum(units[i]["points"] for i in ids if maj_ok(by[i], units[i]["answer"]))
    return exp, per, mj, sum(units[i]["points"] for i in ids)
secs = collections.defaultdict(list)
for i, u in units.items():
    if i in by: secs[(u["exam"], u["section"])].append(i)
groups = []
if ("korean", "common") in secs:
    groups += [("국어 (화법과 작문)", secs[("korean", "common")] + secs[("korean", "hwajak")]),
               ("국어 (언어와 매체)", secs[("korean", "common")] + secs[("korean", "eonmae")])]
if ("english", "reading") in secs:
    groups += [("영어", secs[("english", "listening")] + secs[("english", "reading")]),
               ("  └ 듣기(1~17, 대본 제공)", secs[("english", "listening")]), ("  └ 독해(18~45)", secs[("english", "reading")])]
groups += [(s, ids) for (e, s), ids in sorted(secs.items()) if e == "science"]
for name, ids in groups:
    exp, per, mj, tot = score(ids)
    print(f"{name:<22} 기대 {exp:5.1f}/{tot} | 회차별 {min(per)}~{max(per)} | 다수결 {mj}/{tot}")
print("\n틀린 문항 (정답률 < 100%):")
for i, u in units.items():
    if i not in by: continue
    rs = by[i]; acc = sum(r["correct"] for r in rs) / len(rs)
    if acc < 1:
        v = collections.Counter(r["pred"] for r in rs)
        print(f"  {i:<26} 정답 {u['answer']} ({u['points']}점) 정답률 {acc:4.0%}  응답분포 {dict(v)}")
print(f"\n잘림 {sum(r['finish']=='length' for r in res)}건, 오류 {sum(str(r['finish']).startswith('error') for r in res)}건, 총 {len(res)}건")
