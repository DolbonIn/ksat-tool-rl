"""Build jobs JSON for the 2017-2021 math papers (가형/나형). Paths are container paths (/data/...). Only file names are
inspected here (no parsing of downloaded content). Usage: python3 make_jobs_old.py > /raid/.../work3/jobs.json"""
import glob, json, os, re

D = "/raid/cache/b300-07/ksat-rl/data"
TR = {"ga": "가", "na": "나"}
HELD = {  # 실제 시행일 (KICE: 수능/모평; 교육청: EBSi listing month, 2020 sittings moved)
    "csat": {2017: "2016-11-17", 2018: "2017-11-23", 2019: "2018-11-15", 2020: "2019-11-14", 2021: "2020-12-03"},
}
jobs = []
for sy in range(2017, 2022):
    for ex in ("csat", "kice_06", "kice_09"):
        d = f"{D}/dl3/kice/{sy}_{ex}"
        files = [f for f in glob.glob(d + "/**/*.pdf", recursive=True)]
        for tr, ko in TR.items():
            if ex == "csat":
                pdf = [f for f in files if "unzipped" in f and ko + "형" in f and "홀" in os.path.basename(f)]
                key = [f for f in files if "unzipped" not in f]
            else:
                pdf = [f for f in files if ko + "형" in f and "문제지" in f]
                key = [f for f in files if ko + "형" in f and ("정답" in f or "답안" in f)] or [f for f in files if "정답" in f and "형" not in os.path.basename(f)]
            assert len(pdf) == 1 and len(key) == 1, (sy, ex, tr, pdf, key)
            jobs.append(dict(key=f"{sy}_{ex}_{tr}", exam=ex, school_year=sy, track=tr, track_ko=ko + "형",
                             held=HELD.get(ex, {}).get(sy), pdf=pdf[0].replace(D, "/data"), answer_key=key[0].replace(D, "/data")))
man = json.load(open(f"{D}/dl3/edu/manifest.json"))
for sy in range(2017, 2022):
    for ex in ("edu_03", "edu_04", "edu_07", "edu_10"):
        for tr, ko in TR.items():
            m = {r["kind"]: r for r in man if r["school_year"] == sy and r["exam"] == ex and r["track"] == tr}
            if "paper" not in m:
                continue
            jobs.append(dict(key=f"{sy}_{ex}_{tr}", exam=ex, school_year=sy, track=tr, track_ko=ko + "형", held=m["paper"]["held"],
                             title=m["paper"]["title"], pdf="/data/dl3/edu/" + m["paper"]["path"],
                             answer_img="/data/dl3/edu/" + m["answer"]["path"] if "answer" in m else None,
                             hsj="/data/dl3/edu/" + m["hsj"]["path"] if "hsj" in m else None))
print(json.dumps(jobs, ensure_ascii=False, indent=1))
