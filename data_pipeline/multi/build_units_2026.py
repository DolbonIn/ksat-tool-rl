"""Write units_2026.json (format of experiments/csat2026_baseline/code/{build_units,build_science}.py, consumed by
solve_mc.py / report_mc.py) from the 2026 test JSONL files.

ids: korean-{common|hwajak|eonmae}-n, english-{listening|reading}-n, science-<과목명>-n, social-<과목명>-n
fields: id, exam, section, no, points, answer, text (+ note for 탐구, + dataset_id linking back to the JSONL record).
Note: solve_mc.py maps exam -> subject name via SUBJ = {korean, english, science}; 사탐 units use exam="social".

Usage: python3 build_units_2026.py /raid/.../ksat_multi
"""
import json, os, sys

D = sys.argv[1]
CIRC = "①②③④⑤"
KOR_SEC = {"kor_common": "common", "kor_speech_writing": "hwajak", "kor_lang_media": "eonmae"}
SCI = {"phys1": "물리학Ⅰ", "chem1": "화학Ⅰ", "bio1": "생명과학Ⅰ", "earth1": "지구과학Ⅰ", "phys2": "물리학Ⅱ", "chem2": "화학Ⅱ",
       "bio2": "생명과학Ⅱ", "earth2": "지구과학Ⅱ"}
SOC = {"ethics_life": "생활과 윤리", "ethics_thought": "윤리와 사상", "geo_korea": "한국지리", "geo_world": "세계지리",
       "east_asian_hist": "동아시아사", "world_hist": "세계사", "economics": "경제", "politics_law": "정치와 법",
       "society_culture": "사회·문화"}


def body(r):
    q = f"{r['number']}. {r['question']}\n" + "\n".join(f"{CIRC[i]} {c}" for i, c in enumerate(r["choices"]))
    return (r["passage"] + "\n\n" if r.get("passage") else "") + q


units = []
for area in ("kor", "eng", "sci", "soc"):
    fn = os.path.join(D, f"{area}_test_2026.jsonl")
    if not os.path.exists(fn):
        print("missing", fn); continue
    for l in open(fn):
        r = json.loads(l)
        n = r["number"]
        if area == "kor":
            u = dict(id=f"korean-{KOR_SEC[r['subject']]}-{n}", exam="korean", section=KOR_SEC[r["subject"]])
        elif area == "eng":
            sec = "listening" if n <= 17 else "reading"
            u = dict(id=f"english-{sec}-{n}", exam="english", section=sec)
        elif area == "sci":
            s = SCI[r["subject"]]
            u = dict(id=f"science-{s}-{n}", exam="science", section=s, note=f"({s}, 그림·그래프·표는 문서 OCR 형식의 텍스트로 변환되어 있습니다.)")
        else:
            s = SOC[r["subject"]]
            u = dict(id=f"social-{s}-{n}", exam="social", section=s, note=f"({s}, 그림·그래프·표·지도는 문서 OCR 형식의 텍스트로 변환되어 있습니다.)")
        u.update(no=n, points=r["points"], answer=r["answer"], text=body(r), dataset_id=r["id"])
        units.append(u)
json.dump(units, open(os.path.join(D, "units_2026.json"), "w"), ensure_ascii=False, indent=1)
from collections import Counter
print(len(units), Counter((u["exam"], u["section"]) for u in units))
