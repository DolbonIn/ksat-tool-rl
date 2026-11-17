"""Build MC units for 2026 CSAT 과학탐구 from OCR-style JSON transcriptions + official answer keys."""
import json, glob, os, re
C = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}
units = []
for kf in sorted(glob.glob("data/science_keys/*.txt")):
    tag = os.path.basename(kf)[:-4]            # e.g. 01_물리학Ⅰ
    subj = tag.split("_", 1)[1]
    toks = [t.strip() for t in open(kf).read().split() if t.strip()]
    key = {}
    for i in range(len(toks) - 2):
        a, b, c = toks[i:i+3]
        if a.isdigit() and b in C and c in ("2", "3") and 1 <= int(a) <= 20: key[int(a)] = (C[b], int(c))
    assert len(key) == 20 and sum(p for _, p in key.values()) == 50, (tag, len(key))
    src = f"data/science/{tag}.json"
    if not os.path.exists(src): print("missing transcription:", src); continue
    qs = {q["no"]: q for q in json.load(open(src))["questions"]}
    assert sorted(qs) == list(range(1, 21)), tag
    for n in range(1, 21):
        units.append(dict(id=f"science-{subj}-{n}", exam="science", section=subj, no=n, points=key[n][1], answer=key[n][0],
                          note=f"({subj}, 그림·그래프·표는 문서 OCR 형식의 텍스트로 변환되어 있습니다.)", text=qs[n]["text"]))
json.dump(units, open("units_science.json", "w"), ensure_ascii=False, indent=1)
print(len(units), "units")
