"""Clean page furniture out of the per-crop text-layer references (render index 'ref'; original kept as 'ref_raw'):
'수학 영역(가형)' page headers that fall inside a crop rect, and the '확인 사항' box at the end of the last page
(extracted as '확인 사항 ※ 문제지와 답안지의 해당란을 …' so render_crops' stop regex missed it).
Usage: python3 clean_refs.py render3/crops/*/index.json"""
import json, re, sys

PAT = [re.compile(r"(수\s*학\s*)?영\s*역\s*\(\s*[가나]\s*형\s*\)"), re.compile(r"\(\s*[가나]\s*형\s*\)"),
       re.compile(r"확인\s*사항[\s\S]*$"), re.compile(r"※\s*\n?\s*문제지와\s*답안지[\s\S]*$")]
for p in sys.argv[1:]:
    idx = json.load(open(p)); n = 0
    for c in idx["crops"]:
        raw = c.get("ref_raw", c.get("ref") or "")
        r = raw
        for pat in PAT:
            r = pat.sub("", r)
        if r != raw:
            n += 1
        c["ref_raw"] = raw; c["ref"] = r
    json.dump(idx, open(p, "w"), ensure_ascii=False, indent=1)
    if n:
        print(p.split("/")[-2], "cleaned", n)
