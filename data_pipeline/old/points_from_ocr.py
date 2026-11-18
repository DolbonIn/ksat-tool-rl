"""Points for papers without a text layer (2021_edu_04_ga/na): from the OCR of the printed '[n점]' marker.
Accepted per item only if pass A's <points>, the '[n점]' marker in pass A's stem and in the final (selected) stem agree;
the paper must sum to 100 and number every item 1..30 (OCR <number> = crop order). Output {key: {n: points}} plus a log.
Usage: python3 points_from_ocr.py --ocr ocr3 --keys 2021_edu_04_ga,2021_edu_04_na --out work3/points_e04.json
"""
import argparse, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--ocr", required=True); ap.add_argument("--keys", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
res, ok_all = {}, True
for key in a.keys.split(","):
    tr = key.rsplit("_", 1)[1]
    pts, bad = {}, []
    for n in range(1, 31):
        fn = os.path.join(a.ocr, key, f"{tr}-{n}.json")
        if not os.path.exists(fn):
            bad.append((n, "no ocr")); continue
        o = json.load(open(fn))
        ap_ = o.get("a_parsed") or {}; fi = o.get("final") or {}
        vals = set()
        for x in (ap_.get("points"), fi.get("points")):
            if x and re.fullmatch(r"\s*[234]\s*", str(x)):
                vals.add(int(x))
        for st in (ap_.get("stem") or "", fi.get("stem") or ""):
            vals |= {int(m) for m in re.findall(r"\[\s*([234])\s*점\s*\]", st)}
        num = re.sub(r"\D", "", str(fi.get("number")))
        if num != str(n):
            bad.append((n, f"ocr number {fi.get('number')}"))
        if len(vals) != 1:
            bad.append((n, f"points disagree {vals}")); continue
        pts[n] = vals.pop()
    tot = sum(pts.values())
    print(key, "sum", tot, "4pt", sum(v == 4 for v in pts.values()), "bad", bad)
    ok_all &= tot == 100 and not bad
    res[key] = {str(k): v for k, v in pts.items()} if tot == 100 and not bad else {}
json.dump(res, open(a.out, "w"), indent=1)
print("OK" if ok_all else "CHECK")
