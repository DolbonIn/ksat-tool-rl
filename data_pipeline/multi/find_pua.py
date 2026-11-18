"""Find segments whose text layer contains private-use glyphs that segment.py could not decode (shown as '□' in the
draft): old-Hangul syllables (중세 국어), equation symbols (−, ×, Greek, …). Writes {key/sid: n_unmapped} JSON.
Usage: python -I find_pua.py --jobs J --render /data/render2 --out /data/work2/pua_segments.json
"""
import argparse, json, os, sys
import fitz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import PUA  # noqa: E402


def unmapped(s):
    return sum(1 for c in s if (0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) <= 0x10FFFD) and ord(c) not in PUA)


ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for j in json.load(open(a.jobs)):
    sf = os.path.join(a.render, j["key"], "segments.json")
    if not os.path.exists(sf):
        continue
    doc = fitz.open(j["pdf"])
    for s in json.load(open(sf))["segments"]:
        n = 0
        for r in s["regions"]:
            n += unmapped(doc[r["page"] - 1].get_text("text", clip=fitz.Rect(*r["rect"])))
        if n:
            res[f"{j['key']}/{s['sid']}"] = n
json.dump(res, open(a.out, "w"), indent=1)
from collections import Counter
print(len(res), "segments with undecoded PUA glyphs;", Counter(k.split("/")[0].split("_")[-1] if "_sci_" not in k and "_soc_" not in k else k.split("_")[3] for k in res))
