"""Compare crop rects of render3 (used for OCR) with render3_v2 (fixed footer detection). Lists crops whose bottom moved
down by > 2pt and whether the newly included strip contains text-layer content (= the old crop was cut)."""
import glob, json, os, sys
import fitz
import numpy as np

old_root, new_root = sys.argv[1], sys.argv[2]
out = {}
for f in sorted(glob.glob(os.path.join(new_root, "crops", "*", "index.json"))):
    key = f.split("/")[-2]
    of = os.path.join(old_root, "crops", key, "index.json")
    if not os.path.exists(of):
        continue
    new = {c["number"]: c for c in json.load(open(f))["crops"]}
    oldi = json.load(open(of))
    doc = fitz.open(oldi["job"]["pdf"])
    for c in oldi["crops"]:
        n = new.get(c["number"])
        if not n:
            continue
        if n["rect"][3] > c["rect"][3] + 2:
            p = doc[c["pdf_page"] - 1]
            strip = fitz.Rect(c["rect"][0], c["rect"][3], c["rect"][2], n["rect"][3])
            txt = p.get_text("text", clip=strip).strip()
            pix = p.get_pixmap(dpi=100, clip=strip, colorspace=fitz.csGRAY)
            ink = (np.frombuffer(pix.samples, dtype=np.uint8) < 160).sum()
            out[f"{key}/{c['name']}"] = dict(old_y1=c["rect"][3], new_y1=n["rect"][3], points=c["points_text"], strip_text=txt[:80], ink=int(ink))
            print(key, c["name"], c["points_text"], round(c["rect"][3]), "->", round(n["rect"][3]), "ink", ink, repr(txt[:60]))
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
print("changed", len(out))
