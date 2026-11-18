"""Coverage test for crops: every text-layer span in a page body (below the header rule area, above the page-number
footer, excluding the '확인 사항' box) must lie inside some crop rect or shared-preamble rect. Uncovered spans mean a crop
cut off part of an item (top or bottom). Usage: python -I coverage_check.py /data/render3 /data/work3/preambles.json"""
import glob, json, os, re, sys
import fitz

root = sys.argv[1]
pre = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else []
bad = 0
for f in sorted(glob.glob(os.path.join(root, "crops", "*", "index.json"))):
    idx = json.load(open(f))
    if idx.get("no_text_layer"):
        continue
    key = f.split("/")[-2]
    doc = fitz.open(idx["job"]["pdf"])
    rects = {}
    for c in idx["crops"]:
        rects.setdefault(c["pdf_page"], []).append(fitz.Rect(*c["rect"]))
    for g in pre:
        if g["key"] == key:
            rects.setdefault(g["page"], []).append(fitz.Rect(*g["rect"]))
    for pno, rs in rects.items():
        p = doc[pno - 1]; W, H = p.rect.width, p.rect.height
        top_body = min(r.y0 for r in rs) - 40
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = s["text"].strip()
                    if not t:
                        continue
                    bb = fitz.Rect(s["bbox"]); cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                    if cy < top_body or cy > H * 0.9 or re.fullmatch(r"[\d/ ]+", t) and abs(cx - W / 2) < 60:
                        continue
                    if re.search(r"확인\s*사항|답안지|문제지와|해당란|기입|표기|확인하시오|^[※*◦()\.]$|5\s*지\s*선\s*다\s*형|단\s*답\s*형|수학\s*영역|[가나]\s*형|^\d{1,2}$", t):
                        continue
                    if not any(r.contains(fitz.Point(cx, cy)) for r in rs):
                        bad += 1
                        print(key, f"p{pno}", [round(v) for v in bb], repr(t[:50]))
print("uncovered spans:", bad)
