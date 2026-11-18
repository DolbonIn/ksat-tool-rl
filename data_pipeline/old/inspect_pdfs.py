"""Quick inspection of downloaded PDFs (untrusted, read-only): magic bytes, pages, size, text-layer sample, labels found."""
import glob, json, re, sys
import fitz
sys.path.insert(0, "/pipe")
from render_crops import find_labels, page_form  # noqa: E402

import os
for p in sorted(x for x in glob.glob(sys.argv[1], recursive=True) if os.path.isfile(x) and not x.endswith((".json", ".zip"))):
    head = open(p, "rb").read(5)
    if not p.lower().endswith(".pdf"):
        print(p.split("/data/")[-1], "magic", head[:4]); continue
    if head[:4] != b"%PDF":
        print("NOT PDF", p, head); continue
    d = fitz.open(p)
    pg = d[0]
    labs = []
    for i, page in enumerate(d):
        L, st, bot = find_labels(page)
        labs.append((i + 1, page_form(page), [l["num"] for l in L]))
    txt = re.sub(r"\s+", " ", pg.get_text())[:160]
    print(p.split("/data/")[-1], d.page_count, [round(pg.rect.width), round(pg.rect.height)], "|", txt)
    if "정답" not in p and "answer" not in p and "답안" not in p and "hsj" not in p:
        print("   labels:", labs)
