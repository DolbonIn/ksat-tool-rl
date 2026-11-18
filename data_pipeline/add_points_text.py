"""Add `points_text` (the "[n점]" markers found in the official paper's text layer inside each crop rect) to
existing crop index.json files (for crops rendered before render_crops.py recorded it)."""
import json, re, sys
import fitz

POINTS_RE = re.compile(r"\[\s*(\d)\s*점\s*\]")
for idx_path in sys.argv[1:]:
    idx = json.load(open(idx_path))
    doc = fitz.open(idx["job"]["pdf"])
    for c in idx["crops"]:
        if "points_text" in c:
            continue
        c["points_text"] = [int(x) for x in POINTS_RE.findall(doc[c["pdf_page"] - 1].get_text("text", clip=fitz.Rect(*c["rect"])))]
    json.dump(idx, open(idx_path, "w"), ensure_ascii=False, indent=1)
    print(idx_path.split("/")[-2], sum(len(c["points_text"]) == 1 for c in idx["crops"]), "/", len(idx["crops"]))
