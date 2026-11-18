"""List 0/n solve-check suspects with attributes, and write id lists for follow-up passes.
Usage: python3 suspects.py <solve.jsonl> <items.jsonl[,…]> <out_prefix>
Writes <out_prefix>_length.txt (suspects with a truncated sample -> re-solve with 32k tokens) and
<out_prefix>_reocr.txt (key/sid of suspects with figures/tables or failed text-layer check -> figure proofread / re-OCR).
"""
import json, sys
from collections import defaultdict, Counter

sol, items, pre = sys.argv[1], sys.argv[2], sys.argv[3]
by = defaultdict(list)
for l in open(sol):
    r = json.loads(l)
    if r.get("finish") != "error":
        by[r["id"]].append(r)
recs = {}
for f in items.split(","):
    for l in open(f):
        r = json.loads(l); recs[r["id"]] = r
sus = [i for i, rs in by.items() if i in recs and not any(x["correct"] for x in rs)]
length = [i for i in sus if any(x["finish"] == "length" for x in by[i])]
reocr = []
c = Counter()
for i in sus:
    r = recs[i]
    fig = r["has_figure"] or "[표]" in r["text"]
    tl = r["verify"].get("textlayer_ok")
    c[(("fig" if fig else "text"), ("tl_ok" if tl else "tl_flag"), ("len" if i in length else "nolen"))] += 1
    if (fig or not tl) and i not in length:
        area = r["area"]
        key = f"{r['school_year']}_{r['exam']}_{area}" + (f"_{r['subject']}" if area in ("sci", "soc") else "")
        sec = r["subject"] if area != "eng" else "eng"
        reocr.append(f"{key}/{sec}-q{r['number']:02d}")
open(pre + "_length.txt", "w").write(" ".join(length))
open(pre + "_reocr.txt", "w").write(",".join(reocr))
print(f"{len(by)} solved, {len(sus)} suspects; with truncated sample {len(length)}; to re-OCR/proofread {len(reocr)}")
for k, v in sorted(c.items()):
    print(" ", k, v)
