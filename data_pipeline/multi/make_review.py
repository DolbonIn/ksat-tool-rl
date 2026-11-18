"""Build visual-review batches for the 2026 test sets: every item with a figure/table (and figure passages), plus a
random sample of the others. Each entry lists the rendered crop images and the transcription to compare.
Usage: python3 make_review.py <ksat_multi_dir> <render2_dir> <out_dir> [batch_size] [sample_frac]
"""
import json, os, random, re, sys

D, R, O = sys.argv[1], sys.argv[2], sys.argv[3]
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 25
frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
os.makedirs(O, exist_ok=True)
random.seed(0)
entries, seen_p = [], set()
for area in ("kor", "eng", "sci", "soc"):
    for l in open(os.path.join(D, f"{area}_test_2026.jsonl")):
        r = json.loads(l)
        key = f"{r['school_year']}_{r['exam']}_{area}" + (f"_{r['subject']}" if area in ("sci", "soc") else "")
        segs = {s["sid"]: s for s in json.load(open(os.path.join(R, key, "segments.json")))["segments"]}
        sid = f"{r['subject'] if area != 'eng' else 'eng'}-q{r['number']:02d}"
        s = segs[sid]
        fig = r["has_figure"] or "[표]" in r["question"] or "[그림]" in (r["question"] + " ".join(r["choices"]))
        if fig or random.random() < frac:
            entries.append(dict(id=r["id"], kind="question", reason="figure/table" if fig else "sample",
                                images=[os.path.join(R, key, p) for p in s["pngs"]],
                                transcription=dict(question=r["question"], choices=r["choices"])))
        if r.get("passage_id") and r["passage_id"] not in seen_p and area != "eng" or (area == "eng" and r.get("passage_id") and r["number"] > 17 and r["passage_id"] not in seen_p):
            seen_p.add(r["passage_id"])
            m = re.search(r"-p(\d+)_(\d+)$", r["passage_id"])
            psid = f"{r['subject'] if area != 'eng' else 'eng'}-p{int(m.group(1)):02d}_{int(m.group(2)):02d}"
            ps = segs.get(psid)
            pf = "[그림]" in (r["passage"] or "") or "[표]" in (r["passage"] or "")
            if ps and (pf or random.random() < frac):
                entries.append(dict(id=r["passage_id"], kind="passage", reason="figure/table" if pf else "sample",
                                    images=[os.path.join(R, key, p) for p in ps["pngs"]], transcription=dict(passage=r["passage"])))
batches = [entries[i:i + bs] for i in range(0, len(entries), bs)]
for i, b in enumerate(batches):
    json.dump(b, open(os.path.join(O, f"batch_{i:02d}.json"), "w"), ensure_ascii=False, indent=1)
print(len(entries), "entries", len(batches), "batches;", sum(e["reason"] == "figure/table" for e in entries), "figure/table")
