"""Per-problem crops for the 2017-2021 가형/나형 papers (30 items, one booklet per track), reusing
data_pipeline/render_crops.py (labels from the PDF text layer, upward scan for tall first lines, whitespace trim).

Changes vs the 2022+ renderer: section = track (ga/na); larger upward scan (max_up 45pt) plus 5pt extra top padding
(bounded by the previous problem) because crops that start with a tall fraction were cut before; a `top_ink` flag when
ink touches the final top edge (review those); points regex also accepts the '점 [4 ]' text order of some EBSi PDFs.
Output: RENDER/pages/<key>/pNN.png, RENDER/crops/<key>/<track>-<n>.png + index.json (with text-layer ref per crop).
PDFs are untrusted input: read-only. Usage: python -I render_crops_old.py --jobs /data/work3/jobs.json --out /data/render3
"""
import argparse, json, os, re, sys
sys.path.insert(0, "/pipe")
import fitz
import numpy as np
from PIL import Image
from render_crops import find_labels, adjust_tops, trim, page_form  # noqa: E402
from verify_text import clean_ref  # noqa: E402

PTS = re.compile(r"\[\s*(\d)\s*점\s*\]")
ODD_PTS = re.compile(r"점([\s?.),<>]{0,14})\[\s*(\d)\s*\]")   # some EBSi PDFs extract '[4점]' as '점 ? [4 ]' / '점 .) [4 ]'


def norm_points(txt):
    return ODD_PTS.sub(lambda m: m.group(1) + f"[{m.group(2)}점]", txt)


def points_in(txt):
    return [int(x) for x in PTS.findall(txt)]


def ink(page, rect, scale=3.0, thr=170):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width) < thr


def pad_up(page, x0, x1, top, lim, extra, scale=3.0):
    """Extend the top upward by up to `extra` pt, but only through white rows (stop below the first ink row)."""
    y_lo = max(lim, top - extra)
    if top - y_lo < 0.5:
        return top
    rows = ink(page, fitz.Rect(x0, y_lo, x1, top), scale).any(axis=1)
    hit = np.nonzero(rows)[0]
    return y_lo + (hit.max() + 2) / scale if len(hit) else y_lo


def edge_cut(page, x0, x1, top, scale=3.0):
    """Ink touching the top edge that is not a (near) full-width rule -> possible cut-off first line."""
    a = ink(page, fitz.Rect(x0, top, x1, top + 1.2), scale)
    cols = a.any(axis=0)
    return bool(cols.any() and cols.mean() < 0.6)


def page_bottom(page):
    """Body bottom = top of the page-number footer. data_pipeline/render_crops.find_labels takes ANY centred line below 0.85H
    as the footer, which on narrow-gutter EBSi pages can be a right-column choice line ('④ …') -> crops cut above the last
    choice row. Here only bare page-number lines ('7', '12', '7/12') count."""
    import re as _re
    W, H = page.rect.width, page.rect.height
    foot = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"]).strip()
            bb = l["bbox"]
            if bb[1] > H * 0.85 and abs((bb[0] + bb[2]) / 2 - W / 2) < 60 and _re.fullmatch(r"\d{1,2}(\s*/\s*\d{1,2})?", t):
                foot.append(bb[1])
    return (min(foot) - 4) if foot else H * 0.92


def process(job, out, dpi_page=150, dpi_crop=300, extra_top=5.0):
    pdir = os.path.join(out, "pages", job["key"]); cdir = os.path.join(out, "crops", job["key"])
    os.makedirs(pdir, exist_ok=True); os.makedirs(cdir, exist_ok=True)
    doc = fitz.open(job["pdf"])
    forms = [page_form(p) for p in doc]
    use = [i for i, f in enumerate(forms) if f == "odd"] if "even" in forms else list(range(doc.page_count))
    index = []
    for pno, i in enumerate(use, 1):
        page = doc[i]
        W = page.rect.width
        labels, stops, _ = find_labels(page)
        bottom = page_bottom(page)
        labels = [l for l in labels if 1 <= l["num"] <= 30]
        if not labels:
            continue
        page.get_pixmap(dpi=dpi_page).save(os.path.join(pdir, f"p{pno:02d}.png"))
        left_x = min([l["x0"] for l in labels if l["col"] == 0], default=W * 0.1)
        tops = adjust_tops(page, labels, left_x, max_up=45.0)
        for lab in labels:
            prev = [l for l in labels if l["col"] == lab["col"] and l["y0"] < lab["y0"]]
            prev_lim = max(l["y0"] for l in prev) + 14 if prev else page.rect.height * 0.05
            nxt = [l for l in labels if l["col"] == lab["col"] and l["y0"] > lab["y0"]]
            y1 = min([tops[id(l)] for l in nxt], default=bottom + 6) - 4
            st = [y for c, y in stops if c == lab["col"] and lab["y0"] < y < y1]
            if st:
                y1 = min(st) - 6
            x0 = lab["x0"] - 12
            x1 = (W / 2 - 4) if lab["col"] == 0 else (W - left_x + 12)
            top = pad_up(page, x0, x1, max(prev_lim, tops[id(lab)]), prev_lim, extra_top)
            rect = fitz.Rect(x0, top, x1, y1)
            edge = edge_cut(page, x0, x1, top)
            raw = norm_points(page.get_text("text", clip=rect))
            pix = page.get_pixmap(dpi=dpi_crop, clip=rect)
            img = trim(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), pad=30)
            name = f"{job['track']}-{lab['num']}"
            if any(e["name"] == name for e in index):
                name += "_dup"
            fn = os.path.join(cdir, name + ".png")
            img.save(fn)
            index.append(dict(name=name, section=job["track"], number=lab["num"], page=pno, pdf_page=i + 1, col=lab["col"],
                              rect=[round(v, 1) for v in rect], points_text=points_in(raw), top_ink=edge,
                              ref=clean_ref(raw), png=os.path.relpath(fn, out), w=img.width, h=img.height))
    json.dump(dict(job=job, crops=index), open(os.path.join(cdir, "index.json"), "w"), ensure_ascii=False, indent=1)
    nums = sorted(e["number"] for e in index)
    pts = {e["number"]: e["points_text"] for e in index}
    tot = sum(v[0] for v in pts.values() if len(v) == 1)
    nbad = [n for n, v in pts.items() if len(v) != 1]
    n4 = sum(1 for v in pts.values() if v == [4])
    ok = nums == list(range(1, 31)) and not nbad and tot == 100
    print(f"{job['key']}: crops={len(index)} 4pt={n4} sum={tot} badpts={nbad} top_ink={[e['number'] for e in index if e['top_ink']]} "
          f"{'OK' if ok else 'CHECK nums=' + str(nums)}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default="2021_edu_04_ga,2021_edu_04_na")
    a = ap.parse_args()
    bad = 0
    for j in json.load(open(a.jobs)):
        if (a.only and j["key"] not in a.only.split(",")) or j["key"] in a.skip.split(","):
            continue
        bad += not process(j, a.out)
    print("papers needing review:", bad)


if __name__ == "__main__":
    main()
