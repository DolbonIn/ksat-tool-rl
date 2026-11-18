"""Per-problem crops for papers WITHOUT a text layer (2021학년도 4월 학평, held 2020-05-21: vector outlines only).

Layout: 2 columns split by a vertical rule; each problem starts with a bold label flush with the column's left edge,
continuation lines / choices / boxes are indented. Per column: text bands (rows separated by white) -> label bands =
bands whose leftmost ink is within `tol` of the column's minimum and whose first glyph is not a tall vertical box border.
Labels are numbered in reading order (must give exactly 30); the OCR'd <number> is cross-checked later.
Crops are padded (8pt above the band, through white rows only). Output mirrors render_crops_old.py (ref = '' since
there is no text layer; points_text = [] and points come from OCR + the sum-to-100 check).
Usage: python -I segment_notext.py --jobs /data/work3/jobs.json --out /data/render3 --only 2021_edu_04_ga,2021_edu_04_na
"""
import argparse, json, os, sys
sys.path.insert(0, "/pipe")
import fitz
import numpy as np
from PIL import Image
from render_crops import trim  # noqa: E402

DPI = 200


def bands(rows, min_gap=3):
    out, start, gap = [], None, 0
    for y, r in enumerate(list(rows) + [False] * (min_gap + 1)):
        if r:
            if start is None:
                start = y
            gap = 0; end = y
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                out.append((start, end + 1)); start = None
    return out


def page_layout(a):
    H, W = a.shape
    # header rule: first row (top 25%) with ink across > 70% of the width
    hr = [y for y in range(int(H * 0.25)) if a[y].mean() > 0.7]
    top = (max(hr) + 4) if hr else int(H * 0.12)
    # vertical divider: column near the center with the longest ink run below the header
    cx = range(int(W * 0.45), int(W * 0.55))
    vx = max(cx, key=lambda x: a[top:, x].sum())
    col = a[top:, vx]
    ys = np.nonzero(col)[0]
    bottom = min(top + (ys.max() if len(ys) else int(H * 0.85)), int(H * 0.895))   # stop above the page-number box
    return top, bottom, vx


def labels_in(a, x0, x1, top, bottom, tol):
    sub = a[top:bottom, x0:x1]
    res = []
    bs = bands(sub.any(axis=1))
    lefts = []
    for b0, b1 in bs:
        xs = np.nonzero(sub[b0:b1].any(axis=0))[0]
        lefts.append(xs.min() if len(xs) else 10 ** 6)
    if not bs:
        return res
    mn = min(lefts)
    for (b0, b1), lx in zip(bs, lefts):
        h = b1 - b0
        if lx - mn > tol or h > 200 or h < 12:
            continue
        g = sub[b0:b1, lx:lx + 6]
        if g.any(axis=1).mean() > 0.95 and h > 40:     # tall vertical box border
            continue
        res.append(top + b0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--only", required=True)
    ap.add_argument("--tol", type=int, default=6)
    a_ = ap.parse_args()
    for j in json.load(open(a_.jobs)):
        if j["key"] not in a_.only.split(","):
            continue
        doc = fitz.open(j["pdf"])
        pdir = os.path.join(a_.out, "pages", j["key"]); cdir = os.path.join(a_.out, "crops", j["key"])
        os.makedirs(pdir, exist_ok=True); os.makedirs(cdir, exist_ok=True)
        found = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
            a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width) < 160
            top, bottom, vx = page_layout(a)
            W = a.shape[1]
            for c, (x0, x1) in enumerate(((int(W * 0.06), vx - 3), (vx + 3, int(W * 0.96)))):
                ys = labels_in(a, x0, x1, top, bottom, a_.tol)
                for k, y in enumerate(ys):
                    y_end = ys[k + 1] - 3 if k + 1 < len(ys) else bottom - 2
                    found.append(dict(page=i, col=c, y0=y, y1=y_end, x0=x0, x1=x1, top=top, vx=vx))
            page.get_pixmap(dpi=150).save(os.path.join(pdir, f"p{i + 1:02d}.png"))
        s = 72 / DPI
        index = []
        for n, f in enumerate(found, 1):
            page = doc[f["page"]]
            y_top = max(f["top"], f["y0"] - int(8 / s)) * s
            xa, xb = (f["x0"] * s - 4, (f["vx"] - 10) * s) if f["col"] == 0 else ((f["vx"] + 10) * s, f["x1"] * s + 4)   # keep the column rule out
            rect = fitz.Rect(xa, y_top, xb, f["y1"] * s)
            pix = page.get_pixmap(dpi=300, clip=rect)
            img = trim(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), pad=30)
            name = f"{j['track']}-{n}"
            fn = os.path.join(cdir, name + ".png"); img.save(fn)
            index.append(dict(name=name, section=j["track"], number=n, page=f["page"] + 1, pdf_page=f["page"] + 1, col=f["col"],
                              rect=[round(v, 1) for v in rect], points_text=[], top_ink=False, ref="", no_text_layer=True,
                              png=os.path.relpath(fn, a_.out), w=img.width, h=img.height))
        json.dump(dict(job=j, crops=index, no_text_layer=True), open(os.path.join(cdir, "index.json"), "w"), ensure_ascii=False, indent=1)
        print(j["key"], "labels found:", len(found), [(f["page"] + 1, f["col"]) for f in found])


if __name__ == "__main__":
    main()
