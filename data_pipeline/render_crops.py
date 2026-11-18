"""Render exam PDFs to page PNGs and per-problem crops using the PDF text layer for problem labels.

KICE / 교육청 math papers are 2-column A3/B4 pages. Problem labels ("13.") are real text in the PDF
(the math itself is in private-use glyphs), so we locate each label, then crop from that label down to
the next label in the same column (or the column bottom).

Input: a jobs JSON list of {"key": "2026_csat", "pdf": "/abs/path.pdf", "exam": "csat", "school_year": 2026}
Output (under --out):
  pages/<key>/p<NN>.png          150 dpi full pages (for human review)
  crops/<key>/<section>-<num>.png 300 dpi problem crops (whitespace-trimmed)
  crops/<key>/index.json          crop metadata
PDFs are untrusted input: read-only, nothing executed. Run with python -I.
"""
import argparse, json, os, re, sys
import fitz
import numpy as np
from PIL import Image, ImageOps

SECTION_HDR = [("확률과 통계", "prob_stat"), ("확률과통계", "prob_stat"), ("미적분", "calculus"), ("기하", "geometry")]
LABEL_RE = re.compile(r"^\s*(\d{1,2})\s*\.(?:\s|$|[가-힣-(])")
STOP_RE = re.compile(r"^\s*[*※]\s*확인\s*사항")


def page_section(page):
    top = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.16))
    for k, v in SECTION_HDR:
        if f"({k})" in top.replace(" ", "").replace("확률과통계", "확률과 통계") or f"({k})" in top:
            return v
    return "common"


def page_form(page):
    top = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.2))
    if "짝수형" in top:
        return "even"
    if "홀수형" in top:
        return "odd"
    return None


def lines(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"])
            if t.strip():
                out.append((l["bbox"], t, l["spans"][0]["size"]))
    return out


def find_labels(page):
    W, H = page.rect.width, page.rect.height
    ls = lines(page)
    cands = []
    for bbox, t, size in ls:
        m = LABEL_RE.match(t)
        if not m or bbox[1] < H * 0.08 or bbox[1] > H * 0.92:
            continue
        col = 0 if bbox[0] < W / 2 else 1
        cands.append(dict(num=int(m.group(1)), x0=bbox[0], y0=bbox[1], col=col, size=size))
    labels = []
    for col in (0, 1):
        cc = [c for c in cands if c["col"] == col]
        if not cc:
            continue
        xmin = min(c["x0"] for c in cc)
        labels += [c for c in cc if c["x0"] - xmin < 8]
    stops = [(bbox[0] < W / 2 and 0 or 1, bbox[1]) for bbox, t, _ in ls if STOP_RE.match(t)]
    # footer = page-number line near the bottom center
    foot = [bbox[1] for bbox, t, _ in ls if bbox[1] > H * 0.85 and abs((bbox[0] + bbox[2]) / 2 - W / 2) < 40]
    bottom = min(foot) - 4 if foot else H * 0.92
    return sorted(labels, key=lambda c: (c["col"], c["y0"])), stops, bottom


def adjust_tops(page, labels, left_x, gap_pt=6.0, ink=160, max_up=30.0):
    """A label's y0 is the top of its text line, but a tall first-line expression (e.g. a fraction whose numerator
    sits above the label baseline) can start higher. Scan upward from each label until a white band of
    >= gap_pt points and use the topmost ink row as the problem top."""
    W, H = page.rect.width, page.rect.height
    tops = {}
    for lab in labels:
        prev = [l["y0"] for l in labels if l["col"] == lab["col"] and l["y0"] < lab["y0"]]
        y_lo = max(max(prev) + 2 if prev else H * 0.06, lab["y0"] - max_up)
        x0 = lab["x0"] - 10
        x1 = (W / 2 - 4) if lab["col"] == 0 else (W - left_x + 10)
        clip = fitz.Rect(x0, y_lo, x1, lab["y0"] + 1)
        if clip.height < 2:
            tops[id(lab)] = lab["y0"] - 8; continue
        scale = 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, colorspace=fitz.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        rows = (arr < ink).any(axis=1)
        top_ink, white = None, 0
        for r in range(len(rows) - 1, -1, -1):
            if rows[r]:
                top_ink, white = r, 0
            else:
                white += 1
                if white >= gap_pt * scale:
                    break
        t = lab["y0"] - 8
        if top_ink is not None:
            t = min(t, y_lo + top_ink / scale - 4)
        tops[id(lab)] = t
    return tops


def trim(img, pad=24, thr=200, gap=250, thin=15):
    g = ImageOps.invert(img.convert("L").point(lambda p: 255 if p > thr else 0))
    bb = g.getbbox()
    if not bb:
        return img
    x0, y0, x1, y1 = bb
    # drop trailing thin blobs (e.g. the top rule of the "확인 사항" box) separated by a large white gap
    w, h = g.size
    arr = np.asarray(g) > 0
    rows = list(arr.any(axis=1))
    blobs, start = [], None
    for y, r in enumerate(rows + [False]):
        if r and start is None:
            start = y
        elif not r and start is not None:
            blobs.append([start, y]); start = None
    merged = []
    for b in blobs:
        if merged and b[0] - merged[-1][1] < gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    while len(merged) > 1 and merged[-1][1] - merged[-1][0] < thin:
        merged.pop()
    y1 = min(y1, merged[-1][1])
    bb2 = g.crop((0, y0, w, y1)).getbbox()
    if bb2:
        x0, x1 = bb2[0], bb2[2]
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))


POINTS_RE = re.compile(r"\[\s*(\d)\s*점\s*\]")


def process(job, out, dpi_page=150, dpi_crop=300):
    """job: {key, pdf} (KICE: one booklet, sections from page headers) or
    {key, parts:[{pdf, elective, sections}]} (교육청/EBSi: one 12-page PDF per 선택과목; 1-22 common, 23-30 = elective)."""
    parts = job.get("parts") or [dict(pdf=job["pdf"], elective=None, sections=None)]
    pdir = os.path.join(out, "pages", job["key"]); cdir = os.path.join(out, "crops", job["key"])
    os.makedirs(pdir, exist_ok=True); os.makedirs(cdir, exist_ok=True)
    index, pages_used = [], []
    for part in parts:
        doc = fitz.open(part["pdf"])
        forms = [page_form(p) for p in doc]
        use = [i for i, f in enumerate(forms) if f == "odd"] if "even" in forms else list(range(doc.page_count))
        suffix = "" if len(parts) == 1 else "_" + part["elective"]
        for pno, i in enumerate(use, 1):
            page = doc[i]
            W = page.rect.width
            labels, stops, bottom = find_labels(page)
            if part["elective"]:
                secs = ["common" if l["num"] <= 22 else part["elective"] for l in labels]
            else:
                sec = page_section(page); secs = [sec] * len(labels)
            keep = [(l, sc) for l, sc in zip(labels, secs) if part["sections"] is None or sc in part["sections"]]
            if not keep:
                continue
            page.get_pixmap(dpi=dpi_page).save(os.path.join(pdir, f"p{pno:02d}{suffix}.png"))
            pages_used.append(f"{os.path.basename(part['pdf'])}:{i + 1}")
            left_x = min([l["x0"] for l in labels if l["col"] == 0], default=W * 0.1)
            tops = adjust_tops(page, labels, left_x)
            for lab, sec in keep:
                nxt = [l for l in labels if l["col"] == lab["col"] and l["y0"] > lab["y0"]]
                y1 = min([tops[id(l)] for l in nxt], default=bottom + 6) - 6
                st = [y for c, y in stops if c == lab["col"] and lab["y0"] < y < y1]
                if st:
                    y1 = min(st) - 6
                x0 = lab["x0"] - 10
                x1 = (W / 2 - 4) if lab["col"] == 0 else (W - left_x + 10)
                rect = fitz.Rect(x0, tops[id(lab)], x1, y1)
                pts = POINTS_RE.findall(page.get_text("text", clip=rect))
                pix = page.get_pixmap(dpi=dpi_crop, clip=rect)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                img = trim(img)
                name = f"{sec}-{lab['num']}"
                if any(e["name"] == name for e in index):
                    name += "_dup"
                fn = os.path.join(cdir, name + ".png")
                img.save(fn)
                index.append(dict(name=name, section=sec, number=lab["num"], page=pno, pdf=os.path.basename(part["pdf"]),
                                  pdf_page=i + 1, col=lab["col"], rect=[round(v, 1) for v in rect],
                                  points_text=[int(x) for x in pts], png=os.path.relpath(fn, out), w=img.width, h=img.height))
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(dict(job=job, pages_used=pages_used, crops=index), f, ensure_ascii=False, indent=1)
    exp = {"common": list(range(1, 23)), "prob_stat": list(range(23, 31)), "calculus": list(range(23, 31)), "geometry": list(range(23, 31))}
    got = {s: sorted(e["number"] for e in index if e["section"] == s) for s in exp}
    ok = all(got[s] == exp[s] for s in exp)
    npts = sum(len(e["points_text"]) == 1 for e in index)
    print(f"{job['key']}: crops={len(index)} points_in_text={npts} {'OK' if ok else 'MISMATCH ' + json.dumps(got)}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    bad = 0
    for j in jobs:
        if a.only and j["key"] not in a.only.split(","):
            continue
        bad += not process(j, a.out)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
