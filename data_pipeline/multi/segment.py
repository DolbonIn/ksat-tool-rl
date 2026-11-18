"""Segment 국어/영어/탐구 exam PDFs into passage / question segments using the PDF text layer, render padded crops,
and build a text-layer "draft" per segment (used as the authoritative character source for transcription).

KICE / 교육청 papers: A3-ish pages, 2 columns, question labels "n." (font ~12.7-13pt) at the column's left edge, set
headers "[n～m] ..." at the same x. Only 홀수형 pages are used when a PDF has both forms.

Per segment we record: kind (passage|question), section, number / range, regions [(page, x0, y0, x1, y1)], the text-layer
lines inside the regions (PUA equation digits mapped, underlines marked <u>..</u> from vector strokes, [A]-style margin
brackets turned into "[A] ... [/A]" spans), and graphic counts (raster images, non-underline vector drawings).

Usage: python -I segment.py --jobs /data/work2/jobs.json --out /data/render2 [--only key1,key2]
PDFs are untrusted input: read-only, nothing executed.
"""
import argparse, json, os, re, sys
from collections import Counter
import fitz

LABEL_RE = re.compile(r"^\s*(\d{1,2})\s*\.(?:\s|$)")
SET_RE = re.compile(r"^\s*\[\s*(\d{1,2})\s*[～~∼〜\-–]\s*(\d{1,2})\s*\]")
STOP_RE = re.compile(r"^\s*[*※]\s*확인\s*사항")
BRACKET_RE = re.compile(r"^\s*\[([A-H])\]\s*$")
PUA = {0xE034 + i: str((i + 1) % 10) for i in range(10)}
PUA.update({0xE053: ".", 0xE044: "(", 0xE045: ")", 0xE048: "+"})
for i in range(26):
    PUA[0xE0E5 + i] = chr(ord("a") + i)        # HWP equation-font italic a-z (h=E0EC, v=E0FA, x=E0FC verified)
START_MARK = re.compile(r"^\s*([①②③④⑤]|◦|∘|ㆍ|•|-\s|ㄱ\.|ㄴ\.|ㄷ\.|ㄹ\.|\([가-하]\)|\[[가-하A-Z]\]|<|〈|\*|※|[A-Z]\s*:|[가-힣]{1,4}\s*:|\d+\)|[ⓐ-ⓩ㉠-㉯]\s)")
SECTION_HDR = [("화법과 작문", "kor_speech_writing"), ("화법과작문", "kor_speech_writing"), ("언어와 매체", "kor_lang_media"),
               ("언어와매체", "kor_lang_media")]


def map_pua(s):
    return "".join(PUA.get(ord(c), "□" if 0xE000 <= ord(c) <= 0xF8FF else c) for c in s)


def page_form(page):
    top = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.2))
    if "짝수형" in top:
        return "even"
    if "홀수형" in top:
        return "odd"
    return None


def page_section(page, default):
    top = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.17))
    found = [v for k, v in SECTION_HDR if f"({k})" in top or f"({k})" in top.replace(" ", "")]
    order = ["kor_speech_writing", "kor_lang_media"]
    if found:   # a template leftover can show the previous elective's name too: take the later section
        return max(found, key=order.index)
    return default


def get_lines(page):
    """Text lines with chars (rawdict). Returns list of dict(bbox, text, size, chars=[(c, bbox)])."""
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            chars = []
            for s in l["spans"]:
                for ch in s["chars"]:
                    chars.append((ch["c"], tuple(ch["bbox"]), s["size"]))
            t = "".join(c for c, _, _ in chars)
            if t.strip():
                sizes = [sz for c, _, sz in chars if c.strip()]
                out.append(dict(bbox=tuple(l["bbox"]), text=t, size=max(sizes) if sizes else 0, chars=chars))
    return out


def page_frame(page, lines):
    """Content y-range: below the header rule, above the footer (page-number box / copyright line)."""
    W, H = page.rect.width, page.rect.height
    top = H * 0.12
    rules = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.height < 2.5 and r.width > W * 0.6 and r.y0 < H * 0.3:
            rules.append(r.y1)
    if rules:
        top = max(rules) + 1
    else:
        hdr = [l["bbox"][3] for l in lines if l["bbox"][1] < H * 0.2 and l["size"] >= 15 * W / 842]
        if hdr:
            top = max(hdr) + 4
    # footer = page number / copyright line (centred, digits only or 저작권); side tabs and column text are not footer
    foot = [l["bbox"][1] for l in lines if l["bbox"][1] > H * 0.9 and (abs((l["bbox"][0] + l["bbox"][2]) / 2 - W / 2) < 60
            or "저작권" in l["text"] or re.fullmatch(r"\s*\d+\s*", l["text"]))]
    bottom = (min(foot) - 3) if foot else H * 0.91
    return top, bottom


def thin_segments(page):
    """Horizontal thin strokes (underline candidates) and vertical strokes (for box/bracket detection)."""
    hs, vs = [], []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.8 and abs(p1.x - p2.x) > 3:
                    hs.append((min(p1.x, p2.x), (p1.y + p2.y) / 2, max(p1.x, p2.x)))
                elif abs(p1.x - p2.x) < 0.8 and abs(p1.y - p2.y) > 3:
                    vs.append(((p1.x + p2.x) / 2, min(p1.y, p2.y), max(p1.y, p2.y)))
            elif it[0] == "re":
                r = it[1]
                if r.height < 1.2 and r.width > 3:
                    hs.append((r.x0, (r.y0 + r.y1) / 2, r.x1))
                elif r.width < 1.2 and r.height > 3:
                    vs.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
                else:   # box edges: count as vertical strokes so its top/bottom edges are not underlines
                    vs.append((r.x0, r.y0, r.y1)); vs.append((r.x1, r.y0, r.y1))
                    hs.append((r.x0, r.y0, r.x1)); hs.append((r.x0, r.y1, r.x1))
            elif it[0] == "qu":
                q = it[1]; r = q.rect
                if r.width >= 3 and r.height >= 3:
                    vs.append((r.x0, r.y0, r.y1)); vs.append((r.x1, r.y0, r.y1))
    return hs, vs


def underline_flags(lines, hs, vs, col_w):
    """Mark chars that sit directly above an isolated horizontal stroke (not a box/table edge)."""
    def is_border(h):
        x0, y, x1 = h
        if x1 - x0 > col_w * 0.8:
            return True
        for vx, vy0, vy1 in vs:
            if (abs(vx - x0) < 1.5 or abs(vx - x1) < 1.5) and vy0 - 1.5 <= y <= vy1 + 1.5:
                return True
        return False
    cands = [h for h in hs if not is_border(h)]
    for l in lines:
        flags = []
        for c, bb, _ in l["chars"]:
            cx = (bb[0] + bb[2]) / 2
            u = False
            for x0, y, x1 in cands:
                if x0 - 0.5 <= cx <= x1 + 0.5 and bb[3] - 3.0 <= y <= bb[3] + 3.5:
                    u = True; break
            flags.append(u)
        # spaces between two underlined chars stay underlined; leading/trailing underlined spaces are dropped
        l["uflags"] = flags


def line_text(l, with_u=True):
    out, inside = [], False
    chars = l["chars"]; fl = l.get("uflags") or [False] * len(chars)
    # trim underline on whitespace at run edges
    fl = list(fl)
    for i, (c, _, _) in enumerate(chars):
        if c.isspace() and fl[i]:
            left = any(fl[j] and not chars[j][0].isspace() for j in range(i - 1, -1, -1) if True) if i else False
            right = any(fl[j] and not chars[j][0].isspace() for j in range(i + 1, len(chars)))
            # keep only if both neighbours (nearest non-space) are underlined
            lj = next((j for j in range(i - 1, -1, -1) if not chars[j][0].isspace()), None)
            rj = next((j for j in range(i + 1, len(chars)) if not chars[j][0].isspace()), None)
            fl[i] = lj is not None and rj is not None and fl[lj] and fl[rj]
    for i, (c, _, _) in enumerate(chars):
        if with_u and fl[i] and not inside:
            out.append("<u>"); inside = True
        elif with_u and not fl[i] and inside:
            out.append("</u>"); inside = False
        out.append(c)
    if inside:
        out.append("</u>")
    s = map_pua("".join(out))
    s = s.replace("<u></u>", "")
    return s


def build_draft(region_lines, right_edge):
    """Join wrapped lines of a region into paragraphs. Junction: ' ' if the line had a trailing space, '¦' otherwise
    (the transcriber decides whether that junction is a space; Korean words may be split at a line end)."""
    out = []
    prev = None
    item_x = None          # x of the current list item's marker line (choices, bullets): continuation lines are indented
    for l in region_lines:
        t = line_text(l)
        if prev is not None:
            pb, lb = prev["bbox"], l["bbox"]
            h = max(1.0, pb[3] - pb[1])
            gap = lb[1] - pb[3]
            full = pb[2] >= right_edge - 16
            same_row = abs(lb[1] - pb[1]) < h * 0.5
            if same_row:
                out[-1] += "  " + t.strip()
                prev = l; continue
            cont_ok = lb[0] <= pb[0] + 3 or (item_x is not None and lb[0] <= item_x + 16)
            if full and 0 - h < gap < h * 0.9 and not START_MARK.match(l["text"]) and cont_ok:
                raw_prev = prev["text"]
                j = " " if raw_prev.endswith((" ", "　")) else "¦"
                out[-1] = out[-1].rstrip() + j + t.lstrip()
                prev = l; continue
        out.append(t.rstrip())
        item_x = l["bbox"][0] if START_MARK.match(l["text"]) else None
        prev = l
    return "\n".join(out)


def process(job, outdir, dpi_max=300, px_cap=1.45e6):
    doc = fitz.open(job["pdf"])
    forms = [page_form(p) for p in doc]
    use = [i for i, f in enumerate(forms) if f == "odd"] if "even" in forms else list(range(doc.page_count))
    key = job["key"]
    rdir = os.path.join(outdir, key); os.makedirs(rdir, exist_ok=True)
    default_sec = job.get("section", job.get("subject"))
    allL = []        # reading-order lines: (pidx, col, line)
    pinfo = {}
    for pi in use:
        page = doc[pi]
        W, H = page.rect.width, page.rect.height
        lines = get_lines(page)
        top, bottom = page_frame(page, lines)
        hs, vs = thin_segments(page)
        sec = page_section(page, default_sec) if job["area"] == "kor" else default_sec
        if job["area"] == "kor" and sec == default_sec and pinfo:
            prev = pinfo[max(pinfo)]["sec"]
            if prev != default_sec:          # elective pages without the elective name in the header keep the section
                sec = prev
        cols = []
        for c in (0, 1):
            cl = [l for l in lines if ((l["bbox"][0] + l["bbox"][2]) / 2 < W / 2) == (c == 0)
                  and l["bbox"][1] >= top - 1 and l["bbox"][3] <= bottom + 1]
            cols.append(cl)
        colx = [min([l["bbox"][0] for l in cols[c]], default=88 if c == 0 else 437) for c in (0, 1)]
        colr = [max([l["bbox"][2] for l in cols[c]], default=W / 2 - 15 if c == 0 else W - 88) for c in (0, 1)]
        for c in (0, 1):
            underline_flags(cols[c], hs, vs, colr[c] - colx[c])
            for l in sorted(cols[c], key=lambda l: (round(l["bbox"][1], 0), l["bbox"][0])):
                allL.append((pi, c, l))
        pinfo[pi] = dict(top=top, bottom=bottom, colx=colx, colr=colr, sec=sec, W=W, H=H, vs=vs, hs=hs)
        page.get_pixmap(dpi=110).save(os.path.join(rdir, f"page{pi + 1:02d}.png"))
    # global label x per column (min over pages)
    gx = [min(pinfo[p]["colx"][c] for p in pinfo) for c in (0, 1)]
    # stop markers ("※ 확인 사항" box at the end of each section): drop the box and anything after it on that page
    stop_at = None
    drop = set()
    stops = {}
    for k, (pi, c, l) in enumerate(allL):
        if STOP_RE.match(l["text"]):
            stop_at = k
            y = l["bbox"][1]
            stops[(pi, c)] = y
            for k2, (pj, c2, l2) in enumerate(allL):
                if pj == pi and ((c2 == c and l2["bbox"][1] >= y - 1) or c2 > c):
                    drop.add(k2)
    allL = [x for k, x in enumerate(allL) if k not in drop]
    # label font: larger than body text (KICE A3: body 11.2 / label 12.7-13; 교육청 A4: body ~8.2 / label 9.5)
    szs = sorted(round(l["size"], 1) for _, _, l in allL)
    body = Counter(szs).most_common(1)[0][0] if szs else 11.2
    label_min = body + 0.6
    # label x per column = most common x0 among label-like lines (robust to stray left-margin lines)
    for c in (0, 1):
        xs = [round(l["bbox"][0]) for _, cc, l in allL if cc == c and l["size"] >= label_min and (LABEL_RE.match(l["text"]) or SET_RE.match(l["text"]))]
        if xs:
            gx[c] = Counter(xs).most_common(1)[0][0] - 0.5
    # anchors
    anchors = []
    exp_by_sec = {}
    for k, (pi, c, l) in enumerate(allL):
        x0 = l["bbox"][0]
        if abs(x0 - gx[c]) > 6:
            continue
        sec = pinfo[pi]["sec"]
        m = SET_RE.match(l["text"])
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a <= 10:
                anchors.append(dict(k=k, kind="passage", a=a, b=b, sec=sec))
            continue
        m = LABEL_RE.match(l["text"])
        if m and l["size"] >= label_min:
            n = int(m.group(1))
            last = exp_by_sec.get(sec, job.get("expect", {}).get(sec, [1])[0] - 1)
            if n > last and n - last <= 3:
                anchors.append(dict(k=k, kind="question", n=n, sec=sec))
                exp_by_sec[sec] = n
    # segments
    segs = []
    for i, a in enumerate(anchors):
        k0 = a["k"]; k1 = anchors[i + 1]["k"] if i + 1 < len(anchors) else len(allL)
        seg_lines = allL[k0:k1]
        # group consecutive lines by (page, col) into regions
        regions = []
        for pi, c, l in seg_lines:
            if regions and regions[-1]["page"] == pi and regions[-1]["col"] == c:
                regions[-1]["lines"].append(l)
            else:
                regions.append(dict(page=pi, col=c, lines=[l]))
        # columns between this anchor and the next one that contain no text lines of this segment (raster-only
        # figures / image text) must still be covered: insert empty regions for every (page, col) in between
        if i + 1 < len(anchors):
            pc0 = (allL[k0][0], allL[k0][1]); pc1 = (allL[k1][0], allL[k1][1])
            order = [(p_, c_) for p_ in use for c_ in (0, 1)]
            i0, i1 = order.index(pc0), order.index(pc1)
            have = {(r["page"], r["col"]) for r in regions}
            full = []
            for pc in order[i0:i1 + 1]:
                if pc in have:
                    full.append(next(r for r in regions if (r["page"], r["col"]) == pc))
                elif pc == pc1:
                    gap = allL[k1][2]["bbox"][1] - pinfo[pc[0]]["top"]
                    if gap > 25:
                        full.append(dict(page=pc[0], col=pc[1], lines=[]))
                else:
                    full.append(dict(page=pc[0], col=pc[1], lines=[]))
            regions = full
        for r in regions:
            P = pinfo[r["page"]]; c = r["col"]
            first = r is regions[0]
            ys = [l["bbox"][1] for l in r["lines"]] or [P["top"] + 10]; ye = [l["bbox"][3] for l in r["lines"]] or [P["top"] + 10]
            y0 = min(ys) - 9 if first else P["top"] + 1
            # do not climb into the previous segment's last line in this column
            prev_lines = [l["bbox"][3] for (pi, cc, l) in allL[:k0] if pi == r["page"] and cc == c]
            if first and prev_lines:
                y0 = max(y0, max(prev_lines) + 1.5)
            # bottom: next anchor in same column/page, else column bottom
            nxt = [allL[an["k"]][2]["bbox"][1] for an in anchors[i + 1:i + 2]
                   if allL[an["k"]][0] == r["page"] and allL[an["k"]][1] == c]
            y1 = (nxt[0] - 4) if nxt else P["bottom"]
            if not nxt and (r["page"], c) in stops:
                y1 = min(y1, stops[(r["page"], c)] - 14)   # stop box has a frame above its text line
            x0 = gx[c] - 8
            x1 = (P["W"] / 2 - 2) if c == 0 else (P["W"] - gx[0] + 10)
            r["rect"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        # brackets [A] in the left margin -> spans
        for r in regions:
            P = pinfo[r["page"]]
            for l in list(r["lines"]):
                m = BRACKET_RE.match(l["text"])
                if not m:
                    continue
                lx, ly = l["bbox"][0], (l["bbox"][1] + l["bbox"][3]) / 2
                if lx > gx[r["col"]] + 14:
                    continue
                cx = (lx + l["bbox"][2]) / 2
                ly0, ly1 = l["bbox"][1], l["bbox"][3]
                spans = [(vy0, vy1) for vx, vy0, vy1 in P["vs"] if abs(vx - cx) < 6 and vy1 - vy0 > 12 and
                         (vy0 - 2 <= ly <= vy1 + 2 or abs(vy1 - ly0) < 14 or abs(vy0 - ly1) < 14)]
                if not spans:
                    continue
                y0s, y1s = min(s[0] for s in spans), max(s[1] for s in spans)
                inside = [x for x in r["lines"] if x is not l and y0s - 3 <= (x["bbox"][1] + x["bbox"][3]) / 2 <= y1s + 3]
                if inside:
                    inside[0]["pre"] = f"[{m.group(1)}] "
                    inside[-1]["post"] = f" [/{m.group(1)}]"
                    r["lines"].remove(l)
        draft_parts, gfx = [], dict(images=0, drawings=0)
        page_objs = {}
        for r in regions:
            P = pinfo[r["page"]]
            page = page_objs.setdefault(r["page"], doc[r["page"]])
            rl = r["lines"]
            for l in rl:
                if l.get("pre") or l.get("post"):
                    l["chars"] = [(ch, (0, 0, 0, 0), 0) for ch in l.get("pre", "")] + l["chars"] + [(ch, (0, 0, 0, 0), 0) for ch in l.get("post", "")]
                    if "uflags" in l:
                        l["uflags"] = [False] * len(l.get("pre", "")) + l["uflags"] + [False] * len(l.get("post", ""))
                    l["text"] = l.get("pre", "") + l["text"] + l.get("post", "")
                    l.pop("pre", None); l.pop("post", None)
            draft_parts.append(build_draft(rl, P["colr"][r["col"]]))
            R = fitz.Rect(*r["rect"])
            for info in page.get_image_info():
                bb = fitz.Rect(info["bbox"])
                if bb.intersects(R) and bb.width > 15 and bb.height > 15:
                    gfx["images"] += 1
            nd = 0
            if "drw" not in P:
                P["drw"] = page.get_drawings()
            for d in P["drw"]:
                rr = d["rect"]
                if R.contains(rr) and not (rr.height < 1.5) and (rr.width > 8 or rr.height > 8):
                    nd += 1
            gfx["drawings"] += nd
        # crops
        pngs = []
        for j, r in enumerate(regions):
            page = page_objs[r["page"]]
            R = fitz.Rect(*r["rect"])
            scale = min(dpi_max / 72, (px_cap / max(1, R.width * R.height)) ** 0.5)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=R)
            name = (f"{a['sec']}-q{a['n']:02d}" if a["kind"] == "question" else f"{a['sec']}-p{a['a']:02d}_{a['b']:02d}") + f"_{j}.png"
            pix.save(os.path.join(rdir, name))
            pngs.append(name)
        raw = "\n".join(map_pua(l["text"]) for r in regions for l in r["lines"])
        pts = re.findall(r"\[\s*(\d)\s*점\s*\]", raw)
        sid = f"{a['sec']}-q{a['n']:02d}" if a["kind"] == "question" else f"{a['sec']}-p{a['a']:02d}_{a['b']:02d}"
        segs.append(dict(sid=sid, kind=a["kind"], section=a["sec"], number=a.get("n"), range=[a["a"], a["b"]] if a["kind"] == "passage" else None,
                         regions=[dict(page=r["page"] + 1, col=r["col"], rect=r["rect"]) for r in regions], pngs=pngs,
                         draft="\n".join(draft_parts), raw=raw, points_text=[int(x) for x in pts], gfx=gfx,
                         n_lines=sum(len(r["lines"]) for r in regions)))
    meta = dict(job=job, pages_used=[p + 1 for p in use], segments=segs)
    json.dump(meta, open(os.path.join(rdir, "segments.json"), "w"), ensure_ascii=False, indent=1)
    # checks
    qs = Counter(s["section"] for s in segs if s["kind"] == "question")
    nums = {sec: sorted(s["number"] for s in segs if s["kind"] == "question" and s["section"] == sec) for sec in qs}
    exp = job.get("expect", {})
    ok = all(nums.get(sec) == list(range(lo, hi + 1)) for sec, (lo, hi) in exp.items())
    print(f"{key}: pages={len(use)} segs={len(segs)} q={dict(qs)} passages={sum(s['kind'] == 'passage' for s in segs)} "
          f"{'OK' if ok else 'MISMATCH ' + json.dumps({k: v for k, v in nums.items()})}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    bad = []
    for j in json.load(open(a.jobs)):
        if a.only and not any(j["key"].startswith(x) for x in a.only.split(",")):
            continue
        try:
            if not process(j, a.out):
                bad.append(j["key"])
        except Exception as e:
            import traceback; traceback.print_exc()
            print("ERROR", j["key"], e); bad.append(j["key"])
    print("bad:", bad)


if __name__ == "__main__":
    main()
