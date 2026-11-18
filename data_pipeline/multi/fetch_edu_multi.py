"""Download 교육청 고3 학력평가 (3·4·7·10월) papers for 국어/영어/사탐/과탐 from EBSi (official distributor).

Listing: POST https://www.ebsi.co.kr/ebs/xip/xipc/previousPaperListAjax.ajax (targetCd=D300, yearList=<calendar year>,
monthList=<MM>, arOrd=1..8, subjIdList=firstEnter, currentPage=N). Each entry's buttons carry the file paths:
goDownLoadP (문제 PDF), goDownLoadJ (정답 PNG), goDownLoadH (해설 PDF), goDownLoadD (영어 듣기 대본 PDF); files are on
https://wdown.ebsi.co.kr/W61001/01exam<path>. Audio (goDownLoadR) is skipped. Files are untrusted data: saved only.
EBSi notice: 무단 전재·재배포 금지 -> internal use only.

school_year = calendar year + 1. Usage: python -I fetch_edu_multi.py --out /data/dl2/edu --years 2022-2026
"""
import argparse, hashlib, html, json, os, re, time
import requests

BASE = "https://www.ebsi.co.kr"
WD = "https://wdown.ebsi.co.kr/W61001/01exam"
AREA = {"국어": "kor", "영어": "eng", "사탐": "soc", "과탐": "sci"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ksat-data-fetch"}


def parse(page):
    out = []
    for box in re.split(r'<div class="qus_box', page)[1:]:
        flags = re.findall(r'<span class="flag_subject_col_\w+">([^<]*)</span>', box)
        tit = re.search(r'<div class="qus_tit">(.*?)</div>', box, re.S)
        tit = html.unescape(re.sub(r"<[^>]+>", "", tit.group(1))).replace("\xa0", " ").strip() if tit else ""
        files = {}
        for kind, args in re.findall(r"goDownLoad([PJHDR])\(([^)]*)\)", box):
            first = re.match(r"\s*'([^']*)'", args)
            if first:
                files.setdefault(kind, first.group(1))
        out.append(dict(flags=flags, title=tit, files=files))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--years", default="2022-2026")
    ap.add_argument("--months", default="03,04,07,10")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    s = requests.Session(); s.headers.update(UA)
    s.get(f"{BASE}/ebs/xip/xipc/previousPaperList.ebs?targetCd=D300", timeout=60)
    manifest = []
    for sy in range(y0, y1 + 1):
        cy = sy - 1
        for mm in a.months.split(","):
            entries, seen = [], set()
            for pg in range(1, 10):
                d = dict(targetCd="D300", yearList=str(cy), monthList=mm, arOrd="1,2,3,4,5,6,7,8", subjIdList="firstEnter",
                         sort="recent", paperId="", paperNo="", lvl="", currentPage=str(pg))
                r = s.post(f"{BASE}/ebs/xip/xipc/previousPaperListAjax.ajax", data=d, timeout=60,
                           headers={"X-Requested-With": "XMLHttpRequest", "Referer": f"{BASE}/ebs/xip/xipc/previousPaperList.ebs?targetCd=D300"})
                r.raise_for_status()
                es = parse(r.text)
                new = [e for e in es if (e["title"], e["files"].get("P")) not in seen]
                if not new:
                    break
                for e in new:
                    seen.add((e["title"], e["files"].get("P")))
                entries += new
                time.sleep(0.5)
            for e in entries:
                area = next((AREA[f] for f in e["flags"] if f in AREA), None)
                if not area or "고3" not in e["title"]:
                    continue
                subj = re.sub(r"^고3\s*\d+월\s*학평\([^)]*\)\s*", "", e["title"]).strip() or e["title"]
                sitting = re.search(r"학평\(([^)]*)\)", e["title"])
                d = os.path.join(a.out, f"{sy}_edu_{mm}", area)
                os.makedirs(d, exist_ok=True)
                for kind, name in (("P", "paper"), ("J", "answer"), ("H", "hsj"), ("D", "script")):
                    p = e["files"].get(kind)
                    if not p:
                        continue
                    url = p if p.startswith("http") else WD + p
                    ext = os.path.splitext(url)[1].lower()
                    if ext not in (".pdf", ".png", ".jpg"):
                        continue
                    fn = os.path.join(d, re.sub(r"[^\w가-힣]", "_", subj) + f"_{name}{ext}")
                    if not os.path.exists(fn):
                        fr = s.get(url, timeout=120)
                        if fr.status_code != 200:
                            print("FAIL", url, fr.status_code); continue
                        open(fn, "wb").write(fr.content)
                        time.sleep(0.3)
                    manifest.append(dict(school_year=sy, exam=f"edu_{mm}", held=f"{cy}-{mm}", area=area, subject=subj, kind=name,
                                         path=os.path.relpath(fn, a.out), url=url, bytes=os.path.getsize(fn),
                                         sitting=sitting.group(1) if sitting else None, source_site="EBSi (www.ebsi.co.kr 기출문제 -> wdown.ebsi.co.kr)",
                                         official=True, sha256=hashlib.sha256(open(fn, "rb").read()).hexdigest(),
                                         note="EBSi: 무단 전재·재배포 금지 -> internal use only"))
                print(sy, mm, area, subj, sorted(e["files"]), flush=True)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print("files:", len(manifest))


if __name__ == "__main__":
    main()
