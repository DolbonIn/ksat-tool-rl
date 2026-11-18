"""Download 교육청 고3 학력평가 (3·4·7·10월) 수학 가형/나형 papers for 2017-2021학년도 (held 2016-2020) from EBSi.

Listing reuses data_pipeline/multi/fetch_edu_multi.py (POST previousPaperListAjax.ajax, goDownLoadP/J/H buttons).
Keeps entries whose title mentions 고3 and 수학 (가형/나형). Files are untrusted data: saved only, never executed.
EBSi notice: 무단 전재·재배포 금지 -> internal use only. school_year = calendar year + 1.
Usage: python -I fetch_edu_old.py --out /data/dl3/edu --years 2017-2021 [--list-only]
"""
import argparse, hashlib, json, os, re, sys, time
sys.path.insert(0, "/pipe/multi")
import requests
from fetch_edu_multi import parse, BASE, WD, UA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--years", default="2017-2021")
    ap.add_argument("--months", default="03,04,05,07,10"); ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    s = requests.Session(); s.headers.update(UA)
    s.get(f"{BASE}/ebs/xip/xipc/previousPaperList.ebs?targetCd=D300", timeout=60)
    manifest, listing = [], []
    for sy in range(y0, y1 + 1):
        cy = sy - 1
        for mm in a.months.split(","):
            entries, seen = [], set()
            for pg in range(1, 12):
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
                time.sleep(0.4)
            maths = [e for e in entries if "수학" in e["title"] or "수학" in "".join(e["flags"])]
            listing.append(dict(school_year=sy, held=f"{cy}-{mm}", n_entries=len(entries),
                                math=[dict(title=e["title"], flags=e["flags"], files=e["files"]) for e in maths]))
            print(sy, f"{cy}-{mm}", "entries", len(entries), "math:", [(e["title"], sorted(e["files"])) for e in maths], flush=True)
            if a.list_only:
                continue
            for e in maths:
                if "고3" not in e["title"]:
                    continue
                tr = "ga" if "가형" in e["title"] else "na" if "나형" in e["title"] else None
                if tr is None:
                    print("  skip (no track)", e["title"]); continue
                tm = re.search(r"고3\s*(\d{1,2})월\s*학평", e["title"])
                if not tm:
                    print("  skip (not a 학평)", e["title"]); continue
                ex = f"edu_{int(tm.group(1)):02d}"          # exam month from the title (2020 sittings were moved: 3월->4.24, 4월->5.21)
                dd = os.path.join(a.out, f"{sy}_{ex}")
                os.makedirs(dd, exist_ok=True)
                for kind, name in (("P", "paper"), ("J", "answer"), ("H", "hsj")):
                    p = e["files"].get(kind)
                    if not p:
                        continue
                    url = p if p.startswith("http") else WD + p
                    ext = os.path.splitext(url)[1].lower()
                    if ext not in (".pdf", ".png", ".jpg", ".gif"):
                        print("  skip ext", url); continue
                    fn = os.path.join(dd, f"{sy}_{ex}_math_{tr}_{name}{ext}")
                    if not os.path.exists(fn):
                        fr = s.get(url, timeout=180)
                        if fr.status_code != 200:
                            print("FAIL", url, fr.status_code); continue
                        open(fn, "wb").write(fr.content)
                        time.sleep(0.3)
                    manifest.append(dict(school_year=sy, exam=ex, held=f"{cy}-{mm}", track=tr, kind=name, title=e["title"],
                                         path=os.path.relpath(fn, a.out), url=url, bytes=os.path.getsize(fn),
                                         source_site="EBSi (www.ebsi.co.kr 기출문제 -> wdown.ebsi.co.kr)", official=True,
                                         sha256=hashlib.sha256(open(fn, "rb").read()).hexdigest(),
                                         note="EBSi: 무단 전재·재배포 금지 -> internal use only"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(listing, open(os.path.join(a.out, "listing.json"), "w"), ensure_ascii=False, indent=1)
    if not a.list_only:
        json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print("files:", len(manifest))


if __name__ == "__main__":
    main()
