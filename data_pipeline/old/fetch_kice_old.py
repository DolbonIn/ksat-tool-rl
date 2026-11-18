"""Download official KICE (suneung.re.kr) math papers + answer keys for 수능 / 6·9월 모의평가, 2017-2021학년도 (가형/나형).

Reuses the board parser of data_pipeline/fetch_kice.py (boards 1500234 = 수능, 1500236 = 모의평가). Subject filter is
'수학' prefix (the boards list 수학 / 수학(가형) / 수학(나형) depending on the year). Files are untrusted data: saved only.
Usage: python -I fetch_kice_old.py --out /data/dl3/kice --years 2017-2021 [--list-only]
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, "/pipe")
import requests
from fetch_kice import parse_rows, BASE, BOARDS, UA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--years", default="2017-2021")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    os.makedirs(a.out, exist_ok=True)
    s = requests.Session(); s.headers.update(UA)
    manifest, rows_all = [], []
    for board, kind in BOARDS.items():
        page = 1
        while page < 80:
            url = f"{BASE}/boardCnts/list.do?type=default&page={page}&m=0403&boardID={board}&s=suneung"
            r = s.get(url, timeout=60); r.raise_for_status()
            rows = parse_rows(r.text, board)
            if not rows:
                break
            yrs = [int(x["year"]) for x in rows if x["year"].isdigit()]
            for x in rows:
                if not x["year"].isdigit() or not (y0 <= int(x["year"]) <= y1) or not x["subject"].startswith("수학"):
                    continue
                yr = int(x["year"])
                exam = "csat" if kind == "csat" else {"6월": "kice_06", "9월": "kice_09"}.get(x["month"], "kice_" + x["month"])
                rows_all.append(dict(x, exam=exam, board=board))
                print(yr, exam, x["subject"], "|", x["title"], "|", [t for _, t in x["files"]], flush=True)
                if a.list_only:
                    continue
                d = os.path.join(a.out, f"{yr}_{exam}")
                os.makedirs(d, exist_ok=True)
                for fseq, title in x["files"]:
                    fn = re.sub(r"[^\w.\-가-힣()]", "_", title)
                    path = os.path.join(d, fn)
                    if not os.path.exists(path):
                        fr = s.get(f"{BASE}/boardCnts/fileDown.do?fileSeq={fseq}", timeout=180)
                        fr.raise_for_status()
                        open(path, "wb").write(fr.content)
                        time.sleep(0.5)
                    manifest.append(dict(school_year=yr, exam=exam, board=board, boardSeq=x["boardSeq"], subject=x["subject"],
                                         post_title=x["title"], title=title, fileSeq=fseq, path=os.path.relpath(path, a.out),
                                         bytes=os.path.getsize(path), posted=x["date"], url=f"{BASE}/boardCnts/fileDown.do?fileSeq={fseq}"))
            if yrs and min(yrs) < y0:
                break
            page += 1
            time.sleep(0.4)
    json.dump(rows_all, open(os.path.join(a.out, "rows.json"), "w"), ensure_ascii=False, indent=1)
    if not a.list_only:
        json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print("rows:", len(rows_all), "files:", len(manifest))


if __name__ == "__main__":
    main()
