"""Download official KICE (suneung.re.kr) math papers + answer keys for 수능 and 6/9월 모의평가.

Boards: 1500234 = 기출문제-대학수학능력시험, 1500236 = 기출문제-수능 모의평가.
Writes files under OUT/<school_year>_<exam>/ and a manifest.json. Files are untrusted data:
they are only saved, never executed.

Usage (inside the ksat-data-tools image):
  python -I fetch_kice.py --out /data/dl/kice --years 2022-2026
"""
import argparse, html, json, os, re, time
import requests

BASE = "https://www.suneung.re.kr"
BOARDS = {"1500234": "csat", "1500236": "mock"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ksat-data-fetch"}


def parse_rows(page_html, board):
    rows = []
    tbody = re.search(r"<tbody>(.*?)</tbody>", page_html, re.S)
    if not tbody:
        return rows
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody.group(1), re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        txt = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip() for t in tds]
        files = re.findall(r"fn_fileDown\('([0-9a-f]+)'\);\"\s*title='([^']*)'", tr)
        seq = re.search(r"goView\('\d+','(\d+)'", tr)
        if board == "1500234":   # 번호 학년도 영역 제목 등록일 파일 조회
            if len(txt) < 6:
                continue
            rows.append(dict(seq=txt[0], year=txt[1], month="수능", subject=txt[2], title=txt[3], date=txt[4], files=files,
                             boardSeq=seq.group(1) if seq else txt[0]))
        else:                    # 번호 학년도 월 영역 제목 등록일 파일 조회
            if len(txt) < 7:
                continue
            rows.append(dict(seq=txt[0], year=txt[1], month=txt[2], subject=txt[3], title=txt[4], date=txt[5], files=files,
                             boardSeq=seq.group(1) if seq else txt[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="2022-2026")
    ap.add_argument("--subject", default="수학")
    args = ap.parse_args()
    y0, y1 = map(int, args.years.split("-"))
    os.makedirs(args.out, exist_ok=True)
    s = requests.Session(); s.headers.update(UA)
    manifest = []
    for board, kind in BOARDS.items():
        page = 1
        while True:
            url = f"{BASE}/boardCnts/list.do?type=default&page={page}&m=0403&boardID={board}&s=suneung"
            r = s.get(url, timeout=60); r.raise_for_status()
            rows = parse_rows(r.text, board)
            if not rows:
                break
            min_year = min(int(x["year"]) for x in rows if x["year"].isdigit())
            for x in rows:
                if x["subject"] != args.subject or not x["year"].isdigit():
                    continue
                yr = int(x["year"])
                if not (y0 <= yr <= y1):
                    continue
                exam = "csat" if kind == "csat" else {"6월": "kice_06", "9월": "kice_09"}.get(x["month"], "kice_" + x["month"])
                d = os.path.join(args.out, f"{yr}_{exam}")
                os.makedirs(d, exist_ok=True)
                for fseq, title in x["files"]:
                    fn = re.sub(r"[^\w.\-가-힣]", "_", title)
                    path = os.path.join(d, fn)
                    if not os.path.exists(path):
                        fr = s.get(f"{BASE}/boardCnts/fileDown.do?fileSeq={fseq}", timeout=120)
                        fr.raise_for_status()
                        with open(path, "wb") as f:
                            f.write(fr.content)
                        time.sleep(0.5)
                    manifest.append(dict(school_year=yr, exam=exam, board=board, boardSeq=x["boardSeq"], title=title,
                                         fileSeq=fseq, path=os.path.relpath(path, args.out), bytes=os.path.getsize(path),
                                         posted=x["date"], url=f"{BASE}/boardCnts/fileDown.do?fileSeq={fseq}"))
                    print(yr, exam, title, os.path.getsize(path), flush=True)
            if min_year < y0:
                break
            page += 1
            time.sleep(0.5)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("files:", len(manifest))


if __name__ == "__main__":
    main()
