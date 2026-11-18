"""Build jobs JSON for KICE 국어/영어/과탐/사탐 papers downloaded by fetch_kice_multi.py (+ unzip_safe.py).

Usage: python3 make_jobs_kice.py --dl /raid/.../data/dl2/kice --out /raid/.../data/work2/jobs_kice.json
Each job: {key, area, subject, section, exam, school_year, pdf, answer_key, answer_subject, script (eng), expect}
"""
import argparse, glob, json, os, re, unicodedata

SCI = [("물리학1", "phys1"), ("화학1", "chem1"), ("생명과학1", "bio1"), ("지구과학1", "earth1"),
       ("물리학2", "phys2"), ("화학2", "chem2"), ("생명과학2", "bio2"), ("지구과학2", "earth2")]
SOC = [("생활과윤리", "ethics_life"), ("윤리와사상", "ethics_thought"), ("한국지리", "geo_korea"), ("세계지리", "geo_world"),
       ("동아시아사", "east_asian_hist"), ("세계사", "world_hist"), ("경제", "economics"), ("정치와법", "politics_law"),
       ("사회문화", "society_culture")]
KNAME = {v: k for k, v in SCI + SOC}


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("Ⅱ", "2").replace("Ⅰ", "1").replace("·", "").replace("ㆍ", "").replace("․", "").replace("‧", "").replace(" ", "").replace("_", "")
    s = re.sub(r"II(?![A-Za-z])", "2", s)
    s = re.sub(r"I(?![A-Za-z])", "1", s)
    return s


def subj_of(path, table):
    b = norm(os.path.basename(path))
    hits = [code for name, code in table if name in b]
    # 세계사 is a substring of nothing else; 경제 vs 정치와법 fine; 사회문화 ok
    hits = sorted(set(hits), key=lambda c: -len(KNAME[c]))
    return hits[0] if hits else None


def pick(files, *must, avoid=("짝수",)):
    c = [f for f in files if all(m in os.path.basename(f) for m in must) and not any(a in os.path.basename(f) for a in avoid)]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs, problems = [], []
    for d in sorted(glob.glob(os.path.join(a.dl, "20*_*"))):
        yr, exam = os.path.basename(d).split("_", 1)
        yr = int(yr)
        for area in ("kor", "eng", "sci", "soc"):
            ad = os.path.join(d, area)
            files = sorted(glob.glob(os.path.join(ad, "*.pdf")) + glob.glob(os.path.join(ad, "*_unzipped", "*.pdf")))
            def rel(f):
                return f[len(ad) + 1:]
            keys = [f for f in files if "정답" in rel(f) and "짝수" not in rel(f)]
            papers = [f for f in files if "정답" not in rel(f) and "대본" not in rel(f) and "짝수" not in rel(f)]
            base = dict(area=area, exam=exam, school_year=yr)
            if area in ("kor", "eng"):
                if area == "eng":
                    papers = [f for f in papers if "대본" not in f]
                if len(papers) != 1 or len(keys) != 1:
                    problems.append((d, area, papers, keys)); continue
                j = dict(base, key=f"{yr}_{exam}_{area}", subject=area, section="kor_common" if area == "kor" else "eng",
                         pdf=papers[0], answer_key=keys[0])
                if area == "kor":
                    j["expect"] = {"kor_common": [1, 34], "kor_speech_writing": [35, 45], "kor_lang_media": [35, 45]}
                else:
                    sc = [f for f in files if "대본" in os.path.basename(f)]
                    j["script"] = sc[0] if sc else None
                    j["expect"] = {"eng": [1, 45]}
                    if not sc:
                        problems.append((d, area, "no script"))
                jobs.append(j)
            else:
                table = SCI if area == "sci" else SOC
                pp = [f for f in papers if subj_of(f, table)]
                kk = [f for f in keys]
                for name, code in table:
                    p = [f for f in pp if subj_of(f, table) == code]
                    k = [f for f in kk if subj_of(f, table) == code] or [f for f in kk if subj_of(f, table) is None]
                    if len(p) != 1 or len(k) != 1:
                        problems.append((d, code, p, k)); continue
                    jobs.append(dict(base, key=f"{yr}_{exam}_{area}_{code}", subject=code, section=code, pdf=p[0], answer_key=k[0],
                                     answer_subject=name, expect={code: [1, 20]}))
    json.dump(jobs, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(len(jobs), "jobs")
    for p in problems:
        print("PROBLEM", p)


if __name__ == "__main__":
    main()
