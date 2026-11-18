"""Build jobs JSON for 교육청 학평 papers downloaded by fetch_edu_multi.py.
국어: two PDFs per sitting (화법과 작문 / 언어와 매체), each = 공통 1-34 + that elective 35-45. The 화작 PDF supplies the
common items; the 언매 PDF only its elective (job flag skip_sections).
Usage: python3 make_jobs_edu.py --dl /data/dl2/edu --out /data/work2/jobs_edu.json
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_jobs_kice import SCI, SOC, subj_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs, problems = [], []
    for d in sorted(glob.glob(os.path.join(a.dl, "20*_edu_*"))):
        sy, _, mm = os.path.basename(d).split("_")
        sy = int(sy); exam = f"edu_{mm}"
        base = dict(exam=exam, school_year=sy)
        for area in ("kor", "eng", "sci", "soc"):
            ad = os.path.join(d, area)
            if not os.path.isdir(ad):
                problems.append((d, area, "missing")); continue
            fs = sorted(glob.glob(os.path.join(ad, "*")))
            def f(prefix, kind):
                c = [x for x in fs if os.path.basename(x).startswith(prefix) and os.path.basename(x).rsplit("_", 1)[1].startswith(kind)]
                return c[0] if c else None
            if area == "kor":
                for pre, code, sec in (("화법과_작문", "kor_sw", "kor_speech_writing"), ("언어와_매체", "kor_lm", "kor_lang_media")):
                    p, k, h = f(pre, "paper"), f(pre, "answer"), f(pre, "hsj")
                    if not (p and k and h):
                        problems.append((d, code, p, k, h)); continue
                    j = dict(base, key=f"{sy}_{exam}_{code}", area="kor", subject=code, section="kor_common", pdf=p, answer_png=k, hsj=h,
                             expect={"kor_common": [1, 34], sec: [35, 45]})
                    if code == "kor_lm":
                        j["skip_sections"] = ["kor_common"]
                    jobs.append(j)
            elif area == "eng":
                p, k, h, s = f("영어", "paper"), f("영어", "answer"), f("영어", "hsj"), f("영어", "script")
                if not (p and k and h and s):
                    problems.append((d, "eng", p, k, h, s)); continue
                jobs.append(dict(base, key=f"{sy}_{exam}_eng", area="eng", subject="eng", section="eng", pdf=p, answer_png=k, hsj=h,
                                 script=s, expect={"eng": [1, 45]}))
            else:
                table = SCI if area == "sci" else SOC
                for name, code in table:
                    p = [x for x in fs if x.endswith("_paper.pdf") and subj_of(x, table) == code]
                    k = [x for x in fs if x.endswith("_answer.png") and subj_of(x, table) == code]
                    h = [x for x in fs if x.endswith("_hsj.pdf") and subj_of(x, table) == code]
                    if not p:
                        continue
                    if not (k and h):
                        problems.append((d, code, p, k, h)); continue
                    jobs.append(dict(base, key=f"{sy}_{exam}_{area}_{code}", area=area, subject=code, section=code, pdf=p[0],
                                     answer_png=k[0], hsj=h[0], answer_subject=name, expect={code: [1, 20]}))
    json.dump(jobs, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(len(jobs), "jobs")
    for p in problems:
        print("PROBLEM", p)


if __name__ == "__main__":
    main()
