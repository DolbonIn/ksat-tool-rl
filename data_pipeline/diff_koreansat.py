"""Diff our OCR of 수능 2022-2025 against cfpark00/KoreanSAT (answers + text similarity).

Text similarity is computed on a normalized "skeleton": Hangul characters and digits/latin letters only
(LaTeX commands, whitespace and punctuation removed), so formatting differences don't count.
Writes a per-id verify JSON {id: {"koreansat_answer_match": bool, "koreansat_points_match": bool,
"koreansat_sim": float, "koreansat_digits_equal": bool}} and prints the lowest-similarity items.

Usage (tools image): python -I diff_koreansat.py --hf /data/hf/KoreanSAT --train /data/ksat_math/train.jsonl --out x.json
"""
import argparse, difflib, json, re
import pandas as pd

SEC = {"prob": "prob_stat", "calc": "calculus", "geom": "geometry"}


def skel(s):
    s = re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", " ", s, flags=re.S)   # KoreanSAT choices block
    s = re.sub(r"\[그림\].*?\[/그림\]", " ", s, flags=re.S)
    s = re.sub(r"\\(text|mathrm|operatorname)\{", "{", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)
    s = re.sub(r"\[\s*\d\s*점\s*\]", "", s)
    return re.sub(r"[^가-힣0-9a-zA-Z]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ours = {}
    for l in open(a.train):
        r = json.loads(l)
        if r["exam"] == "csat":
            ours[r["id"]] = r
    res, rows = {}, []
    for y in (2022, 2023, 2024, 2025):
        df = pd.read_parquet(f"{a.hf}/{y}_math.parquet")
        for r in df.itertuples():
            n, _, s = r.name.partition("_")
            pid = f"{y}-csat-{SEC.get(s, 'common')}-{int(n)}"
            o = ours.get(pid)
            if not o:
                continue
            stem_o = o["text"].split("\n①")[0]
            a1, b1 = skel(stem_o), skel(r.problem)
            sim = difflib.SequenceMatcher(None, a1, b1, autojunk=False).ratio()
            dig_o = re.sub(r"\D", "", a1); dig_k = re.sub(r"\D", "", b1)
            res[pid] = dict(koreansat_answer_match=int(r.answer) == o["answer"], koreansat_points_match=int(r.score) == o["points"],
                            koreansat_sim=round(sim, 3), koreansat_digits_equal=dig_o == dig_k)
            rows.append((sim, pid, dig_o == dig_k))
    json.dump(res, open(a.out, "w"), indent=1)
    rows.sort()
    print(f"compared {len(res)}; answer mismatches: {[k for k,v in res.items() if not v['koreansat_answer_match']]}")
    print(f"mean sim {sum(r[0] for r in rows)/max(1,len(rows)):.3f}; digits-equal {sum(r[2] for r in rows)}/{len(rows)}")
    for sim, pid, de in rows[:25]:
        print(f"  {pid:28s} sim={sim:.3f} digits_equal={de}")


if __name__ == "__main__":
    main()
