"""Spot-check against a public copy: HRM8K KSM (HAERAE-HUB/HRM8K, ksm_test.csv, category '수능/모의고사').
KSM has no source ids, so items are matched by Hangul+digit character-trigram Jaccard similarity of the problem text.
For matches above --thr the KSM answer is compared with ours (mc: KSM gives the choice value -> compared with the
text of our official-answer choice, normalised; sa: integer compare). Output JSON {id: {...}} for verify merging.
Usage: python -I cross_hrm8k.py --ksm /data/dl3/hf_hrm8k/ksm_test.csv --in /data/ksat_math_old/X.jsonl --out /data/work3/verify_hrm8k.json
"""
import argparse, json, re
import pandas as pd


def grams(s):
    s = re.sub(r"\\[a-zA-Z]+|[^가-힣0-9a-z]", "", s)
    return {s[i:i + 3] for i in range(len(s) - 2)}


def norm(x):
    return re.sub(r"\s+|\$|\\left|\\right|\\dfrac|\\frac|[{}()]|\\,", "", str(x)).replace("\\displaystyle", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksm", required=True); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.55)
    a = ap.parse_args()
    d = pd.read_csv(a.ksm)
    d = d[d["category"] == "수능/모의고사"].reset_index(drop=True)
    kg = [grams(q) for q in d["question"]]
    out, n = {}, 0
    for l in open(a.inp):
        r = json.loads(l)
        g = grams(re.sub(r"\[그림\].*?\[/그림\]", "", r["text"], flags=re.S))
        if not g:
            continue
        best, bi = 0, -1
        for i, h in enumerate(kg):
            s = len(g & h) / max(1, len(g | h))
            if s > best:
                best, bi = s, i
        if best < a.thr:
            continue
        n += 1
        ka = str(d.loc[bi, "answer"]).strip()
        if r["kind"] == "sa":
            try:
                match = int(float(ka)) == r["answer"]
            except ValueError:
                match = None
        else:
            ours = r["choices"][r["answer"] - 1] if r.get("choices") else None
            match = norm(ka) == norm(ours) if ours else None
            if not match and re.fullmatch(r"[1-5]", ka):
                match = int(ka) == r["answer"]
        out[r["id"]] = dict(hrm8k_ksm=dict(sim=round(best, 3), ksm_index=int(bi), ksm_difficulty=d.loc[bi, "difficulty"],
                                           ksm_answer=ka, answer_match=match))
        print(r["id"], round(best, 3), "ksm_ans", ka, "ours", r["answer"], "match", match, "|", d.loc[bi, "question"][:60].replace("\n", " "))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("matched", n)


if __name__ == "__main__":
    main()
