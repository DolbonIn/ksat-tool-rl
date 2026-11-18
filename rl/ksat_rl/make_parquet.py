"""Convert KSAT math JSONL (data_pipeline schema) into veRL RLHFDataset parquet.

Prompt matches the baseline eval prompt (experiments/csat2026_baseline/code/solve.py) with the exam name
filled in, so the train-time instruction and the 2026 eval instruction differ only in the exam label.
"""
import argparse
import json

import pandas as pd

EXAM_NAME = {
    "csat": "대학수학능력시험", "kice_06": "대학수학능력시험 6월 모의평가", "kice_09": "대학수학능력시험 9월 모의평가",
    "edu_03": "3월 전국연합학력평가", "edu_04": "4월 전국연합학력평가", "edu_07": "7월 전국연합학력평가",
    "edu_10": "10월 전국연합학력평가",
}


def instruction(p):
    head = f"다음은 {p['school_year']}학년도 {EXAM_NAME.get(p['exam'], '대학수학능력시험')} 수학 영역"
    if p.get("track"):  # 2017–2021학년도 가형/나형
        head += f"({p['track']})"
    if p["exam"] == "synth":  # generated variants: no real sitting/year
        head = "다음은 대학수학능력시험 수학 영역"
    if p["kind"] == "mc":
        return f"{head} 객관식 문제입니다. 단계적으로 풀이한 뒤, 마지막에 정답 선택지 번호(1~5 중 하나)만 \\boxed{{}} 안에 넣으세요. 예: \\boxed{{3}}"
    return f"{head} 단답형 문제입니다. 단계적으로 풀이한 뒤, 마지막에 최종 답(0 이상 999 이하의 정수)만 \\boxed{{}} 안에 넣으세요."


def to_row(p, split):
    gt = {"kind": p["kind"], "answer": int(p["answer"]), "choices": p.get("choices")}
    return {
        "data_source": "ksat_math",
        "prompt": [{"role": "user", "content": f"{instruction(p)}\n\n{p['text']}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
        "extra_info": {"id": p["id"], "split": split, "points": int(p.get("points") or 0), "section": p.get("section", "")},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", default=None, help="optional file with one id per line to keep")
    a = ap.parse_args()
    probs = [json.loads(l) for l in open(a.inp) if l.strip()]
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        probs = [p for p in probs if p["id"] in keep]
    pd.DataFrame([to_row(p, a.split) for p in probs]).to_parquet(a.out)
    print(f"wrote {len(probs)} rows -> {a.out}")
