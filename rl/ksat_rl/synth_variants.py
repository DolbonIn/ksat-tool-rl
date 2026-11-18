"""Generate verified KSAT-style math variants with DeepSeek V4.1 Flash (opengateway.ai).

Pipeline per seed problem (train split only; never the 2026 test):
  1. GENERATE  deepseek-v4.1-flash-eco (batches 1-2 used ultrafast) writes K new 단답형 problems in the seed's 수능 style
               (same concept family, structurally changed, integer answer 0..999, text-only), plus
               its own answer.
  2. VERIFY    deepseek-v4.1-flash-eco solves each new problem N times independently (it never sees
               the generator's answer or solution).
  3. ACCEPT    keep only if >= MIN_AGREE of N independent answers equal the generator's answer.
Accepted items get exam="synth", family_id=<seed id> so train/test family separation can be enforced
(ksat_tool_rl_v2/README.md §3: generator families must not cross splits).

Answers of accepted items are model-consensus labels, not official keys; they are marked
`label_source: "deepseek_consensus"` and kept separate from official items in every report.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ksat_rl.reward import last_boxed  # noqa: E402

GEN_MODEL = "deepseek/deepseek-v4.1-flash-eco"  # user instruction 2026-11-18: dataset work uses eco only (batches 1-2 used ultrafast; recorded per item in verify.gen_model)
VER_MODEL = "deepseek/deepseek-v4.1-flash-eco"

SECTION_KO = {"common": "공통(수학Ⅰ·수학Ⅱ)", "prob_stat": "확률과 통계", "calculus": "미적분", "geometry": "기하",
              "ga": "수학 가형(2017–2021학년도)", "na": "수학 나형(2017–2021학년도)"}

GEN_PROMPT = """당신은 대학수학능력시험 수학 영역 출제위원입니다. 아래 [원 문항]을 참고하여 **새로운 단답형 문항 1개**를 출제하세요.

[과목] {section}
[원 문항]
{text}
[원 문항 정답] {answer}

출제 조건:
1. 원 문항과 같은 핵심 개념과 사고 과정을 요구하되, 숫자만 바꾸지 말고 조건·구조·함수·상황 중 적어도 하나를 본질적으로 바꾸세요. 변형 방향: {direction}
2. 난이도는 원 문항과 같거나 더 높게(수능 4점 문항 수준) 하세요.
3. 그림·표 없이 텍스트와 LaTeX 수식만으로 완결되어야 합니다.
4. 정답은 0 이상 999 이하의 정수 하나로 유일하게 정해져야 합니다. 분수·무리수 답이 나오면 "p+q의 값을 구하시오"처럼 정수로 바꾸세요.
5. 출제 후 스스로 끝까지 풀어 정답이 유일하고 조건에 모순이 없는지 검산하세요.
6. 실제 수능 문항처럼 한국어로 쓰고, 문항 끝에 배점 표기는 넣지 마세요.

최종 출력은 아래 JSON 하나만 쓰세요(코드블록 없이):
{{"problem": "<문항 본문>", "answer": <정수>, "concept": "<핵심 개념 한 줄>"}}"""

DIRECTIONS = [
    "조건을 하나 추가하거나 바꿔 경우를 나누는 과정이 필요하게",
    "주어진 함수나 수열의 형태를 바꿔 같은 아이디어를 다른 대상에 적용하게",
    "묻는 대상을 바꿔(예: 값 → 개수, 최댓값 → 최솟값과 그때의 값) 역으로 추론하게",
    "두 개념을 결합해 한 단계 더 깊은 추론이 필요하게",
]

# --mode easier: stepping-stone variants for items the policy (almost) never solves (pass@8 <= 1/8).
EASY_PROMPT = """당신은 대학수학능력시험 수학 영역 출제위원입니다. 아래 [원 문항]은 학생들이 거의 풀지 못한 매우 어려운 문항입니다.
이 문항을 풀기 위한 **디딤돌 문항 1개**(단답형)를 출제하세요.

[과목] {section}
[원 문항]
{text}
[원 문항 정답] {answer}

출제 조건:
1. 원 문항의 핵심 아이디어(결정적인 관찰·기법)를 반드시 쓰게 하되, 원 문항보다 **쉽게** 만드세요. 변형 방향: {direction}
2. 난이도는 수능 3점~쉬운 4점 수준. 원 문항을 그대로 베끼거나 숫자만 바꾸지 마세요.
3. 그림·표 없이 텍스트와 LaTeX 수식만으로 완결되어야 합니다.
4. 정답은 0 이상 999 이하의 정수 하나로 유일해야 합니다. 분수·무리수 답이면 "p+q의 값을 구하시오"처럼 정수로 바꾸세요.
5. 출제 후 스스로 끝까지 풀어 정답이 유일하고 조건에 모순이 없는지 검산하세요.
6. 실제 수능 문항처럼 한국어로 쓰고, 배점 표기는 넣지 마세요.

최종 출력은 아래 JSON 하나만 쓰세요(코드블록 없이):
{{"problem": "<문항 본문>", "answer": <정수>, "concept": "<원 문항에서 가져온 핵심 아이디어 한 줄>"}}"""

EASY_DIRECTIONS = [
    "원 문항의 풀이 단계 중 앞부분만 떼어 내 그 단계의 결과를 묻게",
    "원 문항의 조건 중 하나를 없애거나 특수한 경우(작은 수, 구체적인 함수)로 바꿔 핵심 관찰만 남게",
    "원 문항의 중간 결과 하나를 조건으로 주고 나머지 과정을 풀게",
    "원 문항의 결정적인 기법을 가장 단순한 상황에 적용하게",
]

# --mode notation: Korean-curriculum notation drills (중복조합 nHr, 중복순열 nΠr), which the policy does not know.
NOTATION_PROMPT = """당신은 대학수학능력시험 수학 영역(확률과 통계) 출제위원입니다. 아래 [원 문항]과 같은 **한국 교육과정의 표기**를 사용하는 새로운 문항 1개를 출제하세요.

[원 문항]
{text}
[원 문항 정답] {answer}

출제 조건:
1. 중복조합 ${{}}_{{n}}\\mathrm{{H}}_{{r}}$ 또는 중복순열 ${{}}_{{n}}\\Pi_{{r}}$ 표기를 문항 본문에 그대로 사용하세요(이 표기를 알아야 풀 수 있게). 변형 방향: {direction}
2. 난이도는 수능 2점~3점 수준.
3. 정답은 0 이상 999 이하의 정수 하나로 유일해야 하며, 단답형으로 출제하세요.
4. 출제 후 스스로 풀어 정답을 검산하세요. 배점 표기는 넣지 마세요.

최종 출력은 아래 JSON 하나만 쓰세요(코드블록 없이):
{{"problem": "<문항 본문>", "answer": <정수>, "concept": "<사용한 표기와 개념 한 줄>"}}"""

NOTATION_DIRECTIONS = [
    "표기의 값을 직접 계산하는 식(두 표기의 합·차·곱 등)으로",
    "중복조합 표기를 방정식의 음이 아닌 정수해의 개수와 연결해",
    "중복순열 표기를 함수의 개수나 숫자 배열의 개수와 연결해",
    "두 표기를 함께 써서 비교하거나 미지수를 구하게",
]

# --mode terms: Korean school-math vocabulary drills (the policy e.g. read 사차함수 as "cubic").
TERMS_PROMPT = """당신은 대학수학능력시험 수학 영역 출제위원입니다. 아래 [용어]를 **정확히 알아야만** 풀 수 있는 새로운 단답형 문항 1개를 출제하세요.

[과목] {section}
[용어와 뜻] {text}

출제 조건:
1. 문항 본문에 이 용어를 한국어 그대로 쓰고, 용어를 잘못 이해하면(예: 차수를 하나 틀리게 읽으면) 다른 답이 나오도록 만드세요. 변형 방향: {direction}
2. 난이도는 수능 2점~3점 수준, 그림 없이 텍스트와 LaTeX만 사용하세요.
3. 정답은 0 이상 999 이하의 정수 하나로 유일해야 합니다. 출제 후 스스로 풀어 검산하세요. 배점 표기는 넣지 마세요.

최종 출력은 아래 JSON 하나만 쓰세요(코드블록 없이):
{{"problem": "<문항 본문>", "answer": <정수>, "concept": "<사용한 용어 한 줄>"}}"""

TERMS_DIRECTIONS = [
    "용어가 정한 차수·개수·조건을 써서 미지수를 결정하게",
    "용어의 정의를 그대로 적용해 값을 계산하게",
    "비슷한 용어(예: 삼차/사차, 극댓값/최댓값)와 혼동하면 틀리게",
    "용어가 들어간 두 조건을 함께 써서 답을 구하게",
]

SOLVE_PROMPT = """다음은 대학수학능력시험 수학 영역 단답형 문제입니다. 단계적으로 풀이한 뒤, 마지막에 최종 답(0 이상 999 이하의 정수)만 \\boxed{{}} 안에 넣으세요.

{problem}"""


# A LaTeX command written with an unescaped backslash inside JSON (e.g. "\frac", "\theta", "\neq")
# would otherwise be read as a JSON escape (\f form feed, \t tab, \n newline ...) and corrupt the text.
# Escape any backslash that starts a 2+-letter ASCII word (except a real \uXXXX escape) before parsing.
_LATEX_CMD = re.compile(r'(?<!\\)\\(?=[A-Za-z]{2,})(?!u[0-9a-fA-F]{4})')


def parse_json(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    s = s[:i] + _LATEX_CMD.sub(r"\\\\", s[i:j + 1]) + s[j + 1:]
    j = s.rfind("}")
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        # LaTeX backslashes are often not JSON-escaped; escape lone backslashes and retry
        t = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", s[i:j + 1])
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return None


def int_answer(x):
    if x is None:
        return None
    x = str(x).strip().replace(",", "")
    x = re.sub(r"^\{|\}$", "", x)
    return int(x) if re.fullmatch(r"-?\d+", x) else None


class Client:
    def __init__(self, concurrency):
        self.c = AsyncOpenAI(base_url=os.environ["OPENGATEWAY_BASE_URL"], api_key=os.environ["OPENGATEWAY_API_KEY"], timeout=1800)
        self.sem = asyncio.Semaphore(concurrency)
        self.usage = {"prompt": 0, "completion": 0, "calls": 0, "errors": 0}

    async def chat(self, model, prompt, max_tokens=32768, temperature=0.7):
        for attempt in range(6):
            async with self.sem:
                try:
                    r = await self.c.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens, temperature=temperature)
                    u = r.usage
                    self.usage["prompt"] += u.prompt_tokens; self.usage["completion"] += u.completion_tokens; self.usage["calls"] += 1
                    return r.choices[0].message.content or "", r.choices[0].finish_reason
                except Exception as e:  # 429 / 5xx / timeouts: back off and retry
                    self.usage["errors"] += 1
                    msg = str(e)
                    if "budget" in msg.lower():
                        raise
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        return None, "error"


MODES = {"harder": (GEN_PROMPT, DIRECTIONS), "easier": (EASY_PROMPT, EASY_DIRECTIONS), "notation": (NOTATION_PROMPT, NOTATION_DIRECTIONS),
         "terms": (TERMS_PROMPT, TERMS_DIRECTIONS)}


async def make_variant(cli, seed, k, args):
    template, directions = MODES[args.mode]
    direction = directions[k % len(directions)]
    prompt = template.format(section=SECTION_KO.get(seed["section"], seed["section"]), text=seed["text"],
                             answer=seed["answer"], direction=direction)
    content, finish = await cli.chat(GEN_MODEL, prompt, max_tokens=args.max_tokens, temperature=0.9)
    rec = {"seed_id": seed["id"], "k": k, "direction": direction, "gen_finish": finish, "mode": args.mode}
    obj = parse_json(content or "")
    ans = int_answer(obj.get("answer")) if obj else None
    if not obj or not obj.get("problem") or ans is None or not (0 <= ans <= 999):
        rec["status"] = "gen_parse_fail"; rec["raw"] = (content or "")[-2000:]
        return rec
    rec.update(problem=obj["problem"].strip(), gen_answer=ans, concept=obj.get("concept", ""))
    sols = await asyncio.gather(*[cli.chat(VER_MODEL, SOLVE_PROMPT.format(problem=rec["problem"]), max_tokens=args.max_tokens, temperature=0.7)
                                  for _ in range(args.n_verify)])
    preds = [int_answer(last_boxed(c or "")) for c, _ in sols]
    rec["verify_preds"] = preds
    agree = sum(p == ans for p in preds)
    rec["agree"] = agree
    rec["status"] = "accepted" if agree >= args.min_agree else "rejected"
    return rec


def to_problem(rec, seed):
    return {
        "id": f"synth{'' if rec.get('mode', 'harder') == 'harder' else '_' + rec['mode']}-{seed['id']}-{rec['k']}", "exam": "synth", "school_year": seed.get("school_year"),
        "section": seed["section"], "number": seed.get("number"), "kind": "sa", "points": {"harder": 4, "easier": 3, "notation": 2, "terms": 2}[rec.get("mode", "harder")],
        "text": rec["problem"], "choices": None, "answer": rec["gen_answer"], "has_figure": False,
        "source_file": None, "family_id": seed["id"], "label_source": "deepseek_consensus",
        "verify": {"agree": rec["agree"], "n": len(rec["verify_preds"]), "preds": rec["verify_preds"],
                   "gen_model": GEN_MODEL, "ver_model": VER_MODEL, "direction": rec["direction"], "mode": rec.get("mode", "harder")},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="JSONL of seed problems (train split)")
    ap.add_argument("--ids", default=None, help="optional file: seed ids to use")
    ap.add_argument("--exclude-test", default=None, help="test JSONL; seeds whose id appears there are refused")
    ap.add_argument("--k", type=int, default=3, help="variants per seed")
    ap.add_argument("--mode", choices=list(MODES), default="harder", help="harder: same-or-harder variants; easier: stepping stones; notation: nHr/nPir drills")
    ap.add_argument("--n-verify", type=int, default=4)
    ap.add_argument("--min-agree", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=65536, help="per call; hard seeds exhaust 32k while reasoning")
    ap.add_argument("--out", required=True, help="accepted problems JSONL; <out>.log.jsonl keeps every attempt")
    a = ap.parse_args()
    seeds = [json.loads(l) for l in open(a.seeds) if l.strip()]
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        seeds = [s for s in seeds if s["id"] in keep]
    if a.exclude_test:
        test_ids = {json.loads(l)["id"] for l in open(a.exclude_test) if l.strip()}
        bad = [s["id"] for s in seeds if s["id"] in test_ids or str(s.get("id", "")).startswith("2026-csat")]
        assert not bad, f"test items in seeds: {bad[:5]}"
    if a.limit:
        seeds = seeds[: a.limit]
    cli = Client(a.concurrency)
    t0 = time.time()
    attempted = set()
    if os.path.exists(a.out + ".log.jsonl"):  # resume: keep finished attempts, run only the rest
        for l in open(a.out + ".log.jsonl"):
            r = json.loads(l); attempted.add((r["seed_id"], r["k"]))
    tasks = [make_variant(cli, s, k, a) for s in seeds for k in range(a.k) if (s["id"], k) not in attempted]
    print(f"resume: {len(attempted)} attempts already logged, {len(tasks)} to run", flush=True)
    by_id = {s["id"]: s for s in seeds}
    n_acc = 0
    with open(a.out, "a") as fo, open(a.out + ".log.jsonl", "a") as fl:
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            fl.write(json.dumps(rec, ensure_ascii=False) + "\n"); fl.flush()
            if rec["status"] == "accepted":
                fo.write(json.dumps(to_problem(rec, by_id[rec["seed_id"]]), ensure_ascii=False) + "\n"); fo.flush()
                n_acc += 1
    print(json.dumps({"seeds": len(seeds), "attempts": len(tasks), "accepted": n_acc, "secs": round(time.time() - t0),
                      "usage": cli.usage}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
