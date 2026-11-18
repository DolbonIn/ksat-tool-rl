"""Text-layer-assisted transcription of segments with DeepSeek V4.1 Flash (image + PDF text-layer draft).

The text layer is the authoritative character source (Korean/English/digits/symbols); the image is used for reading
order, structure (boxes, tables, brackets, underlines), formulas and figure/table descriptions.
Results: OCR_DIR/<key>/<sid>.json {"attempts": [...], "final": parsed, "model", ...}

Usage: python -I ocr_assist.py --render /data/render2 --out /data/ocr2 --keys 2026_csat_kor,... [--sids a,b] [--force]
       [--attempt-tag retry1 --temperature 0.3 --thinking]
"""
import argparse, asyncio, base64, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw  # noqa: E402

AREA = {"kor": "국어", "eng": "영어", "sci": "과학탐구", "soc": "사회탐구"}
SUBJ_KO = {"phys1": "물리학Ⅰ", "chem1": "화학Ⅰ", "bio1": "생명과학Ⅰ", "earth1": "지구과학Ⅰ", "phys2": "물리학Ⅱ", "chem2": "화학Ⅱ",
           "bio2": "생명과학Ⅱ", "earth2": "지구과학Ⅱ", "ethics_life": "생활과 윤리", "ethics_thought": "윤리와 사상",
           "geo_korea": "한국지리", "geo_world": "세계지리", "east_asian_hist": "동아시아사", "world_hist": "세계사",
           "economics": "경제", "politics_law": "정치와 법", "society_culture": "사회·문화",
           "kor_common": "공통", "kor_speech_writing": "화법과 작문", "kor_lang_media": "언어와 매체", "eng": ""}

RULES = r"""규칙:
1. [텍스트 레이어]의 단어와 문장을 바꾸거나 빼거나 보태지 마세요. 맞춤법이 이상해 보여도 고치지 말고 원문대로 적습니다. 순서는 이미지의 읽기 순서를 따릅니다.
2. '¦'는 원문의 줄바꿈 지점입니다. 한국어 띄어쓰기(영어는 단어 경계)에 맞게 공백 하나 또는 빈 문자열로 바꾸세요. 예: '파악¦하지'→'파악하지', '단어를¦인식하는'→'단어를 인식하는', 'the¦results'→'the results'. 출력에 '¦'를 남기지 마세요.
3. 문단이 바뀌는 곳은 이미지대로 줄을 바꿉니다. 시(운문)는 행마다 줄을 바꾸고 연 사이는 빈 줄로 둡니다. 대화·대사·항목은 이미지처럼 줄을 나눕니다.
4. <u>…</u>(밑줄)와 [A] … [/A](대괄호로 묶인 구간) 표시는 그대로 유지합니다. 이미지에 밑줄이나 [B] 같은 묶음 표시가 분명히 보이는데 빠져 있으면 같은 방식으로 추가합니다.
5. <보기> 같은 상자 자료는 '<보기>' 줄 다음에 내용을 적습니다('<보 기>'는 '<보기>'로 씁니다). ㄱ. ㄴ. ㄷ. / (가) (나) / ◦ / A, B 같은 항목은 각각 줄을 바꿉니다. 상자 안의 빈칸(㉠, (가) 등 표시가 있는 네모 칸)은 원문 표기대로 둡니다.
6. 표는 그 자리에 [표] 다음 줄부터 마크다운 표로 적고 [/표]로 닫습니다. 칸의 글자와 숫자는 [텍스트 레이어]와 이미지를 대조해 정확히 옮기고, 병합된 칸·빈칸도 이미지대로 나타냅니다.
7. 그림·그래프·도표·지도·사진·삽화·회로도·모식도·실험 장치·화학 구조식 같은 시각 자료는 그 자리에 [그림] … [/그림] 블록으로 넣고, 그림을 보지 않고도 문제를 풀 수 있도록 보이는 정보를 빠짐없이 정확히 기술합니다: 자료의 종류, 축의 이름·단위·눈금 값, 점·선·막대의 위치와 값(눈금에서 직접 읽히는 값만), 범례와 색·무늬 구분, 기호·문자 레이블과 위치 관계(위/아래/왼쪽/오른쪽, 연결, 포함), 화살표 방향, 그림 속 글자와 숫자. 그림에서 직접 읽을 수 없는 값을 추론·계산하거나 정답을 암시하는 해석을 덧붙이지 마세요. 그림 속 글자는 [텍스트 레이어]의 표기를 따르세요.
8. 수식·화학식·단위·변수는 이미지에서 읽어 $…$ 안에 LaTeX로 적습니다(예: $h_0$, $\frac{1}{2}mv^2$, $\mathrm{H_2O}$, $2\times10^{3}\,\mathrm{K}$). [텍스트 레이어]의 '□'는 추출되지 않은 수식 기호이니 이미지에서 읽습니다. 수식 속 숫자는 [텍스트 레이어]의 숫자와 같아야 합니다.
9. 쪽 머리말(영역·과목명·쪽 번호·홀수형), 쪽 번호, 저작권 문구, '※ 확인 사항' 안내는 적지 않습니다.
10. 풀이하지 말고 전사만 하세요."""

FMT_Q = r"""출력 형식(이 형식만 출력):
<number>문항 번호</number>
<stem>
문항 번호를 뺀 발문과 그 문항의 자료(<보기>, 지문, 표, 그림 포함). '[3점]' 표시는 원래 자리에 그대로 둡니다.
</stem>
<choices>
① …
② …
③ …
④ …
⑤ …
</choices>
<has_figure>그림·그래프·지도·사진 등 시각 자료가 있으면 yes, 없으면 no (표만 있으면 no)</has_figure>

선택지 규칙: 선택지가 표 형식(머리글이 ㄱ ㄴ ㄷ, (가) (나), A B C 등)이면 각 선택지를 '① A: 강자성체, B: 상자성체, C: 반자성체'처럼 머리글을 붙여 한 줄로 적습니다. 선택지가 그림이면 각 선택지 그림을 글로 기술합니다. 선택지 번호(①~⑤)가 그림 속 대상을 가리키는 표시이면 각 선택지에 그 번호가 가리키는 대상을 짧게 적습니다(예: '① 기차 앞부분의 난간'). 선택지가 영어 단어·구이면 원문 그대로 적습니다."""

FMT_P = r"""이 영역은 여러 문항이 함께 쓰는 지문(세트)입니다. 맨 앞의 '[n～m] …' 안내문부터 지문 끝까지 전부 옮깁니다.
출력 형식(이 형식만 출력):
<passage>
안내문과 지문 전체
</passage>
<has_figure>지문에 그림·그래프·사진 등 시각 자료가 있으면 yes, 없으면 no (표만 있으면 no)</has_figure>"""


def img_part(path):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()}}


def build_prompt(job, seg):
    area = AREA[job["area"]]
    sub = SUBJ_KO.get(seg["section"], "")
    where = f"{area} 영역" + (f"({sub})" if sub and job["area"] in ("sci", "soc") else (f" {sub}" if sub and sub != "공통" else ""))
    what = f"{seg['number']}번 문항" if seg["kind"] == "question" else f"[{seg['range'][0]}～{seg['range'][1]}] 지문"
    head = (f"이미지는 대한민국 대학수학능력시험(또는 모의평가) {where} 문제지에서 {what} 부분을 잘라낸 것입니다. "
            "이미지가 여러 장이면 순서대로 이어지는 하나의 영역입니다(단, 첫 이미지 위쪽이나 마지막 이미지 아래쪽에 다른 문항의 일부가 조금 보이면 무시하세요).\n"
            "아래 [텍스트 레이어]는 같은 영역에서 PDF 텍스트 레이어를 추출한 원문입니다. 글자(한글·영문·숫자·기호)는 [텍스트 레이어]가 정확하므로 그대로 사용하고, "
            "이미지로는 읽기 순서·배치·구조·수식·그림을 확인하세요.\n\n작업: 이 영역을 원문 그대로, 구조를 살려 텍스트로 옮겨 적으세요.\n")
    fmt = FMT_Q if seg["kind"] == "question" else FMT_P
    text = f"{head}\n{RULES}\n\n{fmt}\n\n[텍스트 레이어]\n{seg['draft']}\n[/텍스트 레이어]"
    return text


TAGS = ["number", "stem", "choices", "has_figure", "passage"]


def parse(raw):
    out = {}
    for t in TAGS:
        m = re.search(rf"<{t}>(.*?)</{t}>", raw or "", re.S)
        out[t] = m.group(1).strip("\n") if m else None
    ch = out.get("choices")
    if ch and re.search(r"[①②③④⑤]", ch):
        parts = re.split(r"\s*([①②③④⑤])\s*", ch)
        lst = {}
        for i in range(1, len(parts) - 1, 2):
            lst.setdefault(parts[i], parts[i + 1].strip())
        out["choices_list"] = [lst.get(c) for c in "①②③④⑤"]
    else:
        out["choices_list"] = None
    return out


async def do_one(args, job, seg, render):
    od = os.path.join(args.out, job["key"]); os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, seg["sid"] + ".json")
    rec = json.load(open(fn)) if os.path.exists(fn) else {"sid": seg["sid"], "attempts": []}
    if rec.get("final") and not args.force and not args.attempt_tag:
        return
    if args.attempt_tag and any(a.get("tag") == args.attempt_tag and a.get("finish") == "stop" for a in rec["attempts"]) and not args.force:
        return
    imgs = [img_part(os.path.join(render, job["key"], p)) for p in seg["pngs"]]
    prompt = build_prompt(job, seg)
    if args.extra_note:
        prompt += "\n\n[추가 확인] " + args.extra_note
    hint = (args.hints or {}).get(f"{job['key']}/{seg['sid']}")
    if hint:
        prompt += "\n\n[이전 전사 검토 결과 — 반드시 반영] " + hint
    msgs = [{"role": "user", "content": imgs + [{"type": "text", "text": prompt}]}]
    t0 = time.time()
    r = await gw.chat(msgs, max_tokens=args.max_tokens, temperature=args.temperature, thinking=args.thinking,
                      tag=f"ocr:{job['key']}/{seg['sid']}", prefer=args.prefer,
                      strict=args.strict, retries=30 if args.strict else 8)
    att = dict(tag=args.attempt_tag or "a", model=r["model"], finish=r["finish"], usage=r["usage"], error=r["error"],
               content=r["content"], reasoning_len=len(r["reasoning"]), thinking=args.thinking, temperature=args.temperature,
               parsed=parse(r["content"]))
    rec["attempts"].append(att)
    if not args.attempt_tag:
        rec["final"] = att["parsed"]; rec["final_from"] = len(rec["attempts"]) - 1
    json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
    print(f"{job['key']}/{seg['sid']}: {time.time()-t0:.0f}s {r['model']} {r['finish']} {r['usage']}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", default=None)
    ap.add_argument("--sids", default=None, help="comma list of key/sid")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--attempt-tag", default=None)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=12000); ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--prefer", default=None); ap.add_argument("--extra-note", default=None)
    ap.add_argument("--strict", action="store_true", help="use only --prefer model (retry on 429)")
    ap.add_argument("--hints", default=None, help="JSON {key/sid: note} appended to the prompt")
    ap.add_argument("--usage-log", default="/data/work2/usage.jsonl")
    args = ap.parse_args()
    args.hints = json.load(open(args.hints)) if args.hints else None
    gw.init(args.concurrency, args.usage_log)
    jobs = {j["key"]: j for f in args.jobs.split(",") for j in json.load(open(f))}
    keys = args.keys.split(",") if args.keys else None
    sids = set(args.sids.split(",")) if args.sids else None
    tasks = []
    for key, job in jobs.items():
        if keys and not any(key == k or (k.endswith("*") and key.startswith(k[:-1])) for k in keys):
            continue
        sf = os.path.join(args.render, key, "segments.json")
        if not os.path.exists(sf):
            print("no segments", key); continue
        for seg in json.load(open(sf))["segments"]:
            if sids and f"{key}/{seg['sid']}" not in sids:
                continue
            if seg["section"] in (job.get("skip_sections") or []):
                continue
            tasks.append(do_one(args, job, seg, args.render))
    t0 = time.time()
    await asyncio.gather(*tasks)
    print(f"done {len(tasks)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
