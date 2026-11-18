"""Transcribe per-problem crops with the local VLM (SGLang, OpenAI-compatible), then proofread.

Pass A (transcribe, thinking off, greedy) -> pass B (proofread vs image, thinking on).
Raw outputs: OCR_DIR/<key>/<name>.json  {"a": raw, "b": raw, "final": parsed fields, ...}

Usage (container with --network host, data mounted at /data):
  python -I ocr_vlm.py --render /data/render --out /data/ocr --keys 2026_csat,2025_csat [--passes a,b]
"""
import argparse, asyncio, base64, json, os, re, time
from openai import AsyncOpenAI

PROMPT_A = r"""이 이미지는 대한민국 대학수학능력시험(또는 모의평가·학력평가) 수학 영역 문제지에서 문항 하나를 잘라낸 것입니다. 문항을 빠짐없이, 원문 그대로 텍스트로 옮겨 적으세요.

규칙:
1. 한국어 문장은 원문 그대로 옮기고, 모든 수식·기호·변수·수식 글꼴의 숫자는 LaTeX로 적습니다. 인라인 수식은 $...$, 따로 한 줄을 차지하는 수식은 $$...$$로 감쌉니다.
2. 문항 번호(예: '13.')는 <number>에만 적고 본문에서는 뺍니다. 배점 표시(예: '[4점]')는 본문의 원래 위치에 그대로 둡니다.
3. 조건 상자의 '(가)', '(나)', '(다)'와 <보기>의 'ㄱ.', 'ㄴ.', 'ㄷ.' 같은 항목은 각각 줄을 바꿔 적습니다. <보기>는 '<보기>' 줄 다음에 항목을 적습니다. 상자로 둘러싸인 조건은 원문 순서대로 그대로 적습니다.
4. 조건에 따라 정의된 함수는 \begin{cases} ... \end{cases}로, 표는 마크다운 표로 적습니다. 선분 AB의 길이처럼 윗줄이 있는 기호는 \overline{AB}, 벡터는 \vec{a} 또는 \overrightarrow{AB}로 적습니다.
5. 그림·그래프·도형(좌표평면의 그래프, 기하 도형, 공·상자 같은 삽화 등)이 그려져 있으면 <has_figure>yes</has_figure>로 표시하고, <figure> 안에 그림에 실제로 나타난 정보만 문서 OCR처럼 자세히 기술합니다: 도형의 종류, 꼭짓점·점의 이름과 상대적 위치(어느 점이 어느 선분/곡선 위에 있는지), 좌표축과 원점 O, 곡선·직선의 모양(증가/감소, 볼록, 극값 위치 등 보이는 대로)과 교점, 그림에 적힌 길이·각·좌표·눈금 값과 문자, 색칠된 영역, 점선, 직각 표시 등. 그림에서 직접 읽을 수 없는 값을 계산하거나 추론해 덧붙이지 마세요. 테두리 상자로 둘러싸인 조건·시행 설명과 표는 그림이 아니므로 본문(<stem>)에 텍스트로 옮기세요(표는 마크다운 표). 그림이 없으면 <has_figure>no</has_figure>, <figure>없음</figure>으로 적습니다.
6. 5지선다형이면 선택지 ①~⑤를 <choices> 안에 한 줄에 하나씩 '① ...' 형식으로 적고, 단답형이면 <choices>없음</choices>으로 적습니다.
7. 문항과 관계없는 머리말·쪽 번호·'5지선다형'/'단답형' 표시·확인 사항 안내는 적지 않습니다.
8. 풀이하지 말고 전사만 하세요.

출력 형식(이 형식만 출력):
<number>문항 번호</number>
<points>배점 숫자</points>
<stem>
선택지를 제외한 문제 본문
</stem>
<has_figure>yes 또는 no</has_figure>
<figure>
그림 설명 또는 없음
</figure>
<choices>
① ...
② ...
③ ...
④ ...
⑤ ...
</choices>"""

PROMPT_B = r"""아래는 이 문항 이미지를 텍스트(한국어 + LaTeX)로 전사한 결과입니다. 이미지와 한 글자, 한 기호씩 꼼꼼히 대조해 전사 오류를 찾으세요.
확인할 것: 숫자, 부호(+, -), 지수·첨자, 분수의 분자·분모, 근호, 부등호 방향과 등호 포함 여부, 괄호, 함수·변수 이름, 로그의 밑, 극한·적분 구간, 빠지거나 중복된 문장·조건, 선택지 값과 순서, 배점, 그림 설명의 누락·오류(그림에 없는 정보를 지어낸 것 포함).
LaTeX 표기 방식의 차이(예: \dfrac와 \frac, 띄어쓰기)는 오류가 아닙니다. 의미가 달라지는 차이만 오류입니다.
본문의 배점 표시(예: '[4점]')는 원문에 있는 그대로 본문에 두는 것이 규칙이므로 오류가 아닙니다. 오류 목록에는 오류 설명만 쓰고, 태그 이름(<stem> 등)을 쓰지 마세요.

출력 형식:
<errors>
찾은 오류를 한 줄에 하나씩(없으면 '없음')
</errors>
오류가 하나라도 있으면 이어서 수정된 전체 전사를 원래와 같은 형식(<number>, <points>, <stem>, <has_figure>, <figure>, <choices>)으로 출력하세요. 오류가 없으면 <errors>없음</errors>만 출력하세요.

[전사 결과]
"""

TAGS = ["number", "points", "stem", "has_figure", "figure", "choices"]


def parse(raw):
    out = {}
    for t in TAGS:
        m = re.search(rf"<{t}>(.*?)</{t}>", raw or "", re.S)
        out[t] = m.group(1).strip() if m else None
    ch = out["choices"]
    if ch is None or ch in ("없음", "") or not re.search(r"[①②③④⑤]", ch):
        out["choices_list"] = None
    else:
        parts = re.split(r"\s*([①②③④⑤])\s*", ch)
        lst = {}
        for i in range(1, len(parts) - 1, 2):
            lst[parts[i]] = parts[i + 1].strip()
        out["choices_list"] = [lst.get(c) for c in "①②③④⑤"]
    return out


def fmt(p):
    return (f"<number>{p['number']}</number>\n<points>{p['points']}</points>\n<stem>\n{p['stem']}\n</stem>\n"
            f"<has_figure>{p['has_figure']}</has_figure>\n<figure>\n{p.get('figure') or '없음'}\n</figure>\n<choices>\n{p['choices'] if p['choices'] else '없음'}\n</choices>")


def img_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


async def call(client, sem, content, thinking, max_tokens, temperature=0.0, retries=3):
    async with sem:
        for k in range(retries):
            try:
                r = await client.chat.completions.create(
                    model="vlm", messages=[{"role": "user", "content": content}], temperature=temperature,
                    top_p=1.0 if temperature == 0 else 0.95, max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": thinking}, "top_k": 1 if temperature == 0 else 20})
                m = r.choices[0].message
                return dict(content=m.content or "", reasoning=getattr(m, "reasoning_content", None) or "",
                            finish=r.choices[0].finish_reason, tokens=r.usage.completion_tokens)
            except Exception as e:
                err = str(e)
                await asyncio.sleep(5 * (k + 1))
        return dict(content="", reasoning="", finish="error: " + err, tokens=0)


def finalize(rec):
    """Decide the final transcription from pass A + pass B (proofread). The corrected transcription is only
    parsed from text AFTER </errors> (the error list itself may mention tag names). The '[n점]' marker is
    restored if the proofreader dropped it."""
    content = rec["b"]["content"]
    em = re.search(r"<errors>(.*?)</errors>", content, re.S)
    errs = em.group(1).strip() if em else None
    rec["b_errors"] = errs
    tail = content[em.end():] if em else content
    fixed = parse(tail) if "<stem>" in tail else None
    no_err = errs is not None and re.sub(r"[\s.']", "", errs) in ("없음", "")
    rec["b_changed"] = bool(fixed and fixed.get("stem")) and not no_err
    if rec["b_changed"]:
        a_stem = rec["a_parsed"].get("stem") or ""
        ma = re.search(r"\[\s*\d\s*점\s*\]\s*$", a_stem)
        if ma and not re.search(r"\[\s*\d\s*점\s*\]", fixed["stem"]):
            fixed["stem"] = fixed["stem"].rstrip() + " " + ma.group(0).strip()
            rec["b_marker_restored"] = True
        for t in ("number", "points", "has_figure", "figure", "choices", "choices_list"):
            if fixed.get(t) is None:
                fixed[t] = rec["a_parsed"].get(t)
    rec["final"] = fixed if rec["b_changed"] else rec["a_parsed"]
    return rec


async def do_one(client, sem, args, key, crop, render):
    od = os.path.join(args.out, key); os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, crop["name"] + ".json")
    rec = json.load(open(fn)) if os.path.exists(fn) else {"name": crop["name"], "section": crop["section"], "number": crop["number"], "png": crop["png"]}
    url = img_url(os.path.join(render, crop["png"]))
    t0 = time.time()
    if "a" in args.passes and ("a" not in rec or args.force):
        rec["a"] = await call(client, sem, [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT_A}], False, 8192)
        rec["a_parsed"] = parse(rec["a"]["content"])
        rec.pop("b", None)
    if "b" in args.passes and "a" in rec and ("b" not in rec or args.force):
        content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT_B + fmt(rec["a_parsed"])}]
        rec["b"] = await call(client, sem, content, True, 32768, temperature=0.6)
        finalize(rec)
    elif "a" in rec and "final" not in rec:
        rec["final"] = rec["a_parsed"]
    json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
    print(f"{key}/{crop['name']}: {time.time()-t0:.0f}s changed={rec.get('b_changed')}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keys", required=True); ap.add_argument("--passes", default="a,b")
    ap.add_argument("--names", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--items", default=None, help="comma list of key/name (overrides --keys/--names selection)")
    ap.add_argument("--reparse", action="store_true", help="re-derive final from saved raw outputs, no server calls")
    ap.add_argument("--base-url", default="http://127.0.0.1:31801/v1"); ap.add_argument("--concurrency", type=int, default=96)
    args = ap.parse_args()
    args.passes = args.passes.split(",")
    if args.reparse:
        import glob
        n = 0
        for key in args.keys.split(","):
            for fn in glob.glob(os.path.join(args.out, key, "*.json")):
                rec = json.load(open(fn))
                if "a" in rec:
                    rec["a_parsed"] = parse(rec["a"]["content"])
                    if "b" in rec:
                        finalize(rec)
                    else:
                        rec["final"] = rec["a_parsed"]
                    json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1); n += 1
        print("reparsed", n)
        return
    client = AsyncOpenAI(base_url=args.base_url, api_key="x", timeout=3 * 3600)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for key in args.keys.split(","):
        idx = json.load(open(os.path.join(args.render, "crops", key, "index.json")))
        for c in idx["crops"]:
            if args.names and c["name"] not in args.names.split(","):
                continue
            if args.items and f"{key}/{c['name']}" not in args.items.split(","):
                continue
            tasks.append(do_one(client, sem, args, key, c, args.render))
    t0 = time.time()
    await asyncio.gather(*tasks)
    print(f"done {len(tasks)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
