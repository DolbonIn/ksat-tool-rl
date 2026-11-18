"""Shared stems ('[13～14] … 13번과 14번의 두 물음에 답하시오.') in the 2017학년도 교육청 papers: crop the preamble region
(group header .. label of the group's first item, same column), keep its text-layer ref, and transcribe it with DeepSeek
(two pass-A reads, one per model, plus a text-layer-assisted read; best by text-layer score). The build step prepends the
preamble (without the '…번과 …번의 두 물음에 답하시오' sentence) to every item of the group.
Output: RENDER/crops/<key>/pre-<a>-<b>.png, OCR/<key>/pre-<a>-<b>.json, WORK/preambles.json
Usage: python -I preamble_old.py --jobs /data/work3/jobs.json --render /data/render3 --ocr /data/ocr3 --out /data/work3/preambles.json
"""
import argparse, asyncio, json, os, re, sys
sys.path.insert(0, "/pipe"); sys.path.insert(0, "/pipe/multi"); sys.path.insert(0, "/code")
import fitz
from PIL import Image
import gw  # noqa: E402
# model policy (user instruction 2026-11-18): eco only from now on; override with GW_MODELS=a,b
gw.MODELS = os.environ.get("GW_MODELS", "deepseek/deepseek-v4.1-flash-eco").split(",")
from render_crops import find_labels, trim  # noqa: E402
from verify_text import clean_ref  # noqa: E402
from repair_ocr import score  # noqa: E402
from ocr_old import img, tl_text  # noqa: E402

HDR = re.compile(r"\[\s*(\d{1,2})\s*[～~∼\-]\s*(\d{1,2})\s*\]|(\d{1,2})\s*번\s*과\s*(\d{1,2})\s*번")
PROMPT = r"""이 이미지는 수학 문제지에서 여러 문항이 함께 쓰는 공통 발문(예: '[13～14] … 13번과 14번의 두 물음에 답하시오.') 부분입니다. 빠짐없이 원문 그대로 옮기세요.
규칙: 한국어는 원문 그대로, 수식·기호·변수·수식 글꼴 숫자는 LaTeX($...$, 따로 줄을 차지하면 $$...$$)로 적습니다. 조건 상자는 원문 순서대로 줄을 바꿔 적습니다.
그림·그래프가 있으면 <has_figure>yes</has_figure>로 표시하고 <figure> 안에 그림에 실제로 보이는 정보(곡선·직선의 모양, 점의 이름과 위치, 축과 원점, 적힌 값)만 자세히 적습니다. 그림이 없으면 no/없음.
이미지 아래쪽에 다른 문항 번호(예: '13.')로 시작하는 내용이 조금 보이면 무시하세요. 풀이하지 마세요.
출력 형식(이 형식만):
<stem>
공통 발문 전체
</stem>
<has_figure>yes 또는 no</has_figure>
<figure>
그림 설명 또는 없음
</figure>"""


def parse(raw):
    out = {}
    for t in ("stem", "has_figure", "figure"):
        m = re.search(rf"<{t}>(.*?)</{t}>", raw or "", re.S)
        out[t] = m.group(1).strip() if m else None
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--ocr", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--usage-log", default="/data/work3/usage.jsonl")
    a = ap.parse_args()
    gw.init(3, a.usage_log)
    groups = []
    for j in json.load(open(a.jobs)):
        if not j["exam"].startswith("edu") or j["key"].startswith("2021_edu_04"):
            continue
        doc = fitz.open(j["pdf"])
        for pi, page in enumerate(doc):
            W = page.rect.width
            blocks = page.get_text("blocks")
            hits = []
            for b in blocks:
                t = re.sub(r"\s+", " ", b[4])
                if re.search(r"\[\s*\d{1,2}\s*[～~∼\-]", t) or re.search(r"번과\s*\d*\s*번의\s*두\s*물음", t.replace(" ", " ")) or "두 물음에 답하시오" in t:
                    hits.append(b)
            if not hits:
                continue
            labels, _, _ = find_labels(page)
            for col in (0, 1):
                hc = [b for b in hits if (b[0] < W / 2) == (col == 0)]
                if not hc:
                    continue
                y_top = min(b[1] for b in hc)
                nxt = sorted([l for l in labels if l["col"] == col and l["y0"] > y_top], key=lambda l: l["y0"])
                if not nxt:
                    continue
                first = nxt[0]
                txt = " ".join(re.sub(r"\s+", " ", b[4]) for b in hc)
                m = re.search(r"\[\s*(\d{1,2})\s*[～~∼\-]\s*(?:[^\]]*?\s)?(\d{1,2})\s*\]", txt)
                lo, hi = (int(m.group(1)), int(m.group(2))) if m else (first["num"], first["num"] + 1)
                if lo != first["num"]:
                    print("WARN group start mismatch", j["key"], lo, first["num"])
                x0 = first["x0"] - 12; x1 = (W / 2 - 4) if col == 0 else (W - 40)
                rect = fitz.Rect(x0, y_top - 8, x1, first["y0"] - 3)
                pix = page.get_pixmap(dpi=300, clip=rect)
                im = trim(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), pad=30)
                name = f"pre-{lo}-{hi}"
                png = os.path.join(a.render, "crops", j["key"], name + ".png"); im.save(png)
                ref = clean_ref(page.get_text("text", clip=rect))
                groups.append(dict(key=j["key"], track=j["track"], lo=lo, hi=hi, page=pi + 1, rect=[round(v, 1) for v in rect],
                                   png=os.path.relpath(png, a.render), ref=ref))
    print("groups:", [(g["key"], g["lo"], g["hi"]) for g in groups], flush=True)

    async def ocr(g):
        path = os.path.join(a.render, g["png"])
        reqs = [gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": PROMPT}]}], max_tokens=6000, temperature=0.0,
                        thinking=False, tag=f"pre:{g['key']}:{m}", prefer=m, strict=True, retries=20) for m in gw.MODELS]
        reqs.append(gw.chat([{"role": "user", "content": [img(path), {"type": "text", "text": PROMPT + "\n\n아래 [텍스트 레이어]는 이 영역의 PDF 텍스트입니다(수식 기호는 □). 한국어는 이를 그대로 따르세요.\n[텍스트 레이어]\n" + tl_text(g["ref"]) + "\n[/텍스트 레이어]"}]}],
                            max_tokens=6000, temperature=0.0, thinking=False, tag=f"pre:{g['key']}:tl"))
        outs = await asyncio.gather(*reqs)
        cands = {k: parse(o["content"]) for k, o in zip(("eco", "ultrafast", "tl"), outs) if o["finish"] == "stop"}
        sc = {k: score(dict(stem=v["stem"], choices="", figure=v.get("figure")), g["ref"])[0] for k, v in cands.items() if v.get("stem")}
        best = min(sc, key=lambda k: (sc[k], k != "tl"))
        g.update(final=cands[best], chosen=best, scores=sc, raw={k: o["content"] for k, o in zip(("eco", "ultrafast", "tl"), outs)})
        json.dump(g, open(os.path.join(a.ocr, g["key"], f"pre-{g['lo']}-{g['hi']}.json"), "w"), ensure_ascii=False, indent=1)
        print(g["key"], g["lo"], g["hi"], "chosen", best, sc, flush=True)
    await asyncio.gather(*[ocr(g) for g in groups])
    json.dump(groups, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
