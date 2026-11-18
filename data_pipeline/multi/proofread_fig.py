"""Figure/table proofreading pass (DeepSeek with thinking): image + current transcription -> corrected [그림]/[표] blocks.

Only figure/table blocks may change; the corrected transcription is accepted only if its text-layer score (verify_seg)
is not worse than before, so non-figure text cannot be corrupted. Result stored as an attempt with tag "fig".
Usage: python -I proofread_fig.py --render /data/render2 --ocr /data/ocr2 --jobs J --sids key/sid,... [--concurrency 12]
"""
import argparse, asyncio, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw  # noqa: E402
from ocr_assist import img_part, parse, AREA, SUBJ_KO  # noqa: E402
import verify_seg  # noqa: E402

PROMPT = r"""이미지는 대한민국 수능(또는 모의평가) {where} 문항이고, 아래는 그 전사입니다. 전사의 본문 글자는 PDF 텍스트 레이어로 검증되었으므로 그대로 두고,
그림·그래프·지도·도표를 설명한 [그림] … [/그림] 블록과 표 [표] … [/표] 블록만 이미지와 꼼꼼히 대조하세요.
확인할 것: 축 이름·단위·눈금 값, 점·선·막대의 위치와 값(격자·눈금에서 읽히는 값), 곡선의 시작점·끝점·교점·극값의 위치, 레이블과 위치 관계,
화살표 방향, 범례, 색칠·무늬, 그림 속 글자·숫자·기호(첨자 포함), 표의 모든 칸. 그림에 없는 값·해석을 지어낸 부분은 빼고, 문제 풀이에 필요한데 빠진 정보는 보탭니다.
정답을 암시하는 해석은 쓰지 마세요.

출력: 고칠 것이 없으면 <ok/> 한 줄만 출력합니다. 고칠 것이 있으면 <errors>찾은 오류 목록</errors> 다음에 전사 전체를 원래 형식(<number>, <stem>, <choices>, <has_figure>)으로 다시 출력하되, [그림]/[표] 블록 밖의 글자는 한 글자도 바꾸지 마세요.

[전사]
{tx}"""


def fmt(p):
    ch = p.get("choices_list") or []
    return (f"<number>{p.get('number')}</number>\n<stem>\n{p.get('stem')}\n</stem>\n<choices>\n" +
            "\n".join(f"{'①②③④⑤'[i]} {c}" for i, c in enumerate(ch)) + f"\n</choices>\n<has_figure>{p.get('has_figure')}</has_figure>")


async def one(a, job, seg):
    fn = os.path.join(a.ocr, job["key"], seg["sid"] + ".json")
    rec = json.load(open(fn))
    if any(x.get("tag") == "fig" for x in rec["attempts"]) and not a.force:
        return
    cur = rec.get("final_fixed")
    if not cur or seg["kind"] != "question":
        return
    sub = SUBJ_KO.get(seg["section"], "")
    where = f"{AREA[job['area']]} 영역" + (f"({sub})" if sub else "")
    msgs = [{"role": "user", "content": [img_part(os.path.join(a.render, job["key"], p)) for p in seg["pngs"]] +
             [{"type": "text", "text": PROMPT.format(where=where, tx=fmt(cur))}]}]
    r = await gw.chat(msgs, max_tokens=24000, temperature=0.3, thinking=True, tag=f"fig:{job['key']}/{seg['sid']}")
    c = r["content"] or ""
    att = dict(tag="fig", model=r["model"], finish=r["finish"], usage=r["usage"], error=r["error"], content=c,
               temperature=0.3, thinking=True)
    status = "error"
    if r["finish"] == "stop":
        if "<ok/>" in c and "<stem>" not in c:
            status = "no_change"
        else:
            tail = c[c.find("</errors>") + 9:] if "</errors>" in c else c
            p = parse(tail)
            if p.get("stem"):
                s0, _ = verify_seg.compare(seg, cur)
                s1, _ = verify_seg.compare(seg, p)
                e1 = verify_seg.structural(seg, p, job["area"])
                if s1 <= s0 and not e1:
                    att["parsed"] = p; status = "changed"
                else:
                    status = f"rejected (score {s0}->{s1}, {e1})"
    att["status"] = status
    m = re.search(r"<errors>(.*?)</errors>", c, re.S)
    att["errors_found"] = m.group(1).strip()[:2000] if m else None
    rec["attempts"].append(att)
    json.dump(rec, open(fn, "w"), ensure_ascii=False, indent=1)
    print(job["key"], seg["sid"], status, r["usage"], flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True); ap.add_argument("--ocr", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--sids", required=True); ap.add_argument("--concurrency", type=int, default=12); ap.add_argument("--force", action="store_true")
    ap.add_argument("--usage-log", default="/data/work2/usage.jsonl")
    a = ap.parse_args()
    gw.init(a.concurrency, a.usage_log)
    jobs = {j["key"]: j for f in a.jobs.split(",") for j in json.load(open(f))}
    sids = open(a.sids).read().split(",") if os.path.exists(a.sids) else a.sids.split(",")
    segs = {}
    tasks = []
    for ks in sids:
        key, sid = ks.split("/")
        if key not in segs:
            segs[key] = {s["sid"]: s for s in json.load(open(os.path.join(a.render, key, "segments.json")))["segments"]}
        tasks.append(one(a, jobs[key], segs[key][sid]))
    t0 = time.time()
    await asyncio.gather(*tasks)
    print(f"done {len(tasks)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
