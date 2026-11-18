"""Read official 정답표 pages that have no text layer (2025 수능) from their rendered images.

Each image is read by BOTH DeepSeek models (temperature 0, no thinking); a third independent read is done manually
(Claude viewing the same images, recorded in --manual JSON). The answer is accepted only when all reads agree and
points sum correctly; disagreements are reported. Output merges into the answer-keys JSON (method="image_read").

Usage: python -I read_key_images.py --jobs /data/work2/jobs_kice.json --keys /data/work2/answer_keys_kice.json \
        --manual /data/work2/keys_manual_2025.json --raw /data/work2/key_image_reads.json
"""
import argparse, asyncio, base64, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw  # noqa: E402
from parse_keys_multi import check  # noqa: E402

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
P_KOR = ('이 이미지는 국어 영역 정답표입니다. 표의 모든 칸을 읽어 JSON 하나로만 출력하세요. 형식: '
         '{"common": {"1": ["③", 2], ...34까지}, "speech_writing": {"35": ["②", 2], ...45까지}, "lang_media": {"35": [...], ...45까지}}. '
         '값은 [정답 원문자, 배점 숫자]입니다. 화법과 작문과 언어와 매체 열을 구분하세요. 다른 설명은 쓰지 마세요.')
P_ONE = ('이 이미지는 {what} 정답표입니다. 표의 모든 문항(1~{n}번)의 정답과 배점을 읽어 JSON 하나로만 출력하세요. '
         '형식: {{"1": ["③", 2], "2": ["④", 3], ...}}. 값은 [정답 원문자, 배점 숫자]입니다. 다른 설명은 쓰지 마세요.')


def to_items(job, js):
    items = {}
    def put(sec, n, v):
        a = CIRC.get(str(v[0]).strip()); p = int(v[1])
        items[f"{sec}-{int(n)}"] = dict(section=sec, number=int(n), answer=a, points=p)
    if job["area"] == "kor":
        for g, sec in (("common", "kor_common"), ("speech_writing", "kor_speech_writing"), ("lang_media", "kor_lang_media")):
            for n, v in (js.get(g) or {}).items():
                put(sec, n, v)
    else:
        for n, v in js.items():
            put(job["section"], n, v)
    return items


async def read(job, png, model):
    url = "data:image/png;base64," + base64.b64encode(open(png, "rb").read()).decode()
    if job["area"] == "kor":
        prompt = P_KOR
    else:
        what = {"eng": "영어 영역"}.get(job["area"], f"{job.get('answer_subject', '')} 과목")
        prompt = P_ONE.format(what=what, n=45 if job["area"] == "eng" else 20)
    r = await gw.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}],
                      max_tokens=4000, temperature=0.0, thinking=False, tag=f"key:{job['key']}", prefer=model, strict=True, retries=20)
    m = re.search(r"\{.*\}", r["content"], re.S)
    try:
        return to_items(job, json.loads(m.group(0))) if m else {}, r["model"]
    except Exception as e:
        return {}, f"{r['model']} parse error {e}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", required=True)
    ap.add_argument("--manual", default=None); ap.add_argument("--raw", required=True)
    ap.add_argument("--usage-log", default="/data/work2/usage.jsonl")
    a = ap.parse_args()
    gw.init(8, a.usage_log)
    jobs = {j["key"]: j for j in json.load(open(a.jobs))}
    keys = json.load(open(a.keys))
    todo = [k for k, v in keys.items() if "no text layer" in v.get("errors", [])]
    raw = json.load(open(a.raw)) if os.path.exists(a.raw) else {}
    raw = {k: {m: v for m, v in d.items() if v and m in gw.MODELS} for k, d in raw.items()}
    need = [k for k in todo if len(raw.get(k, {})) < 2]
    res = await asyncio.gather(*[read(jobs[k], keys[k]["png"], m) for k in need for m in gw.MODELS])
    for i, k in enumerate(need):
        raw.setdefault(k, {})
        for items, model in (res[2 * i], res[2 * i + 1]):
            if items and model in gw.MODELS:
                raw[k][model] = items
    json.dump(raw, open(a.raw, "w"), ensure_ascii=False, indent=1)
    manual = json.load(open(a.manual)) if a.manual and os.path.exists(a.manual) else {}
    for k in todo:
        reads = list(raw[k].values())
        srcs = list(raw[k].keys())
        if k in manual:
            reads.append(to_items(jobs[k], manual[k])); srcs.append("manual(claude)")
        base = reads[0]
        dis = []
        for e in sorted(set().union(*[set(r) for r in reads])):
            vals = [(r.get(e) or {}).get("answer") for r in reads], [(r.get(e) or {}).get("points") for r in reads]
            if len(set(vals[0])) != 1 or len(set(vals[1])) != 1:
                dis.append((e, vals))
        errs = check(jobs[k], base)
        # >= 3 agreeing reads (2 models + manual), or 2 agreeing reads where one is the independent manual read
        enough = len(reads) >= 3 or (len(reads) == 2 and "manual(claude)" in srcs)
        status = "OK" if not dis and not errs and enough else ("NEED_MANUAL" if not enough else "DISAGREE")
        print(k, status, len(base), errs, dis[:6])
        if status == "OK":
            keys[k].update(items=base, errors=[], method="image_read", reads=srcs)
        else:
            keys[k]["image_read_status"] = dict(status=status, disagreements=dis, errors=errs, reads=srcs)
    json.dump(keys, open(a.keys, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
